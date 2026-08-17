"""Class order as a gauge fix: spending at most ``d_out - 1`` free directions.

The unsupervised objective fixes the layout's shape but not its orientation.
:func:`~leanmap.losses.fuzzy_cross_entropy` sees only distances,
:func:`~leanmap.losses.local_rigidity_loss` is explicitly rotation- and
reflection-invariant, and :func:`~leanmap.losses.procrustes_anchor_loss` aligns
its own target before measuring the residual. Rotation and reflection are
therefore free parameters that the data never constrains, and successive refits
spend them arbitrarily -- which is why a reader cannot be told "left means
earlier" about a map produced this way.

A user ordering of the class labels is the natural thing to spend that freedom
on. Asking one coordinate to respect the order of the classes fixes one
direction and its sign, and costs nothing that the data had an opinion about.
In ``d`` dimensions at most ``d - 1`` coordinates may be spent this way: the
remainder has to stay free, or the labels have stopped choosing among
equivalent layouts and started dictating the layout, at which point the map can
no longer disagree with them and the class structure it shows is a restatement
of the input rather than a finding. :func:`validate_class_axes` enforces that
ceiling rather than warning about it.

Two strengths of request, because not every ordering deserves an axis
-------------------------------------------------------------------

A *pinned* axis (``ClassAxis(axis=j, ...)``) names the coordinate and thereby
fixes both a direction and its sign. Spend this on the one ordering a reader
will navigate by -- the thing that makes "further right means later" true.

A *free-direction* axis (``ClassAxis(axis=None, ...)``, see
:func:`class_direction_loss`) asks only that its groups come apart along *some*
direction, recomputed each step and oriented low-to-high, so neither the
direction nor the sign is constrained. This is the honest form for a secondary
or coarse factor: you want even and odd to be tellable apart in the map, but you
have no basis for claiming which way round the map should lay them out, and
claiming one anyway is friction you get nothing for.

Its angle to the pinned axes is discovered too, for the same reason -- two
orderings of the same labels are under no obligation to be square to each other,
and insisting on it would decide an answer the data is entitled to give.
:func:`direction_tilt` reports the lean, in degrees, as ``tilt_<name>``: near
``0`` the two orderings are independent in these features, and near ``90`` the
secondary one has been absorbed into a pinned coordinate and is not a separate
direction at all. Passing ``orthogonal=True`` holds the tilt at zero and buys
back a guarantee -- the term then cannot move a pinned coordinate however hard it
is weighted -- which is worth having when the pinned ordering is the deliverable,
and is the wrong default otherwise.

Per-axis :attr:`ClassAxis.weight` scales ``lambda_class``, since a secondary
factor generally wants a fraction of the force the primary one gets.

Three properties hold the friction at that minimum.

**Order only.** The term constrains the *sign* of ``z[hi, axis] - z[lo, axis]``
for class pairs the user ordered, and nothing else -- not how far apart the
classes sit, not where along the axis any of them lands. Spacing and internal
structure stay entirely a property of the neighbour graph. Only the order of
:attr:`ClassAxis.rank` is read, so any monotone relabelling of it is the same
constraint, and equal ranks mean "these two are not ordered relative to each
other", which is how a partial order is expressed.

**Zero force once satisfied.** A hinge, not a pull: pairs already ordered past a
small margin contribute exactly zero gradient, not a small one. This is the one
place where following the neighbouring :func:`~leanmap.losses.ordinal_triplet_loss`
would be wrong -- its ``logsigmoid`` form never reaches zero because it is a
ranking objective that is meant to keep pressing. A gauge fix should stop.

**Scale free.** The margin is a fraction of the constrained coordinate's own
running spread, so the constraint cannot be satisfied by inflating the layout
and never argues with the attraction/repulsion equilibrium about size.

Labels stay out of the graph. Nothing in this module touches the metric, the
kNN, the ``epsilon``-net or the fuzzy memberships, so what "neighbour" means is
unchanged and the result is an honest feature embedding that happens to be
oriented. The price of that restraint is that the ordering may be
unsatisfiable: if the features do not separate the classes in the requested
order, the hinge stays active and the layout does not comply.
:func:`class_axis_report` measures exactly that residual and it is meant to be
reported, not tuned away -- an ordering accuracy near ``0.5`` says the features
carry no such order, which is a result about the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_logger

# Fraction of the constrained coordinate's spread that a correctly ordered pair
# must clear before the hinge releases. Small enough to be a robustness margin
# rather than a spacing target, large enough that "satisfied" is not a knife
# edge at exactly zero gap (where the hinge would have no gradient and pairs
# could sit on top of each other and still score zero).
CLASS_MARGIN: float = 0.05

# EMA horizon for the coordinate spread that makes the margin scale free,
# matching the running mean in ``ordinal_triplet_loss``.
SPREAD_MOMENTUM: float = 0.9

# EMA horizon for a free-direction axis's direction, slower than the spread
# because an unsmoothed direction does not merely add variance, it stalls the
# term outright: before the groups separate, the per-step estimate is noise, so
# its sign flips from step to step and successive pushes cancel. A ~20-step
# memory lets whatever asymmetry exists at initialisation reinforce itself, after
# which the estimate agrees with the layout and the EMA only tracks it.
DIRECTION_MOMENTUM: float = 0.95

# Ridge on the within-group covariance before inverting it, as a fraction of its
# mean diagonal. The original reason for preferring a raw difference of means was
# that a covariance inverse is the fragile part of a step estimated from a few
# hundred pairs -- true in the ambient dimension, but this covariance is
# ``d_out x d_out`` with ``d_out`` a handful, so a relative ridge is enough to
# make it safe and the correction it buys is not optional (see
# :func:`free_direction`).
DIRECTION_RIDGE: float = 1e-3

# Pairs drawn per step. The constraint is a gauge fix over a handful of
# directions, not a per-point objective, so it needs far fewer samples than the
# edge batch; this keeps the extra forward pass a few percent of a step.
CLASS_PAIRS_PER_STEP: int = 256

# Ordering accuracy below which the requested order is not present in the
# layout. 0.5 is exact chance for a pairwise AUC, so unlike ``retention_f``
# there is nothing to estimate empirically here.
ORDER_CHANCE: float = 0.5
ORDER_WARN: float = 0.6


@dataclass(frozen=True)
class ClassAxis:
    """A user ordering of the classes, on a named coordinate or a free direction.

    Parameters
    ----------
    axis : int or None
        Which coordinate of ``z`` carries the order. ``None`` asks for the much
        weaker constraint: the groups must be ordered along *some* direction,
        which the fit chooses (see :func:`class_direction_loss`). Pinned axes are
        capped at ``d_out - 1``; a free direction names no coordinate and is
        counted separately.
    rank : (K,) float tensor
        The user's ordering value for class label ``k``, indexed by label, where
        labels are integer codes ``0..K-1``. Only the order of these values is
        used and never their spacing, so ``(0, 1, 2)`` and ``(0, 10, 11)`` are
        the same constraint. Equal values leave that pair of classes mutually
        unconstrained, which is how a coarse or partial order is expressed.
    weight : float
        Multiplier on ``config.lambda_class`` for this axis alone. A secondary
        factor usually wants less force than the primary one, and one weight for
        all axes would make that inexpressible.
    orthogonal : bool
        For a free-direction axis, whether to *force* the direction square to the
        pinned axes instead of letting the fit discover its angle. The default
        discovers it: an ordering may genuinely be best expressed along a
        direction that leans into the pinned one, and forbidding that is an
        imposition of the same kind the mechanism exists to avoid. The cost is
        that the term can then push a pinned coordinate, so read ``tilt_<name>``
        to see how far it leans. Setting this to ``True`` restores the guarantee
        that the term cannot move a pinned axis at all, at the price of deciding
        the answer in advance.
    whiten : bool
        For a free-direction axis, whether to take the separating direction in the
        metric of the within-group covariance instead of the Euclidean one. On by
        default and rarely worth turning off: without it the direction leans
        toward whichever coordinate the other loss terms have stretched, which is
        the pinned one by construction. ``False`` recovers a plain difference of
        means, which is useful mainly for seeing the artefact.
    name : str
        Label for diagnostics and logging.
    """

    axis: Optional[int]
    rank: torch.Tensor
    weight: float = 1.0
    orthogonal: bool = False
    whiten: bool = True
    name: str = "class"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rank", torch.as_tensor(self.rank, dtype=torch.float32).reshape(-1)
        )

    @property
    def is_pinned(self) -> bool:
        return self.axis is not None


def ordinal_class_axis(
    n_classes: int,
    axis: Optional[int] = 0,
    order: Optional[Sequence[int]] = None,
    name: str = "class",
    weight: float = 1.0,
    orthogonal: bool = False,
    whiten: bool = True,
) -> ClassAxis:
    """A totally ordered chain of classes, on a named coordinate or a free direction.

    ``order`` lists label codes from low to high; the default is ``0..K-1``, i.e.
    the label codes already are the order. This is the common case -- a stage, a
    severity, a reading order -- where the classes come with a sequence attached.

    One ordering is one direction, however many classes it has. ``axis=None`` lets
    the fit discover that direction; ``axis=j`` pins it to coordinate ``j``. The
    default pins to ``z0`` for back-compatibility; pass ``axis=None`` when the
    request is "preserve this order along some projection", which is the usual one.
    """
    if order is None:
        rank = torch.arange(n_classes, dtype=torch.float32)
    else:
        if sorted(int(o) for o in order) != list(range(n_classes)):
            raise ValueError(
                f"order must be a permutation of 0..{n_classes - 1}, got {list(order)}"
            )
        rank = torch.empty(n_classes, dtype=torch.float32)
        for position, label in enumerate(order):
            rank[int(label)] = float(position)
    return ClassAxis(
        axis=axis,
        rank=rank,
        name=name,
        weight=weight,
        orthogonal=orthogonal,
        whiten=whiten,
    )


def grouped_class_axis(
    groups: Sequence[Sequence[int]],
    axis: Optional[int] = 0,
    n_classes: Optional[int] = None,
    name: str = "group",
    weight: float = 1.0,
    orthogonal: bool = False,
    whiten: bool = True,
) -> ClassAxis:
    """An ordered sequence of *groups* of classes on one coordinate.

    ``groups`` lists label codes bucketed low to high along the axis; labels
    inside a bucket share a rank and are therefore never ordered against each
    other. Use this whenever the thing you want ordered is coarser than the
    label -- a parity, a treatment arm, a broad stage that several labels belong
    to -- which is the common secondary axis.

    ``axis=None`` is usually what a coarse secondary factor wants: the groups get
    ordered along a direction the fit picks rather than one you name, which is a
    weaker request and costs the layout correspondingly less. Its angle to the
    pinned axes is discovered as well; pass ``orthogonal=True`` to force it
    square to them.

    >>> grouped_class_axis([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], axis=None, name="parity")
    ...  # doctest: +SKIP

    Two labels in the same bucket cost nothing: tied ranks are skipped by
    :class:`ClassOrderSampler`, so a coarse axis constrains only the between-group
    order and leaves each group's internal arrangement to the graph.
    """
    flat = [int(c) for g in groups for c in g]
    K = int(n_classes) if n_classes is not None else (max(flat) + 1 if flat else 0)
    if sorted(flat) != list(range(K)):
        raise ValueError(
            f"groups must partition the labels 0..{K - 1} exactly once each; got "
            f"{sorted(flat)}"
        )
    if len(groups) < 2:
        raise ValueError("need at least two groups for an order to exist")
    rank = torch.empty(K, dtype=torch.float32)
    for position, g in enumerate(groups):
        for c in g:
            rank[int(c)] = float(position)
    return ClassAxis(
        axis=axis,
        rank=rank,
        name=name,
        weight=weight,
        orthogonal=orthogonal,
        whiten=whiten,
    )


def validate_class_axes(
    axes: Sequence[ClassAxis], d_out: int, n_classes: int
) -> None:
    """Enforce the ``d_out - 1`` ceiling and the shape/range contract.

    The ceiling on *pinned* axes is the whole point of the mechanism and is
    raised, not warned: naming every coordinate would leave the layout no
    direction of its own to organise, so the classes would be arranging the map
    rather than orienting one the data produced.

    Free-direction axes (``axis=None``) are counted separately because they ask
    for less. They fix no coordinate and no sign -- only that the groups come
    apart along some direction the fit gets to choose -- so one of them does not
    use up a named coordinate the way a pinned axis does. The total is still
    capped at ``d_out``, and hitting that cap is warned about rather than
    refused: it is a weaker request than pinning every axis, and it is one a user
    can legitimately want in ``d_out=2``, where the complement of the pinned
    coordinate is one dimensional and "some direction" can only mean ``+-z1``.
    """
    if not axes:
        return
    pinned = [ax for ax in axes if ax.is_pinned]
    free = [ax for ax in axes if not ax.is_pinned]
    n_left = d_out - len(pinned)
    if n_left < 1:
        raise ValueError(
            f"{len(pinned)} pinned axes in d_out={d_out} leaves {n_left} unnamed "
            "coordinates; at most d_out - 1 axes may name a coordinate so the layout "
            "keeps a direction it organises itself. Drop an axis, raise d_out, or "
            "pass axis=None to ask for an ordering along a direction of the fit's "
            "choosing instead."
        )
    if len(pinned) + len(free) > d_out:
        raise ValueError(
            f"{len(pinned)} pinned and {len(free)} free-direction axes exceed "
            f"d_out={d_out}; there are not that many independent directions to order "
            "along."
        )
    if free and len(pinned) + len(free) == d_out:
        get_logger().warning(
            "%d pinned + %d free-direction axes fills all %d dimensions, so no "
            "direction is left entirely to the data. The free-direction axes still "
            "choose their own direction and sign, which is why this is allowed, but "
            "read order_%s alongside a lambda_class=0 baseline before believing the "
            "layout.",
            len(pinned),
            len(free),
            d_out,
            free[0].name,
        )
    seen: Dict[int, str] = {}
    for ax in axes:
        if ax.is_pinned:
            if not 0 <= ax.axis < d_out:
                raise ValueError(
                    f"class axis {ax.name!r} targets coordinate {ax.axis}, outside "
                    f"[0, {d_out})"
                )
            if ax.axis in seen:
                raise ValueError(
                    f"coordinate {ax.axis} is claimed by both {seen[ax.axis]!r} and "
                    f"{ax.name!r}; two orderings on one axis are contradictory unless "
                    "they agree, in which case one of them is redundant"
                )
            seen[ax.axis] = ax.name
        if not ax.weight > 0.0:
            raise ValueError(
                f"class axis {ax.name!r} has weight={ax.weight}, which switches it "
                "off silently; omit the axis instead"
            )
        if ax.rank.numel() != n_classes:
            raise ValueError(
                f"class axis {ax.name!r} has rank of length {ax.rank.numel()} but "
                f"there are {n_classes} classes"
            )
        if not torch.isfinite(ax.rank).all():
            raise ValueError(f"class axis {ax.name!r} has non-finite ranks")
        if float(ax.rank.max() - ax.rank.min()) == 0.0:
            raise ValueError(
                f"class axis {ax.name!r} gives every class the same rank, which "
                "constrains nothing; omit the axis instead"
            )


class ClassOrderSampler:
    """Draws point pairs whose classes the user placed in a definite order.

    Sampling is uniform over *ordered class pairs* and then uniform within each
    class, which deliberately ignores how many points each class has: the
    constraint is about the arrangement of the classes, so letting a large class
    dominate the batch would weight the gauge by sample size. Pairs of classes
    with equal rank are never drawn, so a partial order costs nothing to express.

    Uniform over ordered pairs is also what makes the term self-limiting.
    Well-separated ranks are satisfied early and drop out of the hinge, so the
    surviving gradient concentrates on adjacent classes -- the only pairs where
    the requested order is genuinely in question -- without any scheduling.
    """

    def __init__(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        rank: torch.Tensor,
        seed: int = 0,
    ):
        self.X = X
        self.rng = np.random.default_rng(seed + 7)
        lab = labels.detach().cpu().numpy().astype(np.int64)
        rnk = rank.detach().cpu().numpy().astype(np.float64)
        K = int(rnk.shape[0])

        order = np.argsort(lab, kind="stable")
        counts = np.bincount(lab, minlength=K)
        offsets = np.zeros(K + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self._flat = order
        self._offset = offsets[:K]
        self._count = counts

        lo: List[int] = []
        hi: List[int] = []
        for a in range(K):
            if counts[a] == 0:
                continue
            for b in range(K):
                if counts[b] == 0 or rnk[a] >= rnk[b]:
                    continue
                lo.append(a)
                hi.append(b)
        if not lo:
            raise ValueError(
                "no ordered class pair is present in the training split: every "
                "populated class has the same rank, or only one class occurs"
            )
        self._pair_lo = np.asarray(lo, dtype=np.int64)
        self._pair_hi = np.asarray(hi, dtype=np.int64)
        self.n_pairs = int(self._pair_lo.shape[0])

    def _member(self, cls: np.ndarray) -> np.ndarray:
        within = (self.rng.random(cls.shape[0]) * self._count[cls]).astype(np.int64)
        return self._flat[self._offset[cls] + within]

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_lo, x_hi)``, each ``(n, D)``, with ``rank(lo) < rank(hi)``."""
        pick = self.rng.integers(0, self.n_pairs, size=int(n))
        i_lo = self._member(self._pair_lo[pick])
        i_hi = self._member(self._pair_hi[pick])
        return (
            self.X[torch.as_tensor(i_lo, dtype=torch.int64)],
            self.X[torch.as_tensor(i_hi, dtype=torch.int64)],
        )


def class_order_loss(
    z_lo: torch.Tensor,
    z_hi: torch.Tensor,
    axis: int,
    spread_state: Optional[Dict[str, float]] = None,
    margin: float = CLASS_MARGIN,
) -> Tuple[torch.Tensor, Dict[str, float], float]:
    """Hinge on the sign of the gap along one coordinate; zero once ordered.

    ``z_lo`` and ``z_hi`` embed point pairs whose classes the user placed in
    that order, so ``z_hi[:, axis] - z_lo[:, axis]`` should be positive. The gap
    is divided by a detached running spread of the coordinate before the hinge,
    which makes the margin a fraction of the layout's own extent rather than an
    absolute distance -- otherwise the term would be satisfiable by growing the
    map, and its strength would depend on the units the embedding happened to
    settle into.

    Only ``z[:, axis]`` enters, so ``dL/dz`` is exactly zero on every other
    coordinate: the term expresses no preference at all about the free
    directions. (Parameter gradients still mix, since the encoder is shared --
    the guarantee is about what the objective asks for, not about isolating the
    network.)

    Returns
    -------
    loss : scalar
    spread_state : updated running scale
    active_frac : fraction of pairs still inside the margin, i.e. the residual
        friction this step. Zero means the gauge is satisfied and the term is
        contributing nothing.
    """
    if spread_state is None:
        spread_state = {}
    if z_lo.shape[0] == 0:
        return z_lo.sum() * 0.0, spread_state, 0.0

    u_lo = z_lo[:, axis]
    u_hi = z_hi[:, axis]
    gap = u_hi - u_lo
    with torch.no_grad():
        batch_spread = float(torch.cat([u_lo, u_hi]).std().item())
        if not np.isfinite(batch_spread) or batch_spread <= 0.0:
            batch_spread = 1.0
        prev = spread_state.get("spread")
        spread_state["spread"] = (
            batch_spread
            if prev is None
            else SPREAD_MOMENTUM * prev + (1.0 - SPREAD_MOMENTUM) * batch_spread
        )
    s = max(spread_state["spread"], 1e-6)
    short = F.relu(margin - gap / s)
    active = float((short > 0).float().mean().item())
    return short.mean(), spread_state, active


def _pinned_mask(d_out: int, pinned_axes: Sequence[int]) -> torch.Tensor:
    keep = torch.ones(d_out, dtype=torch.bool)
    for a in pinned_axes:
        keep[int(a)] = False
    return keep


def free_direction(
    z_lo: torch.Tensor,
    z_hi: torch.Tensor,
    pinned_axes: Sequence[int] = (),
    orthogonal: bool = False,
    whiten: bool = True,
) -> torch.Tensor:
    """The direction along which the ordered groups currently separate best.

    ``Sigma_w^-1 (mu_hi - mu_lo)``, normalised: the difference of the group means
    taken in the metric of the *within-group* covariance rather than the Euclidean
    one. For a two-group factor this is Fisher's discriminant, and the
    ``Sigma_w^-1`` is not a refinement but a correction for the frame the
    difference is measured in.

    Why it matters here specifically. A raw difference of means is in the units of
    the embedding, so it over-weights whichever coordinate happens to be stretched
    -- and with a pinned ordering present, the stretched coordinate is *always* the
    pinned one, because stretching it is what ordering it does. On digits the
    within-group spread runs about ``[1.33, 0.68]``, twice as wide along the pinned
    axis, and the unwhitened estimate leans 11 degrees into it where the whitened
    one leans 4.8 and with the opposite sign. That lean is an artefact of the
    metric, and following it makes the term fight the pinned ordering for nothing.

    Whitening does not fix a second, unrelated confound: a coarse grouping can be
    arithmetically correlated with the pinned ordering, as even/odd is with digit
    value (mean digit 4 against 5), so *some* of the separation genuinely lies
    along the pinned axis and is already accounted for there. Only
    ``orthogonal=True`` removes that, by residualising it away entirely.

    By default the angle to the pinned axes is discovered: whatever direction
    separates the groups is used, including one that leans into a pinned
    coordinate, because an ordering has no obligation to be square to another
    one. :func:`class_axis_report` reports that lean as ``tilt_<name>``.

    ``orthogonal=True`` instead works wholly inside the unpinned coordinates --
    whitening by the conditional covariance there, not the marginal one -- which
    forces the lean to zero and buys a guarantee: the hinge then sees only
    ``z @ u`` for a ``u`` that is exactly zero on every pinned coordinate, so
    ``dL/dz`` vanishes there and the term cannot push a named axis however hard it
    is weighted.

    Degenerate case: if the group means coincide there is no separating direction
    to find, and a coordinate is used instead -- the first unpinned one, since
    reaching for a pinned one would put the term straight into a fight it has no
    reason to pick. Any direction is equally wrong at that point, and an
    arbitrary one at least produces a gradient that starts pulling the groups
    apart.
    """
    d = z_lo.shape[1]
    keep = _pinned_mask(d, pinned_axes).to(z_lo.device)
    delta = z_hi.mean(dim=0) - z_lo.mean(dim=0)
    idx = torch.nonzero(keep).reshape(-1) if orthogonal else torch.arange(
        d, device=z_lo.device
    )

    u = torch.zeros_like(delta)
    sub = delta[idx]
    if whiten and idx.numel() > 1:
        centred = torch.cat(
            [z_lo - z_lo.mean(dim=0, keepdim=True), z_hi - z_hi.mean(dim=0, keepdim=True)]
        )[:, idx]
        n_obs = centred.shape[0]
        S = (centred.T @ centred) / max(n_obs - 1, 1)
        ridge = DIRECTION_RIDGE * torch.diagonal(S).mean().clamp_min(1e-12)
        S = S + ridge * torch.eye(idx.numel(), device=S.device, dtype=S.dtype)
        try:
            sub = torch.linalg.solve(S, sub)
        except Exception:
            pass  # keep the Euclidean direction rather than fail a step
    u[idx] = sub

    n = torch.linalg.vector_norm(u)
    if not torch.isfinite(n) or float(n) < 1e-8:
        u = torch.zeros_like(u)
        u[int(torch.nonzero(keep)[0])] = 1.0
        return u
    return u / n


def class_direction_loss(
    z_lo: torch.Tensor,
    z_hi: torch.Tensor,
    pinned_axes: Sequence[int] = (),
    state: Optional[Dict[str, Any]] = None,
    margin: float = CLASS_MARGIN,
    orthogonal: bool = False,
    whiten: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any], float, torch.Tensor]:
    """:func:`class_order_loss` with the direction chosen by the fit, not the user.

    Same hinge, applied to ``z @ u`` for a ``u`` recomputed each step under
    ``no_grad`` as the currently best separating direction. This is the house
    pattern for "measure a residual the layout is allowed to choose the frame
    for": :func:`~leanmap.losses.procrustes_anchor_loss` solves for the optimal
    rotation before scoring, and :func:`~leanmap.losses.local_isometry_loss`
    fits a scale the same way. Solving for the nuisance parameter and freezing it
    means the gradient asks only about the thing being constrained.

    What is being asked for is weaker than an axis, and deliberately so. Because
    ``u`` is re-chosen and oriented low-to-high every step, a layout that
    separates the groups the "wrong way round" is already satisfied -- the sign
    was never constrained. So this term requests *separation with a consistent
    order along one direction*, and stays silent about which direction that is.
    That is the honest form of a secondary factor: you want even and odd to be
    tellable apart in the map, and you have no basis for claiming which way the
    map should lay them out.

    Cost in degrees of freedom is correspondingly one, not one *coordinate*: in
    ``d_out=3`` with ``z0`` pinned, the term is satisfiable by any of a whole
    plane's worth of arrangements, and the geometry keeps a genuinely free
    direction inside that plane.

    The angle to the pinned axes is discovered rather than assumed. That means
    this term *can* move a pinned coordinate, in proportion to how far its
    direction leans into one, and ``tilt_<name>`` in
    :func:`class_axis_report` is how you see that happening. A large tilt is
    informative rather than alarming: it says the two orderings are not
    independent in these features, which is a fact about the data. If the pinned
    ordering is the deliverable and must be protected regardless, pass
    ``orthogonal=True`` and the tilt is held at zero by construction.

    The direction is smoothed across steps (:data:`DIRECTION_MOMENTUM`) rather
    than taken fresh, which is not a variance nicety but load bearing. Before the
    groups separate there is no direction to find, the per-step estimate is noise,
    and its sign flips between steps so the pushes cancel and the term never gets
    started; a run with a fresh estimate every step sits at chance indefinitely.
    Smoothing lets the initial asymmetry reinforce itself into a real separation.

    Returns
    -------
    loss, state, active_frac : as :func:`class_order_loss`
    u : the smoothed direction used this step, for logging
    """
    if state is None:
        state = {}
    if z_lo.shape[0] == 0:
        d = z_lo.shape[1] if z_lo.dim() > 1 else 1
        return z_lo.sum() * 0.0, state, 0.0, torch.zeros(d)

    with torch.no_grad():
        u_batch = free_direction(
            z_lo, z_hi, pinned_axes, orthogonal=orthogonal, whiten=whiten
        )
        prev_u = state.get("u")
        if prev_u is None:
            u = u_batch
        else:
            u = DIRECTION_MOMENTUM * prev_u.to(u_batch) + (
                1.0 - DIRECTION_MOMENTUM
            ) * u_batch
            n = torch.linalg.vector_norm(u)
            u = u / n if float(n) > 1e-8 else u_batch
        state["u"] = u.detach().cpu()
    p_lo = z_lo @ u
    p_hi = z_hi @ u
    gap = p_hi - p_lo
    with torch.no_grad():
        batch_spread = float(torch.cat([p_lo, p_hi]).std().item())
        if not np.isfinite(batch_spread) or batch_spread <= 0.0:
            batch_spread = 1.0
        prev = state.get("spread")
        state["spread"] = (
            batch_spread
            if prev is None
            else SPREAD_MOMENTUM * prev + (1.0 - SPREAD_MOMENTUM) * batch_spread
        )
    s = max(state["spread"], 1e-6)
    short = F.relu(margin - gap / s)
    active = float((short > 0).float().mean().item())
    return short.mean(), state, active, u.detach().cpu()


def _auc(lo: np.ndarray, hi: np.ndarray) -> float:
    """P(hi > lo) with ties at one half, via the rank-sum identity."""
    n_lo, n_hi = lo.shape[0], hi.shape[0]
    if n_lo == 0 or n_hi == 0:
        return float("nan")
    both = np.concatenate([lo, hi])
    order = np.argsort(both, kind="stable")
    ranks = np.empty(both.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, both.shape[0] + 1, dtype=np.float64)
    # Average ranks within ties so exact ties score 0.5 rather than by position.
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(counts.shape[0], dtype=np.float64)
    np.add.at(tie_mean, inv, ranks)
    tie_mean /= counts
    ranks = tie_mean[inv]
    r_hi = ranks[n_lo:].sum()
    return float((r_hi - n_hi * (n_hi + 1) / 2.0) / (n_lo * n_hi))


def direction_tilt(u: np.ndarray, pinned: Sequence[int]) -> float:
    """Degrees by which ``u`` leans into the pinned subspace, in ``[0, 90]``.

    ``arcsin`` of the norm of the pinned components: ``0`` means square to every
    pinned axis (what ``orthogonal=True`` forces), ``90`` means lying entirely
    inside them, i.e. the secondary ordering has been absorbed into the primary
    one's coordinates and is no longer a separate direction at all. With exactly
    one pinned axis, the angle to that axis is ``90 - tilt``.
    """
    if not len(pinned):
        return 0.0
    lean = float(np.linalg.norm(np.asarray(u, dtype=np.float64)[list(pinned)]))
    return float(np.degrees(np.arcsin(min(1.0, max(0.0, lean)))))


def _report_direction(
    z: np.ndarray,
    lab: np.ndarray,
    rnk: np.ndarray,
    pinned: Sequence[int],
    orthogonal: bool = False,
    whiten: bool = True,
) -> np.ndarray:
    """Rank-weighted combination of group means, whitened by the within-group
    covariance -- the batch-free form of :func:`free_direction`.

    Each *group* of tied ranks contributes its mean with equal weight regardless
    of size, matching the sampler's uniform-over-ordered-pairs convention, so a
    populous class cannot tilt the direction. Reduces to Fisher's discriminant
    for two groups.
    """
    present = np.unique(lab)
    groups: Dict[float, List[np.ndarray]] = {}
    members: Dict[float, List[np.ndarray]] = {}
    for c in present:
        r = float(rnk[int(c)])
        groups.setdefault(r, []).append(z[lab == c].mean(axis=0))
        members.setdefault(r, []).append(z[lab == c])
    ranks = np.asarray(sorted(groups), dtype=np.float64)
    means = np.stack([np.mean(groups[r], axis=0) for r in ranks])
    w = ranks - ranks.mean()
    # np.dot, not @: numpy 2.2's matmul emits spurious divide-by-zero warnings on
    # the 2-D-by-1-D gemv path, which np.dot does not take.
    d = np.dot(w, means)

    free = [j for j in range(z.shape[1]) if j not in set(pinned)]
    idx = free if orthogonal else list(range(z.shape[1]))
    if whiten and len(idx) > 1:
        blocks = [
            np.concatenate(members[r])[:, idx]
            - np.concatenate(members[r])[:, idx].mean(axis=0)
            for r in ranks
        ]
        centred = np.concatenate(blocks)
        S = np.cov(centred, rowvar=False)
        S = S + DIRECTION_RIDGE * max(float(np.mean(np.diag(S))), 1e-12) * np.eye(
            len(idx)
        )
        try:
            sub = np.linalg.solve(S, d[idx])
        except np.linalg.LinAlgError:
            sub = d[idx]
        d = np.zeros_like(d)
        d[idx] = sub
    elif orthogonal:
        d[list(pinned)] = 0.0

    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n < 1e-12:
        d = np.zeros_like(d)
        d[free[0]] = 1.0
        return d
    return d / n


def class_axis_report(
    Z: torch.Tensor,
    labels: torch.Tensor,
    axes: Sequence[ClassAxis],
) -> Dict[str, float]:
    """Residual friction per constrained axis: how much of the order took.

    For every class pair the user ordered, the fraction of cross-class point
    pairs that came out in the requested order along the axis -- a pairwise AUC,
    averaged over ordered class pairs with equal weight per pair so a large
    class cannot carry the score. ``0.5`` is exact chance, ``1.0`` is a fully
    ordered layout.

    Reported alongside it is ``adjacent``, the same quantity restricted to class
    pairs that are consecutive in the user's ranking. That is the number worth
    reading: well-separated classes order themselves almost incidentally once
    they separate at all, so the overall mean is optimistic about whether the
    *sequence* was reproduced. A run with ``order=0.95`` and
    ``order_adjacent=0.55`` has classes grouped but their sequence essentially
    unresolved.

    For a free-direction axis the score is read along the direction that best
    separates its groups in the layout being reported, since that is what the
    term asked for. The chosen direction is reported as ``dir_<name>_<j>`` so the
    arrangement the fit settled on is visible rather than implicit, and
    ``tilt_<name>`` gives the angle in degrees by which it leans into the pinned
    axes (see :func:`direction_tilt`): ``0`` is square to them, ``90`` means the
    ordering has been absorbed into a pinned coordinate and is no longer a
    direction of its own. With one pinned axis the angle to it is ``90 - tilt``.

    One caveat that does not apply to the pinned case: because that direction is
    chosen on the same points being scored, chance is *above* ``ORDER_CHANCE``.
    Measured on random layouts it sits near 0.53 at ``n=600, d_out=2`` and 0.54 at
    ``d_out=3``, falling to about 0.515 by ``n=3000`` -- it grows with the number
    of free directions to search and shrinks with sample size, as a fitted
    nuisance parameter does. A pinned axis has nothing to fit and lands on 0.500.
    So for a free-direction axis do not read 0.5 as the null: get the null from a
    shuffled-label refit, which is worth doing anyway and is what
    ``examples/digits_class_axis.py`` reports.
    """
    out: Dict[str, float] = {}
    z = np.ascontiguousarray(Z.detach().cpu().numpy(), dtype=np.float64)
    lab = labels.detach().cpu().numpy().astype(np.int64)
    pinned = [ax.axis for ax in axes if ax.is_pinned]
    for ax in axes:
        rnk = ax.rank.detach().cpu().numpy().astype(np.float64)
        if ax.is_pinned:
            u = z[:, ax.axis]
        else:
            direction = _report_direction(
                z, lab, rnk, pinned, orthogonal=ax.orthogonal, whiten=ax.whiten
            )
            u = np.dot(z, direction)
            for j, v in enumerate(direction):
                out[f"dir_{ax.name}_{j}"] = float(v)
            out[f"tilt_{ax.name}"] = direction_tilt(direction, pinned)
        by_class = {int(c): u[lab == c] for c in np.unique(lab)}
        present = sorted(by_class)
        all_scores: List[float] = []
        adj_scores: List[float] = []
        uniq_rank = sorted({float(rnk[c]) for c in present})
        next_rank = {
            r: uniq_rank[i + 1] for i, r in enumerate(uniq_rank[:-1])
        }
        for a in present:
            for b in present:
                if rnk[a] >= rnk[b]:
                    continue
                score = _auc(by_class[a], by_class[b])
                if not np.isfinite(score):
                    continue
                all_scores.append(score)
                if next_rank.get(float(rnk[a])) == float(rnk[b]):
                    adj_scores.append(score)
        out[f"order_{ax.name}"] = float(np.mean(all_scores)) if all_scores else float("nan")
        out[f"order_adjacent_{ax.name}"] = (
            float(np.mean(adj_scores)) if adj_scores else float("nan")
        )
    return out


@dataclass
class AxisLoadings:
    """What each requested ordering means in terms of the input features.

    See :func:`axis_loadings`.

    Attributes
    ----------
    loading : dict of str to (D,) ndarray
        Per axis, the change in position along that axis per one standard
        deviation of each input feature. Signed: positive entries push toward the
        high-rank end of the ordering.
    stability : dict of str to float
        Per axis, how much the loading direction rotates across the data, as the
        mean resultant length of the per-point unit loadings. ``1.0`` means every
        point agrees and one loading describes the whole map; low values mean the
        mean is a summary of disagreeing evidence rather than a description of
        anything. Read this before reading ``loading``.
    direction : dict of str to (d_out,) ndarray
        The embedding-space direction each loading was taken along.
    feature_std : (D,) ndarray
        Training standard deviation of each input feature, for masking: features
        that never varied have an arbitrary loading.
    """

    loading: Dict[str, np.ndarray]
    stability: Dict[str, float]
    direction: Dict[str, np.ndarray]
    feature_std: np.ndarray


def axis_loadings(
    model: Any,
    X: torch.Tensor,
    axes: Sequence[ClassAxis],
    labels: Optional[torch.Tensor] = None,
    *,
    directions: Optional[Mapping[str, np.ndarray]] = None,
    sample: Optional[int] = 512,
    batch_size: int = 128,
    seed: int = 0,
) -> AxisLoadings:
    """The input-space gradient of each requested ordering: a biplot loading.

    :func:`class_axis_report` says whether an ordering took. This says what it
    *means*: for an axis with embedding direction ``u``, the gradient
    ``d(z . u)/dx`` is a vector in feature space, so it has the shape of a
    measurement and can be read as one -- an image for image inputs, a spectrum for
    spectral inputs. It is the loading of a biplot with the usual roles swapped:
    rather than drawing the features as arrows in the embedding, it draws the axis
    in the space of the features, which is the more useful way round when the axis
    is something the user asked for by name.

    Read ``stability`` first, and expect to be disappointed
    ------------------------------------------------------

    A biplot presumes a linear map and this one is not, so the gradient belongs to
    the point it was taken at and the returned loading is a mean over sampled
    points. Whether that mean stands for anything is an empirical question, and on
    digits with the default configuration the answer is no: resultant length 0.40
    for a pinned digit axis and 0.27 for a free parity direction, with the 10th
    percentile of per-point agreement at ``-0.43`` and the worst point at
    ``-0.996``. That last number is the one to take seriously -- for a substantial
    minority of points the mean loading points the *opposite* way along the axis
    from the gradient they actually have, so the mean image is not a weak
    description of the map, it is an average over evidence that disagrees in sign.
    Restricting to one class raises the resultant only to about 0.66.

    So a low ``stability`` is not a caveat to note and move past. It says the
    question "what does this axis mean in the features" has no single answer for
    this fit, and a rendered mean loading will look perfectly plausible while
    meaning nothing in particular. Two ways to get an answer that holds:
    interpret locally, taking loadings over a region small enough to agree with
    itself; or fit with ``pca_skip=True``, which gives the encoder an explicitly
    linear component whose loading is globally valid by construction rather than by
    luck, and reserve the interpretation for that part.

    Units
    -----

    Loadings are per *standard deviation* of each feature, not per raw unit -- the
    difference between a correlation and a covariance biplot, and here not a
    stylistic choice. The encoder standardises its input by a training ``x_std``
    clamped below at ``1e-6``, so a raw-unit loading carries a ``1/x_std`` factor
    of ``1e6`` for any feature that never varied, such as the dead border pixels of
    an image. Measured on digits, raw-unit gradient norms run to ~1e4 and are
    dominated by exactly those pixels; standardising removes the factor. It cannot
    make a constant feature meaningful, though -- there is no "one standard
    deviation" of a feature that never moved, so its entry is arbitrary rather than
    merely large, which is what ``feature_std`` is returned to let callers mask.

    Parameters
    ----------
    model : PLANE
        A fitted model. ``forward`` is used rather than ``embed``, which is wrapped
        in ``torch.no_grad`` and so cannot carry a gradient.
    X, labels
        Points to take the gradient at, and their class labels. Labels are needed
        only to locate a free-direction axis, and only when ``directions`` is not
        supplied.
    directions
        Precomputed embedding directions by axis name, e.g. built from the
        ``dir_<name>_<j>`` entries of a :func:`class_axis_report`. Passing the
        report's own directions is how to guarantee the loading describes the same
        axis the score was taken along; otherwise it is recomputed here and a
        differently-sampled direction can disagree slightly.
    sample
        Points to draw for the gradient, or ``None`` for all of them. A few hundred
        is plenty for both the mean and its resultant, and the cost is one backward
        pass per axis.
    """
    dev = next(model.parameters()).device
    X = torch.as_tensor(X)
    n = X.shape[0]
    if sample is not None and int(sample) < n:
        idx = torch.as_tensor(
            np.random.default_rng(seed).choice(n, size=int(sample), replace=False)
        )
    else:
        idx = torch.arange(n)
    Xs = X[idx]

    enc = getattr(model, "encoder", model)
    x_std = getattr(enc, "x_std", None)
    x_std = (
        torch.ones(X.shape[1])
        if x_std is None
        else x_std.detach().cpu().reshape(-1).clone()
    )

    with torch.no_grad():
        Z, _ = model.embed(X)
    Z = Z.detach().cpu()
    d_out = Z.shape[1]

    # A pinned axis is a basis vector; a free one has to be located, either from a
    # report the caller already has or from the labels.
    dirs: Dict[str, np.ndarray] = {}
    pinned = [a.axis for a in axes if a.is_pinned]
    for ax in axes:
        if directions is not None and ax.name in directions:
            dirs[ax.name] = np.asarray(
                directions[ax.name], dtype=np.float64
            ).reshape(-1)
        elif ax.is_pinned:
            e = np.zeros(d_out, dtype=np.float64)
            e[ax.axis] = 1.0
            dirs[ax.name] = e
        elif labels is None:
            raise ValueError(
                f"axis {ax.name!r} names no coordinate, so its direction has to be "
                "located: pass labels=, or directions= from class_axis_report"
            )
        else:
            dirs[ax.name] = _report_direction(
                np.ascontiguousarray(Z.numpy(), dtype=np.float64),
                labels.detach().cpu().numpy().astype(np.int64),
                ax.rank.detach().cpu().numpy().astype(np.float64),
                pinned,
                orthogonal=ax.orthogonal,
                whiten=ax.whiten,
            )

    loading: Dict[str, np.ndarray] = {}
    stability: Dict[str, float] = {}
    for ax in axes:
        u = torch.as_tensor(dirs[ax.name], dtype=torch.float32, device=dev)
        rows = []
        for s in range(0, Xs.shape[0], batch_size):
            # forward(), not embed(): the latter is no_grad. It also does not move
            # inputs to the model device the way embed() does, hence the .to(dev).
            xb = Xs[s : s + batch_size].clone().to(dev).requires_grad_(True)
            out = model.forward(xb)
            z = out[0] if isinstance(out, tuple) else out
            (z @ u).sum().backward()
            rows.append(xb.grad.detach().cpu())
        g = torch.cat(rows) * x_std  # per standard deviation, not per raw unit
        gn = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        loading[ax.name] = g.mean(dim=0).numpy().astype(np.float64)
        stability[ax.name] = float(gn.mean(dim=0).norm())
    return AxisLoadings(
        loading=loading,
        stability=stability,
        direction=dirs,
        feature_std=x_std.numpy().astype(np.float64),
    )


@dataclass
class ClassAxisReadout:
    """Inference-time reading of an ordered axis.

    Built from training points, so it inherits the split discipline of
    :class:`~leanmap.conformal.LandmarkSupport`: per-class positions are not
    rank-preserving transforms of a score and must not be fit on the
    calibration set that later supplies p-values.

    The useful output is not a label. ``position`` places a point *on the
    user's ordering*, interpolating between the class positions the training
    data settled into, so a point between two classes reads as between them
    instead of being forced to one side. That statement is only available
    because the axis was ordered in the first place.

    A pinned axis is read along its named coordinate. A free-direction axis
    (``ClassAxis.axis is None``) is read along the direction
    :func:`from_model` locates on the training layout -- the same direction
    :func:`class_axis_report` scores -- so ``position`` still means "where on
    the requested order" rather than "where on ``z0``".
    """

    axis: Optional[int]
    rank: torch.Tensor
    classes: List[int]
    class_pos: Dict[int, float]
    z_by_class: Dict[int, torch.Tensor] = field(default_factory=dict)
    k: int = 5
    name: str = "class"
    direction: Optional[np.ndarray] = None

    def _along(self, Z: torch.Tensor) -> np.ndarray:
        z = np.ascontiguousarray(Z.detach().cpu().numpy(), dtype=np.float64)
        if self.direction is not None:
            return np.dot(z, self.direction)
        if self.axis is None:
            raise ValueError(
                "ClassAxisReadout has neither a pinned axis nor a direction; "
                "pass a pinned ClassAxis or let from_model locate a free one"
            )
        return z[:, self.axis]

    @classmethod
    @torch.no_grad()
    def from_model(
        cls,
        model: torch.nn.Module,
        X_train: torch.Tensor,
        labels: torch.Tensor,
        ax: ClassAxis,
        k: int = 5,
        batch_size: int = 4096,
        axes: Optional[Sequence[ClassAxis]] = None,
    ) -> "ClassAxisReadout":
        model.eval()
        device = next(model.parameters()).device
        zs = []
        for s in range(0, X_train.shape[0], batch_size):
            zb, _, _ = model(X_train[s : s + batch_size].to(device))
            zs.append(zb.detach().cpu())
        Z = torch.cat(zs, dim=0)
        lab = labels.detach().cpu().reshape(-1).to(torch.int64)
        present = sorted({int(c) for c in lab.tolist()})
        z_by_class = {c: Z[lab == c] for c in present}
        direction = None
        if ax.is_pinned:
            coord = Z[:, ax.axis].numpy().astype(np.float64)
        else:
            others = list(axes) if axes is not None else [ax]
            direction = _report_direction(
                np.ascontiguousarray(Z.numpy(), dtype=np.float64),
                lab.numpy().astype(np.int64),
                ax.rank.detach().cpu().numpy().astype(np.float64),
                [a.axis for a in others if a.is_pinned],
                orthogonal=ax.orthogonal,
                whiten=ax.whiten,
            )
            coord = np.dot(Z.numpy().astype(np.float64), direction)
        class_pos = {
            c: float(np.median(coord[lab.numpy() == c])) for c in present
        }
        return cls(
            axis=ax.axis,
            rank=ax.rank.clone(),
            classes=present,
            class_pos=class_pos,
            z_by_class=z_by_class,
            k=int(k),
            name=ax.name,
            direction=direction,
        )

    def _ordered_classes(self) -> List[int]:
        return sorted(self.classes, key=lambda c: float(self.rank[c]))

    def position(self, Z: torch.Tensor) -> torch.Tensor:
        """Where each point sits on the user's ordering, as a continuous rank.

        Linear interpolation of the axis coordinate through the training class
        positions, clamped at the ends. A value of 1.4 means "past class rank 1,
        four tenths of the way to rank 2" -- the reading the ordered axis exists
        to support, and strictly more informative than the nearest label.
        """
        seq = self._ordered_classes()
        xp = np.asarray([self.class_pos[c] for c in seq], dtype=np.float64)
        fp = np.asarray([float(self.rank[c]) for c in seq], dtype=np.float64)
        keep = np.argsort(xp, kind="stable")
        xp, fp = xp[keep], fp[keep]
        u = self._along(Z)
        return torch.as_tensor(np.interp(u, xp, fp), dtype=torch.float32)

    def region_score(self, Z: torch.Tensor, c: int) -> torch.Tensor:
        """Nonconformity w.r.t. class ``c``: distance to its ``k``-th nearest
        embedded training point.

        Class-discriminative by construction, which the shipped scores in
        :mod:`leanmap.conformal` are not -- ``cover`` and ``affinity_entropy``
        measure distance to the landmark support and are almost identical for
        two points sitting in different class regions of the same manifold. A
        per-class Mondrian test calibrated on those would separate nothing.

        A local ``k``-NN distance rather than a centroid distance because class
        regions have no reason to be convex: a centroid can easily fall outside
        the region it names, in the gap between two lobes of the same class.
        """
        if c not in self.z_by_class:
            raise KeyError(f"unknown class {c!r}; have {self.classes}")
        ref = self.z_by_class[c].to(Z.device, Z.dtype)
        k = min(self.k, ref.shape[0])
        d = torch.cdist(Z.detach(), ref)
        return d.topk(k, dim=1, largest=False).values[:, -1]


@dataclass
class ClassRegionConformal:
    """One conformal test per class, on distance to that class's region.

    This is a genuine Mondrian construction -- a separate calibration
    distribution per class -- but it cannot go through
    :class:`~leanmap.conformal.MondrianCalibrator`, which applies a *single*
    score function across all groups. Here each class needs its own score
    (distance to *its* region), so each class needs its own calibrator.

    What the output buys over an ``argmax``: the accepted set may be empty,
    which is the honest answer for a point that is in no class's region and the
    thing a softmax structurally cannot say; and it may hold more than one
    class, which marks genuine ambiguity rather than hiding it behind a margin.
    Combine with :class:`~leanmap.conformal.LandmarkSupport` for the separate
    question of whether the point is near the training manifold at all -- a
    point can be comfortably inside a class region of the *embedding* while
    sitting far off the data manifold, because the encoder maps everything
    somewhere.
    """

    readout: ClassAxisReadout
    s_calib: Dict[int, torch.Tensor] = field(default_factory=dict)

    @torch.no_grad()
    def fit(
        self,
        model: torch.nn.Module,
        X_calib: torch.Tensor,
        labels_calib: torch.Tensor,
        batch_size: int = 4096,
    ) -> "ClassRegionConformal":
        """Calibrate on held-out labelled points, one distribution per class."""
        model.eval()
        device = next(model.parameters()).device
        zs = []
        for s in range(0, X_calib.shape[0], batch_size):
            zb, _, _ = model(X_calib[s : s + batch_size].to(device))
            zs.append(zb.detach().cpu())
        return self.fit_from_embeddings(torch.cat(zs, dim=0), labels_calib)

    def fit_from_embeddings(
        self, Z_calib: torch.Tensor, labels_calib: torch.Tensor
    ) -> "ClassRegionConformal":
        """As :meth:`fit`, when the calibration embeddings are already in hand."""
        Z = Z_calib.detach()
        lab = labels_calib.detach().cpu().reshape(-1).to(torch.int64)
        if Z.shape[0] != lab.shape[0]:
            raise ValueError(
                f"Z_calib has {Z.shape[0]} rows but labels_calib has {lab.shape[0]}"
            )
        log = get_logger()
        self.s_calib = {}
        for c in self.readout.classes:
            sel = lab == c
            n = int(sel.sum().item())
            if n == 0:
                log.warning(
                    "class %r has no calibration points; it cannot be tested and "
                    "will never appear in a prediction set",
                    c,
                )
                continue
            s = self.readout.region_score(Z[sel], c)
            self.s_calib[c] = torch.sort(s).values
            if n < 50:
                log.warning(
                    "class %r has n_calib=%d; alphas below 1/(n+1)=%.4f are "
                    "unreachable for it",
                    c,
                    n,
                    1.0 / (n + 1),
                )
        if not self.s_calib:
            raise ValueError("no class had calibration points")
        return self

    def p_values(self, Z: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Upper-tailed rank p-value per class: small ``p`` ⇒ not that class."""
        if not self.s_calib:
            raise RuntimeError("ClassRegionConformal.fit has not been called")
        out: Dict[int, torch.Tensor] = {}
        for c, s_c in self.s_calib.items():
            s = self.readout.region_score(Z, c).detach().cpu().contiguous()
            n = int(s_c.numel())
            count_ge = n - torch.searchsorted(s_c, s, right=False)
            out[c] = (1.0 + count_ge.float()) / (n + 1)
        return out

    def prediction_set(self, Z: torch.Tensor, alpha: float = 0.05) -> List[Tuple[int, ...]]:
        """``{c : p_c > alpha}`` per point; empty means no class accepted it."""
        pv = self.p_values(Z)
        classes = sorted(pv)
        B = int(Z.shape[0])
        return [
            tuple(c for c in classes if float(pv[c][i]) > float(alpha))
            for i in range(B)
        ]

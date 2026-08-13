"""Class-order gauge fix: the ceiling, the hinge, and what it leaves alone."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from leanmap import PLANEConfig, fit
from leanmap.classaxis import (
    ClassAxis,
    ClassAxisReadout,
    ClassOrderSampler,
    ClassRegionConformal,
    class_axis_report,
    class_direction_loss,
    class_order_loss,
    grouped_class_axis,
    ordinal_class_axis,
    validate_class_axes,
)


def _ordered_blobs(n_per=120, K=4, seed=0):
    """K blobs whose feature geometry already carries the class order."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(K):
        centre = np.array([2.5 * k, 0.0], dtype=np.float32)
        X.append(centre + rng.normal(scale=0.45, size=(n_per, 2)).astype(np.float32))
        y.append(np.full(n_per, k, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y)


# --------------------------------------------------------------------------
# The d_out - 1 ceiling
# --------------------------------------------------------------------------


def test_ceiling_leaves_one_free_direction():
    ax0 = ClassAxis(axis=0, rank=torch.arange(3.0), name="a")
    validate_class_axes([ax0], d_out=2, n_classes=3)  # 1 of 2 spent — fine

    ax1 = ClassAxis(axis=1, rank=torch.arange(3.0), name="b")
    with pytest.raises(ValueError, match="unnamed coordinates"):
        validate_class_axes([ax0, ax1], d_out=2, n_classes=3)

    validate_class_axes([ax0, ax1], d_out=3, n_classes=3)  # 2 of 3 — fine


def test_ceiling_rejects_duplicate_and_out_of_range_axes():
    a = ClassAxis(axis=0, rank=torch.arange(3.0), name="a")
    b = ClassAxis(axis=0, rank=torch.arange(3.0), name="b")
    with pytest.raises(ValueError, match="claimed by both"):
        validate_class_axes([a, b], d_out=3, n_classes=3)
    with pytest.raises(ValueError, match="outside"):
        validate_class_axes(
            [ClassAxis(axis=5, rank=torch.arange(3.0))], d_out=3, n_classes=3
        )
    with pytest.raises(ValueError, match="same rank"):
        validate_class_axes(
            [ClassAxis(axis=0, rank=torch.zeros(3))], d_out=2, n_classes=3
        )
    with pytest.raises(ValueError, match="3 classes"):
        validate_class_axes(
            [ClassAxis(axis=0, rank=torch.arange(4.0))], d_out=2, n_classes=3
        )
    with pytest.raises(ValueError, match="switches it off"):
        validate_class_axes(
            [ClassAxis(axis=0, rank=torch.arange(3.0), weight=0.0)],
            d_out=2,
            n_classes=3,
        )


# --------------------------------------------------------------------------
# Free-direction axes: a weaker request, counted separately
# --------------------------------------------------------------------------


def test_free_direction_axis_is_allowed_where_a_second_pinned_axis_is_not():
    pinned = ClassAxis(axis=0, rank=torch.arange(3.0), name="a")
    free = ClassAxis(axis=None, rank=torch.tensor([0.0, 0.0, 1.0]), name="b")

    # Two pinned axes in d_out=2 is refused; pinned + free is allowed, because
    # the free one names neither a coordinate nor a sign.
    with pytest.raises(ValueError, match="unnamed coordinates"):
        validate_class_axes(
            [pinned, ClassAxis(axis=1, rank=torch.arange(3.0), name="b")],
            d_out=2,
            n_classes=3,
        )
    validate_class_axes([pinned, free], d_out=2, n_classes=3)

    # There still have to be directions to order along at all.
    with pytest.raises(ValueError, match="not that many independent directions"):
        validate_class_axes(
            [pinned, free, ClassAxis(axis=None, rank=torch.arange(3.0), name="c")],
            d_out=2,
            n_classes=3,
        )


def test_free_direction_term_cannot_move_a_pinned_coordinate():
    """The guarantee that lets a secondary factor be weighted without fear."""
    torch.manual_seed(0)
    z_lo = torch.randn(128, 3, requires_grad=True)
    z_hi = torch.randn(128, 3, requires_grad=True)
    loss, _, active, u = class_direction_loss(z_lo, z_hi, pinned_axes=(0,))
    loss.backward()

    assert active > 0.0, "random points should leave some pairs unordered"
    assert float(u[0]) == 0.0
    for g in (z_lo.grad, z_hi.grad):
        assert torch.all(g[:, 0] == 0.0)
        assert float(g[:, 1:].abs().max()) > 0.0


def test_free_direction_does_not_constrain_the_sign():
    """A layout separated 'the wrong way round' is already satisfied.

    This is the whole difference from a pinned axis, and the reason the term is
    cheap: it asks for separation with a consistent order, not for an orientation.
    """
    torch.manual_seed(0)
    lo = torch.zeros(64, 2)
    hi = torch.zeros(64, 2)
    lo[:, 1] = 3.0 + 0.05 * torch.randn(64)
    hi[:, 1] = -3.0 + 0.05 * torch.randn(64)  # reversed

    loss, _, active, u = class_direction_loss(lo, hi, pinned_axes=(0,))
    assert float(loss) == 0.0
    assert active == 0.0
    assert float(u[1]) < 0.0, "the direction should have flipped to suit the layout"


def test_free_direction_falls_back_when_the_groups_coincide():
    z = torch.randn(64, 3) * 0.01
    loss, _, _, u = class_direction_loss(z, z.clone(), pinned_axes=(0,))
    assert float(loss) > 0.0, "coincident groups are not separated, so not satisfied"
    assert float(u[0]) == 0.0
    assert float(torch.linalg.vector_norm(u)) == pytest.approx(1.0, abs=1e-5)


def test_free_direction_report_is_rotation_agnostic():
    lab = torch.arange(10).repeat(60)
    parity = grouped_class_axis([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], axis=None, name="p")
    torch.manual_seed(0)
    Z = torch.stack([lab.float(), (lab % 2).float() * 4.0], 1)
    Z = Z + 0.2 * torch.randn(Z.shape)
    theta = 0.7
    R = torch.tensor(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=torch.float32,
    )

    upright = class_axis_report(Z, lab, [parity])["order_p"]
    rotated = class_axis_report(Z @ R, lab, [parity])["order_p"]
    assert upright > 0.99
    assert rotated == pytest.approx(upright, abs=0.02)


def test_free_direction_report_null_is_above_one_half():
    """Documented bias: the direction is fitted on the points it scores.

    Locks in the caveat that a free-direction score must be read against a
    shuffled baseline rather than against 0.5.
    """
    parity = grouped_class_axis([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], axis=None, name="p")
    pinned = ordinal_class_axis(10, axis=1, name="q")
    lab = torch.arange(10).repeat(60)
    free, pin = [], []
    for s in range(20):
        g = torch.Generator().manual_seed(s)
        Z = torch.randn(lab.shape[0], 2, generator=g)
        free.append(class_axis_report(Z, lab, [parity])["order_p"])
        pin.append(class_axis_report(Z, lab, [pinned])["order_q"])
    assert np.mean(pin) == pytest.approx(0.5, abs=0.02), "a pinned axis has no bias"
    assert 0.5 < np.mean(free) < 0.6, "fitted direction is optimistic but only mildly"


# --------------------------------------------------------------------------
# The hinge: zero force once satisfied, and blind to the free coordinates
# --------------------------------------------------------------------------


def test_hinge_is_exactly_zero_once_ordered():
    z_lo = torch.zeros(64, 2)
    z_hi = torch.zeros(64, 2)
    z_hi[:, 0] = 5.0  # ordered, far past any margin
    loss, _, active = class_order_loss(z_lo, z_hi, axis=0, spread_state={})
    assert float(loss) == 0.0
    assert active == 0.0


def test_hinge_gradient_vanishes_when_satisfied():
    z_lo = torch.zeros(32, 2, requires_grad=True)
    z_hi = torch.full((32, 2), 5.0, requires_grad=True)
    loss, _, _ = class_order_loss(z_lo, z_hi, axis=0, spread_state={})
    if loss.requires_grad:
        loss.backward()
        assert z_lo.grad is None or float(z_lo.grad.abs().sum()) == 0.0


def test_hinge_penalises_inversion_and_only_the_constrained_axis():
    torch.manual_seed(0)
    z_lo = torch.randn(128, 3, requires_grad=True)
    z_hi = torch.randn(128, 3, requires_grad=True)
    with torch.no_grad():  # force an inversion on axis 0
        z_lo[:, 0] = 3.0
        z_hi[:, 0] = -3.0
    loss, _, active = class_order_loss(z_lo, z_hi, axis=0, spread_state={})
    assert float(loss.detach()) > 0.0
    assert active == 1.0

    loss.backward()
    # The objective expresses no preference at all about the free coordinates.
    assert float(z_lo.grad[:, 1:].abs().sum()) == 0.0
    assert float(z_hi.grad[:, 1:].abs().sum()) == 0.0
    assert float(z_lo.grad[:, 0].abs().sum()) > 0.0


def test_hinge_is_scale_free():
    """Inflating the layout must not change the loss: the margin is relative."""
    torch.manual_seed(0)
    z_lo = torch.randn(256, 2)
    z_hi = torch.randn(256, 2)
    small, _, _ = class_order_loss(z_lo, z_hi, 0, spread_state={})
    big, _, _ = class_order_loss(z_lo * 1000.0, z_hi * 1000.0, 0, spread_state={})
    assert float(small) == pytest.approx(float(big), rel=1e-5)


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------


def test_sampler_only_draws_ordered_pairs_and_respects_ties():
    X = torch.arange(60, dtype=torch.float32).reshape(60, 1)
    labels = torch.arange(60) % 3
    # Ranks 0 and 1 tie, so no pair between classes 0 and 1 may be drawn.
    rank = torch.tensor([0.0, 0.0, 1.0])
    samp = ClassOrderSampler(X, labels, rank, seed=0)
    assert samp.n_pairs == 2  # (0,2) and (1,2) only
    x_lo, x_hi = samp.sample(400)
    lab_lo = labels[x_lo.squeeze(-1).long()]
    lab_hi = labels[x_hi.squeeze(-1).long()]
    assert torch.all(rank[lab_lo] < rank[lab_hi])
    assert set(lab_hi.tolist()) == {2}


def test_sampler_rejects_an_unorderable_split():
    X = torch.zeros(10, 1)
    labels = torch.zeros(10, dtype=torch.int64)
    with pytest.raises(ValueError, match="no ordered class pair"):
        ClassOrderSampler(X, labels, torch.tensor([0.0, 1.0]), seed=0)


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------


def test_report_separates_overall_from_adjacent_ordering():
    # Three classes correctly ordered on axis 0, noise on axis 1.
    rng = np.random.default_rng(0)
    Z = np.zeros((300, 2), dtype=np.float32)
    labels = np.repeat([0, 1, 2], 100).astype(np.int64)
    Z[:, 0] = labels * 10.0 + rng.normal(scale=0.1, size=300)
    Z[:, 1] = rng.normal(size=300)
    ax = ordinal_class_axis(3, axis=0)
    rep = class_axis_report(torch.as_tensor(Z), torch.as_tensor(labels), [ax])
    assert rep["order_class"] > 0.99
    assert rep["order_adjacent_class"] > 0.99

    # Shuffled labels must land at chance.
    shuffled = torch.as_tensor(rng.permutation(labels))
    rep_null = class_axis_report(torch.as_tensor(Z), shuffled, [ax])
    assert abs(rep_null["order_class"] - 0.5) < 0.1


def test_report_flags_a_reversed_axis():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 100).astype(np.int64)
    Z = np.zeros((300, 2), dtype=np.float32)
    Z[:, 0] = -labels * 10.0 + rng.normal(scale=0.1, size=300)
    rep = class_axis_report(
        torch.as_tensor(Z), torch.as_tensor(labels), [ordinal_class_axis(3, axis=0)]
    )
    assert rep["order_class"] < 0.01


# --------------------------------------------------------------------------
# End to end
#
# The requested order is a permutation the unsupervised layout has no reason to
# produce, which is what makes these tests about the mechanism rather than about
# reading a pre-existing geometry off the data. With the term off, the same data
# and seed score near chance on that order; with it on, they order.
# --------------------------------------------------------------------------

REQUESTED_ORDER = [3, 2, 1, 0]


def _labelled_fit(lam: float, epochs: int = 10):
    """Explicit calibration split: a per-class conformal test needs enough points
    in every class, and the 5% internal split cannot supply that at this size."""
    X, y = _ordered_blobs(n_per=100, K=4)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(X))
    n_cal = 160
    cal_i, tr_i = perm[:n_cal], perm[n_cal:]
    cfg = PLANEConfig.for_scale(len(tr_i))
    cfg.epochs = epochs
    cfg.dedup = False
    # Off to keep the test cheap and the comparison about one term.
    cfg.lambda_geo = 0.0
    cfg.lambda_density = 0.0
    cfg.lambda_class = lam
    cfg.class_ramp = (0.0, 0.1)
    ax = ordinal_class_axis(4, order=REQUESTED_ORDER)
    res = fit(
        X[tr_i],
        dist_fn="l2",
        config=cfg,
        X_calib=X[cal_i],
        class_labels=y[tr_i],
        class_axes=[ax],
    )
    # With an explicit X_calib the caller owns the calibration labels, so fit
    # cannot fill them in and leaves the attribute None.
    assert res.class_labels_calib is None
    res.class_labels_calib = torch.as_tensor(y[cal_i])
    return res, ax, X, y


@pytest.fixture(scope="module")
def gauge_on():
    return _labelled_fit(1.0)


@pytest.fixture(scope="module")
def gauge_off():
    return _labelled_fit(0.0)


def _order(res, ax) -> float:
    Z, _ = res.model.embed(res.X_train)
    rep = class_axis_report(Z.detach(), res.class_labels_train, [ax])
    return rep[f"order_{ax.name}"]


def test_gauge_fix_pins_a_requested_order(gauge_on, gauge_off):
    on = _order(*gauge_on[:2])
    off = _order(*gauge_off[:2])
    assert on > 0.9, f"term on scored {on}"
    assert on > off + 0.2, f"term on {on} vs off {off} — the term did nothing"


def test_term_reports_itself_satisfied(gauge_on):
    """Once the order holds the hinge contributes nothing, by construction."""
    res = gauge_on[0]
    Z, _ = res.model.embed(res.X_train)
    ax = gauge_on[1]
    samp = ClassOrderSampler(res.X_train, res.class_labels_train, ax.rank, seed=0)
    x_lo, x_hi = samp.sample(512)
    z_lo, _ = res.model.embed(x_lo)
    z_hi, _ = res.model.embed(x_hi)
    loss, _, active = class_order_loss(z_lo.detach(), z_hi.detach(), ax.axis, {})
    assert active < 0.05, f"{active:.3f} of pairs still inside the margin"
    assert float(loss) < 1e-3


def test_fit_without_labels_is_unchanged():
    """lambda_class defaults to 0, so an unlabelled fit takes no new code path."""
    X, _ = _ordered_blobs(n_per=40, K=3)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 2
    cfg.dedup = False
    cfg.lambda_geo = 0.0
    cfg.lambda_density = 0.0
    res = fit(X, dist_fn="l2", config=cfg)
    assert not hasattr(res, "class_axes")


def test_class_axes_without_labels_raises():
    X, _ = _ordered_blobs(n_per=40, K=2)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 2
    cfg.dedup = False
    with pytest.raises(ValueError, match="nothing to order"):
        fit(X, dist_fn="l2", config=cfg, class_axes=[ordinal_class_axis(2)])


def test_too_many_axes_is_refused_by_fit():
    X, y = _ordered_blobs(n_per=40, K=2)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 2
    cfg.dedup = False
    with pytest.raises(ValueError, match="unnamed coordinates"):
        fit(
            X,
            dist_fn="l2",
            config=cfg,
            class_labels=y,
            class_axes=[
                ClassAxis(axis=0, rank=torch.arange(2.0), name="a"),
                ClassAxis(axis=1, rank=torch.arange(2.0), name="b"),
            ],
        )


# --------------------------------------------------------------------------
# Two orderings at once: a pinned primary and a free-direction secondary
# --------------------------------------------------------------------------

PARITY_GROUPS = [[0, 2], [1, 3]]


def _two_ordering_fit(with_parity: bool, epochs: int = 12):
    X, y = _ordered_blobs(n_per=80, K=4)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = epochs
    cfg.dedup = False
    cfg.lambda_geo = 0.0
    cfg.lambda_density = 0.0
    cfg.lambda_class = 4.0
    cfg.class_ramp = (0.0, 0.1)
    digit = ordinal_class_axis(4, axis=0, order=REQUESTED_ORDER, name="chain")
    parity = grouped_class_axis(PARITY_GROUPS, axis=None, name="parity", weight=0.5)
    res = fit(
        X,
        dist_fn="l2",
        config=cfg,
        class_labels=y,
        class_axes=[digit] + ([parity] if with_parity else []),
    )
    return res, digit, grouped_class_axis(PARITY_GROUPS, axis=None, name="parity")


@pytest.fixture(scope="module")
def two_orderings():
    return _two_ordering_fit(with_parity=True)


@pytest.fixture(scope="module")
def one_ordering():
    return _two_ordering_fit(with_parity=False)


def test_a_free_direction_ordering_is_achieved_on_top_of_a_pinned_one(
    two_orderings, one_ordering
):
    """The classes are interleaved along the pinned chain, so the parity
    grouping can only be expressed in the remaining direction.

    The bar is 0.85 rather than something near 1 because these blobs are an
    adversarial case for a *secondary* axis: four classes strung along a single
    feature direction, so parity is a strict alternation with no feature support
    of its own, and the term has to manufacture the separation rather than orient
    one that exists. On digits the same request reaches 0.999 (see
    ``examples/digits_two_orderings.py``). What is being tested here is that the
    term acts and acts in the right direction, which the comparison below is the
    real statement of.
    """
    on, one = two_orderings, one_ordering
    rep_on = class_axis_report(
        on[0].model.embed(on[0].X_train)[0].detach(),
        on[0].class_labels_train,
        [on[1], on[2]],
    )
    rep_off = class_axis_report(
        one[0].model.embed(one[0].X_train)[0].detach(),
        one[0].class_labels_train,
        [one[1], one[2]],
    )
    assert rep_on["order_parity"] > 0.85, f"parity scored {rep_on['order_parity']}"
    assert rep_on["order_parity"] > rep_off["order_parity"] + 0.1, (
        f"parity {rep_on['order_parity']:.3f} with the term vs "
        f"{rep_off['order_parity']:.3f} without — the term did nothing"
    )


def test_the_free_direction_term_does_not_spoil_the_pinned_ordering(
    two_orderings, one_ordering
):
    """The complement of what the previous test measures, and the practical
    reason the secondary axis is safe to add."""
    both = _order(two_orderings[0], two_orderings[1])
    alone = _order(one_ordering[0], one_ordering[1])
    assert both > 0.9, f"pinned chain scored {both} with a secondary axis present"
    assert both > alone - 0.05, f"adding parity cost the chain {alone - both:.3f}"


def test_readout_position_follows_the_requested_order(gauge_on):
    res, ax, _, _ = gauge_on
    readout = ClassAxisReadout.from_model(
        res.model, res.X_train, res.class_labels_train, ax
    )
    Z_tr, _ = res.model.embed(res.X_train)
    pos = readout.position(Z_tr.detach()).numpy()
    lab = res.class_labels_train.numpy()
    # Mean position must rise along the *requested* sequence, not the label codes.
    means = [pos[lab == k].mean() for k in REQUESTED_ORDER]
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), means


def _synthetic_readout(seed=0, n=200, gap=30.0):
    """Three well-separated class clouds in embedding space, no model needed."""
    rng = np.random.default_rng(seed)
    z_by_class, class_pos = {}, {}
    for c in range(3):
        z = np.stack(
            [rng.normal(gap * c, 1.0, n), rng.normal(0.0, 1.0, n)], axis=1
        ).astype(np.float32)
        z_by_class[c] = torch.as_tensor(z)
        class_pos[c] = float(np.median(z[:, 0]))
    return ClassAxisReadout(
        axis=0,
        rank=torch.arange(3.0),
        classes=[0, 1, 2],
        class_pos=class_pos,
        z_by_class=z_by_class,
        k=5,
    )


def test_conformal_abstains_when_the_embedding_is_far_from_every_class():
    readout = _synthetic_readout()
    rng = np.random.default_rng(1)
    Z_cal = torch.cat([readout.z_by_class[c][:80] for c in range(3)])
    lab_cal = torch.as_tensor(np.repeat([0, 1, 2], 80))
    cal = ClassRegionConformal(readout).fit_from_embeddings(Z_cal, lab_cal)

    # Far from every cloud in embedding space: no class may accept it.
    Z_far = torch.tensor([[0.0, 500.0], [-400.0, 0.0]])
    assert all(len(s) == 0 for s in cal.prediction_set(Z_far, alpha=0.05))

    # Squarely inside class 1: accepted, and p is far from the rejection edge.
    Z_in = readout.z_by_class[1][90:95]
    sets = cal.prediction_set(Z_in, alpha=0.05)
    assert all(1 in s for s in sets), sets
    assert float(cal.p_values(Z_in)[1].min()) > 0.05


def test_conformal_accepts_held_out_points_of_their_own_class(gauge_on):
    res, ax, _, _ = gauge_on
    readout = ClassAxisReadout.from_model(
        res.model, res.X_train, res.class_labels_train, ax
    )
    cal = ClassRegionConformal(readout).fit(
        res.model, res.X_calib, res.class_labels_calib
    )
    Z_cal, _ = res.model.embed(res.X_calib)
    sets = cal.prediction_set(Z_cal.detach(), alpha=0.05)
    own = [int(res.class_labels_calib[i]) in s for i, s in enumerate(sets)]
    assert np.mean(own) > 0.7, np.mean(own)


def test_class_regions_do_not_detect_ambient_outliers(gauge_on):
    """Locks in the documented division of labour, so nobody assumes otherwise.

    The encoder maps every input somewhere, and LayerNorm discards most of the
    scale of an extreme one, so a wildly out-of-range ambient point lands *inside*
    the occupied part of the map. A class-region test in embedding space
    therefore cannot see it; that is what landmark cover is for.
    """
    res, ax, _, _ = gauge_on
    Z_tr, _ = res.model.embed(res.X_train)
    lo, hi = Z_tr.min(0).values, Z_tr.max(0).values
    Z_far, _ = res.model.embed(torch.full((2, 2), 1e4))
    inside = bool(((Z_far.detach() >= lo) & (Z_far.detach() <= hi)).all())
    assert inside, "ambient outlier left the map's range; caveat may be stale"

"""Density: keeping the layout's density *ordering* faithful to the graph's.

A layout's density contrast should not be whatever the attraction/repulsion
equilibrium happens to settle on. Which neighbourhoods are crowded and which are
sparse is a property of the data, measurable from the neighbour graph before any
training, and the fit should be held to it.

What is held, though, is the *ordering* and not the magnitude. An earlier version
of this module targeted magnitude: preserving mass under a dimension change
requires ``log r_embed = (dim / d_out) * log r_ambient + const``, so it pinned the
layout's log-radius spread to ``dim / d_out`` times the ambient spread. That is
correct arithmetic for an equal-area map and a bad objective, for one reason:
when ``dim >> d_out`` the required spread is larger than any equilibrium can
produce. On 64-dimensional digits (intrinsic dim 7.3 into 2-D) it demanded 3.6x
the ambient spread, saturated its own ``min_dist`` search against a ceiling, and
degraded class structure while chasing a number it could never reach. Worse, it
needed a per-dataset switch to decide whether to try -- which is an admission
that the objective was wrong, not that the data was awkward.

Correlating log radii instead is scale free. It asks that crowded ambient
neighbourhoods come out crowded, in order, and says nothing about by how much --
so there is no magnitude to fail to reach, no saturation, and no mode to choose.
The ``dim / d_out`` factor drops out entirely, since correlation is invariant to
it. This is the same shape of objective densMAP uses, which is why densMAP needs
one weight rather than a per-dataset decision.

:func:`density_budget`
    measures the ambient log radius per graph node, before training
:func:`density_correlation_loss`
    holds the layout's log radii in correspondence with it during training

The strength is ``PLANEConfig.lambda_density``; set it to 0 to opt out entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .distance import DistanceFn
from .graph import Graph, _intrinsic_dim_levina_bickel
from .utils import get_logger

# Below this the batch carries no usable density signal -- every star has the
# same radius -- and a correlation computed on it would be noise amplified by a
# vanishing denominator. Such batches contribute nothing instead.
MIN_TARGET_SD: float = 1e-3


@dataclass
class DensityBudget:
    """What contrast the ambient graph licenses, measured before training.

    Attributes
    ----------
    dim : float
        Ambient intrinsic dimension (Levina-Bickel).
    d_out : int
        Embedding dimension.
    log_r : (R,) float32
        Ambient log neighbour radius per graph node, the raw measurement.
    """

    dim: float
    d_out: int
    log_r: torch.Tensor

    def on_stars(self, log_r_stars: torch.Tensor) -> "DensityBudget":
        """Same budget, re-measured on the star neighbourhoods the loss uses."""
        return DensityBudget(dim=self.dim, d_out=self.d_out, log_r=log_r_stars)

    @property
    def target(self) -> torch.Tensor:
        """Centred ambient log neighbour radius, per node -- what to correlate with.

        Deliberately *not* scaled by ``dim / d_out``. That factor is the
        equal-area requirement: preserving mass under a dimension change needs
        ``log R = (dim / d_out) log r``. It is correct arithmetic and a bad
        objective, because when ``dim >> d_out`` it demands a log-radius spread
        no attraction/repulsion equilibrium can produce -- on 64-dim digits it
        asked for 3.6x the ambient spread and the layout tore itself apart trying.
        Correlation is invariant to that factor, so dropping it costs nothing and
        removes the infeasible target along with it. ``dim`` is retained for
        reporting only.
        """
        return self.log_r - self.log_r.mean()

    @property
    def ambient_sd(self) -> float:
        """Spread of the ambient log radius: how much contrast the data carries."""
        return float(self.log_r.std())

    def describe(self) -> str:
        return (
            f"intrinsic dim {self.dim:.2f} into {self.d_out}, ambient log-radius "
            f"sd {self.ambient_sd:.3f} over {int(self.log_r.numel())} nodes"
        )


def ambient_log_radius(
    X: torch.Tensor,
    graph: Graph,
    dist_fn: DistanceFn,
    chunk: int = 512,
) -> torch.Tensor:
    """Log distance to the furthest kNN of every graph node, in ambient metric.

    The k-th radius rather than the mean is used because ``k / r_k**m`` is the
    standard kNN density estimator; taking the max over the stored neighbour
    list also sidesteps whether the list includes the node itself.
    """
    rep = graph.reps.rep_idx.to(torch.long)
    Xr = X.index_select(0, rep)
    knn = graph.knn_idx.to(torch.long)
    R = int(Xr.shape[0])
    out = torch.empty(R, dtype=torch.float32)
    for s in range(0, R, chunk):
        e = min(R, s + chunk)
        D = dist_fn(Xr[s:e], Xr)
        out[s:e] = torch.gather(D, 1, knn[s:e]).max(dim=1).values.to(torch.float32)
    return out.clamp_min(1e-12).log()


def star_log_radius(
    X: torch.Tensor,
    rep_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_mask: torch.Tensor,
    dist_fn: DistanceFn,
    chunk: int = 512,
) -> torch.Tensor:
    """Ambient counterpart of :func:`embedded_log_radius`, per graph node.

    Same estimator (RMS distance to the same neighbour set) on the same stars,
    so the two sides of the density term differ only in which space they were
    measured in. Using a different neighbour count, or a fresh draw each step,
    would leave sampling noise in the comparison, and the optimiser pays for
    that by shrinking the contrast it dares to show.
    """
    rep = rep_idx.to(torch.long)
    Xr = X.index_select(0, rep)
    R = int(Xr.shape[0])
    out = torch.empty(R, dtype=torch.float32)
    w = nbr_mask / nbr_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    for s in range(0, R, chunk):
        e = min(R, s + chunk)
        D = dist_fn(Xr[s:e], Xr)
        d = torch.gather(D, 1, nbr_idx[s:e].to(torch.long)).to(torch.float32)
        out[s:e] = 0.5 * torch.log(((w[s:e] * d * d).sum(dim=1)).clamp_min(1e-24))
    return out


def density_budget(
    X: torch.Tensor,
    graph: Graph,
    dist_fn: DistanceFn,
    d_out: int,
    dim: Optional[float] = None,
    seed: int = 0,
) -> DensityBudget:
    """Measure the licensed density contrast from the ambient graph."""
    if dim is None or not (dim == dim) or dim <= 0:  # NaN-safe
        dim = _intrinsic_dim_levina_bickel(X, dist_fn, seed=seed)
    return DensityBudget(
        dim=float(dim), d_out=int(d_out), log_r=ambient_log_radius(X, graph, dist_fn)
    )


def embedded_log_radius(
    z_c: torch.Tensor, z_nbr: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Log RMS distance from each star centre to its neighbours. Differentiable."""
    d2 = ((z_nbr - z_c.unsqueeze(1)) ** 2).sum(dim=-1)
    w = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return 0.5 * torch.log((w * d2).sum(dim=1).clamp_min(eps))


def density_correlation_loss(
    z_c: torch.Tensor,
    z_nbr: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """``1 - corr(embedded log radius, ambient log radius)`` over the batch.

    Scale free in both arguments: multiplying either side by any positive
    constant leaves the value unchanged. That is the whole point -- the layout is
    asked to put the crowded neighbourhoods in the crowded places, in order, and
    never asked to hit a particular amount of contrast. A magnitude target can be
    unreachable and then does damage while failing; an ordering target cannot be,
    so one weight serves every dataset.

    Ranges over ``[0, 2]`` with 1.0 the score of a layout whose density is
    unrelated to the data's, which puts it on the same footing as the other
    unit-scale terms. Degenerate batches -- fewer than two stars, or no spread on
    either side -- score 1.0 with no gradient rather than dividing by ~0.
    """
    if z_c.shape[0] < 2:
        return z_c.sum() * 0.0 + 1.0
    lr = embedded_log_radius(z_c, z_nbr, mask)
    a = lr - lr.mean()
    b = target - target.mean()
    na, nb = a.norm(), b.norm()
    if float(nb.detach()) < MIN_TARGET_SD or float(na.detach()) < MIN_TARGET_SD:
        return z_c.sum() * 0.0 + 1.0
    return 1.0 - (a * b).sum() / (na * nb)

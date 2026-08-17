"""DDP allreduce helpers (PR-7).

Each helper exists to remove **per-rank vs global statistic bias**: when batches
are sharded across ranks, statistics that enter the loss (mean affinity for
entropy, density moments for correlation centering, path-scale batch means)
must be reduced over the full world. At ``world_size==1`` or when the process
group is not initialized they are no-ops (return the local values unchanged).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

CountLike = Union[int, float, torch.Tensor]


def _dist_ready(group: Optional[object] = None) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    return int(torch.distributed.get_world_size(group)) > 1


def allreduce_mean(t: torch.Tensor, group: Optional[object] = None) -> torch.Tensor:
    """Mean of ``t`` across ranks.

    Fixes per-rank bias when a scalar/vector statistic should reflect the
    global batch rather than one shard. No-op when not distributed / ws==1.
    """
    if not _dist_ready(group):
        return t
    out = t.detach().clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=group)
    out = out / float(torch.distributed.get_world_size(group))
    return out


def allreduce_mean_affinity(
    a_bar_local: torch.Tensor, group: Optional[object] = None
) -> torch.Tensor:
    """Allreduce batch-mean affinity ``ā`` (shape ``(L,)``).

    Landmark entropy uses ``H(ā)`` with ``ā = mean_ℓ a``. Averaging only the
    local shard under- or over-represents landmarks that appear unevenly across
    ranks; this returns the world-mean ``ā``. No-op when not distributed / ws==1.
    """
    return allreduce_mean(a_bar_local, group=group)


def allreduce_density_moments(
    mean_local: torch.Tensor,
    sq_mean_local: torch.Tensor,
    count: CountLike,
    group: Optional[object] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine per-rank first/second moments into global moments.

    Density correlation centers with ``x - mean`` and normalizes by variance.
    Averaging local means across ranks is wrong when shards differ; count-weighted
    global moments are required:

    ``μ = Σ_r n_r μ_r / Σ_r n_r``, ``m₂ = Σ_r n_r m₂,r / Σ_r n_r``.

    No-op when not distributed / ws==1 (returns the local moments).
    """
    if not _dist_ready(group):
        return mean_local, sq_mean_local

    device = mean_local.device
    dtype = mean_local.dtype
    n = float(count.item() if isinstance(count, torch.Tensor) else count)
    # Pack count-weighted moments + count for a single SUM allreduce.
    flat_mean = mean_local.detach().reshape(-1).to(dtype=torch.float64)
    flat_sq = sq_mean_local.detach().reshape(-1).to(dtype=torch.float64)
    pack = torch.cat(
        [
            flat_mean * n,
            flat_sq * n,
            torch.tensor([n], device=device, dtype=torch.float64),
        ]
    )
    torch.distributed.all_reduce(pack, op=torch.distributed.ReduceOp.SUM, group=group)
    n_tot = float(pack[-1].item())
    if n_tot <= 0.0:
        return mean_local, sq_mean_local
    k = flat_mean.numel()
    g_mean = (pack[:k] / n_tot).to(device=device, dtype=dtype).reshape_as(mean_local)
    g_sq = (pack[k : 2 * k] / n_tot).to(device=device, dtype=dtype).reshape_as(sq_mean_local)
    return g_mean, g_sq


def allreduce_path_scale(
    batch_mean: torch.Tensor, group: Optional[object] = None
) -> torch.Tensor:
    """Allreduce path (or ordinal) scale batch-mean before the EMA update.

    Path / ordinal hinges divide by a running scale ``s`` fed by per-batch
    ``||z_a - z_f||`` means. A rank-local mean drifts the gauge across GPUs;
    this returns the world-mean batch scale. No-op when not distributed / ws==1.
    """
    return allreduce_mean(batch_mean, group=group)


__all__ = [
    "allreduce_mean",
    "allreduce_mean_affinity",
    "allreduce_density_moments",
    "allreduce_path_scale",
]

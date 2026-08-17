"""DDP / torchrun entry (PR-7).

``fit_ddp`` is identity with :func:`fit` when not distributed or
``world_size==1`` (bit-compat path). Multi-GPU correctness beyond helpers is
intentionally deferred: call sites in :mod:`leanmap.train.fit` should allreduce
train statistics via :func:`sync_train_stats` / :mod:`leanmap.losses.ddp_stats`.

Geo landmark pairs are **replicated** on every rank — do **not** allreduce geo
pair tensors. Path and class-axis remain available under DDP (not stubbed).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..losses.ddp_stats import (
    allreduce_density_moments,
    allreduce_mean_affinity,
    allreduce_path_scale,
)
from .fit import fit

# Re-export core training entry so ``from leanmap.train.ddp import fit`` works
# without demoting path / class-axis (those live on the package root).
from leanmap.path import (  # noqa: F401
    PathConstraint,
    PathTripletSampler,
    path_constraint_loss,
)
from leanmap.classaxis import (  # noqa: F401
    ClassAxis,
    ClassOrderSampler,
    ordinal_class_axis,
)


def seed_for_rank(seed: int, rank: int) -> int:
    """Sampler seed for a rank: ``seed + rank`` (independent shards, shared base)."""
    return int(seed) + int(rank)


def init_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
) -> Tuple[int, int]:
    """Initialize the process group from ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR``.

    Also reads ``MASTER_PORT`` (default ``29500``). Returns ``(rank, world_size)``.
    No-op (returns ``(0, 1)``) when ``WORLD_SIZE<=1`` and the group is not yet
    initialized. If already initialized, returns the live rank / world size.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.distributed.is_available():
        return 0, 1

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    if init_method is None:
        init_method = f"tcp://{master_addr}:{master_port}"
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size


def sync_train_stats(
    *,
    a_bar_local: Optional[torch.Tensor] = None,
    dens_mean_local: Optional[torch.Tensor] = None,
    dens_sq_mean_local: Optional[torch.Tensor] = None,
    dens_count: Optional[Union[int, float]] = None,
    path_batch_mean: Optional[torch.Tensor] = None,
    group: Optional[object] = None,
) -> Dict[str, Any]:
    """Optional allreduce of train-time statistics (wires :mod:`ddp_stats`).

    Intended call sites inside the fit loop (multi-GPU; PR-7 leaves the loop
    itself on the single-process path):

    - **Affinity entropy** — ``allreduce_mean_affinity(a.mean(dim=0))`` before
      ``H(ā)`` / landmark regularisation logging.
    - **Density correlation** — ``allreduce_density_moments(mean, sq_mean, n)``
      before centering / variance in the density term.
    - **Path / ordinal scale** — ``allreduce_path_scale(batch_mean)`` before the
      EMA that feeds hinge scales.
    - **Geo pairs** — replicated on all ranks; **no** allreduce on geo pair
      index tensors or geodesic distances.
    """
    out: Dict[str, Any] = {}
    if a_bar_local is not None:
        out["a_bar"] = allreduce_mean_affinity(a_bar_local, group=group)
    if dens_mean_local is not None and dens_sq_mean_local is not None:
        n = 1.0 if dens_count is None else dens_count
        g_mean, g_sq = allreduce_density_moments(
            dens_mean_local, dens_sq_mean_local, n, group=group
        )
        out["dens_mean"] = g_mean
        out["dens_sq_mean"] = g_sq
    if path_batch_mean is not None:
        out["path_batch_mean"] = allreduce_path_scale(path_batch_mean, group=group)
    return out


def fit_ddp(*args, **kwargs):
    """Distributed training entry.

    If the process group is unset or ``world_size==1``, delegates to
    :func:`leanmap.train.fit.fit` identically (bit-compat). Multi-GPU wiring of
    the full loop is out of scope for PR-7 beyond :func:`sync_train_stats` and
    the allreduce helpers; geo remains replicated (no allreduce on geo pairs).
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return fit(*args, **kwargs)
    # Multi-GPU: same fit entry for now. Future: wrap model in DDP, shard
    # samplers with seed_for_rank, and call sync_train_stats at the sites
    # documented above. Geo pairs stay replicated — no allreduce.
    return fit(*args, **kwargs)


__all__ = [
    "fit_ddp",
    "fit",
    "init_distributed",
    "seed_for_rank",
    "sync_train_stats",
    "PathConstraint",
    "PathTripletSampler",
    "path_constraint_loss",
    "ClassAxis",
    "ClassOrderSampler",
    "ordinal_class_axis",
]

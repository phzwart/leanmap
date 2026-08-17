"""Path constraint loss (ratio hinge default; optional log-space + distance floor)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from leanmap.paths.constraint import (
    PATH_C,
    PATH_C_HI,
    PATH_MARGIN,
    PATH_ORD_CHANCE,
    SPREAD_MOMENTUM,
)

__all__ = [
    "PATH_C",
    "PATH_C_HI",
    "PATH_MARGIN",
    "PATH_ORD_CHANCE",
    "path_constraint_loss",
]


def path_constraint_loss(
    z_a: torch.Tensor,
    z_n: torch.Tensor,
    z_m: torch.Tensor,
    z_f: torch.Tensor,
    dt_n: torch.Tensor,
    dt_m: torch.Tensor,
    *,
    c: float = PATH_C,
    C: float = PATH_C_HI,
    margin: float = PATH_MARGIN,
    scale_state: Optional[Dict[str, float]] = None,
    log_space: bool = False,
    distance_floor_kappa: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float], float]:
    """Ordinal hinges plus bi-Lipschitz speed hinges; zero once satisfied.

    Returns ``loss, scale_state, ord_frac`` where ``ord_frac`` is the fraction
    of rows with ``d_n < d_m < d_f``.

    Parameters
    ----------
    log_space :
        When ``True``, hinge the log-ratio of path speeds (stable near
        vanishing distances). Default ``False`` keeps the legacy ratio hinge.
    distance_floor_kappa :
        When ``> 0``, floor embedding distances at ``kappa * s`` before speed
        ratios / log-ratios (``s`` is the detached far-pair EMA scale).
    """
    if scale_state is None:
        scale_state = {}
    if z_a.shape[0] == 0:
        return z_a.sum() * 0.0, scale_state, 0.0

    d_n = (z_a - z_n).norm(dim=-1)
    d_m = (z_a - z_m).norm(dim=-1)
    d_f = (z_a - z_f).norm(dim=-1)
    dt_n = dt_n.to(device=z_a.device, dtype=z_a.dtype).clamp_min(1e-6)
    dt_m = dt_m.to(device=z_a.device, dtype=z_a.dtype).clamp_min(1e-6)

    with torch.no_grad():
        batch_s = float(d_f.mean().item())
        if not np.isfinite(batch_s) or batch_s <= 0.0:
            batch_s = 1.0
        prev = scale_state.get("s")
        scale_state["s"] = (
            batch_s if prev is None else SPREAD_MOMENTUM * prev + (1.0 - SPREAD_MOMENTUM) * batch_s
        )
    s = max(float(scale_state["s"]), 1e-6)

    ord1 = F.relu(margin - (d_m - d_n) / s)
    ord2 = F.relu(margin - (d_f - d_m) / s)

    # Optional floor before speed matching (near-duplicate windows).
    kappa = float(distance_floor_kappa)
    if kappa > 0.0:
        floor = kappa * s
        d_n_s = d_n.clamp_min(floor)
        d_m_s = d_m.clamp_min(floor)
    else:
        d_n_s = d_n
        d_m_s = d_m

    # Speed matching on the path: d/Δt of the two lags should stay in [c, C]
    # of each other. Using far-pair scale here would forbid long slow walks
    # (N steps of a chain cannot each be a fraction of the cloud diameter).
    v_n = d_n_s / dt_n
    v_m = d_m_s / dt_m
    if log_space:
        # log(v_m / v_n) ∈ [log c, log C]
        log_rel = torch.log(v_m.clamp_min(1e-8)) - torch.log(v_n.clamp_min(1e-8))
        log_c = float(np.log(c))
        log_C = float(np.log(C))
        lip = F.relu(log_c - log_rel) + F.relu(log_rel - log_C)
    else:
        rel = v_m / v_n.clamp_min(1e-8)
        lip = F.relu(c - rel) + F.relu(rel - C)

    loss = ord1.mean() + ord2.mean() + lip.mean()
    with torch.no_grad():
        ord_frac = float(((d_n < d_m) & (d_m < d_f)).float().mean().item())
    return loss, scale_state, ord_frac

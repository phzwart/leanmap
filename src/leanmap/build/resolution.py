"""Resolution contract: Def-1 ε and δ calibration (``solve_delta``).

ε remains Definition 1 (1-NN quantile) in :func:`estimate_epsilon`. δ is a
coarser net radius chosen so expected representative count ``R`` lands in a
target band when fidelity allows; otherwise δ falls back to ε.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .pipeline import (  # re-export Def-1 unchanged
    _intrinsic_dim_levina_bickel,
    _one_nn_all,
    estimate_epsilon,
)
from ..utils import get_logger

ArrayLike = Union[torch.Tensor, np.ndarray]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nn1_from_probe(probe: np.ndarray) -> np.ndarray:
    """Extract per-point 1-NN distances from a 1-D vector or pairwise matrix."""
    if probe.ndim == 1:
        return probe.astype(np.float64, copy=False)
    if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
        raise ValueError(
            "probe_nn1 must be a 1-D 1-NN vector or a square pairwise distance matrix"
        )
    d = probe.astype(np.float64, copy=True)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _eps_from_nn1(nn1: np.ndarray, quantile: float = 0.01) -> float:
    if nn1.size == 0:
        return 0.0
    eps = float(np.quantile(nn1, quantile))
    if eps <= 0.0:
        pos = nn1[nn1 > 0]
        if pos.size > 0:
            eps = float(np.median(pos))
        else:
            eps = 1e-12
    return eps


def _greedy_ball_cover_count(D: np.ndarray, radius: float) -> int:
    """Count centers in a deterministic greedy ball cover on a distance matrix."""
    n = int(D.shape[0])
    if n == 0:
        return 0
    if radius < 0.0:
        return n
    uncovered = np.ones(n, dtype=bool)
    centers = 0
    # Visit in index order for determinism (matches a fixed RNG shuffle of 0..n-1).
    while uncovered.any():
        i = int(np.flatnonzero(uncovered)[0])
        centers += 1
        uncovered &= D[i] > radius
    return centers


def _estimate_r(
    probe: np.ndarray,
    nn1: np.ndarray,
    radius: float,
    n_rows: int,
) -> float:
    """Estimate net size R at ``radius`` from the probe, scaled to ``n_rows``."""
    p = int(nn1.shape[0])
    if p == 0:
        return float(n_rows)
    if probe.ndim == 2 and probe.shape[0] == probe.shape[1]:
        centers = _greedy_ball_cover_count(probe, radius)
    else:
        # 1-NN packing heuristic: points whose nearest neighbour is farther
        # than ``radius`` cannot be absorbed and must open a cell.
        centers = int(np.sum(nn1 > radius))
        if centers == 0 and np.any(nn1 >= 0):
            # Everything merges at this radius on the probe.
            centers = 1
    scale = float(n_rows) / float(p)
    return max(1.0, float(centers) * scale)


def _pair_fidelity(
    probe: np.ndarray,
    nn1: np.ndarray,
    delta: float,
    eps: float,
) -> float:
    """Fraction of δ-close structure that is also ε-close (α fidelity).

    Pairwise: among off-diagonal pairs with φ ≤ δ, fraction with φ ≤ ε.
    1-NN: among probe points with nn1 ≤ δ, fraction with nn1 ≤ ε.
    """
    if delta <= eps:
        return 1.0
    if probe.ndim == 2 and probe.shape[0] == probe.shape[1]:
        iu = np.triu_indices(probe.shape[0], k=1)
        d = probe[iu]
        within_delta = d <= delta
        n_delta = int(within_delta.sum())
        if n_delta == 0:
            return 1.0
        return float(np.sum(d[within_delta] <= eps) / n_delta)
    within_delta = nn1 <= delta
    n_delta = int(within_delta.sum())
    if n_delta == 0:
        return 1.0
    return float(np.sum(nn1[within_delta] <= eps) / n_delta)


def solve_delta(
    probe_nn1: ArrayLike,
    r_band: Tuple[float, float] = (1e5, 1e6),
    alpha_guard: float = 0.95,
    n_rows: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Choose δ ≥ ε so expected net size R lands in ``r_band`` when possible.

    Parameters
    ----------
    probe_nn1 :
        Per-probe 1-NN distances ``(P,)`` or a square pairwise distance matrix
        ``(P, P)``. Pairwise input enables a greedy ball-cover R estimate.
    r_band :
        Target ``(R_lo, R_hi)`` for the scaled net size.
    alpha_guard :
        Minimum fidelity (fraction of δ-close pairs/points that are also
        ε-close). Failure falls back to δ = ε with ``mode="auto_fallback"``.
    n_rows :
        Ambient row count ``N``. Defaults to the probe size when omitted.

    Returns
    -------
    delta : float
    report : dict
        Keys: ``delta``, ``eps_ref``, ``r_est``, ``r_band``, ``alpha_guard``,
        ``guard_ok``, ``mode`` (``"eps"`` | ``"calibrated"`` | ``"auto_fallback"``).
    """
    log = get_logger()
    probe = _as_numpy(probe_nn1)
    nn1 = _nn1_from_probe(probe)
    p = int(nn1.shape[0])
    n = int(n_rows) if n_rows is not None else max(p, 1)
    r_lo, r_hi = float(r_band[0]), float(r_band[1])
    if r_lo > r_hi:
        r_lo, r_hi = r_hi, r_lo

    eps_ref = _eps_from_nn1(nn1)
    r_at_eps = _estimate_r(probe, nn1, eps_ref, n)

    def _report(
        delta: float,
        *,
        r_est: float,
        guard_ok: bool,
        mode: str,
        fidelity: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "delta": float(delta),
            "eps_ref": float(eps_ref),
            "r_est": float(r_est),
            "r_band": (r_lo, r_hi),
            "alpha_guard": float(alpha_guard),
            "guard_ok": bool(guard_ok),
            "mode": mode,
        }
        if fidelity is not None:
            out["fidelity"] = float(fidelity)
        out.update(extra)
        return out

    # Already in band at ε — no coarsening needed.
    if r_lo <= r_at_eps <= r_hi:
        return eps_ref, _report(
            eps_ref, r_est=r_at_eps, guard_ok=True, mode="eps", fidelity=1.0
        )

    # N itself below the band floor: enlarging δ only shrinks R further.
    if n < r_lo:
        return eps_ref, _report(
            eps_ref,
            r_est=r_at_eps,
            guard_ok=True,
            mode="eps",
            fidelity=1.0,
            note="n_rows_below_band",
        )

    # Candidate radii: ε plus unique probe distances ≥ ε.
    if probe.ndim == 2 and probe.shape[0] == probe.shape[1]:
        iu = np.triu_indices(p, k=1)
        dists = probe[iu]
    else:
        dists = nn1
    cand = np.unique(dists[np.isfinite(dists) & (dists >= eps_ref)])
    if cand.size == 0:
        cand = np.asarray([eps_ref], dtype=np.float64)
    delta_max = float(cand[-1])
    # If R at ε is still above the band, search larger radii.
    # Binary search on [eps_ref, delta_max] for R in band.
    lo, hi = float(eps_ref), delta_max
    best_delta = float(eps_ref)
    best_r = r_at_eps
    best_in_band = r_lo <= r_at_eps <= r_hi
    collapsed = False

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        r_mid = _estimate_r(probe, nn1, mid, n)
        if r_mid < 1.5 or r_mid / max(n, 1) < 1e-6:
            collapsed = True
        if r_lo <= r_mid <= r_hi:
            best_delta, best_r, best_in_band = mid, r_mid, True
            # Prefer the smallest δ that still lands in band (least aggressive).
            hi = mid
        elif r_mid > r_hi:
            # Need more compression → larger δ.
            lo = mid
            if not best_in_band and abs(r_mid - 0.5 * (r_lo + r_hi)) < abs(
                best_r - 0.5 * (r_lo + r_hi)
            ):
                best_delta, best_r = mid, r_mid
        else:
            # r_mid < r_lo: too aggressive → smaller δ.
            hi = mid
            if not best_in_band and abs(r_mid - 0.5 * (r_lo + r_hi)) < abs(
                best_r - 0.5 * (r_lo + r_hi)
            ):
                best_delta, best_r = mid, r_mid
        if hi - lo <= max(1e-15, 1e-9 * max(eps_ref, 1.0)):
            break

    if collapsed or best_r < max(2.0, 0.01 * r_lo):
        log.warning(
            "delta calibration would collapse net size (r_est=%.3g at delta=%.6g, "
            "n_rows=%d, r_band=%s); falling back to delta=eps",
            best_r,
            best_delta,
            n,
            (r_lo, r_hi),
        )
        return eps_ref, _report(
            eps_ref,
            r_est=r_at_eps,
            guard_ok=False,
            mode="auto_fallback",
            fidelity=_pair_fidelity(probe, nn1, best_delta, eps_ref),
            collapse_warned=True,
            candidate_delta=float(best_delta),
            candidate_r_est=float(best_r),
        )

    fidelity = _pair_fidelity(probe, nn1, best_delta, eps_ref)
    guard_ok = fidelity >= float(alpha_guard)

    if not guard_ok or not best_in_band:
        if not guard_ok:
            log.warning(
                "delta alpha_guard failed: fidelity=%.4f < alpha_guard=%.4f "
                "(delta=%.6g, eps=%.6g); falling back to delta=eps",
                fidelity,
                alpha_guard,
                best_delta,
                eps_ref,
            )
        return eps_ref, _report(
            eps_ref,
            r_est=r_at_eps,
            guard_ok=False,
            mode="auto_fallback",
            fidelity=fidelity,
            candidate_delta=float(best_delta),
            candidate_r_est=float(best_r),
            in_band=bool(best_in_band),
        )

    mode = "eps" if abs(best_delta - eps_ref) <= 1e-15 * (1.0 + abs(eps_ref)) else "calibrated"
    return float(best_delta), _report(
        best_delta,
        r_est=best_r,
        guard_ok=True,
        mode=mode,
        fidelity=fidelity,
    )


__all__ = [
    "estimate_epsilon",
    "solve_delta",
    "_one_nn_all",
    "_intrinsic_dim_levina_bickel",
]

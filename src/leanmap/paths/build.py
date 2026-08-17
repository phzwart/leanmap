"""Path triplet construction (vectorized searchsorted + tie / ε filters)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from leanmap.diagnostics.record import DiagnosticsRecord

__all__ = [
    "build_path_triplets",
    "build_path_triplets_with_stats",
    "remap_triplets",
    "record_path_build_stats",
]

_TIE_POLICIES = frozenset({"first", "last", "drop"})


def record_path_build_stats(
    diagnostics: Optional[DiagnosticsRecord],
    stats: Dict[str, Any],
) -> None:
    """Merge path-build ``stats`` into ``DiagnosticsRecord.extra`` when present."""
    if diagnostics is None:
        return
    for key, val in stats.items():
        diagnostics.extra[key] = val


def _resolve_dist_callable(dist_fn: Any) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a paired distance ``(Xa, Xb) -> (n,)`` float64 array."""
    if dist_fn is None:
        raise ValueError("dist_fn is required when eps filtering")

    if isinstance(dist_fn, str):
        from leanmap.metrics import get_metric

        spec = get_metric(dist_fn)
        fn = spec
    else:
        fn = dist_fn

    def _paired(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        import torch

        A = torch.as_tensor(np.asarray(Xa, dtype=np.float32), dtype=torch.float32)
        B = torch.as_tensor(np.asarray(Xb, dtype=np.float32), dtype=torch.float32)
        n = int(A.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        # DistanceFn protocol returns (n, m); take diagonal in chunks.
        chunk = 256
        out = np.empty(n, dtype=np.float64)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d = fn(A[s:e], B[s:e])
            if hasattr(d, "detach"):
                d = d.detach().cpu().numpy()
            d = np.asarray(d, dtype=np.float64)
            if d.ndim == 2:
                out[s:e] = np.diag(d)
            else:
                out[s:e] = d.reshape(-1)[: e - s]
        return out

    return _paired


def _collapse_ties(
    t_s: np.ndarray,
    r_s: np.ndarray,
    tie_policy: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Collapse duplicate ``t`` values per ``tie_policy``.

    Returns unique ``(t_u, r_u)``, ``n_tie_values`` (distinct t with multiplicity>1),
    and ``n_tie_rows`` (rows discarded or overwritten by the policy).
    """
    if t_s.size == 0:
        return t_s, r_s, 0, 0

    # mergesort is stable: first occurrence precedes later ones.
    # Boundaries of equal-t runs.
    change = np.empty(t_s.size, dtype=bool)
    change[0] = True
    change[1:] = t_s[1:] != t_s[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, t_s.size))
    multi = counts > 1
    n_tie_values = int(multi.sum())
    n_tie_rows = int(counts[multi].sum()) if n_tie_values else 0

    if tie_policy == "drop":
        keep_starts = starts[~multi] if n_tie_values else starts
        # one representative per non-tied value (only one row each)
        return t_s[keep_starts], r_s[keep_starts], n_tie_values, n_tie_rows

    if tie_policy == "first":
        # first row of each run
        pick = starts
    elif tie_policy == "last":
        pick = starts + counts - 1
    else:
        raise ValueError(f"tie_policy must be one of {sorted(_TIE_POLICIES)}, got {tie_policy!r}")

    return t_s[pick], r_s[pick], n_tie_values, n_tie_rows


def _build_group_triplets(
    t_u: np.ndarray,
    r_u: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``searchsorted`` exact matches for ``t+1`` and ``t+lag`` within a group."""
    if t_u.size == 0:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )
    target_n = t_u + 1.0
    target_m = t_u + float(lag)
    j_n = np.searchsorted(t_u, target_n, side="left")
    j_m = np.searchsorted(t_u, target_m, side="left")
    n = t_u.size
    ok = (
        (j_n < n)
        & (j_m < n)
        & (t_u[np.minimum(j_n, n - 1)] == target_n)
        & (t_u[np.minimum(j_m, n - 1)] == target_m)
    )
    # Guard indices (ok already requires j < n, but keep safe for empty).
    j_n = j_n[ok]
    j_m = j_m[ok]
    a = r_u[ok]
    rows = np.stack([a, r_u[j_n], r_u[j_m]], axis=1).astype(np.int64, copy=False)
    dts = np.column_stack(
        [
            np.ones(rows.shape[0], dtype=np.float32),
            np.full(rows.shape[0], float(lag), dtype=np.float32),
        ]
    )
    return rows, dts


def build_path_triplets_with_stats(
    group: np.ndarray,
    index: np.ndarray,
    lag: int = 8,
    *,
    tie_policy: str = "first",
    eps: Optional[float] = None,
    X: Optional[np.ndarray] = None,
    dist_fn: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Same-group triples ``(i, i+1, i+lag)`` in *index* units, with build stats.

    Within each group, rows are sorted by ``index`` and near/mid endpoints are
    resolved with ``searchsorted`` (exact ``t`` match). Duplicate ``t`` values
    are handled by ``tie_policy`` ∈ {``first``, ``last``, ``drop``}.

    Optional ε-filter: when ``eps``, ``X``, and ``dist_fn`` are all given, drop
    triples with ``φ(x_a, x_near) <= eps``.
    """
    group = np.asarray(group)
    index = np.asarray(index, dtype=np.float64)
    lag = int(lag)
    if lag < 2:
        raise ValueError("lag must be >= 2")
    if tie_policy not in _TIE_POLICIES:
        raise ValueError(
            f"tie_policy must be one of {sorted(_TIE_POLICIES)}, got {tie_policy!r}"
        )

    stats: Dict[str, Any] = {
        "path_tie_policy": tie_policy,
        "path_tie_values": 0,
        "path_tie_rows": 0,
        "path_eps_dropped": 0,
        "path_n_triplets": 0,
    }

    row_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        t = index[idx]
        order = np.argsort(t, kind="mergesort")
        t_s = t[order]
        r_s = idx[order]
        t_u, r_u, n_tie_val, n_tie_rows = _collapse_ties(t_s, r_s, tie_policy)
        stats["path_tie_values"] = int(stats["path_tie_values"]) + n_tie_val
        stats["path_tie_rows"] = int(stats["path_tie_rows"]) + n_tie_rows
        rows, dts = _build_group_triplets(t_u, r_u, lag)
        if rows.shape[0]:
            row_parts.append(rows)
            dt_parts.append(dts)

    if not row_parts:
        empty_tri = np.zeros((0, 3), dtype=np.int64)
        empty_dt = np.zeros((0, 2), dtype=np.float32)
        return empty_tri, empty_dt, stats

    triplets = np.concatenate(row_parts, axis=0)
    triplet_dt = np.concatenate(dt_parts, axis=0)

    if eps is not None:
        if X is None or dist_fn is None:
            raise ValueError("eps filtering requires both X= and dist_fn=")
        paired = _resolve_dist_callable(dist_fn)
        X = np.asarray(X)
        d_near = paired(X[triplets[:, 0]], X[triplets[:, 1]])
        keep = d_near > float(eps)
        n_drop = int((~keep).sum())
        stats["path_eps_dropped"] = n_drop
        triplets = triplets[keep]
        triplet_dt = triplet_dt[keep]

    stats["path_n_triplets"] = int(triplets.shape[0])
    return triplets, triplet_dt, stats


def build_path_triplets(
    group: np.ndarray,
    index: np.ndarray,
    lag: int = 8,
    *,
    tie_policy: str = "first",
    eps: Optional[float] = None,
    X: Optional[np.ndarray] = None,
    dist_fn: Any = None,
    diagnostics: Optional[DiagnosticsRecord] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same-group triples ``(i, i+1, i+lag)`` in *index* units, not row order.

    Bit-compatible with the legacy dict lookup on inputs without duplicate ``t``.
    Pass ``diagnostics=`` to record tie / ε counts into ``DiagnosticsRecord.extra``.
    """
    tri, dt, stats = build_path_triplets_with_stats(
        group,
        index,
        lag,
        tie_policy=tie_policy,
        eps=eps,
        X=X,
        dist_fn=dist_fn,
    )
    record_path_build_stats(diagnostics, stats)
    return tri, dt


def remap_triplets(
    triplets: np.ndarray,
    triplet_dt: np.ndarray,
    keep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``keep`` is original row indices in the subset, in subset order."""
    keep = np.asarray(keep, dtype=np.int64)
    triplets = np.asarray(triplets, dtype=np.int64)
    triplet_dt = np.asarray(triplet_dt, dtype=np.float32)
    if keep.size == 0 or triplets.size == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    n_all = int(max(int(keep.max()), int(triplets.max())) + 1)
    inv = np.full(n_all, -1, dtype=np.int64)
    inv[keep] = np.arange(keep.shape[0], dtype=np.int64)
    a, n, m = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    ok = (a < n_all) & (n < n_all) & (m < n_all)
    ok &= (inv[a] >= 0) & (inv[n] >= 0) & (inv[m] >= 0)
    if not bool(ok.any()):
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    tri = np.stack([inv[a[ok]], inv[n[ok]], inv[m[ok]]], axis=1)
    return tri, np.asarray(triplet_dt[ok], dtype=np.float32)

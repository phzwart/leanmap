"""Overlapping epoch active sets (train-time pass regularization).

Each training epoch can restrict sampling to an active raw-row set ``A_t``.
Consecutive epochs keep a tunable fraction of the previous set (default 20%)
and refill the rest from outside ``A_{t-1}``. This regularizes which points
are visited without rebuilding the frozen graph.

See also :func:`estimate_cover_passes` for how many epochs are needed to
expect each of the ``N`` points to land in an active set ``n_visits`` times.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..graph import Representatives


def next_epoch_active_set(
    n: int,
    active_size: int,
    prev_active: Optional[np.ndarray],
    overlap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the next active raw-row index set with controlled overlap.

    Parameters
    ----------
    n :
        Ambient row count.
    active_size :
        Target ``|A_t|`` (clamped to ``[1, n]``).
    prev_active :
        Previous epoch's indices, or ``None`` on the first epoch.
    overlap :
        Fraction of ``A_t`` taken from ``prev_active`` (clamped to ``[0, 1]``).
        Remaining slots are drawn uniformly from rows **not** kept.
    rng :
        NumPy generator.

    Returns
    -------
    active : ``(active_size,)`` int64, sorted for stable diagnostics.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    size = int(max(1, min(int(active_size), n)))
    rho = float(np.clip(overlap, 0.0, 1.0))

    keep = np.empty(0, dtype=np.int64)
    if prev_active is not None and size > 0 and rho > 0.0:
        prev = np.unique(np.asarray(prev_active, dtype=np.int64).reshape(-1))
        prev = prev[(prev >= 0) & (prev < n)]
        n_keep = int(round(rho * size))
        n_keep = min(n_keep, size, int(prev.size))
        if n_keep > 0:
            pick = rng.choice(prev.size, size=n_keep, replace=False)
            keep = prev[pick]

    n_new = size - int(keep.size)
    if n_new <= 0:
        return np.sort(keep[:size])

    # Fresh slots come from outside the *entire* previous active set so that
    # |A_t ∩ A_{t-1}| / |A_t| ≈ overlap (not larger via accidental re-draws).
    if prev_active is not None:
        prev_all = np.unique(np.asarray(prev_active, dtype=np.int64).reshape(-1))
        prev_all = prev_all[(prev_all >= 0) & (prev_all < n)]
        mask = np.ones(n, dtype=bool)
        if prev_all.size:
            mask[prev_all] = False
        pool = np.flatnonzero(mask)
    elif keep.size:
        mask = np.ones(n, dtype=bool)
        mask[keep] = False
        pool = np.flatnonzero(mask)
    else:
        pool = np.arange(n, dtype=np.int64)

    if int(pool.size) < n_new:
        # Degenerate: not enough outsiders (e.g. size close to n). Fill from
        # non-kept first, then allow refill from previous if needed.
        if keep.size:
            outsiders_mask = np.ones(n, dtype=bool)
            outsiders_mask[keep] = False
            outsiders = np.flatnonzero(outsiders_mask)
            need = n_new
            parts = []
            if outsiders.size:
                take = min(need, int(outsiders.size))
                parts.append(rng.choice(outsiders, size=take, replace=False))
                need -= take
            if need > 0:
                parts.append(rng.choice(keep, size=need, replace=False))
            new = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        else:
            new = rng.choice(np.arange(n, dtype=np.int64), size=n_new, replace=False)
    else:
        new = rng.choice(pool, size=n_new, replace=False)

    out = np.concatenate([keep, np.asarray(new, dtype=np.int64)])
    return np.sort(out[:size].astype(np.int64, copy=False))


def estimate_cover_passes(
    n: int,
    active_size: int,
    overlap: float,
    n_visits: int = 1,
) -> Dict[str, Any]:
    """Estimate epochs needed to visit every point about ``n_visits`` times.

    Under a steady overlapping schedule each epoch keeps ``ρ · B`` rows from
    the previous active set and draws ``(1-ρ) · B`` **fresh** rows from
    outside that kept set. Mean fresh introductions per epoch::

        fresh = (1 - overlap) * active_size
        epochs_fresh ≈ ceil(n_visits * N / fresh)

    Slot-fill estimator (counts keep+fresh as active-set visits)::

        epochs_slots ≈ ceil(n_visits * N / active_size)

    ``epochs_fresh`` is the primary planning number when circulating through
    the corpus; ``epochs`` in the returned dict equals ``epochs_fresh``.
    """
    n = int(max(n, 1))
    B = int(max(1, min(int(active_size), n)))
    rho = float(np.clip(overlap, 0.0, 1.0))
    nv = int(max(1, n_visits))
    fresh = (1.0 - rho) * float(B)
    if fresh <= 1e-12:
        epochs_fresh = nv if B >= n else 10**18
    else:
        epochs_fresh = int(math.ceil(nv * n / fresh))
    epochs_slots = int(math.ceil(nv * n / float(B)))
    return {
        "n": n,
        "active_size": B,
        "overlap": rho,
        "n_visits": nv,
        "fresh_per_epoch": fresh,
        "kept_per_epoch": rho * float(B),
        "epochs_fresh": int(epochs_fresh),
        "epochs_slots": int(epochs_slots),
        "epochs": int(epochs_fresh),
        "formula": "epochs ≈ n_visits * N / ((1 - overlap) * active_size)",
    }


def format_cover_passes(report: Dict[str, Any]) -> str:
    """One-line human summary of :func:`estimate_cover_passes`."""
    return (
        f"cover≈{report['epochs']} epochs for {report['n_visits']}× visits "
        f"(N={report['n']}, B={report['active_size']}, overlap={report['overlap']:.2f}, "
        f"fresh/epoch≈{report['fresh_per_epoch']:.0f}; "
        f"slot-fill≈{report['epochs_slots']})"
    )


def cell_intersects_active(
    reps: Representatives,
    active_idx: np.ndarray,
) -> np.ndarray:
    """Return ``(R,)`` bool: cell has at least one member in ``active_idx``."""
    active = np.asarray(active_idx, dtype=np.int64).reshape(-1)
    n = int(reps.member_of.shape[0])
    mask = np.zeros(n, dtype=bool)
    if active.size:
        active = active[(active >= 0) & (active < n)]
        mask[active] = True
    R = int(reps.rep_idx.shape[0])
    hit = np.zeros(R, dtype=bool)
    off = reps.offsets.detach().cpu().numpy().astype(np.int64)
    vals = reps.values.detach().cpu().numpy().astype(np.int64)
    for c in range(R):
        s, e = int(off[c]), int(off[c + 1])
        if e > s and mask[vals[s:e]].any():
            hit[c] = True
    return hit


def edge_weights_for_active_cells(
    edges: np.ndarray,
    base_weights: np.ndarray,
    cell_active: np.ndarray,
    *,
    require_both: bool = True,
) -> np.ndarray:
    """Mask edge mass to cells that intersect the epoch active set."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.maximum(np.asarray(base_weights, dtype=np.float64).reshape(-1), 1e-12)
    ca = np.asarray(cell_active, dtype=bool).reshape(-1)
    if e.shape[0] == 0:
        return w.copy()
    a = ca[e[:, 0]]
    b = ca[e[:, 1]]
    if require_both:
        keep = a & b
        if not bool(keep.any()):
            keep = a | b
    else:
        keep = a | b
    out = w * keep.astype(np.float64)
    if float(out.sum()) <= 0.0:
        return w.copy()
    return np.maximum(out, 1e-12)


def active_member_csr(
    reps: Representatives,
    active_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """CSR over active members only; empty cells get a ``-1`` placeholder."""
    active = np.asarray(active_idx, dtype=np.int64).reshape(-1)
    n = int(reps.member_of.shape[0])
    is_active = np.zeros(n, dtype=bool)
    if active.size:
        active = active[(active >= 0) & (active < n)]
        is_active[active] = True
    R = int(reps.rep_idx.shape[0])
    off_in = reps.offsets.detach().cpu().numpy().astype(np.int64)
    val_in = reps.values.detach().cpu().numpy().astype(np.int64)
    pieces = []
    offsets = np.zeros(R + 1, dtype=np.int64)
    cursor = 0
    for c in range(R):
        s, e = int(off_in[c]), int(off_in[c + 1])
        mem = val_in[s:e]
        hit = mem[is_active[mem]] if mem.size else mem
        if hit.size == 0:
            pieces.append(np.asarray([-1], dtype=np.int64))
            cursor += 1
        else:
            pieces.append(hit.astype(np.int64, copy=False))
            cursor += int(hit.size)
        offsets[c + 1] = cursor
    values = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    return offsets, values

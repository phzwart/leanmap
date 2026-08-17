"""Declared path constraint: bi-Lipschitz in a user index, sampled as triplets.

A path is not a metric neighbour. Consecutive overlapping windows (or frames,
or time steps) may be far in the ambient feature, yet the caller still wants
``z`` to change at a bounded speed along a scalar index ``t``. That request
stays out of the kNN / ε-net / pyramid — the same isolation
:mod:`leanmap.classaxis` uses for label order.

The Lipschitz is in ``t``, not in ambient ``x``. A Jacobian penalty
``||dz/dx||_F ≈ 1`` was deleted from the training objective because it fixes
the embedding scale; every other term is scale-free. Distances here are
divided by a detached EMA of unrelated-pair ``||z_a - z_far||``.

Stored triples are ``(anchor, near, mid)`` on the *same* path with
``Δt_near < Δt_mid``. Random ``far`` is drawn in the trainer so negatives
track the train split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PATH_MARGIN: float = 0.05
PATH_C: float = 0.25
PATH_C_HI: float = 4.0
PATH_PAIRS_PER_STEP: int = 256
SPREAD_MOMENTUM: float = 0.9
PATH_ORD_CHANCE: float = 1.0 / 6.0


def _as_int64(a: Any, n: int, what: str) -> np.ndarray:
    x = np.asarray(a, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != n:
        raise ValueError(f"{what} must have shape (T, {n}), got {x.shape}")
    if x.size and int(x.min()) < 0:
        raise ValueError(f"{what} contains negative indices")
    return np.ascontiguousarray(x)


def _as_dt(a: Any, t: int) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    if x.ndim != 2 or x.shape != (t, 2):
        raise ValueError(f"triplet_dt must have shape (T, 2), got {x.shape}")
    if np.any(x <= 0) or not np.isfinite(x).all():
        raise ValueError("triplet_dt must be positive and finite")
    if np.any(x[:, 0] >= x[:, 1]):
        raise ValueError("triplet_dt requires dt_near < dt_mid on every row")
    return np.ascontiguousarray(x)


@dataclass
class PathConstraint:
    """One named path regulariser backed by an explicit triplet table.

    ``triplets[k] = (a, near, mid)`` indexes rows of ``X`` *before* the
    calibration split. ``triplet_dt[k] = (|t_near-t_a|, |t_mid-t_a|)``.
    """

    name: str
    triplets: np.ndarray
    triplet_dt: np.ndarray
    weight: float = 1.0
    c: float = PATH_C
    C: float = PATH_C_HI

    def __post_init__(self) -> None:
        self.triplets = _as_int64(self.triplets, 3, "triplets")
        t = int(self.triplets.shape[0])
        self.triplet_dt = _as_dt(self.triplet_dt, t)
        if self.weight <= 0:
            raise ValueError(f"PathConstraint {self.name!r} weight must be > 0")
        if not (0.0 < self.c < 1.0 < self.C):
            raise ValueError(
                f"PathConstraint {self.name!r} needs 0 < c < 1 < C, got c={self.c} C={self.C}"
            )
        if int(self.triplets.shape[0]) == 0:
            raise ValueError(f"PathConstraint {self.name!r} has no triplets")

    def n_rows_required(self) -> int:
        return int(self.triplets.max()) + 1

    def restrict(self, train_idx: np.ndarray, n_all: int) -> Optional["PathConstraint"]:
        """Keep triples whose three endpoints lie in ``train_idx``; remap to 0..n_train-1."""
        inv = np.full(int(n_all), -1, dtype=np.int64)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        inv[train_idx] = np.arange(train_idx.shape[0], dtype=np.int64)
        a, n, m = self.triplets[:, 0], self.triplets[:, 1], self.triplets[:, 2]
        if int(self.triplets.max()) >= n_all:
            raise ValueError(
                f"PathConstraint {self.name!r} indexes row {int(self.triplets.max())} "
                f"but X has {n_all} rows"
            )
        keep = (inv[a] >= 0) & (inv[n] >= 0) & (inv[m] >= 0)
        if not bool(keep.any()):
            return None
        tri = np.stack([inv[a[keep]], inv[n[keep]], inv[m[keep]]], axis=1)
        return PathConstraint(
            name=self.name,
            triplets=tri,
            triplet_dt=self.triplet_dt[keep],
            weight=self.weight,
            c=self.c,
            C=self.C,
        )


class PathTripletSampler:
    def __init__(self, X: torch.Tensor, constraint: PathConstraint, seed: int = 0):
        self.X = X
        self.triplets = constraint.triplets
        self.dt = constraint.triplet_dt
        self.n = int(X.shape[0])
        self.T = int(self.triplets.shape[0])
        self.rng = np.random.default_rng(seed)
        if self.T == 0:
            raise ValueError("PathTripletSampler: empty triplet table")

    def sample(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pick = self.rng.integers(0, self.T, size=int(n))
        tri = self.triplets[pick]
        dt = self.dt[pick]
        far = self.rng.integers(0, self.n, size=int(n))
        idx_a = torch.as_tensor(tri[:, 0], dtype=torch.int64)
        idx_n = torch.as_tensor(tri[:, 1], dtype=torch.int64)
        idx_m = torch.as_tensor(tri[:, 2], dtype=torch.int64)
        idx_f = torch.as_tensor(far, dtype=torch.int64)
        return (
            self.X[idx_a],
            self.X[idx_n],
            self.X[idx_m],
            self.X[idx_f],
            torch.as_tensor(dt[:, 0], dtype=torch.float32),
            torch.as_tensor(dt[:, 1], dtype=torch.float32),
        )


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
) -> Tuple[torch.Tensor, Dict[str, float], float]:
    """Ordinal hinges plus bi-Lipschitz speed hinges; zero once satisfied.

    Returns ``loss, scale_state, ord_frac`` where ``ord_frac`` is the fraction
    of rows with ``d_n < d_m < d_f``.
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
    # Speed matching on the path: d/Δt of the two lags should stay in [c, C]
    # of each other. Using far-pair scale here would forbid long slow walks
    # (N steps of a chain cannot each be a fraction of the cloud diameter).
    v_n = d_n / dt_n
    v_m = d_m / dt_m
    rel = v_m / v_n.clamp_min(1e-8)
    lip = F.relu(c - rel) + F.relu(rel - C)
    loss = ord1.mean() + ord2.mean() + lip.mean()
    with torch.no_grad():
        ord_frac = float(((d_n < d_m) & (d_m < d_f)).float().mean().item())
    return loss, scale_state, ord_frac


def parse_index(start) -> np.ndarray:
    """Numeric sequence index from PDB-style seqids (``12``, ``12A``)."""
    out = np.empty(len(start), dtype=np.float32)
    for i, s in enumerate(start):
        if isinstance(s, bytes):
            s = s.decode()
        s = str(s).strip()
        num = "".join(ch for ch in s if ch.isdigit() or ch == "-")
        out[i] = float(int(num)) if num not in ("", "-") else float("nan")
    return out


def encode_groups(pdb_id: np.ndarray, chain: np.ndarray) -> np.ndarray:
    keys = []
    for p, c in zip(pdb_id, chain):
        if isinstance(p, bytes):
            p = p.decode()
        if isinstance(c, bytes):
            c = c.decode()
        keys.append(f"{p}|{str(c).strip()}")
    _, group = np.unique(np.array(keys), return_inverse=True)
    return group.astype(np.int32)


def build_path_triplets(
    group: np.ndarray,
    index: np.ndarray,
    lag: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same-group triples ``(i, i+1, i+lag)`` in *index* units, not row order."""
    group = np.asarray(group)
    index = np.asarray(index, dtype=np.float64)
    lag = int(lag)
    if lag < 2:
        raise ValueError("lag must be >= 2")
    rows: list[list[int]] = []
    dts: list[list[float]] = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        t = index[idx]
        order = np.argsort(t, kind="mergesort")
        t_s = t[order]
        r_s = idx[order]
        lookup = {}
        for tt, rr in zip(t_s, r_s):
            lookup[float(tt)] = int(rr)
        for tt, rr in zip(t_s, r_s):
            nkey = float(tt) + 1.0
            mkey = float(tt) + float(lag)
            if nkey in lookup and mkey in lookup:
                rows.append([int(rr), lookup[nkey], lookup[mkey]])
                dts.append([1.0, float(lag)])
    if not rows:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )
    return np.asarray(rows, dtype=np.int64), np.asarray(dts, dtype=np.float32)


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


def subset_by_group(
    group: np.ndarray,
    n_keep: int,
    seed: int = 0,
) -> np.ndarray:
    """Row indices covering whole groups until ``n_keep`` rows (sorted)."""
    group = np.asarray(group)
    rng = np.random.default_rng(seed)
    gids = np.unique(group)
    rng.shuffle(gids)
    chosen: list[np.ndarray] = []
    n = 0
    for g in gids:
        rows = np.flatnonzero(group == g)
        chosen.append(rows)
        n += int(rows.size)
        if n >= n_keep:
            break
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(chosen))

"""Vose alias tables (single-level and two-level)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Any]


def _alias_setup(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vose alias table for weighted sampling. Returns (prob, alias)."""
    n = len(weights)
    prob = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int64)
    w = weights.astype(np.float64)
    s = w.sum()
    if s <= 0:
        w = np.ones(n, dtype=np.float64)
        s = float(n)
    w = w / s * n
    small, large = [], []
    for i, qi in enumerate(w):
        (small if qi < 1.0 else large).append(i)
    while small and large:
        s_i, l_i = small.pop(), large.pop()
        prob[s_i] = w[s_i]
        alias[s_i] = l_i
        w[l_i] = w[l_i] + w[s_i] - 1.0
        (small if w[l_i] < 1.0 else large).append(l_i)
    for rem in large + small:
        prob[rem] = 1.0
        alias[rem] = rem
    return prob, alias


def _alias_draw(
    n_draw: int,
    prob: np.ndarray,
    alias: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(prob)
    kk = rng.integers(0, n, size=n_draw)
    accept = rng.random(n_draw) < prob[kk]
    out = kk.copy()
    out[~accept] = alias[kk[~accept]]
    return out


def build_edge_alias(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Single-level Vose alias table for edge (or arbitrary) weights."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return _alias_setup(w)


class TwoLevelAlias:
    """Two-level Vose alias: shard mass → in-shard index.

    Drawing ``(shard, local_idx)`` is exactly proportional to the flattened
    per-item weights: if ``w`` is the concatenation of per-shard weight arrays
    and ``W = sum(w)``, then

        P(shard=s, local=i) = w_{s,i} / W.
    """

    def __init__(
        self,
        shard_masses: np.ndarray,
        shard_weights: Sequence[np.ndarray],
    ):
        masses = np.asarray(shard_masses, dtype=np.float64).reshape(-1)
        if masses.shape[0] != len(shard_weights):
            raise ValueError(
                f"shard_masses length {masses.shape[0]} != "
                f"number of shards {len(shard_weights)}"
            )
        if masses.shape[0] == 0:
            raise ValueError("TwoLevelAlias requires at least one shard")
        masses = np.maximum(masses, 0.0)
        if float(masses.sum()) <= 0.0:
            masses = np.ones_like(masses)

        self.shard_masses = masses
        self.n_shards = int(masses.shape[0])
        self._shard_prob, self._shard_alias = _alias_setup(masses)

        self._local_prob: List[np.ndarray] = []
        self._local_alias: List[np.ndarray] = []
        self._shard_sizes = np.zeros(self.n_shards, dtype=np.int64)
        flat_parts: List[np.ndarray] = []
        for s, ww in enumerate(shard_weights):
            arr = np.asarray(ww, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                raise ValueError(f"shard {s} has empty weight array")
            arr = np.maximum(arr, 1e-12)
            # Reconcile mass used for top-level with local table (local sum).
            local_sum = float(arr.sum())
            if local_sum <= 0.0:
                arr = np.ones_like(arr)
                local_sum = float(arr.size)
            self.shard_masses[s] = local_sum
            flat_parts.append(arr)
            lp, la = _alias_setup(arr)
            self._local_prob.append(lp)
            self._local_alias.append(la)
            self._shard_sizes[s] = int(arr.size)

        # Rebuild top-level after reconciling masses to local sums.
        self._shard_prob, self._shard_alias = _alias_setup(self.shard_masses)
        self._flat_weights = np.concatenate(flat_parts)
        self._flat_total = float(self._flat_weights.sum())
        offsets = np.zeros(self.n_shards + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(self._shard_sizes)
        self._flat_offsets = offsets

    @property
    def normalized_weights(self) -> np.ndarray:
        """Flattened weights / total mass (draw probabilities)."""
        return self._flat_weights / self._flat_total

    def flat_index(self, shard: ArrayLike, local_idx: ArrayLike) -> np.ndarray:
        """Map ``(shard, local_idx)`` pairs to flattened indices."""
        s = np.asarray(shard, dtype=np.int64)
        loc = np.asarray(local_idx, dtype=np.int64)
        return self._flat_offsets[s] + loc

    def draw(
        self, n_draw: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(shard, local_idx)`` each of shape ``(n_draw,)``."""
        shards = _alias_draw(n_draw, self._shard_prob, self._shard_alias, rng)
        local = np.empty(n_draw, dtype=np.int64)
        # Group by shard so each local table is drawn in a vectorized batch.
        order = np.argsort(shards, kind="mergesort")
        shards_sorted = shards[order]
        boundaries = np.flatnonzero(np.diff(shards_sorted, prepend=-1))
        boundaries = np.append(boundaries, n_draw)
        for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
            s = int(shards_sorted[b0])
            m = b1 - b0
            local_sorted = _alias_draw(
                m, self._local_prob[s], self._local_alias[s], rng
            )
            local[order[b0:b1]] = local_sorted
        return shards, local

    def draw_flat(self, n_draw: int, rng: np.random.Generator) -> np.ndarray:
        """Draw flattened indices with the same law as single-level alias."""
        shards, local = self.draw(n_draw, rng)
        return self.flat_index(shards, local)


def build_two_level_alias(weights: np.ndarray, shard_size: int) -> TwoLevelAlias:
    """Partition ``weights`` into contiguous shards and build a two-level alias."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 1e-12)
    n = int(w.shape[0])
    if n == 0:
        raise ValueError("weights must be non-empty")
    shard_weights = [
        w[i : i + shard_size] for i in range(0, n, shard_size)
    ]
    masses = np.asarray([float(sw.sum()) for sw in shard_weights], dtype=np.float64)
    return TwoLevelAlias(masses, shard_weights)


def freeze_alias_tables(
    graph: Any,
    *,
    shard_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Build alias arrays for a frozen graph store's ``alias/`` directory.

    Returns a dict with at least ``prob`` and ``alias`` (single-level Vose).
    When ``shard_size`` is set, also includes a ``two_level`` :class:`TwoLevelAlias`.
    """
    w = graph.weights
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    prob, alias = build_edge_alias(w)
    out: Dict[str, Any] = {
        "prob": np.asarray(prob, dtype=np.float64),
        "alias": np.asarray(alias, dtype=np.int64),
    }
    if shard_size is not None:
        out["two_level"] = build_two_level_alias(w, int(shard_size))
        out["shard_size"] = int(shard_size)
    return out


__all__ = [
    "_alias_setup",
    "_alias_draw",
    "build_edge_alias",
    "build_two_level_alias",
    "TwoLevelAlias",
    "freeze_alias_tables",
]

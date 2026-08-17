"""Path triplet sampler (memmap gather + optional group-mass alias)."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from leanmap.paths.constraint import PathConstraint, PathTripletSampler

__all__ = ["PathTripletSampler", "MemmapPathSampler"]


class MemmapPathSampler:
    """Gather path endpoints from array / memmap ``X`` by triplet indices.

    Stub for large-N training: returns index tensors (and optional features)
    without requiring a full ``torch.Tensor`` materialization of ``X``.
    Optional ``group_mass`` builds a Vose alias over triplet rows.
    """

    def __init__(
        self,
        X: Any,
        constraint: PathConstraint,
        seed: int = 0,
        group_mass: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.triplets = np.asarray(constraint.triplets, dtype=np.int64)
        self.dt = np.asarray(constraint.triplet_dt, dtype=np.float32)
        self.n = int(np.asarray(X).shape[0]) if not hasattr(X, "shape") else int(X.shape[0])
        self.T = int(self.triplets.shape[0])
        self.rng = np.random.default_rng(seed)
        if self.T == 0:
            raise ValueError("MemmapPathSampler: empty triplet table")
        self._alias_prob: Optional[np.ndarray] = None
        self._alias_alias: Optional[np.ndarray] = None
        if group_mass is not None:
            from leanmap.sampling.alias import build_edge_alias

            mass = np.asarray(group_mass, dtype=np.float64).reshape(-1)
            if mass.shape[0] != self.T:
                raise ValueError(
                    f"group_mass length {mass.shape[0]} != n_triplets {self.T}"
                )
            self._alias_prob, self._alias_alias = build_edge_alias(mass)

    def sample_indices(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(idx_a, idx_n, idx_m, idx_far, dt)`` with ``dt`` shape ``(n, 2)``."""
        if self._alias_prob is not None and self._alias_alias is not None:
            from leanmap.sampling.alias import _alias_draw

            pick = _alias_draw(int(n), self._alias_prob, self._alias_alias, self.rng)
        else:
            pick = self.rng.integers(0, self.T, size=int(n))
        tri = self.triplets[pick]
        dt = self.dt[pick]
        far = self.rng.integers(0, self.n, size=int(n))
        return tri[:, 0], tri[:, 1], tri[:, 2], far, dt

    def sample(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather features for a minibatch (materializes selected rows only)."""
        ia, inn, im, ifr, dt = self.sample_indices(n)
        X = self.X
        if isinstance(X, torch.Tensor):
            return (
                X[torch.as_tensor(ia, dtype=torch.int64)],
                X[torch.as_tensor(inn, dtype=torch.int64)],
                X[torch.as_tensor(im, dtype=torch.int64)],
                X[torch.as_tensor(ifr, dtype=torch.int64)],
                torch.as_tensor(dt[:, 0], dtype=torch.float32),
                torch.as_tensor(dt[:, 1], dtype=torch.float32),
            )
        # numpy / memmap path
        return (
            torch.as_tensor(np.asarray(X[ia]), dtype=torch.float32),
            torch.as_tensor(np.asarray(X[inn]), dtype=torch.float32),
            torch.as_tensor(np.asarray(X[im]), dtype=torch.float32),
            torch.as_tensor(np.asarray(X[ifr]), dtype=torch.float32),
            torch.as_tensor(dt[:, 0], dtype=torch.float32),
            torch.as_tensor(dt[:, 1], dtype=torch.float32),
        )

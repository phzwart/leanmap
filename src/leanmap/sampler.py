"""Edge, negative, and ordinal-triplet samplers."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from .distance import DistanceFn
from .graph import Graph, Representatives
from .landmarks import AnchorAffinity, LandmarkAffinity
from .utils import get_logger


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


def _cell_member(reps: Representatives, cell: int, rng: np.random.Generator) -> int:
    start = int(reps.offsets[cell])
    end = int(reps.offsets[cell + 1])
    if end <= start:
        raise RuntimeError(f"empty cell {cell}")
    j = int(rng.integers(start, end))
    return int(reps.values[j])


class EdgeSampler:
    """Sample graph edges then expand to random raw cell members."""

    def __init__(self, X: torch.Tensor, graph: Graph, seed: int = 0):
        self.X = X
        self.reps = graph.reps
        self.edges = graph.edges.cpu().numpy().astype(np.int64)
        self._weights = graph.weights.cpu().numpy().astype(np.float64)
        self._weights = np.maximum(self._weights, 1e-12)
        self.prob, self.alias = _alias_setup(self._weights)
        self.rng = np.random.default_rng(seed)

    def sample(
        self, batch_edges: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``x_i (B,D), x_j (B,D), w (B,), edge_idx (B,)``."""
        eidx = _alias_draw(batch_edges, self.prob, self.alias, self.rng)
        pairs = self.edges[eidx]
        ii = [_cell_member(self.reps, int(c), self.rng) for c in pairs[:, 0]]
        jj = [_cell_member(self.reps, int(c), self.rng) for c in pairs[:, 1]]
        x_i = self.X[torch.as_tensor(ii, dtype=torch.int64)]
        x_j = self.X[torch.as_tensor(jj, dtype=torch.int64)]
        w = torch.as_tensor(self._weights[eidx], dtype=torch.float32)
        return x_i, x_j, w, torch.as_tensor(eidx, dtype=torch.int64)


class NegativeSampler:
    """Uniform over representatives, then cell-expanded."""

    def __init__(self, X: torch.Tensor, reps: Representatives, seed: int = 0):
        self.X = X
        self.reps = reps
        self.R = int(reps.rep_idx.shape[0])
        self.rng = np.random.default_rng(seed + 1)

    def sample(self, B: int, n_neg: int = 5) -> torch.Tensor:
        """Returns ``x_neg: (B, n_neg, D)``."""
        cells = self.rng.integers(0, self.R, size=(B, n_neg))
        idx = [
            [_cell_member(self.reps, int(c), self.rng) for c in row] for row in cells
        ]
        return self.X[torch.as_tensor(idx, dtype=torch.int64)]


class OrdinalTripletSampler:
    """Anchor-rank mid + uniform far, verified with the factor view metric."""

    def __init__(
        self,
        X: torch.Tensor,
        assign_top1: torch.Tensor,
        dist_fn: DistanceFn,
        seed: int = 0,
        view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.X = X
        self.assign_top1 = assign_top1.cpu()
        self.dist_fn = dist_fn
        self.view = view if view is not None else (lambda x: x)
        self.rng = np.random.default_rng(seed + 2)
        self.N = X.shape[0]
        Lmax = int(assign_top1.max().item()) + 1 if assign_top1.numel() else 1
        self.buckets = [
            torch.where(self.assign_top1 == b)[0].numpy() for b in range(Lmax)
        ]
        self.last_retention: Optional[float] = None

    def sample(
        self,
        x_anchor: torch.Tensor,
        x_near: torch.Tensor,
        affinity: AnchorAffinity,
        shuffle_ranks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return ``x_mid (B,D), x_far (B,D), mask (B,), retention``.

        ``shuffle_ranks`` permutes each row's landmark ordering, which draws the
        mid point from an arbitrary landmark's bucket instead of the r-th
        nearest. Everything else — the log-uniform rank draw, bucket-based mid
        selection, uniform far point — is untouched, so the resulting retention
        is the **chance level** for this sampler on this data. See
        :func:`estimate_retention_null`.
        """
        B = x_anchor.shape[0]
        with torch.no_grad():
            v_a = self.view(x_anchor)
            _, Dm = affinity(v_a)
            order = torch.argsort(Dm, dim=1)  # (B, L) ascending
            if shuffle_ranks:
                perm = torch.as_tensor(
                    np.argsort(self.rng.random(order.shape), axis=1),
                    dtype=torch.int64,
                    device=order.device,
                )
                order = torch.gather(order, 1, perm)
        L = order.shape[1]
        log2, logL = np.log(2.0), np.log(max(L, 2))
        ranks = np.round(np.exp(self.rng.uniform(log2, logL, size=B))).astype(int)
        ranks = np.clip(ranks, 2, L)

        mid_idx = []
        far_idx = []
        for i in range(B):
            r = int(ranks[i])
            ell = int(order[i, r - 1].item())
            bucket = self.buckets[ell] if ell < len(self.buckets) else np.array([], dtype=np.int64)
            if len(bucket) == 0:
                mid_idx.append(int(self.rng.integers(0, self.N)))
            else:
                mid_idx.append(int(bucket[self.rng.integers(0, len(bucket))]))
            far_idx.append(int(self.rng.integers(0, self.N)))

        x_mid = self.X[torch.as_tensor(mid_idx, dtype=torch.int64)].to(x_anchor.device)
        x_far = self.X[torch.as_tensor(far_idx, dtype=torch.int64)].to(x_anchor.device)

        v_near = self.view(x_near)
        v_mid = self.view(x_mid)
        v_far = self.view(x_far)
        d_near = self.dist_fn(v_a, v_near).diag()
        d_mid = self.dist_fn(v_a, v_mid).diag()
        d_far = self.dist_fn(v_a, v_far).diag()
        mask = (d_near < d_mid) & (d_mid < d_far)
        retention = float(mask.float().mean().item())
        if shuffle_ranks:
            return x_mid, x_far, mask, retention
        self.last_retention = retention
        if retention < 0.3:
            get_logger().info(
                "ordinal triplet retention=%.3f < 0.3 — landmark ranking is a "
                "poor proxy for true distances",
                retention,
            )
        return x_mid, x_far, mask, retention


def estimate_retention_null(
    ord_samp: "OrdinalTripletSampler",
    edge_sampler: Any,
    affinity: AnchorAffinity,
    n_batches: int = 8,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> float:
    """Chance retention for the ordinal triplet loss, measured not asserted.

    ``retention_f`` is only interpretable against the rate at which a triplet
    would satisfy ``d_near < d_mid < d_far`` with no information in the
    landmark ranking. That rate depends on the sampler (log-uniform ranks,
    bucket-based mid selection), on the bucket-size distribution, and on the
    data — it is not a universal constant, and the previously asserted 0.475 is
    not recoverable from the stated scheme.

    Measuring it by shuffling the landmark ranks is correct whatever the
    sampling scheme, and stays correct if the scheme changes.
    """
    rates = []
    for _ in range(max(1, n_batches)):
        x_i, x_j, _, _ = edge_sampler.sample(batch_size)
        if device is not None:
            x_i, x_j = x_i.to(device), x_j.to(device)
        _, _, _, r = ord_samp.sample(x_i, x_j, affinity, shuffle_ranks=True)
        rates.append(r)
    return float(np.mean(rates)) if rates else 0.0


class StarSampler:
    """Sample fine-graph neighbourhoods ("stars") for the local-rigidity loss.

    Precomputes symmetric neighbour lists from a (fine) graph's edges. Each
    :meth:`sample` draws ``B`` centre cells and gathers up to ``m`` of their
    graph neighbours, padded to ``m`` with a validity mask. Cells are expanded
    to their deterministic representative member (``reps.rep_idx``) so a star's
    geometry is coherent across steps (unlike the random cell expansion used by
    :class:`EdgeSampler`).
    """

    def __init__(
        self,
        X: torch.Tensor,
        graph: Graph,
        m: int = 6,
        seed: int = 0,
        deterministic: bool = False,
    ):
        """``deterministic`` takes each node's first ``m`` neighbours instead of
        a fresh random ``m``. The density term needs it: its target is the
        ambient radius of a fixed neighbour set, so the layout radius it is
        compared against has to be measured over that same set, not a different
        draw each step (which would attenuate the fitted contrast).
        """
        self.X = X
        self.m = int(m)
        self.deterministic = bool(deterministic)
        self.rep_idx = graph.reps.rep_idx.cpu().numpy().astype(np.int64)  # (R,)
        R = int(self.rep_idx.shape[0])
        edges = graph.edges.cpu().numpy().astype(np.int64)
        adj: list = [[] for _ in range(R)]
        for a, b in edges:
            a = int(a)
            b = int(b)
            if a == b:
                continue
            adj[a].append(b)
            adj[b].append(a)
        self.nbrs = [np.unique(np.asarray(n, dtype=np.int64)) for n in adj]
        self.centers = np.asarray(
            [i for i in range(R) if self.nbrs[i].size > 0], dtype=np.int64
        )
        self.rng = np.random.default_rng(seed + 3)

    def sample(
        self, B: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``x_c (B,D), x_nbr (B,m,D), mask (B,m)``."""
        return self.sample_indexed(B)[:3]

    def padded_neighbours(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every node's neighbour cells as ``(R, m)`` ids plus an ``(R, m)`` mask.

        Uses the same first-``m`` rule as :meth:`sample_indexed` under
        ``deterministic``, so a per-node quantity tabulated from this is exactly
        the one a sampled star will see.
        """
        R = int(self.rep_idx.shape[0])
        m = self.m
        idx = np.zeros((R, m), dtype=np.int64)
        mask = np.zeros((R, m), dtype=np.float32)
        for i in range(R):
            sel = self.nbrs[i][:m]
            k = int(sel.size)
            idx[i, :k] = sel
            mask[i, :k] = 1.0
        return torch.as_tensor(idx), torch.as_tensor(mask)

    def sample_indexed(
        self, B: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """As :meth:`sample`, plus the centre *cell* ids ``(B,)``.

        The density term needs the ids to look up each centre's ambient
        neighbour radius, which is tabulated per graph node.
        """
        m = self.m
        if self.centers.size == 0:
            D = self.X.shape[1]
            return (
                self.X.new_zeros(B, D),
                self.X.new_zeros(B, m, D),
                torch.zeros(B, m, dtype=torch.float32),
                torch.zeros(B, dtype=torch.int64),
            )
        pick = self.rng.integers(0, self.centers.size, size=B)
        cells = self.centers[pick]
        nbr_cells = np.zeros((B, m), dtype=np.int64)
        mask = np.zeros((B, m), dtype=np.float32)
        for r in range(B):
            nb = self.nbrs[int(cells[r])]
            if nb.size <= m:
                sel = nb
            elif self.deterministic:
                sel = nb[:m]
            else:
                sel = self.rng.choice(nb, size=m, replace=False)
            k = int(sel.size)
            nbr_cells[r, :k] = sel
            mask[r, :k] = 1.0
        c_raw = self.rep_idx[cells]  # (B,)
        nbr_raw = self.rep_idx[nbr_cells.reshape(-1)].reshape(B, m)  # (B, m)
        x_c = self.X[torch.as_tensor(c_raw, dtype=torch.int64)]
        x_nbr = self.X[torch.as_tensor(nbr_raw, dtype=torch.int64)]
        return (
            x_c,
            x_nbr,
            torch.as_tensor(mask, dtype=torch.float32),
            torch.as_tensor(cells, dtype=torch.int64),
        )

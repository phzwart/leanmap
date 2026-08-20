"""Edge, negative, and ordinal-triplet samplers."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch

from ..distance import DistanceFn
from ..graph import Graph, Representatives
from ..landmarks import AnchorAffinity, LandmarkAffinity
from ..utils import get_logger
from .alias import _alias_draw, _alias_setup

ArrayLike = Union[np.ndarray, Any]


def _cell_member(reps: Representatives, cell: int, rng: np.random.Generator) -> int:
    start = int(reps.offsets[cell])
    end = int(reps.offsets[cell + 1])
    if end <= start:
        raise RuntimeError(f"empty cell {cell}")
    j = int(rng.integers(start, end))
    return int(reps.values[j])


def _cell_member_csr(
    offsets: ArrayLike,
    values: ArrayLike,
    cell: int,
    rng: np.random.Generator,
) -> int:
    start = int(offsets[cell])
    end = int(offsets[cell + 1])
    if end <= start:
        raise RuntimeError(f"empty cell {cell}")
    j = int(rng.integers(start, end))
    return int(values[j])


def basin_balanced_edge_weights(
    edges: ArrayLike,
    weights: ArrayLike,
    cell_landmark: ArrayLike,
    *,
    mix: float = 1.0,
) -> np.ndarray:
    """Reweight edges toward equal landmark-basin coverage.

    Parameters
    ----------
    edges :
        ``(E, 2)`` representative / cell ids.
    weights :
        ``(E,)`` fuzzy edge masses (``> 0``).
    cell_landmark :
        ``(R,)`` primary landmark id for each cell.
    mix :
        ``0`` keeps ``weights``; ``1`` fully equalizes basins via
        ``0.5 (1/μ_a + 1/μ_b)`` (mean-normalized). Values in between blend.

    Returns
    -------
    weights' : ``(E,)`` float64, strictly positive.
    """
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
    cl = np.asarray(cell_landmark, dtype=np.int64).reshape(-1)
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0 or e.shape[0] == 0:
        return w.copy()
    L = int(cl.max()) + 1 if cl.size else 1
    la = cl[e[:, 0]]
    lb = cl[e[:, 1]]
    mu = np.zeros(L, dtype=np.float64)
    np.add.at(mu, la, w)
    np.add.at(mu, lb, w)
    mu = np.maximum(mu, 1e-12)
    inv = 0.5 * (1.0 / mu[la] + 1.0 / mu[lb])
    inv = inv / max(float(inv.mean()), 1e-12)
    out = w * ((1.0 - mix) + mix * inv)
    return np.maximum(out, 1e-12)


def landmark_epoch_steps(
    n_landmarks: int,
    batch_edges: int,
    *,
    samples_per_landmark: float = 128.0,
) -> int:
    """Steps so each landmark expects ``samples_per_landmark`` edge draws.

    With basin-balanced (or uniform-landmark) sampling, one step of
    ``batch_edges`` draws spreads roughly uniformly across ``L`` basins, so
    ``steps = ceil(L * samples_per_landmark / batch_edges)``.
    Independent of graph edge count ``E`` / δ-net size ``R``.
    """
    L = max(1, int(n_landmarks))
    B = max(1, int(batch_edges))
    s = max(1.0, float(samples_per_landmark))
    return max(1, int(math.ceil(L * s / B)))


class EdgeSampler:
    """Sample graph edges then expand to random raw cell members.

    Default path materializes alias tables and edge endpoints from ``graph``
    (bit-compatible with existing goldens). Optional store path accepts
    precomputed alias arrays and cell CSR (``np.memmap`` or arrays) so training
    need not keep ``graph.edges`` resident in RAM.
    """

    def __init__(
        self,
        X: torch.Tensor,
        graph: Graph,
        seed: int = 0,
        *,
        alias_prob: Optional[ArrayLike] = None,
        alias_alias: Optional[ArrayLike] = None,
        member_offsets: Optional[ArrayLike] = None,
        member_values: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None,
    ):
        self.X = X
        self.rng = np.random.default_rng(seed)
        store_alias = alias_prob is not None or alias_alias is not None
        store_members = member_offsets is not None or member_values is not None
        if store_alias and (alias_prob is None or alias_alias is None):
            raise ValueError("store path requires both alias_prob= and alias_alias=")
        if store_members and (member_offsets is None or member_values is None):
            raise ValueError(
                "store path requires both member_offsets= and member_values="
            )

        if store_alias:
            # Memmap / precomputed alias: do not build from graph.weights.
            self.prob = alias_prob
            self.alias = alias_alias
            if edges is not None:
                self.edges = edges
            else:
                # Still allow callers that only swap alias tables.
                self.edges = graph.edges.cpu().numpy().astype(np.int64)
            if weights is not None:
                self._weights = np.maximum(
                    np.asarray(weights, dtype=np.float64), 1e-12
                )
            else:
                self._weights = np.maximum(
                    graph.weights.cpu().numpy().astype(np.float64), 1e-12
                )
        else:
            self.edges = (
                edges
                if edges is not None
                else graph.edges.cpu().numpy().astype(np.int64)
            )
            if weights is not None:
                self._weights = np.maximum(
                    np.asarray(weights, dtype=np.float64), 1e-12
                )
            else:
                self._weights = np.maximum(
                    graph.weights.cpu().numpy().astype(np.float64), 1e-12
                )
            self.prob, self.alias = _alias_setup(self._weights)

        if store_members:
            self._member_offsets = member_offsets
            self._member_values = member_values
            self.reps = None
            self._use_csr_members = True
        else:
            self.reps = graph.reps
            self._member_offsets = None
            self._member_values = None
            self._use_csr_members = False
        # Base weights before per-epoch active-set masking (for restore / rebuild).
        self._base_weights = np.asarray(self._weights, dtype=np.float64).copy()

    def set_weights(self, weights: ArrayLike) -> None:
        """Replace alias mass (e.g. after epoch active-set masking)."""
        self._weights = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
        if int(self._weights.shape[0]) != int(np.asarray(self.edges).shape[0]):
            raise ValueError(
                f"weights length {self._weights.shape[0]} != E={np.asarray(self.edges).shape[0]}"
            )
        self.prob, self.alias = _alias_setup(self._weights)

    def set_member_csr(
        self,
        offsets: Optional[ArrayLike],
        values: Optional[ArrayLike],
    ) -> None:
        """Restrict cell expansion to a CSR view (e.g. active members only).

        Pass ``offsets=None`` to restore the graph's full cell membership.
        """
        if offsets is None or values is None:
            self._use_csr_members = False
            self._member_offsets = None
            self._member_values = None
            return
        self._member_offsets = offsets
        self._member_values = values
        self._use_csr_members = True

    def _expand_cell(self, cell: int) -> int:
        if self._use_csr_members:
            idx = _cell_member_csr(
                self._member_offsets, self._member_values, cell, self.rng
            )
            if idx < 0:
                # Empty active cell — fall back to full membership if available.
                if self.reps is not None:
                    return _cell_member(self.reps, cell, self.rng)
                raise RuntimeError(f"active cell {cell} has no members")
            return idx
        return _cell_member(self.reps, cell, self.rng)

    def sample(
        self, batch_edges: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``x_i (B,D), x_j (B,D), w (B,), edge_idx (B,)``."""
        eidx = _alias_draw(batch_edges, self.prob, self.alias, self.rng)
        pairs = np.asarray(self.edges[eidx], dtype=np.int64)
        if pairs.ndim == 1:
            pairs = pairs.reshape(1, -1)
        ii = [self._expand_cell(int(c)) for c in pairs[:, 0]]
        jj = [self._expand_cell(int(c)) for c in pairs[:, 1]]
        x_i = self.X[torch.as_tensor(ii, dtype=torch.int64)]
        x_j = self.X[torch.as_tensor(jj, dtype=torch.int64)]
        w = torch.as_tensor(
            np.asarray(self._weights[eidx], dtype=np.float64), dtype=torch.float32
        )
        return x_i, x_j, w, torch.as_tensor(eidx, dtype=torch.int64)


class NegativeSampler:
    """Uniform over representatives, then cell-expanded."""

    def __init__(self, X: torch.Tensor, reps: Representatives, seed: int = 0):
        self.X = X
        self.reps = reps
        self.R = int(reps.rep_idx.shape[0])
        self.rng = np.random.default_rng(seed + 1)
        self._active_cells: Optional[np.ndarray] = None

    def set_active_cells(self, cell_active: Optional[np.ndarray]) -> None:
        """If set, draw negatives only from cells that intersect the active set."""
        if cell_active is None:
            self._active_cells = None
            return
        ca = np.asarray(cell_active, dtype=bool).reshape(-1)
        idx = np.flatnonzero(ca)
        self._active_cells = idx if idx.size else None

    def sample(self, B: int, n_neg: int = 5) -> torch.Tensor:
        """Returns ``x_neg: (B, n_neg, D)``."""
        if self._active_cells is not None and self._active_cells.size > 0:
            cells = self.rng.choice(self._active_cells, size=(B, n_neg), replace=True)
        else:
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

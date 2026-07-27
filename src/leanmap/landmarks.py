"""Farthest-point / quantile anchors, IVF assignment, and AnchorAffinity."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distance import DistanceFn, chunked_cdist, is_differentiable
from .utils import get_logger


def fps_init_indices(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    L: int,
    seed: int = 0,
    sample_pool: int = 200_000,
) -> torch.Tensor:
    """Farthest-point sampling; returns indices into ``X`` (length ``L``)."""
    n = X.shape[0]
    if L <= 0:
        raise ValueError("L must be positive")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if n > sample_pool:
        pool_idx = torch.randperm(n, generator=g)[:sample_pool]
        pool = X[pool_idx]
    else:
        pool_idx = None
        pool = X
    n_pool = pool.shape[0]
    L = min(L, n_pool)
    first = int(torch.randint(0, n_pool, (1,), generator=g).item())
    chosen = [first]
    min_dist = torch.full((n_pool,), float("inf"), dtype=torch.float32, device=X.device)
    d_row = dist_fn(pool[first : first + 1], pool)[0]
    min_dist = torch.minimum(min_dist, d_row)
    for _ in range(L - 1):
        j = int(min_dist.argmax().item())
        chosen.append(j)
        d_row = dist_fn(pool[j : j + 1], pool)[0]
        min_dist = torch.minimum(min_dist, d_row)
    local = torch.tensor(chosen, dtype=torch.int64, device=X.device)
    if pool_idx is None:
        return local
    return pool_idx.to(device=X.device)[local].contiguous()


def fps_init_indices_geodesic(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    L: int,
    n_neighbors: int = 15,
    seed: int = 0,
    sample_pool: int = 20_000,
    chunk: int = 4096,
) -> torch.Tensor:
    """Farthest-point sampling on **geodesic** (kNN shortest-path) distances.

    Builds a symmetric kNN distance graph on ``X`` (edge weights = ``dist_fn``),
    then greedily selects ``L`` points that are maximally far apart *along the
    manifold* (graph shortest paths via Dijkstra) rather than in the ambient
    metric. This spreads landmarks uniformly over the intrinsic geometry, which
    ambient FPS can miss when the manifold folds back on itself (e.g. the
    S-curve / swiss roll). Unreachable nodes (disconnected components) are
    treated as maximally far, so every component gets covered.

    Returns indices into ``X`` (length ``min(L, n)``).
    """
    import numpy as np
    from scipy.sparse.csgraph import dijkstra

    n = X.shape[0]
    if L <= 0:
        raise ValueError("L must be positive")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if n > sample_pool:
        pool_idx = torch.randperm(n, generator=g)[:sample_pool]
        pool = X[pool_idx]
    else:
        pool_idx = None
        pool = X
    n_pool = pool.shape[0]
    L = min(L, n_pool)
    k = min(n_neighbors, n_pool - 1)
    G = _geodesic_knn_graph(pool, dist_fn, k, chunk=chunk)

    first = int(torch.randint(0, n_pool, (1,), generator=g).item())
    min_dist = dijkstra(G, directed=False, indices=first)
    chosen = [first]
    for _ in range(L - 1):
        md = min_dist.copy()
        finite = md[np.isfinite(md)]
        big = (float(finite.max()) * 10.0 + 1.0) if finite.size else 1.0
        md[~np.isfinite(md)] = big  # unreachable => maximally far (cover components)
        j = int(np.argmax(md))
        chosen.append(j)
        dj = dijkstra(G, directed=False, indices=j)
        min_dist = np.minimum(min_dist, dj)

    local = torch.tensor(chosen, dtype=torch.int64, device=X.device)
    if pool_idx is None:
        return local
    return pool_idx.to(device=X.device)[local].contiguous()


def _geodesic_knn_graph(
    pool: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    chunk: int = 4096,
):
    """Symmetric kNN distance graph (CSR) for shortest-path (geodesic) queries."""
    import numpy as np
    from scipy.sparse import csr_matrix

    n_pool = pool.shape[0]
    rows_l, cols_l, data_l = [], [], []
    for s in range(0, n_pool, chunk):
        e = min(n_pool, s + chunk)
        vals, idx = chunked_cdist(dist_fn, pool[s:e], pool, topk=k + 1, out_device=pool.device)
        vals = vals[:, 1:].detach().cpu().numpy().ravel()
        idx = idx[:, 1:].detach().cpu().numpy().ravel()
        rows_l.append(np.repeat(np.arange(s, e), k))
        cols_l.append(idx)
        data_l.append(vals)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    data = np.concatenate(data_l).astype(np.float64)
    G = csr_matrix((data, (rows, cols)), shape=(n_pool, n_pool))
    return G.maximum(G.T)  # symmetrize: keep an edge if either direction has it


def landmark_geodesic_matrix(
    X: torch.Tensor,
    M: torch.Tensor,
    dist_fn: DistanceFn,
    n_neighbors: int = 15,
    chunk: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise graph-geodesic distances among landmarks (Isomap backbone).

    Maps each landmark row of ``M`` to its nearest training point in ``X``,
    builds a symmetric kNN distance graph on ``X``, and runs Dijkstra from each
    landmark. Returns ambient landmark coords (frozen copies of the matched
    training rows), the geodesic matrix, and a finite-pair mask.

    Parameters
    ----------
    X : (N, D) training points
    M : (L, D) landmark coordinates (typically selected from ``X``)
    dist_fn : ambient metric used for the kNN graph
    n_neighbors : k for the geodesic graph

    Returns
    -------
    X_lm : (L, D) matched training rows (stable ambient anchors for the loss)
    G : (L, L) float32 geodesic distances (``inf`` where unreachable)
    finite : (L, L) bool — True for finite off-diagonal pairs
    """
    import numpy as np
    from scipy.sparse.csgraph import dijkstra

    n = X.shape[0]
    L = M.shape[0]
    if L < 2:
        raise ValueError("need at least 2 landmarks for a geodesic backbone")
    k = min(n_neighbors, n - 1)
    # Nearest training index for each landmark (exact match for FPS from X).
    _, nn_idx = chunked_cdist(dist_fn, M, X, topk=1, out_device=X.device)
    lm_idx = nn_idx[:, 0].detach().cpu().numpy().astype(np.int64)
    X_lm = X[torch.as_tensor(lm_idx, dtype=torch.int64)].contiguous()

    Gsp = _geodesic_knn_graph(X, dist_fn, k, chunk=chunk)
    D = dijkstra(Gsp, directed=False, indices=lm_idx)  # (L, N)
    G = D[:, lm_idx].astype(np.float64)  # (L, L)
    finite = np.isfinite(G)
    np.fill_diagonal(finite, False)
    G_t = torch.as_tensor(G, dtype=torch.float32)
    finite_t = torch.as_tensor(finite, dtype=torch.bool)
    return X_lm, G_t, finite_t


def classical_mds(D: torch.Tensor, d: int = 2, finite: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Classical MDS (Isomap embedding) from a pairwise distance matrix.

    Double-centres ``D^2``, takes the top-``d`` eigenvectors. Unreachable
    (non-finite) entries are replaced by the max finite distance so a single
    connected component still embeds; prefer passing a fully-finite ``D``.
    """
    import numpy as np

    Dn = D.detach().cpu().numpy().astype(np.float64)
    if finite is not None:
        fin = finite.detach().cpu().numpy()
        if fin.any():
            fill = float(Dn[fin].max())
            Dn = Dn.copy()
            Dn[~np.isfinite(Dn)] = fill
    else:
        Dn = np.nan_to_num(Dn, nan=0.0, posinf=float(np.nanmax(Dn[np.isfinite(Dn)])))
    n = Dn.shape[0]
    D2 = np.asarray(Dn * Dn, dtype=np.float64)
    J = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (J @ D2 @ J)
    # Symmetrise for numerical hermiticity of the Gram.
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1][:d]
    lam = np.maximum(evals[order], 0.0)
    Z = evecs[:, order] * np.sqrt(lam)[None, :]
    return torch.as_tensor(Z, dtype=torch.float32)


def poisson_disk_indices_geodesic(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    L: int,
    n_neighbors: int = 15,
    seed: int = 0,
    sample_pool: int = 20_000,
    chunk: int = 4096,
    radius_scale: float = 1.0,
) -> torch.Tensor:
    """Geodesic Poisson-disk (blue-noise / Delone) landmark sampling.

    Selects a maximal set of points that are pairwise ``>= r`` apart *along the
    manifold* (kNN graph shortest paths), where ``r`` is auto-calibrated so the
    set has ``~L`` members. Unlike farthest-point sampling — which greedily
    grabs the single most-distant point each step and so over-populates
    boundaries / extreme tips — dart-throwing in a random order yields an
    interior-uniform blue-noise (Poisson-disk) distribution with a guaranteed
    minimum geodesic separation. This gives smoother, more even conditioning
    coverage on folded manifolds (S-curve, swiss roll) than either ambient FPS
    or geodesic FPS.

    Algorithm: build the geodesic graph, estimate the packing radius from a
    short geodesic-FPS pass (the coverage radius at ``L`` points, times
    ``radius_scale``), then dart-throw: iterate points in a random order and
    accept one iff its geodesic distance to every accepted point is ``>= r``,
    running one Dijkstra per acceptance to maintain the min-distance field.

    Returns indices into ``X`` (length ``<= L``; a maximal r-separated set).
    """
    import numpy as np
    from scipy.sparse.csgraph import dijkstra

    n = X.shape[0]
    if L <= 0:
        raise ValueError("L must be positive")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if n > sample_pool:
        pool_idx = torch.randperm(n, generator=g)[:sample_pool]
        pool = X[pool_idx]
    else:
        pool_idx = None
        pool = X
    n_pool = pool.shape[0]
    L = min(L, n_pool)
    k = min(n_neighbors, n_pool - 1)
    G = _geodesic_knn_graph(pool, dist_fn, k, chunk=chunk)

    rng = np.random.default_rng(seed)

    # 1) Estimate the packing radius via a short geodesic-FPS pass: the coverage
    #    radius when L points are placed (the last max-min distance).
    first = int(torch.randint(0, n_pool, (1,), generator=g).item())
    min_dist = dijkstra(G, directed=False, indices=first)
    finite0 = min_dist[np.isfinite(min_dist)]
    big = (float(finite0.max()) * 10.0 + 1.0) if finite0.size else 1.0
    cover = big
    for _ in range(max(1, L - 1)):
        md = min_dist.copy()
        md[~np.isfinite(md)] = big
        j = int(np.argmax(md))
        cover = float(md[j])
        dj = dijkstra(G, directed=False, indices=j)
        min_dist = np.minimum(min_dist, dj)
    r = max(cover * float(radius_scale), 1e-9)

    # 2) Dart-throwing: accept points in random order that are >= r (geodesic)
    #    from every already-accepted point.
    order = rng.permutation(n_pool)
    min_da = np.full(n_pool, np.inf)
    accepted: list = []
    for idx in order:
        if min_da[idx] >= r:
            accepted.append(int(idx))
            da = dijkstra(G, directed=False, indices=int(idx))
            da[~np.isfinite(da)] = big  # unreachable => far (cover components)
            min_da = np.minimum(min_da, da)
            if len(accepted) >= L:
                break

    local = torch.tensor(accepted, dtype=torch.int64, device=X.device)
    if pool_idx is None:
        return local
    return pool_idx.to(device=X.device)[local].contiguous()


def fps_init(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    L: int,
    seed: int = 0,
    sample_pool: int = 200_000,
) -> torch.Tensor:
    """Farthest-point sampling of ``L`` landmarks.

    Parameters
    ----------
    X : (N, D) float32
    dist_fn : DistanceFn
    L : int
        Number of landmarks.
    seed : int
    sample_pool : int
        If ``N > sample_pool``, subsample that many rows first.

    Returns
    -------
    Tensor (L, D) float32
        Landmark coordinates (copies of selected rows).
    """
    idx = fps_init_indices(X, dist_fn, L, seed=seed, sample_pool=sample_pool)
    return X[idx].contiguous()


def quantile_init(
    X: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Initialise ``L`` scalar anchors at quantiles of a 1-D view.

    Parameters
    ----------
    X : (N, 1) float32
    L : int

    Returns
    -------
    Tensor (L, 1) float32
    """
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"quantile_init expects (N, 1), got {tuple(X.shape)}")
    if L <= 0:
        raise ValueError("L must be positive")
    v = X[:, 0].detach().float().cpu()
    # Inclusive endpoints; unique-ish if mass is concentrated
    qs = torch.linspace(0.0, 1.0, L)
    knots = torch.quantile(v, qs)
    return knots.unsqueeze(1).to(device=X.device, dtype=torch.float32)


def init_anchors(
    X_view: torch.Tensor,
    dist_fn: DistanceFn,
    L: int,
    seed: int = 0,
) -> torch.Tensor:
    """FPS for ``D_f > 1``, quantile knots for scalar views."""
    if X_view.shape[1] == 1:
        return quantile_init(X_view, L)
    return fps_init(X_view, dist_fn, L, seed=seed)


def assign_buckets(
    X: torch.Tensor,
    M: torch.Tensor,
    dist_fn: DistanceFn,
    c: int = 8,
    chunk: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each point to its ``c`` nearest landmarks.

    Parameters
    ----------
    X : (N, D) float32
    M : (L, D) float32
    dist_fn : DistanceFn
    c : int
    chunk : int

    Returns
    -------
    assign_top1 : (N,) int64
    assign_topc : (N, c) int64
    """
    n, L = X.shape[0], M.shape[0]
    c = min(c, L)
    top1 = torch.empty(n, dtype=torch.int64, device=X.device)
    topc = torch.empty(n, c, dtype=torch.int64, device=X.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        vals, idx = chunked_cdist(dist_fn, X[s:e], M, topk=c, out_device=X.device)
        topc[s:e] = idx
        top1[s:e] = idx[:, 0]
    return top1, topc


class AnchorAffinity(nn.Module):
    """Soft assignment of points to anchors with per-anchor temperatures.

    Parameters
    ----------
    M_init : (L, D_f) float32
    dist_fn : DistanceFn
        Metric on the *view*, not necessarily on raw ``x``.
    tau_init : (L,) float32 | None
        If None, set to mean distance from each anchor to its
        ``min(32, L-1)`` nearest anchors.
    learn_anchors : bool
    learn_tau : bool
    tau_min : float
    """

    def __init__(
        self,
        M_init: torch.Tensor,
        dist_fn: DistanceFn,
        tau_init: Optional[torch.Tensor] = None,
        learn_anchors: bool = True,
        learn_tau: bool = True,
        tau_min: float = 1e-3,
        tau_scale: float = 1.0,
        probe_differentiable: bool = True,
        *,
        learn_landmarks: Optional[bool] = None,
    ):
        super().__init__()
        if learn_landmarks is not None:
            learn_anchors = learn_landmarks
        self.dist_fn = dist_fn
        self.tau_min = float(tau_min)
        log = get_logger()
        if probe_differentiable and not is_differentiable(
            dist_fn, M_init.shape[1], M_init.device
        ):
            if learn_anchors:
                log.warning(
                    "dist_fn is not differentiable w.r.t. its second argument; "
                    "freezing anchor coordinates after init"
                )
            learn_anchors = False
        self.M = nn.Parameter(M_init.clone().float(), requires_grad=learn_anchors)
        if tau_init is None:
            tau_init = self._default_tau(M_init, dist_fn) * float(tau_scale)
        self.log_tau = nn.Parameter(
            torch.log(tau_init.clamp_min(tau_min)).float(),
            requires_grad=learn_tau,
        )

    @staticmethod
    def _default_tau(M: torch.Tensor, dist_fn: DistanceFn) -> torch.Tensor:
        L = M.shape[0]
        if L == 1:
            return torch.ones(1, dtype=torch.float32, device=M.device)
        k = min(32, L - 1)
        vals, idx = chunked_cdist(dist_fn, M, M, topk=k + 1, out_device=M.device)
        nn_d = vals[:, 1 : k + 1]
        return nn_d.mean(dim=1).clamp_min(1e-3)

    def tau(self) -> torch.Tensor:
        """Return: (L,) float32 temperatures, ``tau_min + exp(log_tau)``."""
        return self.tau_min + torch.exp(self.log_tau)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, D_f). Returns ``a: (B, L)``, ``Dm: (B, L)``."""
        Dm = self.dist_fn(x, self.M)
        tau = self.tau()
        logits = -Dm / tau.unsqueeze(0)
        a = F.softmax(logits, dim=1)
        return a, Dm


# Back-compat alias (one release).
LandmarkAffinity = AnchorAffinity

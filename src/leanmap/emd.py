"""Earth Mover's Distance as a reference geometry for image data.

Pixel L2 is a poor global image metric: once two digits stop overlapping,
``||a - b||`` saturates at ``sqrt(||a||^2 + ||b||^2)`` and says nothing about how
far apart they are. EMD (2-D optimal transport of ink mass across the pixel
grid) does not saturate and is linear in deformation, which makes it a usable
stand-in for perceptual distance -- and, because no embedder here is fit on it,
an *independent* reference for scoring embeddings.

Two conventions are fixed throughout:

* ground cost is Euclidean between pixel centres, in pixel units, so ``W1``
  between a blob and the same blob shifted by ``s`` is exactly ``s``;
* every image is normalised to unit mass, so EMD measures shape and not
  brightness. Callers that care about brightness should compare total mass
  separately.

The scoring helpers take a *precomputed distance matrix* rather than a
``DistanceFn``, which is what lets the same code score L2, EMD, or any other
reference on equal terms.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "grid_cost_matrix",
    "image_masses",
    "image_emd",
    "pairwise_emd",
    "geodesic_from_matrix",
    "geodesic_submatrix",
    "reference_shepard",
    "reference_trust_continuity",
    "reference_knn_overlap",
    "reference_retrieval_overlap",
]

_COST_CACHE: Dict[Tuple[Tuple[int, int], float], np.ndarray] = {}


def grid_cost_matrix(shape: Tuple[int, int], p: float = 1.0) -> np.ndarray:
    """``(P, P)`` ground cost between pixel centres of an ``H x W`` grid.

    Distances are in pixel units and rows are ordered like ``image.ravel()``.
    ``p=1`` gives the Euclidean cost of ``W1`` (classic EMD); ``p=2`` gives the
    squared cost of ``W2``.
    """
    key = ((int(shape[0]), int(shape[1])), float(p))
    cached = _COST_CACHE.get(key)
    if cached is not None:
        return cached
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    C = np.ascontiguousarray(d**p if p != 1.0 else d)
    _COST_CACHE[key] = C
    return C


def image_masses(images: np.ndarray) -> np.ndarray:
    """Flatten to ``(N, P)`` non-negative rows normalised to unit mass."""
    W = np.asarray(images, dtype=np.float64)
    if W.ndim > 2:
        W = W.reshape(len(W), -1)
    elif W.ndim == 1:
        W = W[None, :]
    W = np.clip(W, 0.0, None)
    total = W.sum(axis=1)
    bad = np.flatnonzero(total <= 0)
    if bad.size:
        raise ValueError(
            f"{bad.size} image(s) have zero total mass and no EMD is defined for "
            f"them (first at index {int(bad[0])})"
        )
    return np.ascontiguousarray(W / total[:, None])


def image_emd(
    a: np.ndarray,
    b: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    shape: Optional[Tuple[int, int]] = None,
) -> float:
    """Exact ``W1`` between two images, via the network simplex.

    Either pass a precomputed ``C`` from :func:`grid_cost_matrix` or a ``shape``
    to derive one. No entropic regularisation, so the value is exact rather than
    blurred.
    """
    import ot

    if C is None:
        if shape is None:
            raise ValueError("pass either a cost matrix C or an image shape")
        C = grid_cost_matrix(shape)
    wa, wb = image_masses(np.stack([np.ravel(a), np.ravel(b)]))
    return float(ot.emd2(wa, wb, C))


def _emd_rows(W_query: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """EMD from each row of ``W_query`` to every column of ``B`` (P, m)."""
    import ot

    out = np.empty((len(W_query), B.shape[1]), dtype=np.float64)
    for i, w in enumerate(W_query):
        out[i] = np.atleast_1d(np.asarray(ot.emd2(np.ascontiguousarray(w), B, C)))
    return out


_WORKER: Dict[str, np.ndarray] = {}


def _init_worker(W: np.ndarray, C: np.ndarray) -> None:
    _WORKER["W"] = W
    _WORKER["C"] = C


def _worker_block(task: Tuple[int, int, np.ndarray]) -> Tuple[int, int, np.ndarray]:
    start, stop, row_idx = task
    W = _WORKER["W"]
    C = _WORKER["C"]
    B = np.asfortranarray(W.T)
    return start, stop, _emd_rows(W[row_idx], B, C)


def pairwise_emd(
    images: np.ndarray,
    shape: Tuple[int, int],
    *,
    query_idx: Optional[np.ndarray] = None,
    n_jobs: int = 1,
    block: int = 32,
    progress: bool = True,
) -> np.ndarray:
    """EMD from every query image to every image.

    Returns the full symmetric ``(N, N)`` matrix, or ``(len(query_idx), N)`` when
    ``query_idx`` is given. At roughly 110 us per pair a full 1797-image digit
    matrix is a few minutes single-threaded and 13 MB on disk, so it is worth
    computing once and reusing to score every embedding.

    ``n_jobs > 1`` spreads row blocks over processes.
    """
    W = image_masses(images)
    C = grid_cost_matrix(shape)
    n = len(W)
    if C.shape[0] != W.shape[1]:
        raise ValueError(
            f"image shape {shape} implies {C.shape[0]} pixels but images have "
            f"{W.shape[1]}"
        )
    rows = np.arange(n) if query_idx is None else np.asarray(query_idx, dtype=np.int64)
    out = np.zeros((len(rows), n), dtype=np.float64)

    square = query_idx is None
    blocks = [
        (int(s), int(min(len(rows), s + block)))
        for s in range(0, len(rows), block)
    ]
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(rows), desc="EMD rows", unit="row")

    def _record(s: int, e: int, vals: np.ndarray) -> None:
        out[s:e] = vals
        if bar is not None:
            bar.update(e - s)

    try:
        if n_jobs and n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            tasks = [(s, e, rows[s:e]) for s, e in blocks]
            with ProcessPoolExecutor(
                max_workers=int(n_jobs), initializer=_init_worker, initargs=(W, C)
            ) as pool:
                for s, e, vals in pool.map(_worker_block, tasks):
                    _record(s, e, vals)
        else:
            B = np.asfortranarray(W.T)
            for s, e in blocks:
                _record(s, e, _emd_rows(W[rows[s:e]], B, C))
    finally:
        if bar is not None:
            bar.close()

    if square:
        # Network simplex is symmetric up to solver tolerance; enforce it exactly
        # so downstream kNN ties break consistently.
        out = 0.5 * (out + out.T)
        np.fill_diagonal(out, 0.0)
    return out


def geodesic_from_matrix(
    D: np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Graph geodesics under an arbitrary precomputed metric.

    Builds a symmetric kNN graph from ``D`` and runs Dijkstra on it, so the
    result is "shortest path through the data" measured in whatever metric ``D``
    encodes. Unreachable pairs come back as ``inf``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    k = int(min(n_neighbors, n - 1))
    if k < 1:
        raise ValueError("need at least 2 points for a geodesic graph")
    idx = np.argsort(D, axis=1)[:, 1 : k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = D[rows, cols]
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    return dijkstra(G, directed=False)


def geodesic_submatrix(
    X,
    dist_fn,
    indices: np.ndarray,
    n_neighbors: int = 15,
    chunk: int = 4096,
) -> np.ndarray:
    """Geodesics among ``indices``, chained through *all* of ``X``.

    The graph is built on the full point cloud even though only a subset is
    returned: chaining short hops is the whole point, and restricting the graph
    to the subset first would remove the stepping stones.
    """
    import torch
    from scipy.sparse.csgraph import dijkstra

    from .landmarks import _geodesic_knn_graph

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    idx = np.asarray(indices, dtype=np.int64)
    k = int(min(n_neighbors, len(Xt) - 1))
    G = _geodesic_knn_graph(Xt, dist_fn, k, chunk=chunk)
    D = dijkstra(G, directed=False, indices=idx)
    return np.asarray(D[:, idx], dtype=np.float64)


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _pdist_from_Z(Z: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import squareform, pdist

    return squareform(pdist(np.asarray(Z, dtype=np.float64)))


def reference_shepard(
    D_ref: np.ndarray,
    Z: np.ndarray,
    *,
    prefix: str = "emd",
    n_bands: int = 3,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Spearman between a reference distance and embedding distance.

    Reported overall and split by reference-distance band, because a single
    number hides which scale survived: an embedding can hold local structure
    while destroying global ordering, and on images that is exactly the
    distinction that matters.
    """
    from scipy.stats import spearmanr

    D_ref = np.asarray(D_ref, dtype=np.float64)
    n = len(D_ref)
    iu = np.triu_indices(n, k=1)
    ref = D_ref[iu]
    emb = _pdist_from_Z(Z)[iu]
    ref, emb = _finite_pairs(ref, emb)
    out: Dict[str, float] = {f"{prefix}_pairs": int(ref.size)}
    if ref.size < 32:
        out[f"{prefix}_spearman"] = float("nan")
        return out
    if ref.size > max_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(ref.size, size=max_pairs, replace=False)
        ref, emb = ref[sel], emb[sel]
    out[f"{prefix}_spearman"] = float(spearmanr(ref, emb).correlation)
    names = ("local", "mid", "global") if n_bands == 3 else tuple(
        f"b{i}" for i in range(n_bands)
    )
    edges = np.quantile(ref, np.linspace(0.0, 1.0, n_bands + 1))
    for b, band in enumerate(names):
        m = (ref >= edges[b]) & (ref <= edges[b + 1])
        out[f"{prefix}_spearman_{band}"] = (
            float(spearmanr(ref[m], emb[m]).correlation) if m.sum() > 32 else float("nan")
        )
    return out


def _neighbor_ranks(D: np.ndarray) -> np.ndarray:
    """``rank[i, j]`` = position of ``j`` in ``i``'s ordering, self excluded."""
    n = len(D)
    order = np.argsort(D, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=np.int64)
    rows = np.repeat(np.arange(n), n)
    ranks[rows, order.ravel()] = np.tile(np.arange(n), n)
    # Self lands at 0, so the nearest other point is already 1-based as the
    # trustworthiness estimator expects.
    return ranks


def reference_trust_continuity(
    D_ref: np.ndarray,
    Z: np.ndarray,
    *,
    prefix: str = "emd",
    k_list: Sequence[int] = (5, 15),
) -> Dict[str, float]:
    """Trustworthiness / continuity with a precomputed reference metric.

    Same estimator the L2 battery uses, but the "true" neighbourhoods come from
    ``D_ref``. Trust penalises embedding neighbours that are not reference
    neighbours; continuity penalises the reverse.
    """
    D_ref = np.asarray(D_ref, dtype=np.float64)
    D_emb = _pdist_from_Z(Z)
    n = len(D_ref)
    rank_ref = _neighbor_ranks(D_ref)
    rank_emb = _neighbor_ranks(D_emb)
    nn_ref = np.argsort(D_ref, axis=1, kind="stable")[:, 1:]
    nn_emb = np.argsort(D_emb, axis=1, kind="stable")[:, 1:]
    out: Dict[str, float] = {}
    for k in k_list:
        k = int(k)
        if k < 1 or k >= n - 1:
            continue
        norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
        t_pen = 0.0
        c_pen = 0.0
        for i in range(n):
            set_ref = set(nn_ref[i, :k].tolist())
            set_emb = set(nn_emb[i, :k].tolist())
            for j in set_emb - set_ref:
                t_pen += rank_ref[i, j] - k
            for j in set_ref - set_emb:
                c_pen += rank_emb[i, j] - k
        out[f"{prefix}_trust_{k}"] = float(1.0 - norm * t_pen)
        out[f"{prefix}_cont_{k}"] = float(1.0 - norm * c_pen)
    return out


def reference_knn_overlap(
    D_ref: np.ndarray,
    Z: np.ndarray,
    *,
    prefix: str = "emd",
    k: int = 15,
) -> Dict[str, float]:
    """Mean fraction of each point's reference-kNN that survive in ``Z``."""
    D_ref = np.asarray(D_ref, dtype=np.float64)
    n = len(D_ref)
    k = int(min(k, n - 1))
    nn_ref = np.argsort(D_ref, axis=1, kind="stable")[:, 1 : k + 1]
    nn_emb = np.argsort(_pdist_from_Z(Z), axis=1, kind="stable")[:, 1 : k + 1]
    hits = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(nn_ref, nn_emb)]
    return {f"{prefix}_knn_overlap_{k}": float(np.mean(hits) / k)}


def reference_retrieval_overlap(
    D_qg: np.ndarray,
    Z_query: np.ndarray,
    Z_gallery: np.ndarray,
    *,
    prefix: str = "emd",
    k: int = 15,
) -> Dict[str, float]:
    """Retrieval quality for points placed out of sample.

    ``D_qg`` is ``(n_query, n_gallery)`` reference distances. For each query we
    take its ``k`` nearest gallery items in the embedding and ask how many are
    among its ``k`` nearest gallery items under the reference. This is the
    practical out-of-sample question -- drop a new image on the map, does it
    land among the right neighbours -- and unlike trustworthiness it needs no
    shared index space between query and gallery.
    """
    D_qg = np.asarray(D_qg, dtype=np.float64)
    n_gal = D_qg.shape[1]
    k = int(min(k, n_gal))
    Zq = np.asarray(Z_query, dtype=np.float64)
    Zg = np.asarray(Z_gallery, dtype=np.float64)
    d_emb = np.linalg.norm(Zq[:, None, :] - Zg[None, :, :], axis=2)
    nn_ref = np.argpartition(D_qg, k - 1, axis=1)[:, :k]
    nn_emb = np.argpartition(d_emb, k - 1, axis=1)[:, :k]
    hits = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(nn_ref, nn_emb)]
    return {f"{prefix}_retrieval_overlap_{k}": float(np.mean(hits) / k)}

"""Fuzzy simplicial-set graph construction (UMAP math, no umap-learn).

Reimplements UMAP's graph stage in vectorized NumPy/SciPy and uses FAISS for
approximate k-NN so the pipeline scales to large datasets.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit


@dataclass
class FuzzyGraphData:
    """Bundle of everything the training stage needs.

    Attributes
    ----------
    x_scaled : standardized input, shape (n, d)
    mean, scale : per-feature standardization stats (for raw-input inference)
    knn_indices, knn_distances : neighbor arrays, self in column 0
    sigmas, rhos : per-point UMAP bandwidth and local-connectivity offset
    head, tail, weight : symmetrized fuzzy graph as an edge list
    a, b : low-dimensional kernel parameters 1/(1 + a d^(2b))
    """

    x_scaled: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    knn_indices: np.ndarray
    knn_distances: np.ndarray
    sigmas: np.ndarray
    rhos: np.ndarray
    head: np.ndarray
    tail: np.ndarray
    weight: np.ndarray
    a: float
    b: float


def standardize(
    X: np.ndarray, mode: Literal["zscore", "center", "none"] = "zscore"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features and return (X_scaled, mean, scale).

    mode
    ----
    "zscore" : subtract mean, divide by per-feature std (default).
    "center" : subtract mean only; leave scale at 1. Preferred when features
               already share units (e.g. image pixels) — z-scoring low-variance
               dimensions there amplifies noise and blurs cluster separation.
    "none"   : no-op (mean=0, scale=1).

    ``mean`` and ``scale`` are always returned so the identical transform is
    reapplied to new data at inference time.
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if mode not in ("zscore", "center", "none"):
        raise ValueError("mode must be 'zscore', 'center', or 'none'")

    if mode == "none":
        mean = np.zeros(X.shape[1], dtype=np.float32)
    else:
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)

    if mode == "zscore":
        scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-7] = 1.0
    else:
        scale = np.ones(X.shape[1], dtype=np.float32)

    X_scaled = np.ascontiguousarray((X - mean) / scale, dtype=np.float32)
    return X_scaled, mean, scale


def _remove_self_from_search_batch(
    search_indices: np.ndarray,
    search_squared_distances: np.ndarray,
    row_ids: np.ndarray,
    required_nonself: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each query's own row id and retain the first required neighbors."""
    valid = (search_indices >= 0) & (search_indices != row_ids[:, None])
    ranks = np.cumsum(valid, axis=1, dtype=np.int32) - 1
    take = valid & (ranks < required_nonself)

    out_i = np.full(
        (search_indices.shape[0], required_nonself), -1, dtype=np.int32
    )
    out_d2 = np.full(
        (search_indices.shape[0], required_nonself), np.inf, dtype=np.float32
    )

    rr, cc = np.nonzero(take)
    dst = ranks[rr, cc]
    out_i[rr, dst] = search_indices[rr, cc].astype(np.int32, copy=False)
    out_d2[rr, dst] = search_squared_distances[rr, cc].astype(
        np.float32, copy=False
    )

    if np.any(out_i < 0):
        missing = int(np.sum(out_i < 0))
        raise RuntimeError(
            f"FAISS did not return enough non-self neighbors; {missing} slots "
            "are missing. Increase search_extra or HNSW efSearch."
        )

    return out_i, out_d2


def faiss_knn(
    X: np.ndarray,
    *,
    n_neighbors: int = 50,
    index_kind: Literal["flat", "hnsw"] = "hnsw",
    use_gpu_for_flat: bool = True,
    gpu_id: int = 0,
    search_batch_size: int = 65536,
    search_extra: int = 8,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
    num_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return UMAP-style neighbor arrays with self in column 0.

    n_neighbors follows UMAP's convention: it includes the self entry, so
    n_neighbors=50 produces 49 non-self neighbors.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Install faiss-cpu or a FAISS GPU build before calling faiss_knn"
        ) from exc

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    if not 2 <= n_neighbors < n:
        raise ValueError("n_neighbors must satisfy 2 <= n_neighbors < len(X)")

    if num_threads is None:
        num_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(int(num_threads))

    if index_kind == "flat":
        cpu_index = faiss.IndexFlatL2(d)
        index = cpu_index
        gpu_resources = None

        gpu_available = (
            use_gpu_for_flat
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > gpu_id
        )
        if gpu_available:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, cpu_index)

    elif index_kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        index.hnsw.efConstruction = int(hnsw_ef_construction)
        index.hnsw.efSearch = int(hnsw_ef_search)
        gpu_resources = None
    else:
        raise ValueError("index_kind must be 'flat' or 'hnsw'")

    index.add(X)

    required_nonself = n_neighbors - 1
    search_k = min(n, n_neighbors + search_extra)

    indices = np.empty((n, n_neighbors), dtype=np.int32)
    distances = np.empty((n, n_neighbors), dtype=np.float32)
    indices[:, 0] = np.arange(n, dtype=np.int32)
    distances[:, 0] = 0.0

    for start in range(0, n, search_batch_size):
        stop = min(n, start + search_batch_size)
        d2, idx = index.search(X[start:stop], search_k)

        row_ids = np.arange(start, stop, dtype=np.int64)
        nonself_i, nonself_d2 = _remove_self_from_search_batch(
            idx,
            d2,
            row_ids,
            required_nonself,
        )

        indices[start:stop, 1:] = nonself_i
        # FAISS METRIC_L2 returns squared Euclidean distances.
        distances[start:stop, 1:] = np.sqrt(
            np.maximum(nonself_d2, 0.0)
        ).astype(np.float32, copy=False)

    return indices, distances


def _compute_rhos(
    distances: np.ndarray,
    local_connectivity: float,
    tolerance: float = 1e-5,
) -> np.ndarray:
    if local_connectivity <= 0:
        raise ValueError("local_connectivity must be positive")

    n, width = distances.shape
    positive_count = np.count_nonzero(distances > 0.0, axis=1)
    first_positive = width - positive_count
    rho = np.zeros(n, dtype=np.float32)

    base = int(math.floor(local_connectivity))
    interpolation = local_connectivity - base
    enough = positive_count >= local_connectivity
    rows = np.arange(n)

    if base > 0:
        selected = rows[enough]
        pos = first_positive[selected] + base - 1
        left = distances[selected, pos]
        values = left.copy()
        if interpolation > tolerance:
            right = distances[selected, pos + 1]
            values += interpolation * (right - left)
        rho[selected] = values
    elif interpolation > tolerance:
        selected = rows[enough]
        pos = first_positive[selected]
        rho[selected] = interpolation * distances[selected, pos]

    some_but_not_enough = (positive_count > 0) & (~enough)
    selected = rows[some_but_not_enough]
    if selected.size:
        rho[selected] = distances[selected, -1]

    return rho


def smooth_knn_dist(
    distances: np.ndarray,
    *,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    n_iter: int = 64,
    tolerance: float = 1e-5,
    min_k_dist_scale: float = 1e-3,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row sigma and rho without importing umap-learn."""
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("distances must have shape (n_samples, >=2)")

    n, n_neighbors = distances.shape
    target = float(np.log2(n_neighbors) * bandwidth)
    rhos = _compute_rhos(distances, local_connectivity)
    sigmas = np.empty(n, dtype=np.float32)
    global_mean = float(np.mean(distances, dtype=np.float64))

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        d = distances[start:stop]
        rho = rhos[start:stop]
        delta = d[:, 1:] - rho[:, None]
        positive_delta = np.maximum(delta, 0.0)

        lo = np.zeros(stop - start, dtype=np.float32)
        hi = np.full(stop - start, np.inf, dtype=np.float32)
        mid = np.ones(stop - start, dtype=np.float32)

        for _ in range(n_iter):
            psum = np.exp(-positive_delta / mid[:, None]).sum(
                axis=1, dtype=np.float32
            )
            too_large = psum > target
            hi = np.where(too_large, mid, hi)
            lo = np.where(too_large, lo, mid)
            mid = np.where(np.isfinite(hi), 0.5 * (lo + hi), 2.0 * mid)

            if np.max(np.abs(psum - target)) < tolerance:
                break

        row_mean = np.mean(d, axis=1, dtype=np.float32)
        sigma_floor = np.where(
            rho > 0.0,
            min_k_dist_scale * row_mean,
            min_k_dist_scale * global_mean,
        ).astype(np.float32)

        sigmas[start:stop] = np.maximum(mid, sigma_floor)

    sigmas = np.maximum(sigmas, np.float32(1e-12))
    return sigmas, rhos


def fuzzy_graph_from_knn(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    *,
    local_connectivity: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    prune_below: float = 1e-4,
    sigma_batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct and symmetrize the fuzzy graph."""
    if not 0.0 <= set_op_mix_ratio <= 1.0:
        raise ValueError("set_op_mix_ratio must be in [0, 1]")
    if knn_indices.shape != knn_distances.shape:
        raise ValueError("knn_indices and knn_distances must have the same shape")

    n, n_neighbors = knn_indices.shape
    sigmas, rhos = smooth_knn_dist(
        knn_distances,
        local_connectivity=local_connectivity,
        batch_size=sigma_batch_size,
    )

    delta = knn_distances[:, 1:] - rhos[:, None]
    directed_weight = np.exp(
        -np.maximum(delta, 0.0) / sigmas[:, None]
    ).astype(np.float32, copy=False)

    cols = knn_indices[:, 1:].reshape(-1).astype(np.int32, copy=False)
    vals = directed_weight.reshape(-1)
    if np.any(cols < 0) or not np.isfinite(vals).all():
        raise ValueError("Invalid k-NN index or fuzzy membership value")

    # Every row has exactly n_neighbors - 1 entries, so construct CSR
    # directly and avoid allocating a repeated COO row array.
    indptr = np.arange(
        0, n * (n_neighbors - 1) + 1, n_neighbors - 1, dtype=np.int64
    )
    directed = sp.csr_matrix(
        (vals, cols, indptr), shape=(n, n), dtype=np.float32
    )
    directed.sum_duplicates()
    if prune_below > 0.0:
        directed.data[directed.data < prune_below] = 0.0
    directed.eliminate_zeros()

    transpose = directed.transpose().tocsr()
    product = directed.multiply(transpose)
    fuzzy_union = directed + transpose - product

    if set_op_mix_ratio == 1.0:
        graph = fuzzy_union
    elif set_op_mix_ratio == 0.0:
        graph = product
    else:
        graph = (
            set_op_mix_ratio * fuzzy_union
            + (1.0 - set_op_mix_ratio) * product
        )

    graph = graph.tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    np.clip(graph.data, 0.0, 1.0, out=graph.data)

    if prune_below > 0.0:
        graph.data[graph.data < prune_below] = 0.0
        graph.eliminate_zeros()

    graph = graph.tocoo(copy=False)
    return (
        graph.row.astype(np.int32, copy=False),
        graph.col.astype(np.int32, copy=False),
        graph.data.astype(np.float32, copy=False),
        sigmas,
        rhos,
    )


def fit_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> tuple[float, float]:
    """Fit q(d)=1/(1+a*d^(2b)) to the usual UMAP target curve."""
    if spread <= 0:
        raise ValueError("spread must be positive")
    if not 0 <= min_dist <= spread:
        raise ValueError("Require 0 <= min_dist <= spread")

    x = np.linspace(0.0, 3.0 * spread, 300, dtype=np.float64)
    y = np.where(
        x < min_dist,
        1.0,
        np.exp(-(x - min_dist) / spread),
    )

    def curve(x_value: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * np.power(x_value, 2.0 * b))

    params, _ = curve_fit(
        curve,
        x,
        y,
        p0=(1.0, 1.0),
        bounds=(0.0, np.inf),
        maxfev=10000,
    )
    return float(params[0]), float(params[1])


def build_fuzzy_graph(
    X: np.ndarray,
    *,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    spread: float = 1.0,
    index_kind: Literal["flat", "hnsw"] = "hnsw",
    use_gpu_for_flat: bool = True,
    local_connectivity: float = 1.0,
    prune_below: float = 1e-4,
    scale_mode: Literal["zscore", "center", "none"] = "zscore",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
) -> FuzzyGraphData:
    """Standardize -> FAISS k-NN -> fuzzy graph -> (a, b) fit, in one call."""
    x_scaled, mean, scale = standardize(X, mode=scale_mode)
    knn_indices, knn_distances = faiss_knn(
        x_scaled,
        n_neighbors=n_neighbors,
        index_kind=index_kind,
        use_gpu_for_flat=use_gpu_for_flat,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    head, tail, weight, sigmas, rhos = fuzzy_graph_from_knn(
        knn_indices,
        knn_distances,
        local_connectivity=local_connectivity,
        prune_below=prune_below,
    )
    a, b = fit_ab_params(spread=spread, min_dist=min_dist)
    return FuzzyGraphData(
        x_scaled=x_scaled,
        mean=mean,
        scale=scale,
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        sigmas=sigmas,
        rhos=rhos,
        head=head,
        tail=tail,
        weight=weight,
        a=a,
        b=b,
    )

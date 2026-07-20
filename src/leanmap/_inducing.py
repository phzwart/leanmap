"""Inducing-point (landmark) out-of-sample extension.

A training-free, UMAP-quality embedder for new points: store a small set of
landmarks with known embedding coordinates, then place any query point by its
high-dimensional *fuzzy membership* to the nearest landmarks — the same smooth
k-NN kernel UMAP uses. Because placement uses each query's actual distances to
the landmarks (not a fixed learned function), it generalizes like
``umap.transform`` rather than memorizing a training layout.

Landmark selection offers three strategies:

- ``"fps"`` (default) — farthest-point sampling / greedy k-center. A
  2-approximation to the k-center objective: every point is within a bounded
  radius of some landmark. Coverage-weighted, so sparse regions and rare
  classes still get a landmark. Recommended.
- ``"kmeans"`` — k-means centroids snapped to nearest real points.
  Density-weighted: more landmarks where data is dense.
- ``"hexgrid"`` — hexagonal lattice over the 2D reference embedding, pruned to
  occupied cells. Uniform in the embedding *plane* (not the manifold); can
  under-cover rare classes that occupy little embedding area.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._graph import smooth_knn_dist

LandmarkMethod = Literal["fps", "kmeans", "hexgrid"]


def farthest_point_sampling(Xs: np.ndarray, n_landmarks: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center selection. Returns indices into ``Xs``.

    Each new landmark is the point farthest from all currently chosen
    landmarks, giving a 2-approximation to the minimax coverage radius.
    """
    n = len(Xs)
    if not 1 <= n_landmarks <= n:
        raise ValueError("n_landmarks must satisfy 1 <= n_landmarks <= len(Xs)")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    d2 = ((Xs - Xs[first]) ** 2).sum(1)
    for _ in range(n_landmarks - 1):
        j = int(np.argmax(d2))
        chosen.append(j)
        d2 = np.minimum(d2, ((Xs - Xs[j]) ** 2).sum(1))
    return np.array(chosen, dtype=np.int64)


def _kmeans_landmarks(Xs: np.ndarray, n_landmarks: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_landmarks, random_state=seed, n_init=4).fit(Xs)
    idx = {int(np.argmin(((Xs - c) ** 2).sum(1))) for c in km.cluster_centers_}
    return np.array(sorted(idx), dtype=np.int64)


def _hexgrid_landmarks(
    emb2d: np.ndarray, target: int
) -> np.ndarray:
    """Occupied-cell centres of a hex lattice over the 2D embedding."""
    e = np.asarray(emb2d, dtype=np.float64)
    lo, hi = e.min(0), e.max(0)
    span = np.maximum(hi - lo, 1e-9)
    area = span[0] * span[1]
    s = np.sqrt(area / (max(target, 1) * np.sqrt(3) / 2))
    dx = s
    dy = s * np.sqrt(3) / 2
    row = np.round((e[:, 1] - lo[1]) / dy).astype(int)
    off = (row % 2) * (dx / 2)
    col = np.round((e[:, 0] - lo[0] - off) / dx).astype(int)
    keys = row * 100003 + col
    chosen = []
    for key in np.unique(keys):
        members = np.where(keys == key)[0]
        r = key // 100003
        c = key - r * 100003
        cx = lo[0] + c * dx + (r % 2) * (dx / 2)
        cy = lo[1] + r * dy
        center = np.array([cx, cy])
        best = members[np.argmin(((e[members] - center) ** 2).sum(1))]
        chosen.append(int(best))
    return np.array(sorted(chosen), dtype=np.int64)


def select_landmarks(
    Xs: np.ndarray,
    emb2d: np.ndarray | None,
    n_landmarks: int,
    *,
    method: LandmarkMethod = "fps",
    seed: int = 0,
) -> np.ndarray:
    """Choose landmark indices into ``Xs`` by the requested strategy.

    ``Xs`` is standardized high-dimensional data; ``emb2d`` is the reference 2D
    embedding (only needed for ``"hexgrid"``). For ``"hexgrid"`` the occupied
    cell count depends on the embedding shape, so ``n_landmarks`` is a target
    and the grid spacing is grown until at least that many cells are occupied.
    """
    if method == "fps":
        return farthest_point_sampling(Xs, n_landmarks, seed=seed)
    if method == "kmeans":
        return _kmeans_landmarks(Xs, n_landmarks, seed=seed)
    if method == "hexgrid":
        if emb2d is None:
            raise ValueError("hexgrid landmark selection needs a 2D embedding")
        # grow the target until enough cells are occupied (clustered embeddings
        # leave much of the bounding box empty).
        for factor in (1, 3, 5, 8, 12, 20):
            idx = _hexgrid_landmarks(emb2d, n_landmarks * factor)
            if len(idx) >= 0.85 * n_landmarks:
                return idx
        return idx
    raise ValueError("method must be 'fps', 'kmeans', or 'hexgrid'")


def induce_embed(
    Xs_query: np.ndarray,
    landmark_hd: np.ndarray,
    landmark_emb: np.ndarray,
    *,
    k: int = 5,
    local_connectivity: float = 1.0,
) -> np.ndarray:
    """Embed queries as UMAP-fuzzy-weighted averages of landmark coordinates.

    Parameters
    ----------
    Xs_query : (N, d) standardized query points.
    landmark_hd : (M, d) standardized landmark coordinates.
    landmark_emb : (M, 2) landmark embedding coordinates.
    k : number of nearest landmarks each query attaches to (smaller = sharper
        clusters; 5 is a good default).
    """
    Xs_query = np.ascontiguousarray(Xs_query, dtype=np.float32)
    landmark_hd = np.ascontiguousarray(landmark_hd, dtype=np.float32)
    landmark_emb = np.ascontiguousarray(landmark_emb, dtype=np.float32)
    m = len(landmark_hd)
    k = int(min(k, m))
    if k < 1:
        raise ValueError("k must be >= 1")

    # full query-to-landmark distance matrix (M is small, so this is cheap)
    d2 = (
        (Xs_query[:, None, :] - landmark_hd[None, :, :]) ** 2
    ).sum(-1)
    knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    knn_d2 = np.take_along_axis(d2, knn_idx, axis=1)
    order = np.argsort(knn_d2, axis=1)
    knn_idx = np.take_along_axis(knn_idx, order, axis=1)
    knn_d = np.sqrt(np.maximum(np.take_along_axis(knn_d2, order, axis=1), 0.0)).astype(
        np.float32
    )

    sigmas, rhos = smooth_knn_dist(knn_d, local_connectivity=local_connectivity)
    w = np.exp(-np.maximum(knn_d - rhos[:, None], 0.0) / sigmas[:, None]).astype(
        np.float32
    )
    w /= w.sum(1, keepdims=True) + 1e-12
    return np.einsum("nk,nkd->nd", w, landmark_emb[knn_idx]).astype(np.float32)


def landmark_conditioning(
    Xs_query: np.ndarray,
    landmark_hd: np.ndarray,
    landmark_emb: np.ndarray,
    *,
    k: int = 5,
    local_connectivity: float = 1.0,
) -> np.ndarray:
    """Per-point conditioning vector for FiLM: ``[est_x, est_y, log1p(nn_dist)]``.

    ``est`` is the landmark inducing estimate (where the landmarks place the
    point); ``nn_dist`` is the distance to the nearest landmark. This summarizes
    the point's manifold membership for the FiLM generator. The estimate columns
    carry most of the signal; the distance column adds a small margin.
    """
    est = induce_embed(
        Xs_query, landmark_hd, landmark_emb, k=k, local_connectivity=local_connectivity
    )
    d2 = ((np.asarray(Xs_query, np.float32)[:, None, :] - np.asarray(landmark_hd, np.float32)[None, :, :]) ** 2).sum(-1)
    nn_d = np.sqrt(np.maximum(d2.min(1), 0.0))
    return np.column_stack([est, np.log1p(nn_d)]).astype(np.float32)


def coverage_radius(Xs: np.ndarray, landmark_idx: np.ndarray) -> tuple[float, float]:
    """(max, mean) distance from any point to its nearest landmark."""
    L = Xs[landmark_idx]
    nearest = np.sqrt(((Xs[:, None, :] - L[None, :, :]) ** 2).sum(-1).min(1))
    return float(nearest.max()), float(nearest.mean())

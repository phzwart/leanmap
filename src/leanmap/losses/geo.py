"""Geodesic gauge helpers: pyramid-level Dijkstra with metric edge lengths."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from leanmap.distance import DistanceFn
from leanmap.landmarks import classical_mds

# Default flip: level 0 below this R, level 1 at/above (design §8).
DEFAULT_GAUGE_R_THRESHOLD: float = 3e5

__all__ = [
    "DEFAULT_GAUGE_R_THRESHOLD",
    "select_gauge_level",
    "metric_edge_lengths",
    "landmark_geodesics_on_level",
    "gauge_nu_diagnostic",
]


def select_gauge_level(n_reps: int, threshold: float = DEFAULT_GAUGE_R_THRESHOLD) -> int:
    """Choose pyramid level for the geodesic gauge.

    Returns 0 when ``R < threshold``, else 1. Callers with a single pyramid
    level must clamp to 0 regardless of ``R``.
    """
    return 0 if int(n_reps) < float(threshold) else 1


def metric_edge_lengths(
    X: torch.Tensor,
    edges: torch.Tensor,
    dist_fn: DistanceFn,
    *,
    block: int = 4096,
) -> torch.Tensor:
    """Ambient metric length of each undirected edge.

    Parameters
    ----------
    X :
        Node coordinates ``(R, D)`` aligned with ``edges`` indices (typically
        representative rows for the chosen pyramid level).
    edges :
        ``(E, 2)`` int64 endpoint indices into ``X``.
    dist_fn :
        Ambient distance ``dist_fn(A, B) -> (n, m)``.

    Returns
    -------
    lengths : ``(E,)`` float32 — ``dist_fn`` between endpoints, never fuzzy
        memberships / squashed affinities.
    """
    e = torch.as_tensor(edges, dtype=torch.int64).reshape(-1, 2)
    n_e = int(e.shape[0])
    if n_e == 0:
        return torch.zeros(0, dtype=torch.float32)
    X = torch.as_tensor(X, dtype=torch.float32)
    out = torch.empty(n_e, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n_e, block):
            t = min(n_e, s + block)
            A = X[e[s:t, 0]]
            B = X[e[s:t, 1]]
            D = dist_fn(A, B)
            out[s:t] = torch.diagonal(D).detach().cpu().to(torch.float32)
    return out


def _edges_and_n(
    graph_or_edges: Any,
    landmark_idx: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if hasattr(graph_or_edges, "edges") and hasattr(graph_or_edges, "reps"):
        edges = torch.as_tensor(graph_or_edges.edges, dtype=torch.int64)
        n = int(graph_or_edges.reps.rep_idx.shape[0])
        return edges, n
    edges = torch.as_tensor(graph_or_edges, dtype=torch.int64).reshape(-1, 2)
    lm = torch.as_tensor(landmark_idx, dtype=torch.int64).reshape(-1)
    hi = int(lm.max().item()) if lm.numel() else -1
    if edges.numel():
        hi = max(hi, int(edges.max().item()))
    return edges, hi + 1


def landmark_geodesics_on_level(
    graph_or_edges: Any,
    edge_lengths: Union[torch.Tensor, np.ndarray],
    landmark_idx: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Pairwise graph geodesics among landmarks via Dijkstra on one level.

    Parameters
    ----------
    graph_or_edges :
        A :class:`~leanmap.build.pipeline.Graph` or an ``(E, 2)`` edge index
        tensor into that level's nodes.
    edge_lengths :
        ``(E,)`` non-negative **metric** lengths (see :func:`metric_edge_lengths`).
        Must not be squashed fuzzy weights.
    landmark_idx :
        ``(L,)`` node indices on the same level (e.g. ``member_of`` of matched
        training rows).

    Returns
    -------
    G : ``(L, L)`` float32 geodesic distances (``inf`` where unreachable).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    edges, n = _edges_and_n(graph_or_edges, landmark_idx)
    lengths = np.asarray(
        edge_lengths.detach().cpu() if isinstance(edge_lengths, torch.Tensor) else edge_lengths,
        dtype=np.float64,
    ).reshape(-1)
    lm = np.asarray(
        landmark_idx.detach().cpu() if isinstance(landmark_idx, torch.Tensor) else landmark_idx,
        dtype=np.int64,
    ).reshape(-1)
    L = int(lm.shape[0])
    if L < 2:
        raise ValueError("need at least 2 landmarks for a geodesic gauge")
    if edges.shape[0] != lengths.shape[0]:
        raise ValueError(
            f"edges ({edges.shape[0]}) and edge_lengths ({lengths.shape[0]}) size mismatch"
        )
    if n <= 0:
        G = np.full((L, L), np.inf, dtype=np.float64)
        np.fill_diagonal(G, 0.0)
        return torch.as_tensor(G, dtype=torch.float32)

    e = edges.detach().cpu().numpy().astype(np.int64) if isinstance(edges, torch.Tensor) else np.asarray(edges, dtype=np.int64)
    if e.shape[0] == 0:
        G = np.full((L, L), np.inf, dtype=np.float64)
        np.fill_diagonal(G, 0.0)
        return torch.as_tensor(G, dtype=torch.float32)

    # Clamp non-positive lengths to a tiny positive so Dijkstra stays valid;
    # metric distances should already be >= 0.
    data = np.maximum(lengths, 1e-12)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    vals = np.concatenate([data, data])
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    D = dijkstra(A, directed=False, indices=lm)  # (L, n)
    G = D[:, lm].astype(np.float64)
    return torch.as_tensor(G, dtype=torch.float32)


def gauge_nu_diagnostic(
    Y_mds: torch.Tensor,
    geodesics: torch.Tensor,
    *,
    finite: Optional[torch.Tensor] = None,
) -> float:
    """MDS negative-eigenvalue mass ν of the geodesic Gram (paper diagnostic).

    Wraps :func:`~leanmap.landmarks.classical_mds` diagnostics. ``Y_mds`` sets
    the embedding dimension; ν is independent of the particular MDS coordinates
    up to numerical noise.
    """
    d = int(Y_mds.shape[1]) if Y_mds.ndim == 2 and Y_mds.shape[1] > 0 else 2
    if finite is None:
        finite = torch.isfinite(geodesics)
        if finite.ndim == 2:
            finite = finite.clone()
            finite.fill_diagonal_(False)
    _, diag = classical_mds(
        geodesics, d=d, finite=finite, return_diagnostics=True
    )
    return float(diag["mds_neg_eigen_ratio"])

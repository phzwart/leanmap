"""Graph construction (N/R streaming passes).

Distributed landmark-bunch builds live in :mod:`leanmap.build.bunches`
(optional ``leanmap[hpc]``). Import them directly::

    from leanmap.build.bunches import build_graph_bunches

They are intentionally **not** re-exported here so core / pipeline imports
never pull in the HPC path at module load time.
"""
from __future__ import annotations

from .pipeline import (
    GRAPH_PYRAMID_VERSION,
    Graph,
    GraphStats,
    Representatives,
    build_graph,
    build_graph_pyramid,
    build_representatives,
    check_tensor_fingerprint,
    estimate_epsilon,
    graph_from_state,
    graph_to_state,
    knn_representatives,
    landmark_backbone,
    load_graph_pyramid,
    save_graph_pyramid,
    smooth_knn,
    tensor_fingerprint,
    union_assign_topc,
    validate_precomputed_knn,
)
from .resolution import solve_delta

__all__ = [
    "GRAPH_PYRAMID_VERSION",
    "Graph",
    "GraphStats",
    "Representatives",
    "build_graph",
    "build_graph_pyramid",
    "build_representatives",
    "check_tensor_fingerprint",
    "estimate_epsilon",
    "graph_from_state",
    "graph_to_state",
    "knn_representatives",
    "landmark_backbone",
    "load_graph_pyramid",
    "save_graph_pyramid",
    "smooth_knn",
    "solve_delta",
    "tensor_fingerprint",
    "union_assign_topc",
    "validate_precomputed_knn",
]

"""Versioned graph-store layout."""
from __future__ import annotations

STORE_SCHEMA_VERSION = 1
GRAPH_PT_VERSION = 1

# Auto-select directory store when representative count exceeds this.
DIR_STORE_R_THRESHOLD = 50_000

META_FILENAME = "meta.json"

# Directory layout keys present from day one (may be empty).
STORE_DIRS = (
    "reps",
    "knn",
    "csr",
    "pyramid",
    "alias",
    "density",
    "gauge",
    "paths",
)

# Keys that invalidate a frozen store when they disagree with the build config.
INVALIDATION_KEYS = (
    "metric_name",
    "epsilon",
    "delta",
    "n_neighbors",
    "n_landmarks",
    "seed",
    "dedup",
    "n_pyramid_levels",
)

__all__ = [
    "STORE_SCHEMA_VERSION",
    "GRAPH_PT_VERSION",
    "DIR_STORE_R_THRESHOLD",
    "META_FILENAME",
    "STORE_DIRS",
    "INVALIDATION_KEYS",
]

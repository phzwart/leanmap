"""Shim — implementation in :mod:`leanmap.paths` (core capability)."""
from __future__ import annotations

from leanmap.paths import *  # noqa: F403
from leanmap.paths import (
    PATH_C,
    PATH_C_HI,
    PATH_MARGIN,
    PATH_ORD_CHANCE,
    PATH_PAIRS_PER_STEP,
    PathConstraint,
    PathTripletSampler,
    build_collinear_spatial_triplets,
    build_path_triplets,
    build_path_triplets_with_stats,
    encode_groups,
    parse_index,
    path_constraint_loss,
    record_path_build_stats,
    remap_triplets,
    subset_by_group,
)

__all__ = [
    "PATH_C",
    "PATH_C_HI",
    "PATH_MARGIN",
    "PATH_ORD_CHANCE",
    "PATH_PAIRS_PER_STEP",
    "PathConstraint",
    "PathTripletSampler",
    "build_collinear_spatial_triplets",
    "build_path_triplets",
    "build_path_triplets_with_stats",
    "encode_groups",
    "parse_index",
    "path_constraint_loss",
    "record_path_build_stats",
    "remap_triplets",
    "subset_by_group",
]

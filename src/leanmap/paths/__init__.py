"""Declared path (bi-Lipschitz) constraints — core capability."""
from __future__ import annotations

from .build import (
    build_path_triplets,
    build_path_triplets_with_stats,
    record_path_build_stats,
    remap_triplets,
)
from .constraint import (
    PATH_C,
    PATH_C_HI,
    PATH_MARGIN,
    PATH_ORD_CHANCE,
    PATH_PAIRS_PER_STEP,
    PathConstraint,
    PathTripletSampler,
    encode_groups,
    parse_index,
    path_constraint_loss,
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
    "build_path_triplets",
    "build_path_triplets_with_stats",
    "encode_groups",
    "parse_index",
    "path_constraint_loss",
    "record_path_build_stats",
    "remap_triplets",
    "subset_by_group",
]

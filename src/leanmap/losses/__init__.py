"""Training losses."""
from __future__ import annotations

from .ddp_stats import (
    allreduce_density_moments,
    allreduce_mean,
    allreduce_mean_affinity,
    allreduce_path_scale,
)
from .geo import (
    DEFAULT_GAUGE_R_THRESHOLD,
    gauge_nu_diagnostic,
    landmark_geodesics_on_level,
    metric_edge_lengths,
    select_gauge_level,
)
from .geom import (
    _best_rotation,
    _clamp_prob,
    _fit_ab,
    alignment_ramp,
    find_ab_params,
    fuzzy_cross_entropy,
    geodesic_stress_loss,
    landmark_regularisation,
    local_isometry_loss,
    local_rigidity_loss,
    min_dist_for_b,
    ordinal_triplet_loss,
    procrustes_anchor_loss,
)
from .path import path_constraint_loss

__all__ = [
    "DEFAULT_GAUGE_R_THRESHOLD",
    "alignment_ramp",
    "allreduce_density_moments",
    "allreduce_mean",
    "allreduce_mean_affinity",
    "allreduce_path_scale",
    "find_ab_params",
    "fuzzy_cross_entropy",
    "gauge_nu_diagnostic",
    "geodesic_stress_loss",
    "landmark_geodesics_on_level",
    "landmark_regularisation",
    "local_isometry_loss",
    "local_rigidity_loss",
    "metric_edge_lengths",
    "min_dist_for_b",
    "ordinal_triplet_loss",
    "path_constraint_loss",
    "procrustes_anchor_loss",
    "select_gauge_level",
    "_best_rotation",
    "_clamp_prob",
    "_fit_ab",
]

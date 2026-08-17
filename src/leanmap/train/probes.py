"""Training probes / sufficiency gates for exemplar policy refresh (PR-9)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

# Simple default thresholds — applications may tighten via probe_metrics overrides.
DEFAULT_CELL_COVERAGE_MIN: float = 0.90
DEFAULT_LANDMARK_MASS_RATIO_MIN: float = 0.50
DEFAULT_PATH_COVERAGE_MIN: float = 0.80
DEFAULT_PROBE_GAP_MAX: float = 0.25


def sufficiency_gates(probe_metrics: Optional[Mapping[str, Any]] = None) -> bool:
    """Return True when probe metrics pass simple sufficiency thresholds.

    Recognised keys (all optional; missing keys are treated as passing):

    - ``cell_coverage`` — fraction of occupied cells hit in the epoch stream
    - ``landmark_mass_ratio`` — stream landmark mass / population landmark mass
    - ``path_coverage`` — fraction of path triplets still resolvable
    - ``probe_gap`` — train-vs-probe metric gap (lower is better)
    """
    m = dict(probe_metrics or {})
    if "cell_coverage" in m:
        if float(m["cell_coverage"]) < DEFAULT_CELL_COVERAGE_MIN:
            return False
    if "landmark_mass_ratio" in m:
        if float(m["landmark_mass_ratio"]) < DEFAULT_LANDMARK_MASS_RATIO_MIN:
            return False
    if "path_coverage" in m:
        if float(m["path_coverage"]) < DEFAULT_PATH_COVERAGE_MIN:
            return False
    if "probe_gap" in m:
        if float(m["probe_gap"]) > DEFAULT_PROBE_GAP_MAX:
            return False
    return True


def maybe_refresh_policy(
    policy: Any,
    gates: Union[bool, Mapping[str, Any]],
    stats: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Refresh ``policy`` when sufficiency gates fail.

    ``gates`` may be the boolean from :func:`sufficiency_gates` or a metrics
    mapping (evaluated here). Returns ``True`` if ``refresh`` was called.
    """
    ok = bool(gates) if not isinstance(gates, Mapping) else sufficiency_gates(gates)
    if ok:
        return False
    refresh_stats = stats if stats is not None else getattr(policy, "_last_stats", {}) or {}
    policy.refresh(refresh_stats)
    return True


__all__ = [
    "DEFAULT_CELL_COVERAGE_MIN",
    "DEFAULT_LANDMARK_MASS_RATIO_MIN",
    "DEFAULT_PATH_COVERAGE_MIN",
    "DEFAULT_PROBE_GAP_MAX",
    "maybe_refresh_policy",
    "sufficiency_gates",
]

"""Typed diagnostics record D (filled incrementally across PRs)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DiagnosticsRecord:
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    compression_ratio: Optional[float] = None
    knn_mode: Optional[str] = None
    knn_recall: Optional[float] = None
    gauge_level: Optional[int] = None
    nu: Optional[float] = None
    # Resolution-guard outcome from ``solve_delta`` / build (``True``/``False``).
    guard_ok: Optional[bool] = None
    delta_mode: Optional[str] = None  # "eps" | "calibrated" | "auto_fallback"
    # Exemplar stream measure p_t (PR-9): ``uniform`` | ``sufficient_v1``.
    exemplar_policy: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_graph_stats(
        cls,
        stats: Any,
        *,
        epsilon: Optional[float] = None,
    ) -> "DiagnosticsRecord":
        """Fill ε / δ / guard fields from a :class:`~leanmap.build.pipeline.GraphStats`."""
        eps = float(getattr(stats, "epsilon", epsilon if epsilon is not None else 0.0))
        dlt = getattr(stats, "delta", None)
        extra = dict(getattr(stats, "extra", {}) or {})
        if dlt is None:
            dlt = extra.get("delta", eps)
        gauge_level = extra.get("gauge_level")
        nu = extra.get("nu")
        return cls(
            epsilon=eps,
            delta=float(dlt) if dlt is not None else None,
            compression_ratio=float(getattr(stats, "compression_ratio", 1.0)),
            knn_mode=str(getattr(stats, "knn_mode", "brute")),
            knn_recall=getattr(stats, "knn_recall", None),
            gauge_level=int(gauge_level) if gauge_level is not None else None,
            nu=float(nu) if nu is not None else None,
            guard_ok=extra.get("delta_guard_ok"),
            delta_mode=extra.get("delta_mode"),
            extra=extra,
        )


__all__ = ["DiagnosticsRecord"]

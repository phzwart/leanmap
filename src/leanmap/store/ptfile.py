"""Single-file graph.pt backend (byte-identical wrap of pipeline I/O)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch

from leanmap.build.pipeline import (
    GRAPH_PYRAMID_VERSION,
    Graph,
    check_tensor_fingerprint,
    graph_from_state,
    graph_to_state,
    load_graph_pyramid,
    save_graph_pyramid,
    tensor_fingerprint,
)
from leanmap.diagnostics.record import DiagnosticsRecord

from .schema import GRAPH_PT_VERSION, STORE_SCHEMA_VERSION

PathLike = Union[str, Path]


def _meta_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    graphs = payload.get("graphs") or []
    n_levels = len(graphs)
    stats0 = None
    if graphs:
        g0 = graphs[0]
        stats0 = g0.stats if hasattr(g0, "stats") else (g0.get("stats") if isinstance(g0, dict) else None)
    diagnostics: Dict[str, Any]
    if isinstance(stats0, dict):
        diagnostics = DiagnosticsRecord(
            epsilon=stats0.get("epsilon"),
            delta=stats0.get("delta", stats0.get("epsilon")),
            compression_ratio=stats0.get("compression_ratio"),
            knn_mode=stats0.get("knn_mode"),
            knn_recall=stats0.get("knn_recall"),
            guard_ok=(stats0.get("extra") or {}).get("delta_guard_ok"),
            delta_mode=(stats0.get("extra") or {}).get("delta_mode"),
            extra=dict(stats0.get("extra") or {}),
        ).to_dict()
    elif stats0 is not None:
        diagnostics = DiagnosticsRecord.from_graph_stats(stats0).to_dict()
    else:
        diagnostics = DiagnosticsRecord(epsilon=payload.get("epsilon")).to_dict()

    meta = {
        "schema_version": STORE_SCHEMA_VERSION,
        "backend": "ptfile",
        "version": int(payload.get("version", GRAPH_PT_VERSION)),
        "metric_name": payload.get("metric_name"),
        "n_all": payload.get("n_all"),
        "n_landmarks": payload.get("n_landmarks"),
        "n_neighbors": payload.get("n_neighbors"),
        "epsilon": payload.get("epsilon"),
        "seed": payload.get("seed"),
        "dedup": payload.get("dedup"),
        "fingerprint": payload.get("fingerprint"),
        "n_pyramid_levels": n_levels,
        "diagnostics": diagnostics,
    }
    if payload.get("delta") is not None:
        meta["delta"] = payload["delta"]
    elif isinstance(stats0, dict) and stats0.get("delta") is not None:
        meta["delta"] = stats0["delta"]
    elif stats0 is not None and getattr(stats0, "delta", None) is not None:
        meta["delta"] = float(stats0.delta)
    return meta


class PtFileStore:
    """Graph store backed by a single ``torch.save`` pyramid file."""

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self._cache: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> Dict[str, Any]:
        if self._cache is None:
            self._cache = load_graph_pyramid(self.path)
        return self._cache

    def meta(self) -> dict[str, Any]:
        return _meta_from_payload(self._ensure_loaded())

    def edges(self, level: int = 0) -> torch.Tensor:
        graphs = self._ensure_loaded()["graphs"]
        if level < 0 or level >= len(graphs):
            raise IndexError(f"pyramid level {level} out of range [0, {len(graphs)})")
        return graphs[level].edges

    def load(self) -> dict[str, Any]:
        return self._ensure_loaded()

    def save(
        self,
        *,
        graphs: Sequence[Graph],
        M: torch.Tensor,
        assign_top1: torch.Tensor,
        assign_topc: torch.Tensor,
        train_idx: torch.Tensor,
        calib_idx: torch.Tensor,
        fingerprint: Dict[str, Any],
        metric_name: str,
        n_all: int,
        n_neighbors: int,
        epsilon: float,
        seed: int,
        dedup: bool,
    ) -> Path:
        """Delegate to :func:`save_graph_pyramid` (unchanged on-disk format)."""
        path = save_graph_pyramid(
            self.path,
            graphs=graphs,
            M=M,
            assign_top1=assign_top1,
            assign_topc=assign_topc,
            train_idx=train_idx,
            calib_idx=calib_idx,
            fingerprint=fingerprint,
            metric_name=metric_name,
            n_all=n_all,
            n_neighbors=n_neighbors,
            epsilon=epsilon,
            seed=seed,
            dedup=dedup,
        )
        self._cache = None
        return path

    def save_from_state(self, state: Dict[str, Any]) -> Path:
        """Persist a payload previously returned by :meth:`load` / ``load_graph_pyramid``."""
        return self.save(
            graphs=state["graphs"],
            M=state["M"],
            assign_top1=state["assign_top1"],
            assign_topc=state["assign_topc"],
            train_idx=state["train_idx"],
            calib_idx=state["calib_idx"],
            fingerprint=state["fingerprint"],
            metric_name=state["metric_name"],
            n_all=int(state["n_all"]),
            n_neighbors=int(state["n_neighbors"]),
            epsilon=float(state["epsilon"]),
            seed=int(state["seed"]),
            dedup=bool(state["dedup"]),
        )


__all__ = [
    "GRAPH_PYRAMID_VERSION",
    "GRAPH_PT_VERSION",
    "PtFileStore",
    "check_tensor_fingerprint",
    "graph_from_state",
    "graph_to_state",
    "load_graph_pyramid",
    "save_graph_pyramid",
    "tensor_fingerprint",
]

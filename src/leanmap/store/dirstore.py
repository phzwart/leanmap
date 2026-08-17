"""Directory graph store (meta.json + numpy/torch arrays)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from leanmap.build.pipeline import Graph, graph_from_state, graph_to_state
from leanmap.diagnostics.record import DiagnosticsRecord
from leanmap.utils import get_logger

from .fingerprint import fingerprint_array
from .schema import META_FILENAME, STORE_DIRS, STORE_SCHEMA_VERSION

PathLike = Union[str, Path]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _save_npy(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), _to_numpy(array))


def _load_npy(path: Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def _ensure_store_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in STORE_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def _diagnostics_from_graphs(graphs: Sequence[Graph], epsilon: float) -> Dict[str, Any]:
    if not graphs:
        return DiagnosticsRecord(epsilon=float(epsilon)).to_dict()
    stats = graphs[0].stats
    rec = DiagnosticsRecord.from_graph_stats(stats, epsilon=float(epsilon))
    rec.extra = {
        "n_reps": int(getattr(stats, "n_reps", graphs[0].reps.rep_idx.shape[0])),
        "n_components_before_backbone": int(
            getattr(stats, "n_components_before_backbone", 1)
        ),
        **dict(rec.extra or {}),
    }
    return rec.to_dict()


def _write_level(level_dir: Path, state: Dict[str, Any]) -> None:
    level_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(level_dir / "edges.npy", state["edges"])
    _save_npy(level_dir / "weights.npy", state["weights"])
    _save_npy(level_dir / "knn_idx.npy", state["knn_idx"])
    reps = state["reps"]
    for key in ("rep_idx", "member_of", "weight", "offsets", "values"):
        _save_npy(level_dir / f"reps_{key}.npy", reps[key])
    (level_dir / "stats.json").write_text(
        json.dumps(state["stats"], indent=2, default=_json_default) + "\n"
    )


def _read_level(level_dir: Path) -> Dict[str, Any]:
    reps = {
        key: _load_npy(level_dir / f"reps_{key}.npy")
        for key in ("rep_idx", "member_of", "weight", "offsets", "values")
    }
    stats = json.loads((level_dir / "stats.json").read_text())
    return {
        "edges": _load_npy(level_dir / "edges.npy"),
        "weights": _load_npy(level_dir / "weights.npy"),
        "knn_idx": _load_npy(level_dir / "knn_idx.npy"),
        "reps": reps,
        "stats": stats,
    }


def _mirror_finest(root: Path, state: Dict[str, Any]) -> None:
    """Write finest-level arrays into reps/, knn/, csr/ for the layout contract."""
    reps = state["reps"]
    for key in ("rep_idx", "member_of", "weight", "offsets", "values"):
        _save_npy(root / "reps" / f"{key}.npy", reps[key])
    _save_npy(root / "knn" / "knn_idx.npy", state["knn_idx"])
    _save_npy(root / "csr" / "edges.npy", state["edges"])
    _save_npy(root / "csr" / "weights.npy", state["weights"])


class DirStore:
    """Directory-backed frozen graph store."""

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._state_cache: Optional[Dict[str, Any]] = None

    def _meta_path(self) -> Path:
        return self.path / META_FILENAME

    def meta(self) -> dict[str, Any]:
        if self._meta_cache is not None:
            return dict(self._meta_cache)
        p = self._meta_path()
        if not p.exists():
            raise FileNotFoundError(f"DirStore meta missing: {p}")
        self._meta_cache = json.loads(p.read_text())
        return dict(self._meta_cache)

    def edges(self, level: int = 0) -> torch.Tensor:
        level_dir = self.path / "pyramid" / f"level_{int(level):03d}"
        edges_path = level_dir / "edges.npy"
        if not edges_path.exists():
            # Fall back to full load for a clearer error / compatibility.
            graphs = self.load()["graphs"]
            if level < 0 or level >= len(graphs):
                raise IndexError(
                    f"pyramid level {level} out of range [0, {len(graphs)})"
                )
            return graphs[level].edges
        return torch.as_tensor(_load_npy(edges_path), dtype=torch.int64)

    def load(self) -> dict[str, Any]:
        if self._state_cache is not None:
            return self._state_cache
        meta = self.meta()
        root = self.path
        n_levels = int(meta["n_pyramid_levels"])
        graphs = []
        for i in range(n_levels):
            level_dir = root / "pyramid" / f"level_{i:03d}"
            graphs.append(graph_from_state(_read_level(level_dir)))
        state: Dict[str, Any] = {
            "version": int(meta.get("graph_pt_version", meta.get("version", 1))),
            "metric_name": meta["metric_name"],
            "n_all": int(meta["n_all"]),
            "n_landmarks": int(meta["n_landmarks"]),
            "n_neighbors": int(meta["n_neighbors"]),
            "epsilon": float(meta["epsilon"]),
            "seed": int(meta["seed"]),
            "dedup": bool(meta["dedup"]),
            "fingerprint": meta["fingerprint"],
            "train_idx": torch.as_tensor(
                _load_npy(root / "train_idx.npy"), dtype=torch.int64
            ),
            "calib_idx": torch.as_tensor(
                _load_npy(root / "calib_idx.npy"), dtype=torch.int64
            ),
            "graphs": graphs,
            "M": torch.as_tensor(_load_npy(root / "M.npy")),
            "assign_top1": torch.as_tensor(
                _load_npy(root / "assign_top1.npy"), dtype=torch.int64
            ),
            "assign_topc": torch.as_tensor(
                _load_npy(root / "assign_topc.npy"), dtype=torch.int64
            ),
        }
        if "delta" in meta:
            state["delta"] = meta["delta"]
        self._state_cache = state
        return state

    def save(
        self,
        *,
        graphs: Sequence[Graph],
        M: torch.Tensor,
        assign_top1: torch.Tensor,
        assign_topc: torch.Tensor,
        train_idx: torch.Tensor,
        calib_idx: torch.Tensor,
        fingerprint: Optional[Dict[str, Any]] = None,
        metric_name: str,
        n_all: int,
        n_neighbors: int,
        epsilon: float,
        seed: int,
        dedup: bool,
        X: Any = None,
        delta: Optional[float] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write a directory store round-trippable to :meth:`load`."""
        root = self.path
        _ensure_store_dirs(root)

        if fingerprint is None:
            if X is None:
                raise ValueError("DirStore.save requires fingerprint= or X=")
            fingerprint = fingerprint_array(X)
        elif X is not None and "digest" not in fingerprint:
            # Upgrade legacy fingerprints when ambient X is available.
            fingerprint = fingerprint_array(X)

        graph_states = [graph_to_state(g) for g in graphs]
        for i, state in enumerate(graph_states):
            _write_level(root / "pyramid" / f"level_{i:03d}", state)
        if graph_states:
            _mirror_finest(root, graph_states[0])

        _save_npy(root / "train_idx.npy", train_idx)
        _save_npy(root / "calib_idx.npy", calib_idx)
        _save_npy(root / "M.npy", M)
        _save_npy(root / "assign_top1.npy", assign_top1)
        _save_npy(root / "assign_topc.npy", assign_topc)

        diag = diagnostics if diagnostics is not None else _diagnostics_from_graphs(
            graphs, float(epsilon)
        )
        # Promote gauge fields from graph.stats.extra when the caller did not
        # already put them on the diagnostics record (fit records them there).
        if isinstance(diag, dict) and graphs:
            extra0 = dict(getattr(graphs[0].stats, "extra", {}) or {})
            if diag.get("gauge_level") is None and "gauge_level" in extra0:
                diag["gauge_level"] = int(extra0["gauge_level"])
            if diag.get("nu") is None and "nu" in extra0:
                diag["nu"] = float(extra0["nu"])
        meta: Dict[str, Any] = {
            "schema_version": STORE_SCHEMA_VERSION,
            "backend": "dirstore",
            "metric_name": str(metric_name),
            "n_all": int(n_all),
            "n_landmarks": int(M.shape[0]),
            "n_neighbors": int(n_neighbors),
            "epsilon": float(epsilon),
            "seed": int(seed),
            "dedup": bool(dedup),
            "fingerprint": fingerprint,
            "n_pyramid_levels": len(graphs),
            "diagnostics": diag,
        }
        if delta is not None:
            meta["delta"] = float(delta)
            if isinstance(meta["diagnostics"], dict):
                meta["diagnostics"]["delta"] = float(delta)

        # gauge/ artefact when helpers recorded a level / ν.
        if isinstance(diag, dict) and (
            diag.get("gauge_level") is not None or diag.get("nu") is not None
        ):
            gauge_payload = {
                "gauge_level": diag.get("gauge_level"),
                "nu": diag.get("nu"),
            }
            (root / "gauge" / "gauge.json").write_text(
                json.dumps(gauge_payload, indent=2, default=_json_default) + "\n"
            )

        self._meta_path().write_text(
            json.dumps(meta, indent=2, default=_json_default) + "\n"
        )
        self._meta_cache = meta
        self._state_cache = None
        log = get_logger()
        log.info(
            "saved DirStore %s (%d level(s), R=%d, L=%d)",
            root,
            len(graphs),
            int(graphs[0].reps.rep_idx.shape[0]) if graphs else 0,
            int(M.shape[0]),
        )
        return root

    def save_from_state(self, state: Dict[str, Any], *, X: Any = None) -> Path:
        """Persist a payload previously returned by ``load_graph_pyramid`` / :meth:`load`."""
        return self.save(
            graphs=state["graphs"],
            M=state["M"],
            assign_top1=state["assign_top1"],
            assign_topc=state["assign_topc"],
            train_idx=state["train_idx"],
            calib_idx=state["calib_idx"],
            fingerprint=state.get("fingerprint"),
            metric_name=state["metric_name"],
            n_all=int(state["n_all"]),
            n_neighbors=int(state["n_neighbors"]),
            epsilon=float(state["epsilon"]),
            seed=int(state["seed"]),
            dedup=bool(state["dedup"]),
            X=X,
            delta=state.get("delta"),
        )


__all__ = ["DirStore"]

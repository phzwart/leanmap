"""GraphStore protocol and backend selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .fingerprint import verify_fingerprint
from .schema import DIR_STORE_R_THRESHOLD, INVALIDATION_KEYS

PathLike = Union[str, Path]


@runtime_checkable
class GraphStore(Protocol):
    """Frozen graph artefact (``graph.pt`` or directory store)."""

    def meta(self) -> dict[str, Any]:
        """Scalar / diagnostic metadata (no large arrays)."""
        ...

    def edges(self, level: int = 0) -> Any:
        """Edge index tensor ``(E, 2)`` at pyramid ``level`` (0 = finest)."""
        ...

    def load(self) -> dict[str, Any]:
        """Full pyramid state compatible with :func:`leanmap.train.fit`."""
        ...


def select_backend(path: PathLike, n_reps: Optional[int] = None) -> str:
    """Choose ``\"ptfile\"`` or ``\"dirstore\"`` for ``path``.

    Rules:
    - Existing ``.pt`` file, or path ending in ``.pt`` → ptfile
    - Existing directory → dirstore
    - Creating new: dirstore when ``n_reps > DIR_STORE_R_THRESHOLD``, else ptfile
    """
    p = Path(path)
    if p.suffix == ".pt" or str(p).endswith(".pt"):
        return "ptfile"
    if p.exists() and p.is_dir():
        return "dirstore"
    if p.exists() and p.is_file():
        return "ptfile"
    # Creating new (path does not exist yet).
    if n_reps is not None and int(n_reps) > int(DIR_STORE_R_THRESHOLD):
        return "dirstore"
    return "ptfile"


def open_graph_store(path: PathLike, *, n_reps: Optional[int] = None) -> GraphStore:
    """Open a graph store, auto-selecting the backend."""
    backend = select_backend(path, n_reps=n_reps)
    if backend == "dirstore":
        from .dirstore import DirStore

        return DirStore(path)
    from .ptfile import PtFileStore

    return PtFileStore(path)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _values_differ(stored: Any, wanted: Any, *, key: str) -> bool:
    if stored is None or wanted is None:
        return False
    if key in ("epsilon", "delta"):
        a, b = _as_float(stored), _as_float(wanted)
        if a is None or b is None:
            return stored != wanted
        return abs(a - b) > 1e-9 * (1.0 + abs(a))
    if isinstance(stored, (bool, np.bool_)) or isinstance(wanted, (bool, np.bool_)):
        return bool(stored) != bool(wanted)
    if isinstance(stored, (int, np.integer)) or isinstance(wanted, (int, np.integer)):
        try:
            return int(stored) != int(wanted)
        except (TypeError, ValueError):
            return stored != wanted
    return stored != wanted


def needs_rebuild(
    meta: Mapping[str, Any],
    X: Any,
    config_kwargs: Mapping[str, Any],
) -> bool:
    """Return True when ``meta`` disagrees with ``X`` / build config.

    Detects fingerprint mismatch and invalidation-key mismatches
    (metric, ε/δ, k, L, seed, dedup, pyramid depth).
    """
    if meta.get("fingerprint") is not None:
        if not verify_fingerprint(X, meta, full=False):
            return True
    aliases = {
        "metric": "metric_name",
        "metric_name": "metric_name",
        "epsilon": "epsilon",
        "delta": "delta",
        "n_neighbors": "n_neighbors",
        "k": "n_neighbors",
        "n_landmarks": "n_landmarks",
        "L": "n_landmarks",
        "seed": "seed",
        "dedup": "dedup",
        "n_pyramid_levels": "n_pyramid_levels",
        "pyramid_scales": "n_pyramid_levels",
    }
    for cfg_key, value in config_kwargs.items():
        meta_key = aliases.get(cfg_key, cfg_key)
        if meta_key not in INVALIDATION_KEYS:
            continue
        if meta_key not in meta:
            continue
        if _values_differ(meta[meta_key], value, key=meta_key):
            return True
    return False


__all__ = [
    "GraphStore",
    "needs_rebuild",
    "open_graph_store",
    "select_backend",
]

"""Load feature / color arrays from disk (researcher-style ingest)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_array(path: PathLike) -> np.ndarray:
    """Load a 1-D or 2-D array from ``.npy``, ``.npz``, or CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"array file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as z:
            if "arr_0" in z.files:
                arr = z["arr_0"]
            elif len(z.files) == 1:
                arr = z[z.files[0]]
            elif "X" in z.files:
                arr = z["X"]
            else:
                raise ValueError(
                    f"{path}: npz must contain 'arr_0', 'X', or a single array; "
                    f"got {z.files}"
                )
    elif suffix in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(
            f"unsupported array format {suffix!r} for {path} "
            "(use .npy, .npz, .csv, or .txt)"
        )
    return np.asarray(arr)


def load_features(path: PathLike) -> np.ndarray:
    """Load ``X`` as float32 with shape ``(N, D)``."""
    X = load_array(path)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (N, D); got shape {X.shape} from {path}")
    if X.shape[0] < 3:
        raise ValueError(f"need at least 3 rows; got N={X.shape[0]} from {path}")
    return np.asarray(X, dtype=np.float32)


def load_color(path: PathLike, n: int) -> np.ndarray:
    """Load a length-``n`` color / label vector."""
    y = load_array(path)
    y = np.asarray(y).reshape(-1)
    if y.shape[0] != n:
        raise ValueError(
            f"color length {y.shape[0]} != N={n} (from features)"
        )
    return y


def ingest(
    x_path: PathLike,
    color_path: Optional[PathLike] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(X, color)`` with validated shapes."""
    X = load_features(x_path)
    color = load_color(color_path, len(X)) if color_path is not None else None
    return X, color


def default_run_name(x_path: PathLike, name: Optional[str] = None) -> str:
    """Tag for the output tree: ``--name`` or stem of ``--X``."""
    if name:
        return str(name).strip().replace(" ", "_")
    stem = Path(x_path).stem
    # strip common suffixes like _X / _features
    for suffix in ("_X", "_x", "_features", "_feats"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "run"

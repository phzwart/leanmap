"""Staged Zarr cache for graph construction (landmarks → ε-net → knn)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..graph import Representatives, tensor_fingerprint
from ..utils import get_logger

STAGES_META = "meta.json"


def _zarr():
    try:
        import zarr
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "graph stages require zarr; pip install 'leanmap[hpc]' or zarr"
        ) from e
    return zarr


def stages_root(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_path(root: Path) -> Path:
    return root / STAGES_META


def write_meta(root: Path, meta: Dict[str, Any]) -> None:
    stages_root(root)
    _meta_path(root).write_text(json.dumps(meta, indent=2, default=str) + "\n")


def read_meta(root: Path) -> Optional[Dict[str, Any]]:
    p = _meta_path(root)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fingerprint_matches(root: Path, X: torch.Tensor) -> bool:
    meta = read_meta(root)
    if not meta or "fingerprint" not in meta:
        return False
    try:
        from ..graph import check_tensor_fingerprint

        check_tensor_fingerprint(X, meta["fingerprint"])
        return True
    except ValueError:
        return False


def init_meta(root: Path, X: torch.Tensor, **extra: Any) -> Dict[str, Any]:
    meta = {
        "fingerprint": tensor_fingerprint(X),
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        **extra,
    }
    write_meta(root, meta)
    return meta


def _save_array(path: Path, data: np.ndarray, chunks: Any = True) -> None:
    zarr = _zarr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    arr = np.asarray(data)
    z = zarr.open(
        str(path),
        mode="w",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks,
    )
    z[:] = arr


def _open_array(path: Path, mode: str = "r"):
    zarr = _zarr()
    return zarr.open(str(path), mode=mode)


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(_open_array(path, mode="r"))


def save_landmarks(
    root: Path,
    M: torch.Tensor,
    assign_top1: torch.Tensor,
    assign_topc: torch.Tensor,
) -> None:
    base = stages_root(root) / "landmarks"
    _save_array(base / "M", M.detach().cpu().numpy().astype(np.float32))
    _save_array(base / "assign_top1", assign_top1.detach().cpu().numpy().astype(np.int64))
    _save_array(base / "assign_topc", assign_topc.detach().cpu().numpy().astype(np.int64))
    get_logger().info("stages: wrote landmarks L=%d", int(M.shape[0]))


def load_landmarks(
    root: Path,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    base = Path(root) / "landmarks"
    if not (base / "M").exists():
        return None
    M = torch.as_tensor(_load_array(base / "M"), dtype=torch.float32)
    top1 = torch.as_tensor(_load_array(base / "assign_top1"), dtype=torch.int64)
    topc = torch.as_tensor(_load_array(base / "assign_topc"), dtype=torch.int64)
    get_logger().info("stages: loaded landmarks L=%d", int(M.shape[0]))
    return M, top1, topc


def save_enet(root: Path, reps: Representatives, epsilon: float) -> None:
    base = stages_root(root) / "enet"
    _save_array(base / "rep_idx", reps.rep_idx.detach().cpu().numpy().astype(np.int64))
    _save_array(base / "member_of", reps.member_of.detach().cpu().numpy().astype(np.int64))
    _save_array(base / "weight", reps.weight.detach().cpu().numpy().astype(np.float32))
    _save_array(base / "offsets", reps.offsets.detach().cpu().numpy().astype(np.int64))
    _save_array(base / "values", reps.values.detach().cpu().numpy().astype(np.int64))
    (base / "epsilon.txt").write_text(repr(float(epsilon)) + "\n")
    get_logger().info("stages: wrote ε-net R=%d", int(reps.rep_idx.shape[0]))


def load_enet(root: Path) -> Optional[Tuple[Representatives, float]]:
    base = Path(root) / "enet"
    if not (base / "rep_idx").exists():
        return None
    reps = Representatives(
        rep_idx=torch.as_tensor(_load_array(base / "rep_idx"), dtype=torch.int64),
        member_of=torch.as_tensor(_load_array(base / "member_of"), dtype=torch.int64),
        weight=torch.as_tensor(_load_array(base / "weight"), dtype=torch.float32),
        offsets=torch.as_tensor(_load_array(base / "offsets"), dtype=torch.int64),
        values=torch.as_tensor(_load_array(base / "values"), dtype=torch.int64),
    )
    eps = float((base / "epsilon.txt").read_text().strip())
    get_logger().info("stages: loaded ε-net R=%d", int(reps.rep_idx.shape[0]))
    return reps, eps


class _KnnStore:
    """Thin wrapper so callers can write ``store.idx[i] = ...``."""

    def __init__(self, idx, dist):
        self.idx = idx
        self.dist = dist


def create_knn_store(root: Path, R: int, k: int, row_chunk: int = 8192) -> _KnnStore:
    zarr = _zarr()
    base = stages_root(root) / "knn"
    base.mkdir(parents=True, exist_ok=True)
    chunks = (min(row_chunk, R), k)
    for name, dtype, fill in (
        ("knn_idx", "i8", -1),
        ("knn_dist", "f4", np.inf),
    ):
        path = base / name
        if path.exists():
            import shutil

            shutil.rmtree(path) if path.is_dir() else path.unlink()
        zarr.open(
            str(path),
            mode="w",
            shape=(R, k),
            dtype=dtype,
            chunks=chunks,
            fill_value=fill,
        )
    (base / "complete").unlink(missing_ok=True)
    idx = _open_array(base / "knn_idx", mode="r+")
    dist = _open_array(base / "knn_dist", mode="r+")
    return _KnnStore(idx, dist)


# Back-compat names used by graph.py spill helper
ARR_KNN_IDX = "idx"
ARR_KNN_DIST = "dist"


def knn_store_complete(root: Path) -> bool:
    return (Path(root) / "knn" / "complete").exists()


def load_knn(root: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not knn_store_complete(root):
        return None
    base = Path(root) / "knn"
    idx = torch.as_tensor(_load_array(base / "knn_idx"), dtype=torch.int64)
    dist = torch.as_tensor(_load_array(base / "knn_dist"), dtype=torch.float32)
    get_logger().info("stages: loaded knn R=%d k=%d", int(idx.shape[0]), int(idx.shape[1]))
    return idx, dist


def mark_knn_complete(root: Path) -> None:
    (Path(root) / "knn" / "complete").write_text("1\n")

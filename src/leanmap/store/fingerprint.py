"""Streamed X fingerprints and sampled-block verification."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch

from leanmap.build.pipeline import check_tensor_fingerprint, tensor_fingerprint

ArrayLike = Union[np.ndarray, torch.Tensor]

_CHUNK_BYTES = 1 << 20  # 1 MiB
_N_SAMPLE_BLOCKS = 8
_SAMPLE_BLOCK_BYTES = 4096


def _as_numpy(X: ArrayLike) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return np.ascontiguousarray(X.detach().cpu().numpy())
    return np.ascontiguousarray(np.asarray(X))


def _hasher():
    try:
        import xxhash  # type: ignore

        return xxhash.xxh64(), "xxh64"
    except ImportError:
        return hashlib.sha256(), "sha256"


def _update_streamed(hasher: Any, flat: np.ndarray) -> None:
    if flat.size == 0:
        return
    itemsize = int(flat.itemsize)
    chunk_elems = max(1, _CHUNK_BYTES // max(itemsize, 1))
    for start in range(0, int(flat.size), chunk_elems):
        hasher.update(flat[start : start + chunk_elems].tobytes())


def _block_digest(flat: np.ndarray, offset: int, nbytes: int, algo: str) -> str:
    if flat.size == 0 or nbytes <= 0:
        raw = b""
    else:
        itemsize = int(flat.itemsize)
        start = int(offset) // itemsize
        n_elem = max(1, int(nbytes) // max(itemsize, 1))
        end = min(int(flat.size), start + n_elem)
        start = min(start, int(flat.size))
        raw = flat[start:end].tobytes()
    if algo == "xxh64":
        import xxhash  # type: ignore

        return xxhash.xxh64(raw).hexdigest()
    return hashlib.sha256(raw).hexdigest()


def _sample_offsets(nbytes: int, n_blocks: int, block_bytes: int) -> list[int]:
    if nbytes <= 0:
        return [0]
    usable = max(0, nbytes - block_bytes)
    if n_blocks <= 1 or usable == 0:
        return [0]
    return [int(round(i * usable / (n_blocks - 1))) for i in range(n_blocks)]


def fingerprint_array(X: ArrayLike) -> Dict[str, Any]:
    """Streamed content hash of ``X`` (xxhash if available, else SHA-256)."""
    arr = _as_numpy(X)
    flat = arr.reshape(-1)
    hasher, algo = _hasher()
    _update_streamed(hasher, flat)
    nbytes = int(arr.nbytes)
    block_bytes = min(_SAMPLE_BLOCK_BYTES, max(nbytes, 1))
    offsets = _sample_offsets(nbytes, _N_SAMPLE_BLOCKS, block_bytes)
    samples = [
        {
            "offset": int(off),
            "nbytes": int(block_bytes),
            "digest": _block_digest(flat, off, block_bytes, algo),
        }
        for off in offsets
    ]
    return {
        "algo": algo,
        "digest": hasher.hexdigest(),
        "shape": [int(s) for s in arr.shape],
        "dtype": str(arr.dtype),
        "nbytes": nbytes,
        "samples": samples,
    }


def _legacy_ok(X: ArrayLike, fingerprint: Mapping[str, Any]) -> bool:
    if "mean" not in fingerprint or "head" not in fingerprint:
        return False
    try:
        t = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X))
        check_tensor_fingerprint(t, dict(fingerprint))
        return True
    except ValueError:
        return False


def verify_fingerprint(
    X: ArrayLike,
    meta: Mapping[str, Any],
    *,
    full: bool = False,
) -> bool:
    """Verify ``X`` against a store fingerprint.

    When ``full=False``, only shape/dtype and sampled blocks are checked.
    When ``full=True``, the streamed digest must match.
    Legacy mean/head/tail fingerprints are accepted for compatibility.
    """
    fp = meta.get("fingerprint", meta)
    if not isinstance(fp, Mapping):
        return False
    if "digest" not in fp:
        return _legacy_ok(X, fp)

    arr = _as_numpy(X)
    want_shape = [int(s) for s in fp.get("shape", [])]
    if list(arr.shape) != want_shape:
        return False
    if "dtype" in fp and str(arr.dtype) != str(fp["dtype"]):
        return False
    if "nbytes" in fp and int(arr.nbytes) != int(fp["nbytes"]):
        return False

    algo = str(fp.get("algo", "sha256"))
    flat = arr.reshape(-1)

    if full:
        hasher, got_algo = _hasher()
        # Prefer the algorithm recorded in the fingerprint when possible.
        if algo == "sha256":
            hasher = hashlib.sha256()
            got_algo = "sha256"
        elif algo == "xxh64":
            try:
                import xxhash  # type: ignore

                hasher = xxhash.xxh64()
                got_algo = "xxh64"
            except ImportError:
                return False
        _update_streamed(hasher, flat)
        return hasher.hexdigest() == str(fp["digest"]) and got_algo == algo

    samples = fp.get("samples") or []
    if not samples:
        # No sample blocks stored — fall back to a cheap full check of digest
        # length / shape already done; require full verify for digest.
        return True
    for sample in samples:
        off = int(sample["offset"])
        nbytes = int(sample["nbytes"])
        want = str(sample["digest"])
        if _block_digest(flat, off, nbytes, algo) != want:
            return False
    return True


__all__ = [
    "check_tensor_fingerprint",
    "fingerprint_array",
    "tensor_fingerprint",
    "verify_fingerprint",
]

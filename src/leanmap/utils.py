"""Seeding, chunking helpers, and logging for PLANE."""

from __future__ import annotations

import logging
import random
from typing import Callable, Iterator, Optional, Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")

_LOGGER = logging.getLogger("leanmap")


def get_logger() -> logging.Logger:
    """Return the package logger (`leanmap`)."""
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s leanmap: %(message)s")
        )
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = True
    return _LOGGER


def rss_mb() -> float:
    """Current process resident set size in MiB, or ``-1`` if unavailable."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        import sys

        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:  # noqa: BLE001
        return -1.0


class BuildProgress:
    """Thread-safe phase string for :class:`Heartbeat` detail callbacks.

    Long graph-build stages update ``phase`` / ``detail`` so periodic heartbeats
    report *where* work is, not only that the process is alive.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._phase = ""
        self._detail = ""

    def set(self, phase: str, detail: str = "") -> None:
        with self._lock:
            self._phase = str(phase or "")
            self._detail = str(detail or "")

    def clear(self) -> None:
        with self._lock:
            self._phase = ""
            self._detail = ""

    def format(self) -> str:
        with self._lock:
            if not self._phase:
                return self._detail
            if not self._detail:
                return f"[{self._phase}]"
            return f"[{self._phase}] {self._detail}"


#: Process-wide progress shared by graph-build stages and CLI heartbeats.
BUILD_PROGRESS = BuildProgress()


class Heartbeat:
    """Background ``I'm still running`` logger for long graph-build phases.

    Starts a daemon thread that emits an INFO line every ``interval`` seconds
    with elapsed wall time and RSS. Optional ``detail`` callable may return an
    extra suffix (e.g. ``\"knn 12000/507390\"``).
    """

    def __init__(
        self,
        label: str,
        *,
        interval: float = 10.0,
        detail: Optional[Callable[[], str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        import threading
        import time

        self.label = label
        self.interval = float(interval)
        self.detail = detail
        self.log = logger or get_logger()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=f"leanmap-heartbeat-{label}", daemon=True
        )

    def _run(self) -> None:
        import time

        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self._t0
            extra = ""
            if self.detail is not None:
                try:
                    msg = self.detail()
                    if msg:
                        extra = f" {msg}"
                except Exception:  # noqa: BLE001
                    extra = " (detail error)"
            self.log.info(
                "heartbeat: %s still running  elapsed=%.0fs  RSS≈%.0f MiB%s",
                self.label,
                elapsed,
                rss_mb(),
                extra,
            )

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval))


def seed_everything(seed: int = 0) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` for reproducibility.

    Parameters
    ----------
    seed : int
        Global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        # warn_only: MPS (and some CUDA kernels) lack deterministic impls;
        # hard True would crash training on those devices (§11: "where feasible").
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick CUDA → MPS → CPU unless ``device`` is given explicitly.

    On MPS, enables ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so ops missing on Metal
    (notably spectral-norm power iteration ``aten::vdot``) fall back to CPU
    instead of raising.

    Parameters
    ----------
    device : str | None
        Explicit device string, or None for auto.

    Returns
    -------
    torch.device
    """
    import os

    if device is not None:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if dev.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return dev


def as_float32_tensor(
    x: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert array-like input to a contiguous float32 tensor.

    Parameters
    ----------
    x : (...,) array or tensor
    device : torch.device | None

    Returns
    -------
    torch.Tensor, float32
    """
    if isinstance(x, torch.Tensor):
        t = x.detach() if not x.requires_grad else x
        t = t.to(dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def chunk_ranges(n: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` half-open ranges covering ``0..n``.

    Parameters
    ----------
    n : int
        Total length.
    chunk : int
        Chunk size.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    for start in range(0, n, chunk):
        yield start, min(n, start + chunk)


def ensure_2d_float32(X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate and return a contiguous ``(N, D)`` float32 numpy array.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    np.ndarray, shape (N, D), float32
    """
    if isinstance(X, torch.Tensor):
        arr = X.detach().cpu().numpy()
    else:
        arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)

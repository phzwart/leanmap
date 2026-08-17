"""Collective transports for multi-worker bunches builds.

Default multi-node path is :class:`FileStoreTransport` (shared directory;
no mpi4py). Optional :class:`TorchDistTransport` and :class:`MPITransport`
adapters share the same :class:`BunchTransport` API.
"""
from __future__ import annotations

import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence

__all__ = [
    "BunchTransport",
    "FileStoreTransport",
    "TorchDistTransport",
    "MPITransport",
    "make_transport",
]


class BunchTransport(ABC):
    """Minimal collectives needed by bunches graph construction."""

    @property
    @abstractmethod
    def rank(self) -> int: ...

    @property
    @abstractmethod
    def world_size(self) -> int: ...

    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    def broadcast_obj(self, obj: Any, root: int = 0) -> Any: ...

    @abstractmethod
    def allgather_obj(self, obj: Any) -> List[Any]: ...

    @abstractmethod
    def gather_obj(self, obj: Any, root: int = 0) -> Optional[List[Any]]: ...


class FileStoreTransport(BunchTransport):
    """Coordinate workers via a shared filesystem directory.

    Each collective writes under ``root/phase_<name>/`` with per-rank payloads
    and a root-assembled result. Suitable for Slurm/K8s array jobs that share
    NFS/Lustre (or a local temp dir for in-process tests).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        poll_s: float = 0.05,
        timeout_s: float = 600.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._rank = int(os.environ["RANK"]) if rank is None else int(rank)
        self._world = (
            int(os.environ["WORLD_SIZE"]) if world_size is None else int(world_size)
        )
        if self._world < 1:
            raise ValueError("world_size must be >= 1")
        if not (0 <= self._rank < self._world):
            raise ValueError(f"rank {self._rank} out of range for world_size={self._world}")
        self.poll_s = float(poll_s)
        self.timeout_s = float(timeout_s)
        self._seq = 0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world

    def _phase_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _wait_file(self, path: Path) -> None:
        deadline = time.monotonic() + self.timeout_s
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"FileStoreTransport timed out waiting for {path}")
            time.sleep(self.poll_s)

    def _wait_files(self, paths: Sequence[Path]) -> None:
        deadline = time.monotonic() + self.timeout_s
        pending = set(Path(p) for p in paths)
        while pending:
            pending = {p for p in pending if not p.exists()}
            if not pending:
                return
            if time.monotonic() > deadline:
                missing = ", ".join(str(p) for p in sorted(pending))
                raise TimeoutError(
                    f"FileStoreTransport timed out waiting for: {missing}"
                )
            time.sleep(self.poll_s)

    def _atomic_write(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{self._rank}")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _read(self, path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _next_name(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}_{self._seq:05d}"

    def barrier(self) -> None:
        name = self._next_name("barrier")
        d = self._phase_dir(name)
        marker = d / f"rank_{self._rank}.done"
        marker.write_text("1\n")
        self._wait_files([d / f"rank_{r}.done" for r in range(self._world)])

    def broadcast_obj(self, obj: Any, root: int = 0) -> Any:
        name = self._next_name("bcast")
        d = self._phase_dir(name)
        payload = d / "payload.pkl"
        if self._rank == int(root):
            self._atomic_write(payload, obj)
        self._wait_file(payload)
        return self._read(payload)

    def allgather_obj(self, obj: Any) -> List[Any]:
        name = self._next_name("allgather")
        d = self._phase_dir(name)
        mine = d / f"rank_{self._rank}.pkl"
        self._atomic_write(mine, obj)
        paths = [d / f"rank_{r}.pkl" for r in range(self._world)]
        self._wait_files(paths)
        return [self._read(p) for p in paths]

    def gather_obj(self, obj: Any, root: int = 0) -> Optional[List[Any]]:
        name = self._next_name("gather")
        d = self._phase_dir(name)
        mine = d / f"rank_{self._rank}.pkl"
        self._atomic_write(mine, obj)
        if self._rank != int(root):
            # Non-root still waits until root has read everyone (ack marker).
            ack = d / "root_ack.done"
            self._wait_file(ack)
            return None
        paths = [d / f"rank_{r}.pkl" for r in range(self._world)]
        self._wait_files(paths)
        out = [self._read(p) for p in paths]
        (d / "root_ack.done").write_text("1\n")
        return out


class TorchDistTransport(BunchTransport):
    """``torch.distributed`` object collectives (gloo/nccl already initialized)."""

    def __init__(self) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "TorchDistTransport requires an initialized torch.distributed "
                "process group (e.g. launch with torchrun)"
            )
        self._dist = dist

    @property
    def rank(self) -> int:
        return int(self._dist.get_rank())

    @property
    def world_size(self) -> int:
        return int(self._dist.get_world_size())

    def barrier(self) -> None:
        self._dist.barrier()

    def broadcast_obj(self, obj: Any, root: int = 0) -> Any:
        obj_list = [obj if self.rank == int(root) else None]
        self._dist.broadcast_object_list(obj_list, src=int(root))
        return obj_list[0]

    def allgather_obj(self, obj: Any) -> List[Any]:
        out: List[Any] = [None] * self.world_size
        self._dist.all_gather_object(out, obj)
        return out

    def gather_obj(self, obj: Any, root: int = 0) -> Optional[List[Any]]:
        if self.rank == int(root):
            out: List[Any] = [None] * self.world_size
            self._dist.gather_object(obj, object_gather_list=out, dst=int(root))
            return out
        self._dist.gather_object(obj, object_gather_list=None, dst=int(root))
        return None


class MPITransport(BunchTransport):
    """Optional ``mpi4py`` adapter (``leanmap[hpc]``)."""

    def __init__(self) -> None:
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mpi4py is required for --bunch-partition mpi; "
                "install with: pip install leanmap[hpc]"
            ) from exc
        self._comm = MPI.COMM_WORLD

    @property
    def rank(self) -> int:
        return int(self._comm.Get_rank())

    @property
    def world_size(self) -> int:
        return int(self._comm.Get_size())

    def barrier(self) -> None:
        self._comm.Barrier()

    def broadcast_obj(self, obj: Any, root: int = 0) -> Any:
        return self._comm.bcast(obj, root=int(root))

    def allgather_obj(self, obj: Any) -> List[Any]:
        return list(self._comm.allgather(obj))

    def gather_obj(self, obj: Any, root: int = 0) -> Optional[List[Any]]:
        gathered = self._comm.gather(obj, root=int(root))
        if self.rank != int(root):
            return None
        return list(gathered)


def make_transport(
    kind: str,
    *,
    stages_dir: Optional[Path | str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> BunchTransport:
    """Factory: ``fs`` | ``ddp`` | ``mpi`` | ``local`` (single-process FileStore)."""
    kind = str(kind).lower()
    if kind in ("local", "none", "off"):
        return FileStoreTransport(
            stages_dir or Path(".") / "_bunch_local",
            rank=0,
            world_size=1,
        )
    if kind in ("fs", "file", "filestore"):
        if stages_dir is None:
            raise ValueError("--bunch-partition fs requires a shared --stages directory")
        return FileStoreTransport(stages_dir, rank=rank, world_size=world_size)
    if kind in ("ddp", "torch", "torchdist"):
        return TorchDistTransport()
    if kind == "mpi":
        return MPITransport()
    raise ValueError(f"unknown bunch transport kind: {kind!r}")

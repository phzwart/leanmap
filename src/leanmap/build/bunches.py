"""Distributed landmark-bunch graph build (optional ``leanmap[hpc]``).

Entry point::

    from leanmap.build.bunches import build_graph_bunches

Core build paths (:mod:`leanmap.build.pipeline`) must **not** import this
module at load time. ``mpi4py`` is imported lazily only when an MPI-backed
helper runs; top-level ``import leanmap.build.bunches`` works without it.

Pipeline (design §4.3 / v2 §9): probe → reconcile landmarks → partition
bunches → margin halo → owned nets → kNN fill → distributed union-find →
stitch. At MPI world size 1, :func:`build_graph_bunches` delegates to
:func:`~leanmap.build.pipeline.build_graph` for bit-compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from leanmap.distance import DistanceFn
from leanmap.utils import get_logger

__all__ = [
    "probe_shards",
    "reconcile_landmarks",
    "partition_bunches",
    "margin_halo",
    "owned_net",
    "uf_link",
    "distributed_union_find",
    "fill_knn_rows",
    "stitch_graph",
    "build_graph_bunches",
    "halo_fraction",
    "cut_mass",
    "knn_completeness_audit",
    "mpi_world_size",
]


# ---------------------------------------------------------------------------
# Lazy MPI
# ---------------------------------------------------------------------------


def _require_mpi4py():
    """Import ``mpi4py.MPI`` or raise with an install hint."""
    try:
        from mpi4py import MPI  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when mpi4py absent
        raise ImportError(
            "mpi4py is required for distributed leanmap builds; "
            "install with: pip install leanmap[hpc]"
        ) from exc
    return MPI


def mpi_world_size() -> int:
    """Return MPI world size, or ``1`` when MPI is unused / unavailable.

    Does **not** require ``mpi4py`` when ``WORLD_SIZE`` is unset or ``1``.
    """
    import os

    ws_env = int(os.environ.get("WORLD_SIZE", "1"))
    if ws_env <= 1:
        return 1
    MPI = _require_mpi4py()
    return int(MPI.COMM_WORLD.Get_size())


def mpi_rank() -> int:
    """Return MPI rank, or ``0`` when world size is 1."""
    if mpi_world_size() <= 1:
        return 0
    MPI = _require_mpi4py()
    return int(MPI.COMM_WORLD.Get_rank())


# ---------------------------------------------------------------------------
# Probe / reconcile / partition / halo / owned net
# ---------------------------------------------------------------------------


def probe_shards(
    X: torch.Tensor,
    n_probe: int,
    seed: int = 0,
) -> torch.Tensor:
    """Sample ``n_probe`` row indices from ``X`` for a landmark probe.

    Embarrassingly parallel across shards: each rank calls this on its local
    shard with a rank-offset seed. Returns a 1-D ``int64`` index tensor.
    """
    n = int(X.shape[0])
    if n_probe <= 0:
        raise ValueError("n_probe must be positive")
    n_probe = min(int(n_probe), n)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g)[:n_probe].to(dtype=torch.int64)


def reconcile_landmarks(
    local_landmarks: torch.Tensor,
    *,
    world_size: int = 1,
    max_landmarks: Optional[int] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Reconcile per-rank landmark pools into a global set.

    At ``world_size == 1`` this is identity (optionally truncated). Multi-rank
    gather / dedupe requires MPI and is structured for later wiring.
    """
    local = torch.as_tensor(local_landmarks)
    if int(world_size) <= 1:
        if max_landmarks is not None and local.shape[0] > int(max_landmarks):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            pick = torch.randperm(local.shape[0], generator=g)[: int(max_landmarks)]
            return local[pick].contiguous()
        return local.contiguous()

    MPI = _require_mpi4py()
    comm = MPI.COMM_WORLD
    gathered = comm.allgather(local.detach().cpu().numpy())
    if mpi_rank() != 0:
        # Non-root still needs the broadcast result below.
        pass
    pool = np.concatenate([np.asarray(g) for g in gathered], axis=0)
    # Row-wise unique (exact duplicate landmarks).
    if pool.ndim == 1:
        uniq = np.unique(pool)
    else:
        # Structured view for unique rows.
        dtype = np.dtype((np.void, pool.dtype.itemsize * pool.shape[1]))
        uniq = np.unique(pool.view(dtype)).view(pool.dtype).reshape(-1, pool.shape[1])
    if max_landmarks is not None and uniq.shape[0] > int(max_landmarks):
        rng = np.random.default_rng(int(seed))
        choose = rng.choice(uniq.shape[0], size=int(max_landmarks), replace=False)
        uniq = uniq[np.sort(choose)]
    out = torch.as_tensor(uniq, dtype=local.dtype)
    obj = comm.bcast(out.numpy() if mpi_rank() == 0 else None, root=0)
    return torch.as_tensor(obj, dtype=local.dtype).contiguous()


def partition_bunches(
    landmark_idx: torch.Tensor,
    n_bunches: int,
) -> torch.Tensor:
    """Assign each landmark to a bunch id in ``[0, n_bunches)``.

    Contiguous blocks balanced by landmark count (mass-aware balancing is a
    later refinement). Returns ``(L,)`` int64 bunch ids aligned with
    ``landmark_idx``.
    """
    L = int(torch.as_tensor(landmark_idx).shape[0])
    n_bunches = max(1, int(n_bunches))
    if L == 0:
        return torch.empty(0, dtype=torch.int64)
    # Contiguous blocks: landmark i -> floor(i * n_bunches / L)
    ids = (torch.arange(L, dtype=torch.int64) * n_bunches) // max(L, 1)
    return ids.clamp(max=n_bunches - 1)


def margin_halo(
    assign: torch.Tensor,
    margin: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute owned vs halo membership from a bunch (or top-c) assignment.

    Parameters
    ----------
    assign :
        ``(N,)`` primary bunch id per row, **or** ``(N, c)`` top-c bunch /
        landmark ids. For 2-D input, a row is in the halo of bunch ``b`` when
        any of its top-c entries equals ``b`` while its primary (column 0) is
        not ``b``.
    margin :
        For 1-D ``assign``, expand ownership by ``±margin`` bunch ids (circular
        over the observed id range). Ignored for the 2-D top-c path (1-ring
        shortlist intersection is the margin).

    Returns
    -------
    owned_mask, halo_mask : ``(N,)`` bool tensors
        ``owned_mask`` is primary ownership of the *minimum* bunch id present
        (caller typically slices by rank-owned bunches first). When ``assign``
        is 1-D, ``owned_mask`` marks rows whose id is in the closed interval
        around the median bunch after margin expansion is applied relative to
        each row's own id — more usefully, we return:

        - ``owned_mask``: True where the row's primary id equals its value
          (always True for valid 1-D ids); use together with a caller-supplied
          owned-bunch set via boolean indexing on ``assign``.

        For a practical single-process API we return masks relative to the
        full assignment: ``halo_mask[i]`` is True when row ``i`` is within
        ``margin`` of a *different* bunch than its primary (1-D) or when
        top-c intersects a foreign bunch (2-D).
    """
    a = torch.as_tensor(assign)
    margin = max(0, int(margin))

    if a.ndim == 1:
        n = a.shape[0]
        owned = torch.ones(n, dtype=torch.bool, device=a.device)
        ids = a.to(dtype=torch.int64)
        if margin <= 0 or n == 0:
            return owned, torch.zeros(n, dtype=torch.bool, device=a.device)
        occupied = {int(x) for x in torch.unique(ids).tolist()}
        if len(occupied) <= 1:
            return owned, torch.zeros(n, dtype=torch.bool, device=a.device)
        # Rows whose bunch shares a ±margin boundary with another occupied bunch.
        boundary_bunches = set()
        for b in occupied:
            for d in range(1, margin + 1):
                if (b - d) in occupied or (b + d) in occupied:
                    boundary_bunches.add(b)
                    break
        halo = torch.tensor(
            [int(ids[i].item()) in boundary_bunches for i in range(n)],
            dtype=torch.bool,
            device=a.device,
        )
        return owned, halo

    if a.ndim != 2:
        raise ValueError("assign must be 1-D or 2-D")
    primary = a[:, 0].to(dtype=torch.int64)
    n = primary.shape[0]
    owned = torch.ones(n, dtype=torch.bool, device=a.device)
    # Halo: top-c contains a foreign bunch id.
    foreign = (a.to(dtype=torch.int64) != primary.unsqueeze(1)) & (a >= 0)
    halo = foreign.any(dim=1)
    return owned, halo


def owned_net(
    X: torch.Tensor,
    owned_idx: torch.Tensor,
    radius: float,
    dist_fn: Optional[DistanceFn] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy radius-net on owned row indices (single-process / per-bunch).

    Returns
    -------
    rep_idx : ``(R_local,)`` int64 indices into ``X``
    member_of : ``(len(owned_idx),)`` int64 local cell id per owned row
    """
    from leanmap.distance import EuclideanDistance

    owned_idx = torch.as_tensor(owned_idx, dtype=torch.int64)
    if owned_idx.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    if dist_fn is None:
        dist_fn = EuclideanDistance()

    rng = np.random.default_rng(int(seed))
    # Local import to avoid cycles at module import of pipeline helpers.
    from leanmap.build.pipeline import _epsilon_net_bucket

    reps, member_map = _epsilon_net_bucket(
        X, owned_idx, dist_fn, float(radius), rng
    )
    rep_idx = torch.as_tensor(reps, dtype=torch.int64)
    member_of = torch.as_tensor(
        [member_map[int(i)] for i in owned_idx.tolist()],
        dtype=torch.int64,
    )
    return rep_idx, member_of


# ---------------------------------------------------------------------------
# Deterministic union-find
# ---------------------------------------------------------------------------


def uf_find(parent: List[int], a: int) -> int:
    """Path-compressed find."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def uf_link(parent: List[int], a: int, b: int) -> None:
    """Union two elements, always linking the **higher** root onto the lower.

    Tie-break is total on root indices, so the same undirected edge set yields
    identical roots regardless of union order (bit-identical forests).
    """
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra == rb:
        return
    if rb < ra:
        ra, rb = rb, ra
    parent[rb] = ra


def distributed_union_find(
    parents: Union[torch.Tensor, np.ndarray, Sequence[int], Sequence[Sequence[int]]],
) -> torch.Tensor:
    """Deterministic union-find roots from parent pointer(s).

    Parameters
    ----------
    parents :
        - 1-D length-``n`` parent pointers, or
        - a sequence of 1-D parent arrays (one per rank / shard). Every
          ``(i, parents_r[i])`` edge is united with :func:`uf_link`
          (higher root → lower root). Merge order does not affect roots.

    Returns
    -------
    roots : ``(n,)`` int64
        Root id for each element after path compression.
    """
    # Detect sequence-of-arrays vs single parent vector.
    if isinstance(parents, (list, tuple)) and parents and not isinstance(
        parents[0], (int, np.integer)
    ):
        arrays = [torch.as_tensor(p, dtype=torch.int64).reshape(-1) for p in parents]
        n = int(arrays[0].shape[0])
        if any(int(a.shape[0]) != n for a in arrays):
            raise ValueError("all parent arrays must share length n")
        parent = list(range(n))
        for arr in arrays:
            for i in range(n):
                uf_link(parent, i, int(arr[i].item()))
    else:
        arr = torch.as_tensor(parents, dtype=torch.int64).reshape(-1)
        n = int(arr.shape[0])
        parent = list(range(n))
        for i in range(n):
            uf_link(parent, i, int(arr[i].item()))

    roots = [uf_find(parent, i) for i in range(n)]
    return torch.as_tensor(roots, dtype=torch.int64)


# ---------------------------------------------------------------------------
# kNN fill (absorbs former scripts/mpi_knn_fill.py sketch)
# ---------------------------------------------------------------------------


def fill_knn_rows(
    X: torch.Tensor,
    row_start: int,
    row_end: int,
    *,
    k: int,
    dist_fn: Optional[DistanceFn] = None,
    X_corpus: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fill exact kNN for query rows ``[row_start, row_end)``.

    Each MPI rank owns a contiguous query-row range against a shared corpus
    (default: all of ``X``). Self-matches are excluded when the corpus is
    ``X`` and the query index falls in range.

    Returns
    -------
    knn_idx : ``(row_end - row_start, k)`` int64 into the corpus
    knn_dist : ``(row_end - row_start, k)`` float32
    """
    from leanmap.distance import EuclideanDistance

    if dist_fn is None:
        dist_fn = EuclideanDistance()
    corpus = X if X_corpus is None else X_corpus
    n = int(X.shape[0])
    row_start = int(row_start)
    row_end = int(row_end)
    if not (0 <= row_start <= row_end <= n):
        raise ValueError(f"invalid row range [{row_start}, {row_end}) for N={n}")
    m = row_end - row_start
    R = int(corpus.shape[0])
    k = min(int(k), max(R - 1, 1))
    knn_idx = torch.empty(m, k, dtype=torch.int64, device=X.device)
    knn_dist = torch.empty(m, k, dtype=torch.float32, device=X.device)
    same = X_corpus is None
    for local_i, global_i in enumerate(range(row_start, row_end)):
        d = dist_fn(X[global_i : global_i + 1], corpus)[0]
        if same and 0 <= global_i < R:
            d = d.clone()
            d[global_i] = float("inf")
        vals, idx = torch.topk(d, k=k, largest=False)
        knn_idx[local_i] = idx
        knn_dist[local_i] = vals
    return knn_idx, knn_dist


# ---------------------------------------------------------------------------
# Stitch + build entry
# ---------------------------------------------------------------------------


def stitch_graph(
    parts: Sequence[Dict[str, Any]],
    *,
    n_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Stitch per-bunch / per-rank partial graphs into one artefact dict.

    At a single part (world size 1), returns that part unchanged (copy of
    keys). Multi-part stitching concatenates ``rep_idx``, remaps ``member_of`` /
    edges, and stacks kNN rows when present.
    """
    if not parts:
        raise ValueError("parts must be non-empty")
    if len(parts) == 1:
        return dict(parts[0])

    rep_chunks: List[torch.Tensor] = []
    edge_chunks: List[torch.Tensor] = []
    weight_chunks: List[torch.Tensor] = []
    knn_idx_chunks: List[torch.Tensor] = []
    knn_dist_chunks: List[torch.Tensor] = []
    rep_offset = 0
    for part in parts:
        rep_idx = torch.as_tensor(part["rep_idx"], dtype=torch.int64)
        rep_chunks.append(rep_idx)
        if "edges" in part and part["edges"] is not None:
            e = torch.as_tensor(part["edges"], dtype=torch.int64).clone()
            if e.numel():
                e = e + rep_offset
            edge_chunks.append(e)
            if "weights" in part and part["weights"] is not None:
                weight_chunks.append(torch.as_tensor(part["weights"]))
        if "knn_idx" in part and part["knn_idx"] is not None:
            ki = torch.as_tensor(part["knn_idx"], dtype=torch.int64).clone()
            if ki.numel():
                # Local rep-space indices → global after offset.
                valid = ki >= 0
                ki = ki + rep_offset
                ki = torch.where(valid, ki, torch.full_like(ki, -1))
            knn_idx_chunks.append(ki)
            if "knn_dist" in part:
                knn_dist_chunks.append(torch.as_tensor(part["knn_dist"]))
        rep_offset += int(rep_idx.shape[0])

    out: Dict[str, Any] = {
        "rep_idx": torch.cat(rep_chunks, dim=0) if rep_chunks else torch.empty(0, dtype=torch.int64),
        "n_total": n_total,
    }
    if edge_chunks:
        out["edges"] = torch.cat(edge_chunks, dim=0)
    if weight_chunks:
        out["weights"] = torch.cat(weight_chunks, dim=0)
    if knn_idx_chunks:
        out["knn_idx"] = torch.cat(knn_idx_chunks, dim=0)
    if knn_dist_chunks:
        out["knn_dist"] = torch.cat(knn_dist_chunks, dim=0)
    return out


def build_graph_bunches(X: torch.Tensor, metric: Any, **kwargs: Any):
    """Distributed-aware graph build.

    When MPI world size is 1 (the default / CI path), delegates directly to
    :func:`~leanmap.build.pipeline.build_graph` so results are bit-identical
    to the standard pipeline. The multi-rank path is structured around the
    helpers in this module and is exercised with mocks / manual HPC runs.
    """
    ws = mpi_world_size()
    if ws <= 1:
        from leanmap.build.pipeline import build_graph

        return build_graph(X, metric, **kwargs)

    # Multi-rank skeleton: probe → reconcile → partition → owned work → stitch.
    # Full production wiring (halo exchange, CSR assemble) is HPC-manual.
    log = get_logger()
    log.info("build_graph_bunches: MPI world_size=%d — distributed path", ws)
    _require_mpi4py()
    seed = int(kwargs.get("seed", 0))
    n_landmarks = int(kwargs.get("n_landmarks", 256))
    n_probe = int(kwargs.get("n_probe", min(40_000, X.shape[0])))
    rank = mpi_rank()

    probe_idx = probe_shards(X, n_probe, seed=seed + rank)
    # Local landmark indices via FPS on the probe subset.
    from leanmap.landmarks import fps_init_indices

    dist_fn: DistanceFn = metric
    local_L = max(1, n_landmarks // ws + (1 if rank < (n_landmarks % ws) else 0))
    local_idx = fps_init_indices(X[probe_idx], dist_fn, local_L, seed=seed + rank)
    local_landmarks = X[probe_idx[local_idx]]
    M = reconcile_landmarks(
        local_landmarks, world_size=ws, max_landmarks=n_landmarks, seed=seed
    )
    bunch_ids = partition_bunches(torch.arange(M.shape[0]), n_bunches=ws)
    # Assignment of every row to nearest landmark (primary).
    # Chunked for memory; exact for small X.
    assign = torch.empty(X.shape[0], dtype=torch.int64, device=X.device)
    chunk = 4096
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        D = dist_fn(X[s:e], M)
        assign[s:e] = D.argmin(dim=1)
    # Map landmark id → bunch id.
    row_bunch = bunch_ids[assign]
    owned_mask = row_bunch == rank
    _, halo_mask = margin_halo(row_bunch.unsqueeze(1), margin=1)
    # Owned interior plus halo fringe (foreign shortlist intersection).
    work_idx = torch.where(owned_mask | halo_mask)[0]
    radius = kwargs.get("epsilon")
    if radius is None:
        radius = 0.0
    rep_idx, member_of = owned_net(
        X, work_idx, float(radius), dist_fn=dist_fn, seed=seed + rank
    )
    part = {
        "rep_idx": rep_idx,
        "member_of": member_of,
        "owned_idx": torch.where(owned_mask)[0],
        "halo_idx": torch.where(halo_mask & ~owned_mask)[0],
        "M": M,
        "bunch_ids": bunch_ids,
    }
    # Gather parts on root and stitch; non-roots return None placeholders.
    MPI = _require_mpi4py()
    gathered = MPI.COMM_WORLD.gather(part, root=0)
    if rank != 0:
        return None
    assert gathered is not None
    stitched = stitch_graph(gathered, n_total=int(X.shape[0]))
    stitched["halo_fraction"] = halo_fraction(
        int(owned_mask.sum().item()),
        int((halo_mask & ~owned_mask).sum().item()),
    )
    return stitched


# ---------------------------------------------------------------------------
# Freeze-time metrics
# ---------------------------------------------------------------------------


def halo_fraction(n_owned: int, n_halo: int) -> float:
    """Halo duplication rate: ``n_halo / (n_owned + n_halo)``."""
    n_owned = int(n_owned)
    n_halo = int(n_halo)
    denom = n_owned + n_halo
    if denom <= 0:
        return 0.0
    return float(n_halo) / float(denom)


def cut_mass(
    edge_weights: torch.Tensor,
    cut_mask: torch.Tensor,
) -> float:
    """Fraction of edge mass on the cut (``sum(w[cut]) / sum(w)``)."""
    w = torch.as_tensor(edge_weights, dtype=torch.float64).reshape(-1)
    mask = torch.as_tensor(cut_mask, dtype=torch.bool).reshape(-1)
    if w.numel() == 0:
        return 0.0
    if mask.shape[0] != w.shape[0]:
        raise ValueError("cut_mask must match edge_weights length")
    total = float(w.sum().item())
    if total <= 0.0:
        return 0.0
    return float(w[mask].sum().item()) / total


def knn_completeness_audit(
    expected_k: int,
    knn_idx: torch.Tensor,
    *,
    strict: bool = False,
) -> float:
    """Return kNN incompleteness in ``[0, 1]`` (≈0 when every row is full).

    A neighbour slot is incomplete when it is negative (sentinel) or when a
    row has fewer than ``expected_k`` columns. With ``strict=True``, raises
    :class:`RuntimeError` if incompleteness is positive — used to fail loudly
    on deliberately thin-halo fixtures.
    """
    idx = torch.as_tensor(knn_idx)
    if idx.ndim != 2:
        raise ValueError("knn_idx must be 2-D (n_rows, k)")
    expected_k = int(expected_k)
    n_rows, k_cols = int(idx.shape[0]), int(idx.shape[1])
    if n_rows == 0:
        return 0.0
    if k_cols < expected_k:
        miss = 1.0
    else:
        cols = idx[:, :expected_k]
        n_missing = int((cols < 0).sum().item())
        miss = float(n_missing) / float(n_rows * expected_k)
    if strict and miss > 0.0:
        raise RuntimeError(
            f"kNN completeness audit failed: incompleteness={miss:.6f} "
            f"(expected_k={expected_k}, shape={tuple(idx.shape)})"
        )
    return miss

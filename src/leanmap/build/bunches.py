"""Distributed landmark-bunch graph build (transport-agnostic).

Entry points::

    from leanmap.build.bunches import build_graph_bunches, build_graph_pyramid_bunches

Multi-node coordination uses :class:`~leanmap.build.transport.BunchTransport`.
**FileStore** (shared ``--stages`` directory + ``RANK`` / ``WORLD_SIZE``) is the
default multi-node transport and does **not** require ``mpi4py``. Optional
``torch.distributed`` and ``mpi4py`` adapters share the same API via
:func:`~leanmap.build.transport.make_transport`.

Core build paths (:mod:`leanmap.build.pipeline`) must **not** import this
module at load time. Top-level ``import leanmap.build.bunches`` works without
``mpi4py``; MPI is only pulled in when ``transport_kind=\"mpi\"``.

Pipeline (design §4.3 / v2 §9): probe → reconcile landmarks → mass-aware
partition → owned+halo nets → union-find merge → kNN fill → root reduce
(fuzzy + pyramid). At world size 1 / ``transport_kind=\"local\"``,
:func:`build_graph_pyramid_bunches` delegates to
:func:`~leanmap.build.pipeline.build_graph_pyramid` for bit-compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from leanmap.config import BETA_MULTIPLICITY, C_BUCKETS, LAMBDA_BACKBONE, PYRAMID_REP_RATIO
from leanmap.distance import DistanceFn
from leanmap.utils import get_logger

__all__ = [
    "probe_shards",
    "reconcile_landmarks",
    "partition_bunches",
    "partition_bunches_by_mass",
    "margin_halo",
    "owned_net",
    "uf_link",
    "uf_find",
    "distributed_union_find",
    "fill_knn_rows",
    "fill_knn_rep_slice",
    "stitch_graph",
    "build_graph_bunches",
    "build_graph_pyramid_bunches",
    "poisson_landmarks_from_reps",
    "relabel_pyramid_landmarks_poisson",
    "halo_fraction",
    "cut_mass",
    "knn_completeness_audit",
    "mpi_world_size",
    "mpi_rank",
]


# ---------------------------------------------------------------------------
# Env / lazy MPI
# ---------------------------------------------------------------------------


def _require_mpi4py():
    """Import ``mpi4py.MPI`` or raise with an install hint."""
    try:
        from mpi4py import MPI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mpi4py is required for distributed leanmap builds; "
            "install with: pip install leanmap[hpc]"
        ) from exc
    return MPI


def mpi_world_size() -> int:
    """Return worker world size from ``WORLD_SIZE`` (default ``1``).

    Env-based: does **not** require ``mpi4py``. Use a :class:`BunchTransport`
    when you need a live process-group size.
    """
    import os

    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def mpi_rank() -> int:
    """Return worker rank from ``RANK`` (default ``0``).

    Env-based: does **not** require ``mpi4py``.
    """
    import os

    return max(0, int(os.environ.get("RANK", "0")))


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
    transport: Optional[Any] = None,
) -> torch.Tensor:
    """Reconcile per-rank landmark pools into a global set.

    At ``world_size == 1`` (and no multi-worker transport) this is identity
    (optionally truncated). With ``transport``, uses
    ``allgather_obj`` / ``broadcast_obj``. Otherwise falls back to mpi4py when
    ``world_size > 1``.
    """
    local = torch.as_tensor(local_landmarks)
    ws = int(transport.world_size) if transport is not None else int(world_size)

    def _dedupe_truncate(pool: np.ndarray) -> np.ndarray:
        if pool.size == 0:
            return pool
        if pool.ndim == 1:
            uniq = np.unique(pool)
        else:
            dtype = np.dtype((np.void, pool.dtype.itemsize * pool.shape[1]))
            uniq = np.unique(pool.view(dtype)).view(pool.dtype).reshape(-1, pool.shape[1])
        if max_landmarks is not None and uniq.shape[0] > int(max_landmarks):
            rng = np.random.default_rng(int(seed))
            choose = rng.choice(uniq.shape[0], size=int(max_landmarks), replace=False)
            uniq = uniq[np.sort(choose)]
        return uniq

    if transport is not None:
        gathered = transport.allgather_obj(local.detach().cpu().numpy())
        pool = np.concatenate([np.asarray(g) for g in gathered], axis=0)
        if transport.rank == 0:
            uniq = _dedupe_truncate(pool)
            out_np = np.asarray(uniq)
        else:
            out_np = None
        obj = transport.broadcast_obj(out_np, root=0)
        return torch.as_tensor(obj, dtype=local.dtype).contiguous()

    if ws <= 1:
        if max_landmarks is not None and local.shape[0] > int(max_landmarks):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            pick = torch.randperm(local.shape[0], generator=g)[: int(max_landmarks)]
            return local[pick].contiguous()
        return local.contiguous()

    MPI = _require_mpi4py()
    comm = MPI.COMM_WORLD
    gathered = comm.allgather(local.detach().cpu().numpy())
    pool = np.concatenate([np.asarray(g) for g in gathered], axis=0)
    uniq = _dedupe_truncate(pool)
    out = torch.as_tensor(uniq, dtype=local.dtype)
    obj = comm.bcast(out.numpy() if mpi_rank() == 0 else None, root=0)
    return torch.as_tensor(obj, dtype=local.dtype).contiguous()


def partition_bunches(
    landmark_idx: torch.Tensor,
    n_bunches: int,
) -> torch.Tensor:
    """Assign each landmark to a bunch id in ``[0, n_bunches)``.

    Contiguous blocks balanced by landmark count. Returns ``(L,)`` int64 bunch
    ids aligned with ``landmark_idx``. Prefer :func:`partition_bunches_by_mass`
    for load-balanced multi-worker builds.
    """
    L = int(torch.as_tensor(landmark_idx).shape[0])
    n_bunches = max(1, int(n_bunches))
    if L == 0:
        return torch.empty(0, dtype=torch.int64)
    ids = (torch.arange(L, dtype=torch.int64) * n_bunches) // max(L, 1)
    return ids.clamp(max=n_bunches - 1)


def partition_bunches_by_mass(
    assign_top1: torch.Tensor,
    n_bunches: int,
    *,
    n_landmarks: Optional[int] = None,
) -> torch.Tensor:
    """Greedy LPT partition of landmarks by ``assign_top1`` point mass.

    Landmarks are sorted by descending mass (ties: lower landmark id first) and
    each is assigned to the bunch with the smallest current load (ties: lowest
    bunch id). Returns ``(L,)`` int64 bunch ids.
    """
    top1 = torch.as_tensor(assign_top1, dtype=torch.int64).reshape(-1)
    n_bunches = max(1, int(n_bunches))
    if top1.numel() == 0:
        L = int(n_landmarks or 0)
        return torch.zeros(L, dtype=torch.int64) if L else torch.empty(0, dtype=torch.int64)
    L = int(n_landmarks) if n_landmarks is not None else int(top1.max().item()) + 1
    L = max(L, int(top1.max().item()) + 1 if top1.numel() else 0)
    if L == 0:
        return torch.empty(0, dtype=torch.int64)
    mass = torch.bincount(top1, minlength=L).to(dtype=torch.int64)
    # Stable LPT: sort by (-mass, landmark_id).
    order = sorted(range(L), key=lambda i: (-int(mass[i].item()), i))
    loads = [0] * n_bunches
    bunch = [0] * L
    for lid in order:
        b = min(range(n_bunches), key=lambda j: (loads[j], j))
        bunch[lid] = b
        loads[b] += int(mass[lid].item())
    return torch.as_tensor(bunch, dtype=torch.int64)


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
        For 1-D ``assign``, expand ownership by ``±margin`` bunch ids.
        Ignored for the 2-D top-c path.
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


def _owned_net_halo(
    X: torch.Tensor,
    owned_idx: torch.Tensor,
    halo_idx: torch.Tensor,
    radius: float,
    dist_fn: DistanceFn,
    seed: int = 0,
    *,
    assign_top1: Optional[torch.Tensor] = None,
    owned_landmarks: Optional[torch.Tensor] = None,
    max_bucket: int = 20_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radius-net on owned landmark buckets (same algorithm as single-node).

    Nets each owned top-1 landmark bucket independently (with ``max_bucket``
    splits), matching :func:`~leanmap.build.pipeline.build_representatives`.
    Only owned points become representatives. Halo is retained for API /
    diagnostics.

    Returns ``rep_idx`` (owned only) and ``member_of`` aligned with ``owned_idx``.
    """
    owned_idx = torch.as_tensor(owned_idx, dtype=torch.int64).reshape(-1)
    halo_idx = torch.as_tensor(halo_idx, dtype=torch.int64).reshape(-1)
    _ = halo_idx
    if owned_idx.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    rng = np.random.default_rng(int(seed))
    from leanmap.build.pipeline import _epsilon_net_bucket

    # Fast path: no landmark structure → single bucket (small tests).
    if assign_top1 is None or owned_landmarks is None:
        rep_idx, member_of = owned_net(
            X, owned_idx, float(radius), dist_fn=dist_fn, seed=seed
        )
        return rep_idx, member_of

    assign_top1 = torch.as_tensor(assign_top1, dtype=torch.int64).reshape(-1)
    owned_landmarks = torch.as_tensor(owned_landmarks, dtype=torch.int64).reshape(-1)
    all_reps: List[int] = []
    member_of_full = torch.full((int(X.shape[0]),), -1, dtype=torch.int64)
    owned_set = set(int(x) for x in owned_idx.tolist())

    for b in owned_landmarks.tolist():
        P = torch.where(assign_top1 == int(b))[0]
        if P.numel() == 0:
            continue
        # Restrict to owned rows in this bucket (halo rows do not seed reps).
        P_owned = torch.as_tensor(
            [int(i) for i in P.tolist() if int(i) in owned_set],
            dtype=torch.int64,
        )
        if P_owned.numel() == 0:
            continue
        if P_owned.numel() > max_bucket:
            perm = P_owned.cpu().numpy().copy()
            rng.shuffle(perm)
            sub_reps: List[int] = []
            for start in range(0, len(perm), max_bucket):
                block = torch.as_tensor(
                    perm[start : start + max_bucket], dtype=torch.int64
                )
                reps_b, _ = _epsilon_net_bucket(X, block, dist_fn, float(radius), rng)
                sub_reps.extend(reps_b)
            reps_t = torch.tensor(sub_reps, dtype=torch.int64)
            reps, _ = _epsilon_net_bucket(X, reps_t, dist_fn, float(radius), rng)
            Xr = X[torch.tensor(reps, dtype=torch.int64, device=X.device)]
            base = len(all_reps)
            for r in reps:
                all_reps.append(int(r))
            for s in range(0, P_owned.numel(), 4096):
                e = min(P_owned.numel(), s + 4096)
                d = dist_fn(X[P_owned[s:e]], Xr)
                local = d.argmin(dim=1)
                for ii, j in enumerate(local.tolist()):
                    member_of_full[int(P_owned[s + ii])] = base + int(j)
        else:
            reps, mem = _epsilon_net_bucket(
                X, P_owned, dist_fn, float(radius), rng
            )
            base = len(all_reps)
            for r in reps:
                all_reps.append(int(r))
            for raw_i, local_j in mem.items():
                member_of_full[int(raw_i)] = base + int(local_j)

    if not all_reps:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.zeros(owned_idx.shape[0], dtype=torch.int64),
        )
    rep_idx = torch.as_tensor(all_reps, dtype=torch.int64)
    member_of = member_of_full[owned_idx]
    if (member_of < 0).any():
        miss = (member_of < 0).nonzero(as_tuple=False).reshape(-1)
        Xr = X[rep_idx]
        for s in range(0, miss.numel(), 4096):
            e = min(miss.numel(), s + 4096)
            idx = owned_idx[miss[s:e]]
            d = dist_fn(X[idx], Xr)
            member_of[miss[s:e]] = d.argmin(dim=1)
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


def _merge_owned_reps(
    X: torch.Tensor,
    parts: Sequence[Dict[str, Any]],
    dist_fn: DistanceFn,
    radius: float,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """UF-merge per-rank owned nets into global ``rep_idx`` + ``member_of``.

    Each part must provide:

    - ``rep_idx``: ``(R_r,)`` global row indices of owned representatives
    - ``owned_idx``: ``(n_r,)`` owned row indices
    - ``member_of``: ``(n_r,)`` local cell ids into that part's ``rep_idx``

    Cross-rank reps within ``radius`` are united (lowest root wins). Returns
    global ``rep_idx`` ``(R,)`` and ``member_of`` ``(n,)``.
    """
    if not parts:
        raise ValueError("parts must be non-empty")
    n = int(n)
    radius = float(radius)

    rep_chunks: List[torch.Tensor] = []
    offsets: List[int] = []
    off = 0
    for part in parts:
        ri = torch.as_tensor(part["rep_idx"], dtype=torch.int64).reshape(-1)
        rep_chunks.append(ri)
        offsets.append(off)
        off += int(ri.shape[0])

    if off == 0:
        member_of = torch.full((n,), -1, dtype=torch.int64)
        return torch.empty(0, dtype=torch.int64), member_of

    all_reps = torch.cat(rep_chunks, dim=0)
    R_all = int(all_reps.shape[0])
    parent = list(range(R_all))

    # Always unite duplicate raw indices (same point born on overlapping work).
    raw_to_slots: Dict[int, List[int]] = {}
    for i, raw in enumerate(all_reps.tolist()):
        raw_to_slots.setdefault(int(raw), []).append(i)
    for slots in raw_to_slots.values():
        for j in range(1, len(slots)):
            uf_link(parent, slots[0], slots[j])

    # Cross-rank radius merge. Exact pairwise is O(R^2); above a soft cap use
    # Faiss IVF ball queries (or skip if faiss unavailable).
    if R_all >= 2 and radius > 0.0:
        X_rep = X[all_reps]
        if R_all <= 8_000:
            chunk = 512
            for s in range(0, R_all, chunk):
                e = min(s + chunk, R_all)
                D = dist_fn(X_rep[s:e], X_rep)
                close = (D <= radius).nonzero(as_tuple=False)
                for a_pos, b in close.tolist():
                    a = s + int(a_pos)
                    b = int(b)
                    if a >= b:
                        continue
                    uf_link(parent, a, b)
        else:
            try:
                import faiss  # type: ignore

                faiss.omp_set_num_threads(1)
                xb = np.ascontiguousarray(
                    X_rep.detach().cpu().numpy().astype(np.float32)
                )
                d = int(xb.shape[1])
                nlist = min(max(int(np.sqrt(R_all)), 16), R_all // 39 + 1)
                quant = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quant, d, nlist)
                index.train(xb)
                index.add(xb)
                index.nprobe = min(32, nlist)
                lims, _D, I = index.range_search(xb, float(radius) ** 2)
                for i in range(R_all):
                    for j in I[lims[i] : lims[i + 1]]:
                        j = int(j)
                        if i < j:
                            uf_link(parent, i, j)
            except Exception as exc:  # pragma: no cover - optional acceleration
                get_logger().warning(
                    "_merge_owned_reps: skipping radius merge at R=%d (%s)",
                    R_all,
                    exc,
                )

    roots = [uf_find(parent, i) for i in range(R_all)]
    uniq = sorted(set(roots))
    remap = {old: new for new, old in enumerate(uniq)}
    new_of_old = torch.as_tensor([remap[r] for r in roots], dtype=torch.int64)
    rep_idx = all_reps[torch.as_tensor(uniq, dtype=torch.int64)].contiguous()

    member_of = torch.full((n,), -1, dtype=torch.int64)
    for part, base in zip(parts, offsets):
        owned_idx = torch.as_tensor(part["owned_idx"], dtype=torch.int64).reshape(-1)
        local_mem = torch.as_tensor(part["member_of"], dtype=torch.int64).reshape(-1)
        if owned_idx.numel() == 0:
            continue
        if local_mem.shape[0] != owned_idx.shape[0]:
            raise ValueError("member_of length must match owned_idx")
        global_local = local_mem + int(base)
        member_of[owned_idx] = new_of_old[global_local]

    if (member_of < 0).any():
        missing = int((member_of < 0).sum().item())
        raise RuntimeError(
            f"_merge_owned_reps: {missing} row(s) lack membership after merge"
        )
    return rep_idx, member_of


# ---------------------------------------------------------------------------
# kNN fill
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

    Each worker owns a contiguous query-row range against a shared corpus
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


def fill_knn_rep_slice(
    X: torch.Tensor,
    rep_idx: torch.Tensor,
    row_start: int,
    row_end: int,
    *,
    k: int,
    dist_fn: Optional[DistanceFn] = None,
    mode: str = "brute",
    faiss_index: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fill kNN for representative rows ``[row_start, row_end)``.

    ``mode=\"brute\"`` uses exact distances. ``mode=\"ann\"`` uses a shared
    Faiss IVF index (``faiss_index``) when provided, else builds a temporary
    one on ``X[rep_idx]`` (expensive if every worker rebuilds — prefer root
    broadcast of a serialized index for multi-worker runs).
    """
    from leanmap.distance import EuclideanDistance

    if dist_fn is None:
        dist_fn = EuclideanDistance()
    rep_idx = torch.as_tensor(rep_idx, dtype=torch.int64).reshape(-1)
    R = int(rep_idx.shape[0])
    row_start = int(row_start)
    row_end = int(row_end)
    if not (0 <= row_start <= row_end <= R):
        raise ValueError(f"invalid rep-row range [{row_start}, {row_end}) for R={R}")
    m = row_end - row_start
    if R == 0 or m == 0:
        kk = min(int(k), max(R - 1, 0))
        return (
            torch.empty(m, kk, dtype=torch.int64, device=X.device),
            torch.empty(m, kk, dtype=torch.float32, device=X.device),
        )
    k = min(int(k), max(R - 1, 1))
    use_ann = str(mode).lower() in ("ann", "ivf", "auto") and R > 2_000
    if use_ann:
        try:
            import faiss  # type: ignore

            faiss.omp_set_num_threads(1)
            X_rep = np.ascontiguousarray(
                X[rep_idx].detach().cpu().numpy().astype(np.float32)
            )
            index = faiss_index
            if index is None:
                d = int(X_rep.shape[1])
                nlist = min(max(int(np.sqrt(R)), 16), R // 39 + 1)
                quant = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quant, d, nlist)
                get_logger().info(
                    "fill_knn_rep_slice ANN train: R=%d nlist=%d rows=[%d,%d)",
                    R,
                    nlist,
                    row_start,
                    row_end,
                )
                index.train(X_rep)
                index.add(X_rep)
            index.nprobe = min(32, getattr(index, "nlist", 32))
            q = X_rep[row_start:row_end]
            # k+1 then drop self
            D, I = index.search(q, k + 1)
            knn_idx = torch.empty(m, k, dtype=torch.int64)
            knn_dist = torch.empty(m, k, dtype=torch.float32)
            for i in range(m):
                global_r = row_start + i
                cols: List[int] = []
                dists: List[float] = []
                for j in range(I.shape[1]):
                    nb = int(I[i, j])
                    if nb < 0 or nb == global_r:
                        continue
                    cols.append(nb)
                    dists.append(float(np.sqrt(max(D[i, j], 0.0))))
                    if len(cols) >= k:
                        break
                while len(cols) < k:
                    cols.append(0 if R > 1 else 0)
                    dists.append(float("inf"))
                knn_idx[i] = torch.as_tensor(cols[:k], dtype=torch.int64)
                knn_dist[i] = torch.as_tensor(dists[:k], dtype=torch.float32)
            return knn_idx.to(device=X.device), knn_dist.to(device=X.device)
        except Exception as exc:
            get_logger().warning(
                "fill_knn_rep_slice ANN failed (%s); falling back to brute", exc
            )

    X_rep = X[rep_idx]
    knn_idx = torch.empty(m, k, dtype=torch.int64, device=X.device)
    knn_dist = torch.empty(m, k, dtype=torch.float32, device=X.device)
    for local_i, global_r in enumerate(range(row_start, row_end)):
        d = dist_fn(X_rep[global_r : global_r + 1], X_rep)[0].clone()
        d[global_r] = float("inf")
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


def _resolve_worker_transport(
    transport: Optional[Any],
    transport_kind: str,
    stages_dir: Optional[Union[str, Path]],
) -> Any:
    from leanmap.build.transport import make_transport

    if transport is not None:
        return transport
    return make_transport(transport_kind, stages_dir=stages_dir)


def _net_radius_from_kwargs(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    *,
    epsilon: Optional[float],
    delta: Optional[Union[float, str]],
    dedup: bool,
    seed: int,
    r_band: Tuple[float, float],
    alpha_guard: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """Return ``(eps, net_radius, delta_report)`` using pipeline helpers."""
    from leanmap.build.pipeline import _resolve_net_radius, estimate_epsilon

    if not dedup:
        return 0.0, 0.0, {
            "delta": 0.0,
            "eps_ref": 0.0,
            "mode": "eps",
            "guard_ok": True,
        }
    if epsilon is None:
        eps, _diag = estimate_epsilon(X, dist_fn, seed=seed)
    else:
        eps = float(epsilon)
    net_radius, report = _resolve_net_radius(
        X,
        dist_fn,
        eps,
        delta,
        seed=seed,
        r_band=r_band,
        alpha_guard=alpha_guard,
    )
    return float(eps), float(net_radius), report


def build_graph_bunches(X: torch.Tensor, metric: Any, **kwargs: Any):
    """Distributed-aware single-scale graph build.

    Delegates to :func:`build_graph_pyramid_bunches` with ``pyramid_scales=0``
    and unwraps ``graphs[0]``. Non-root workers return ``None``.
    """
    kw = dict(kwargs)
    kw["pyramid_scales"] = 0
    result = build_graph_pyramid_bunches(X, metric, **kw)
    if result is None:
        return None
    graphs, M, assign_top1, assign_topc = result
    return graphs[0], M, assign_top1, assign_topc


def build_graph_pyramid_bunches(
    X: torch.Tensor,
    metric: Any,
    *,
    transport: Optional[Any] = None,
    transport_kind: str = "local",
    stages_dir: Optional[Union[str, Path]] = None,
    pyramid_scales: int = 3,
    pyramid_rep_ratio: float = PYRAMID_REP_RATIO,
    pyramid_min_reps: int = 256,
    pyramid_coarse_backbone: float = 1.0,
    pyramid_squash: str = "rational_q99",
    n_neighbors: int = 15,
    n_landmarks: int = 256,
    c_buckets: int = C_BUCKETS,
    epsilon: Optional[float] = None,
    delta: Optional[Union[float, str]] = None,
    dedup: bool = True,
    local_connectivity: int = 1,
    beta_multiplicity: float = BETA_MULTIPLICITY,
    lambda_backbone: float = LAMBDA_BACKBONE,
    seed: int = 0,
    n_probe: Optional[int] = None,
    r_band: Tuple[float, float] = (1e5, 1e6),
    alpha_guard: float = 0.95,
    knn_mode: str = "auto",
    fps_poisson: bool = False,
    fps_geodesic: bool = False,
    fps_geodesic_k: Optional[int] = None,
    **_extra: Any,
):
    """Multi-worker landmark-bunch pyramid build (transport-agnostic).

    World size 1 / ``transport_kind=\"local\"`` delegates to
    :func:`~leanmap.build.pipeline.build_graph_pyramid` (bit-compat).

    Multi-worker path: shared landmarks (geodesic **Poisson** when
    ``fps_poisson=True``, else legacy per-rank probe FPS) → assign buckets →
    mass-aware partition → owned+halo nets → gather → root UF merge →
    partitioned / root ANN kNN → root
    :func:`~leanmap.build.pipeline.assemble_graph_from_knn` +
    :func:`~leanmap.build.pipeline.pyramid_from_finest`.

    Non-root workers return ``None`` after collectives complete.
    """
    from leanmap.build.pipeline import (
        GraphStats,
        assemble_graph_from_knn,
        build_graph_pyramid,
        pyramid_from_finest,
        representatives_from_membership,
    )
    from leanmap.landmarks import (
        assign_buckets,
        fps_init_indices,
        poisson_disk_indices_geodesic,
    )

    log = get_logger()
    dist_fn: DistanceFn = metric
    kind = str(transport_kind).lower()
    build_passthrough = dict(
        n_neighbors=n_neighbors,
        n_landmarks=n_landmarks,
        c_buckets=c_buckets,
        epsilon=epsilon,
        delta=delta,
        dedup=dedup,
        local_connectivity=local_connectivity,
        beta_multiplicity=beta_multiplicity,
        lambda_backbone=lambda_backbone,
        seed=seed,
        stages_dir=stages_dir,
        r_band=r_band,
        alpha_guard=alpha_guard,
        knn_mode=knn_mode,
        fps_poisson=fps_poisson,
        fps_geodesic=fps_geodesic,
        fps_geodesic_k=fps_geodesic_k,
    )

    # Resolve transport; local / ws=1 → single-node pyramid (bit-compat).
    if transport is None and kind in ("local", "none", "off"):
        return build_graph_pyramid(
            X,
            metric,
            pyramid_scales=pyramid_scales,
            pyramid_rep_ratio=pyramid_rep_ratio,
            pyramid_min_reps=pyramid_min_reps,
            pyramid_coarse_backbone=pyramid_coarse_backbone,
            pyramid_squash=pyramid_squash,
            **build_passthrough,
        )

    t = _resolve_worker_transport(transport, kind, stages_dir)
    ws = int(t.world_size)
    rank = int(t.rank)

    if ws <= 1:
        return build_graph_pyramid(
            X,
            metric,
            pyramid_scales=pyramid_scales,
            pyramid_rep_ratio=pyramid_rep_ratio,
            pyramid_min_reps=pyramid_min_reps,
            pyramid_coarse_backbone=pyramid_coarse_backbone,
            pyramid_squash=pyramid_squash,
            **build_passthrough,
        )

    log.info(
        "build_graph_pyramid_bunches: world_size=%d rank=%d transport=%s "
        "fps_poisson=%s",
        ws,
        rank,
        type(t).__name__,
        bool(fps_poisson),
    )
    n = int(X.shape[0])
    if n_probe is None:
        n_probe = min(40_000, n)
    n_probe = int(n_probe)
    gk = int(fps_geodesic_k) if fps_geodesic_k is not None else int(n_neighbors)

    # --- Shared landmarks (Poisson preferred; probe-FPS is legacy parallel sketch) ---
    if fps_poisson:
        if rank == 0:
            idx = poisson_disk_indices_geodesic(
                X, dist_fn, n_landmarks, n_neighbors=gk, seed=seed
            )
            M_np = X[idx].detach().cpu().numpy()
            log.info(
                "landmarks: shared geodesic Poisson-disk -> L=%d", int(idx.shape[0])
            )
        else:
            M_np = None
        M_np = t.broadcast_obj(M_np, root=0)
        M = torch.as_tensor(M_np, dtype=X.dtype, device=X.device).contiguous()
    else:
        probe_idx = probe_shards(X, n_probe, seed=seed + rank)
        local_L = max(1, n_landmarks // ws + (1 if rank < (n_landmarks % ws) else 0))
        local_sel = fps_init_indices(
            X[probe_idx], dist_fn, local_L, seed=seed + rank
        )
        local_landmarks = X[probe_idx[local_sel]]
        M = reconcile_landmarks(
            local_landmarks,
            world_size=ws,
            max_landmarks=n_landmarks,
            seed=seed,
            transport=t,
        )
        M = M.to(device=X.device, dtype=X.dtype)
        if rank == 0:
            log.info("landmarks: reconciled probe-FPS -> L=%d", int(M.shape[0]))

    # --- Assign + mass-aware partition (root computes, broadcast) ---
    assign_top1, assign_topc = assign_buckets(X, M, dist_fn, c=c_buckets)
    if rank == 0:
        bunch_ids = partition_bunches_by_mass(
            assign_top1, n_bunches=ws, n_landmarks=int(M.shape[0])
        )
        bunch_np = bunch_ids.detach().cpu().numpy()
    else:
        bunch_np = None
    bunch_np = t.broadcast_obj(bunch_np, root=0)
    bunch_ids = torch.as_tensor(bunch_np, dtype=torch.int64, device=X.device)

    row_bunch = bunch_ids[assign_top1]
    owned_mask = row_bunch == rank
    # Halo: not owned by this rank, but top-c shortlist intersects this bunch.
    topc_bunch = bunch_ids[assign_topc.clamp(min=0)]
    if topc_bunch.ndim == 2:
        touches_rank = (topc_bunch == rank).any(dim=1)
    else:
        touches_rank = topc_bunch == rank
    foreign_halo = (~owned_mask) & touches_rank
    owned_idx = torch.where(owned_mask)[0]
    halo_idx = torch.where(foreign_halo)[0]

    eps, net_radius, delta_report = _net_radius_from_kwargs(
        X,
        dist_fn,
        epsilon=epsilon,
        delta=delta,
        dedup=dedup,
        seed=seed,
        r_band=r_band,
        alpha_guard=alpha_guard,
    )

    rep_idx_local, member_local = _owned_net_halo(
        X,
        owned_idx,
        halo_idx,
        radius=net_radius,
        dist_fn=dist_fn,
        seed=seed + rank,
        assign_top1=assign_top1,
        owned_landmarks=torch.where(bunch_ids == rank)[0],
    )
    part = {
        "rep_idx": rep_idx_local.detach().cpu(),
        "owned_idx": owned_idx.detach().cpu(),
        "member_of": member_local.detach().cpu(),
        "halo_idx": halo_idx.detach().cpu(),
        "n_owned": int(owned_idx.numel()),
        "n_halo": int(halo_idx.numel()),
    }

    gathered = t.gather_obj(part, root=0)
    if rank == 0:
        assert gathered is not None
        rep_idx, member_of = _merge_owned_reps(
            X.cpu() if X.device.type != "cpu" else X,
            gathered,
            dist_fn,
            net_radius,
            n,
        )
        payload = {
            "rep_idx": rep_idx.detach().cpu().numpy(),
            "member_of": member_of.detach().cpu().numpy(),
        }
        n_owned_tot = sum(int(p["n_owned"]) for p in gathered)
        n_halo_tot = sum(int(p["n_halo"]) for p in gathered)
        halo_frac = halo_fraction(n_owned_tot, n_halo_tot)
    else:
        payload = None
        halo_frac = 0.0
    payload = t.broadcast_obj(payload, root=0)
    rep_idx = torch.as_tensor(payload["rep_idx"], dtype=torch.int64, device=X.device)
    member_of = torch.as_tensor(
        payload["member_of"], dtype=torch.int64, device=X.device
    )
    R = int(rep_idx.shape[0])

    # --- kNN over rep rows (ANN on root for large R; else partitioned brute) ---
    knn_mode_l = str(knn_mode).lower()
    use_root_ann = knn_mode_l in ("ann", "ivf", "auto") and R > 2_000
    if use_root_ann:
        if rank == 0:
            knn_idx, knn_dist = fill_knn_rep_slice(
                X,
                rep_idx,
                0,
                R,
                k=n_neighbors,
                dist_fn=dist_fn,
                mode="ann",
            )
            knn_payload = {
                "knn_idx": knn_idx.detach().cpu().numpy(),
                "knn_dist": knn_dist.detach().cpu().numpy(),
            }
        else:
            knn_payload = None
        knn_payload = t.broadcast_obj(knn_payload, root=0)
        knn_idx = torch.as_tensor(knn_payload["knn_idx"], dtype=torch.int64)
        knn_dist = torch.as_tensor(knn_payload["knn_dist"], dtype=torch.float32)
        if rank != 0:
            return None
    else:
        r0 = (R * rank) // ws
        r1 = (R * (rank + 1)) // ws
        knn_idx_loc, knn_dist_loc = fill_knn_rep_slice(
            X,
            rep_idx,
            r0,
            r1,
            k=n_neighbors,
            dist_fn=dist_fn,
            mode="brute",
        )
        knn_part = {
            "row_start": r0,
            "row_end": r1,
            "knn_idx": knn_idx_loc.detach().cpu(),
            "knn_dist": knn_dist_loc.detach().cpu(),
        }
        knn_gathered = t.gather_obj(knn_part, root=0)

        if rank != 0:
            return None

        assert knn_gathered is not None
        k_eff = (
            int(knn_gathered[0]["knn_idx"].shape[1]) if knn_gathered else n_neighbors
        )
        knn_idx = torch.empty(R, k_eff, dtype=torch.int64)
        knn_dist = torch.empty(R, k_eff, dtype=torch.float32)
        for kp in knn_gathered:
            s, e = int(kp["row_start"]), int(kp["row_end"])
            if e > s:
                knn_idx[s:e] = torch.as_tensor(kp["knn_idx"], dtype=torch.int64)
                knn_dist[s:e] = torch.as_tensor(kp["knn_dist"], dtype=torch.float32)

    knn_completeness_audit(n_neighbors, knn_idx, strict=True)

    stats = GraphStats(dedup=bool(dedup))
    stats.epsilon = float(eps)
    stats.delta = float(net_radius)
    stats.knn_mode = "bunches_ann" if use_root_ann else "bunches_brute"
    stats.extra["delta"] = float(net_radius)
    stats.extra["epsilon"] = float(eps)
    stats.extra["delta_mode"] = delta_report.get("mode")
    stats.extra["delta_guard_ok"] = delta_report.get("guard_ok")
    stats.extra["halo_fraction"] = float(halo_frac)
    stats.extra["bunch_transport"] = type(t).__name__
    stats.n_reps = R
    if n > 0 and R > 0:
        stats.compression_ratio = float(n) / float(R)

    reps = representatives_from_membership(rep_idx, member_of, n=n)
    graph0 = assemble_graph_from_knn(
        X,
        metric,
        reps,
        knn_idx,
        knn_dist,
        M,
        stats=stats,
        local_connectivity=local_connectivity,
        beta_multiplicity=beta_multiplicity,
        lambda_backbone=lambda_backbone,
    )
    graphs = pyramid_from_finest(
        graph0,
        X,
        metric,
        pyramid_scales=pyramid_scales,
        pyramid_rep_ratio=pyramid_rep_ratio,
        pyramid_min_reps=pyramid_min_reps,
        pyramid_coarse_backbone=pyramid_coarse_backbone,
        pyramid_squash=pyramid_squash,
        seed=seed,
    )
    return graphs, M, assign_top1, assign_topc


def poisson_landmarks_from_reps(
    X: torch.Tensor,
    rep_idx: torch.Tensor,
    dist_fn: DistanceFn,
    n_landmarks: int,
    *,
    n_neighbors: int = 15,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Geodesic Poisson landmarks whose support is the δ-net corpus.

    Runs :func:`~leanmap.landmarks.poisson_disk_indices_geodesic` on
    ``X[rep_idx]`` and maps accepted indices back to ambient row ids.

    Returns
    -------
    M : ``(L, D)`` landmark feature rows
    landmark_idx : ``(L,)`` int64 indices into ``X``
    """
    from leanmap.landmarks import poisson_disk_indices_geodesic

    rep_idx = torch.as_tensor(rep_idx, dtype=torch.int64).reshape(-1)
    if rep_idx.numel() == 0:
        raise ValueError("rep_idx must be non-empty")
    X_rep = X[rep_idx]
    L = min(int(n_landmarks), int(rep_idx.shape[0]))
    local = poisson_disk_indices_geodesic(
        X_rep, dist_fn, L, n_neighbors=n_neighbors, seed=seed
    )
    landmark_idx = rep_idx[local].contiguous()
    M = X[landmark_idx].contiguous()
    return M, landmark_idx


def relabel_pyramid_landmarks_poisson(
    graphs: Sequence[Any],
    X: torch.Tensor,
    metric: Any,
    *,
    n_landmarks: int,
    n_neighbors: int = 15,
    c_buckets: int = C_BUCKETS,
    seed: int = 0,
    refresh_backbone: bool = True,
    lambda_backbone: float = LAMBDA_BACKBONE,
) -> Tuple[List[Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Option B: resample Poisson ``M`` from the finest δ-net; keep the net.

    Replaces landmarks / assignments on an already-built pyramid. When
    ``refresh_backbone`` is True, re-merges the landmark backbone into
    ``graphs[0]`` edge weights (fuzzy kNN structure unchanged).
    """
    from leanmap.build.pipeline import landmark_backbone
    from leanmap.landmarks import assign_buckets
    from scipy import sparse

    if not graphs:
        raise ValueError("graphs must be non-empty")
    g0 = graphs[0]
    dist_fn: DistanceFn = metric
    M, _lm_idx = poisson_landmarks_from_reps(
        X,
        g0.reps.rep_idx,
        dist_fn,
        n_landmarks,
        n_neighbors=n_neighbors,
        seed=seed,
    )
    assign_top1, assign_topc = assign_buckets(X, M, dist_fn, c=c_buckets)

    out_graphs = list(graphs)
    if refresh_backbone and int(g0.edges.shape[0]) > 0:
        R = int(g0.reps.rep_idx.shape[0])
        e = g0.edges.detach().cpu().numpy()
        w = g0.weights.detach().cpu().numpy().astype(np.float32)
        P = sparse.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(R, R))
        P = (P + P.T).tocsr()
        bb = landmark_backbone(
            M, dist_fn, g0.reps.rep_idx, X, lambda_bb=lambda_backbone
        ).tocsr()
        diff = bb - P
        diff.data = np.maximum(diff.data, 0)
        P = P + diff
        if P.data.size:
            mx = float(P.data.max())
            if mx > 0:
                P.data /= mx
        P = P.tocoo()
        mask = P.row < P.col
        edges = torch.stack(
            [
                torch.as_tensor(P.row[mask], dtype=torch.int64),
                torch.as_tensor(P.col[mask], dtype=torch.int64),
            ],
            dim=1,
        )
        weights = torch.as_tensor(P.data[mask], dtype=torch.float32)
        keep = weights > 0
        g0.edges = edges[keep]
        g0.weights = weights[keep]
        g0.stats.extra["landmarks_resampled"] = "poisson_from_reps"
        g0.stats.extra["n_landmarks_resampled"] = int(M.shape[0])
        out_graphs[0] = g0

    get_logger().info(
        "relabel_pyramid_landmarks_poisson: L=%d R=%d refresh_backbone=%s",
        int(M.shape[0]),
        int(g0.reps.rep_idx.shape[0]),
        bool(refresh_backbone),
    )
    return out_graphs, M, assign_top1, assign_topc


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
    :class:`RuntimeError` if incompleteness is positive.
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

"""Streaming cover graph construction for large N.

Seed a subgraph on a random subsample, then ingest the remaining rows in
batches: assign to landmarks, absorb into δ-cells or spawn reps, promote
landmark novelty, refresh dirty kNN rows, and finally assemble the same
:func:`~leanmap.build.pipeline.assemble_graph_from_knn` artefact as a
single-pass build.

See ``docs/design/streaming_graph_build.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..config import (
    BETA_MULTIPLICITY,
    C_BUCKETS,
    C_SEARCH,
    LAMBDA_BACKBONE,
    PYRAMID_REP_RATIO,
)
from ..distance import DistanceFn, chunked_cdist
from ..landmarks import assign_buckets, fps_init_indices
from ..metrics import MetricSpec
from ..utils import BUILD_PROGRESS, get_logger, rss_mb
from .pipeline import (
    Graph,
    GraphStats,
    _halo_merge,
    assemble_graph_from_knn,
    build_graph,
    knn_representatives,
    pyramid_from_finest,
    representatives_from_membership,
)


@dataclass
class StreamingBuildReport:
    """Diagnostics for a streaming cover build."""

    ingest_batch: int = 0
    seed_size: int = 0
    n_rounds: int = 0
    n_absorbed: int = 0
    n_spawned: int = 0
    n_novelty_landmarks: int = 0
    compression_ratio: float = 1.0
    n_reps: int = 0
    knn_overlap: Optional[float] = None
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def knn_overlap_jaccard(
    knn_a: torch.Tensor,
    knn_b: torch.Tensor,
    *,
    rep_idx_a: Optional[torch.Tensor] = None,
    rep_idx_b: Optional[torch.Tensor] = None,
    n_sample: int = 512,
    seed: int = 0,
) -> float:
    """Mean Jaccard overlap of neighbor sets.

    When ``rep_idx_a`` / ``rep_idx_b`` are given (ambient row ids of each
    representative), neighbors are mapped to ambient ids and only reps that
    appear in **both** covers are compared. Otherwise falls back to
    row-aligned comparison on ``min(Ra, Rb)`` (only meaningful for identical
    representative orderings).
    """
    a = torch.as_tensor(knn_a, dtype=torch.int64)
    b = torch.as_tensor(knn_b, dtype=torch.int64)
    k = min(int(a.shape[1]) if a.ndim == 2 else 0, int(b.shape[1]) if b.ndim == 2 else 0)
    if k == 0:
        return 0.0

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if rep_idx_a is not None and rep_idx_b is not None:
        ra = torch.as_tensor(rep_idx_a, dtype=torch.int64).reshape(-1)
        rb = torch.as_tensor(rep_idx_b, dtype=torch.int64).reshape(-1)
        map_b = {int(v): i for i, v in enumerate(rb.tolist())}
        shared = [i for i, v in enumerate(ra.tolist()) if int(v) in map_b]
        if not shared:
            return 0.0
        take = min(int(n_sample), len(shared))
        pick = torch.randperm(len(shared), generator=g)[:take]
        scores: List[float] = []
        for p in pick.tolist():
            ia = int(shared[p])
            ib = int(map_b[int(ra[ia].item())])
            sa = {int(ra[int(j)].item()) for j in a[ia, :k].tolist() if 0 <= int(j) < int(ra.numel())}
            sb = {int(rb[int(j)].item()) for j in b[ib, :k].tolist() if 0 <= int(j) < int(rb.numel())}
            if not sa and not sb:
                scores.append(1.0)
                continue
            inter = len(sa & sb)
            union = len(sa | sb)
            scores.append(float(inter) / float(max(union, 1)))
        return float(sum(scores) / max(len(scores), 1))

    r = min(int(a.shape[0]), int(b.shape[0]))
    if r == 0:
        return 0.0
    take = min(int(n_sample), r)
    rows = torch.randperm(r, generator=g)[:take]
    scores = []
    for i in rows.tolist():
        sa = set(int(x) for x in a[i, :k].tolist())
        sb = set(int(x) for x in b[i, :k].tolist())
        if not sa and not sb:
            scores.append(1.0)
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        scores.append(float(inter) / float(max(union, 1)))
    return float(sum(scores) / max(len(scores), 1))


def _novelty_radius(
    M: torch.Tensor,
    dist_fn: DistanceFn,
    delta: float,
) -> float:
    """Threshold for promoting a point to a landmark candidate."""
    L = int(M.shape[0])
    if L < 2:
        return max(float(delta), 0.0)
    vals, _ = chunked_cdist(dist_fn, M, M, topk=2, out_device=M.device)
    nn1 = vals[:, 1]
    med = float(nn1.median().item()) if nn1.numel() else float(delta)
    return max(float(delta), med)


def _basin_rep_lists(
    rep_top1: torch.Tensor,
    n_landmarks: int,
) -> List[List[int]]:
    """Inverted index: landmark id → list of rep ids with that top-1."""
    lists: List[List[int]] = [[] for _ in range(int(n_landmarks))]
    for r, b in enumerate(rep_top1.tolist()):
        b = int(b)
        if 0 <= b < len(lists):
            lists[b].append(int(r))
    return lists


def _ingest_batch_absorb_spawn(
    X: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    M: torch.Tensor,
    rep_idx: torch.Tensor,
    member_of: torch.Tensor,
    rep_top1: torch.Tensor,
    dist_fn: DistanceFn,
    delta: float,
    c_buckets: int,
    novelty_radius: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    List[int],
    List[int],
]:
    """Absorb batch rows into existing reps or spawn new ones."""
    log = get_logger()
    batch_idx = torch.as_tensor(batch_idx, dtype=torch.int64).reshape(-1)
    if batch_idx.numel() == 0:
        empty_c = torch.empty(0, c_buckets, dtype=torch.int64, device=X.device)
        return (
            rep_idx,
            member_of,
            rep_top1,
            empty_c,
            0,
            0,
            [],
            [],
        )

    Xb = X[batch_idx]
    top1, topc = assign_buckets(Xb, M, dist_fn, c=min(int(c_buckets), int(M.shape[0])))
    L = int(M.shape[0])
    basin = _basin_rep_lists(rep_top1, L)

    rep_list = rep_idx.detach().cpu().tolist()
    top1_list = rep_top1.detach().cpu().tolist()
    member = member_of.clone()
    n_absorbed = 0
    n_spawned = 0
    dirty: List[int] = []
    novelty: List[int] = []

    chunk = 2048
    for s in range(0, int(batch_idx.numel()), chunk):
        e = min(int(batch_idx.numel()), s + chunk)
        for local_i in range(s, e):
            gi = int(batch_idx[local_i].item())
            buckets = [int(b) for b in topc[local_i].tolist() if int(b) >= 0]
            cand_reps: List[int] = []
            seen = set()
            for b in buckets:
                for r in basin[b] if b < len(basin) else ():
                    if r not in seen:
                        seen.add(r)
                        cand_reps.append(r)
            absorbed = False
            if cand_reps and float(delta) > 0.0:
                ri = torch.as_tensor(cand_reps, dtype=torch.int64, device=X.device)
                rep_t = torch.as_tensor(rep_list, dtype=torch.int64, device=X.device)
                Xr = X[rep_t[ri]]
                d = dist_fn(X[gi : gi + 1], Xr)[0]
                j = int(d.argmin().item())
                dmin = float(d[j].item())
                if dmin <= float(delta):
                    r_id = int(cand_reps[j])
                    member[gi] = r_id
                    n_absorbed += 1
                    dirty.append(r_id)
                    absorbed = True
            if not absorbed:
                r_new = len(rep_list)
                rep_list.append(gi)
                b0 = int(top1[local_i].item())
                top1_list.append(b0)
                if 0 <= b0 < len(basin):
                    basin[b0].append(r_new)
                member[gi] = r_new
                n_spawned += 1
                dirty.append(r_new)

            dm = dist_fn(X[gi : gi + 1], M)[0]
            d_m = float(dm.min().item())
            if d_m > float(novelty_radius):
                novelty.append(gi)

    if n_absorbed or n_spawned:
        log.info(
            "streaming ingest: batch=%d absorbed=%d spawned=%d novelty=%d R→%d",
            int(batch_idx.numel()),
            n_absorbed,
            n_spawned,
            len(novelty),
            len(rep_list),
        )

    new_rep_idx = torch.as_tensor(rep_list, dtype=torch.int64, device=X.device)
    new_rep_top1 = torch.as_tensor(top1_list, dtype=torch.int64, device=X.device)
    return (
        new_rep_idx,
        member,
        new_rep_top1,
        topc,
        n_absorbed,
        n_spawned,
        sorted(set(dirty)),
        novelty,
    )


def _reconcile_landmark_indices(
    X: torch.Tensor,
    landmark_idx: torch.Tensor,
    novelty_rows: Sequence[int],
    *,
    n_landmarks: int,
    dist_fn: DistanceFn,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Merge novelty rows into the landmark set; FPS-truncate to ``n_landmarks``."""
    lm = torch.as_tensor(landmark_idx, dtype=torch.int64).reshape(-1)
    if not novelty_rows:
        return lm, X[lm].contiguous(), 0
    pool = torch.unique(
        torch.cat(
            [
                lm,
                torch.as_tensor(list(novelty_rows), dtype=torch.int64),
            ]
        )
    )
    n_added = int(pool.numel()) - int(lm.numel())
    if int(pool.numel()) <= int(n_landmarks):
        return pool, X[pool].contiguous(), max(n_added, 0)
    Xp = X[pool]
    local = fps_init_indices(Xp, dist_fn, int(n_landmarks), seed=seed)
    out = pool[local].contiguous()
    return out, X[out].contiguous(), max(n_added, 0)


def _expand_knn_storage(
    knn_idx: Optional[torch.Tensor],
    knn_dist: Optional[torch.Tensor],
    R: int,
    k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grow (or allocate) knn buffers to shape ``(R, k)``."""
    if knn_idx is None or knn_dist is None or int(knn_idx.shape[0]) == 0:
        return (
            torch.full((R, k), -1, dtype=torch.int64, device=device),
            torch.full((R, k), float("inf"), dtype=torch.float32, device=device),
        )
    old_R = int(knn_idx.shape[0])
    old_k = int(knn_idx.shape[1])
    kk = min(int(k), max(R - 1, 1))
    if old_R == R and old_k == kk:
        return knn_idx, knn_dist
    new_idx = torch.full((R, kk), -1, dtype=torch.int64, device=device)
    new_dist = torch.full((R, kk), float("inf"), dtype=torch.float32, device=device)
    copy_r = min(old_R, R)
    copy_k = min(old_k, kk)
    if copy_r > 0 and copy_k > 0:
        new_idx[:copy_r, :copy_k] = knn_idx[:copy_r, :copy_k]
        new_dist[:copy_r, :copy_k] = knn_dist[:copy_r, :copy_k]
    return new_idx, new_dist


def _refresh_dirty_knn(
    X: torch.Tensor,
    rep_idx: torch.Tensor,
    dirty: Sequence[int],
    knn_idx: torch.Tensor,
    knn_dist: torch.Tensor,
    *,
    k: int,
    dist_fn: DistanceFn,
    knn_mode: str,
    M: torch.Tensor,
    metric: MetricSpec,
    c_buckets: int,
    c_search: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Recompute kNN rows for dirty rep ids against the current rep set."""
    R = int(rep_idx.shape[0])
    kk = min(int(k), max(R - 1, 1))
    knn_idx, knn_dist = _expand_knn_storage(knn_idx, knn_dist, R, kk, X.device)
    if R < 2 or not dirty:
        return knn_idx, knn_dist, 0

    dirty_u = sorted({int(d) for d in dirty if 0 <= int(d) < R})
    expanded = set(dirty_u)
    for d in dirty_u:
        for nb in knn_idx[d].tolist():
            nb = int(nb)
            if 0 <= nb < R:
                expanded.add(nb)
    dirty_u = sorted(expanded)
    n_dirty = len(dirty_u)

    X_rep = X[rep_idx]
    if n_dirty >= max(64, R // 4) or n_dirty == R:
        _rep_top1, rep_topc = assign_buckets(
            X_rep, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
        )
        knn_dist_full, knn_idx_full, _info = knn_representatives(
            X_rep,
            dist_fn,
            kk,
            mode=knn_mode,
            landmarks=M,
            assign_topc=rep_topc,
            c_search=c_search,
            metric=metric,
        )
        return knn_idx_full, knn_dist_full, n_dirty

    for r in dirty_u:
        d = dist_fn(X_rep[r : r + 1], X_rep)[0].clone()
        d[r] = float("inf")
        vals, idx = torch.topk(d, k=kk, largest=False)
        knn_idx[r] = idx
        knn_dist[r] = vals
    return knn_idx, knn_dist, n_dirty


def build_graph_streaming(
    X: torch.Tensor,
    metric: MetricSpec,
    *,
    ingest_batch: int = 50_000,
    seed_size: Optional[int] = None,
    n_neighbors: int = 15,
    n_landmarks: int = 256,
    c_buckets: int = C_BUCKETS,
    epsilon: Optional[float] = None,
    delta: Optional[Union[float, str]] = None,
    dedup: bool = True,
    local_connectivity: int = 1,
    beta_multiplicity: float = BETA_MULTIPLICITY,
    lambda_backbone: float = LAMBDA_BACKBONE,
    knn_mode: str = "auto",
    c_search: int = C_SEARCH,
    seed: int = 0,
    r_band: Tuple[float, float] = (1e5, 1e6),
    alpha_guard: float = 0.95,
    compute_knn_overlap: bool = False,
) -> Tuple[Graph, torch.Tensor, torch.Tensor, torch.Tensor, StreamingBuildReport]:
    """Build a fuzzy neighbour graph via streaming cover ingest.

    Parameters
    ----------
    ingest_batch :
        Max uncovered rows processed per round.
    seed_size :
        Initial subsample size (default: ``ingest_batch``).
    compute_knn_overlap :
        If True and ``N ≤ 20_000``, also run single-pass :func:`build_graph`
        and record mean kNN Jaccard in the report.

    Returns
    -------
    graph, M, assign_top1, assign_topc, report
    """
    log = get_logger()
    dist_fn: DistanceFn = metric
    X = torch.as_tensor(X, dtype=torch.float32)
    n = int(X.shape[0])
    if n < 2:
        raise ValueError("streaming build requires N >= 2")

    batch_n = max(1, int(ingest_batch))
    seed_n = int(seed_size) if seed_size is not None else batch_n
    seed_n = max(2, min(seed_n, n))

    report = StreamingBuildReport(ingest_batch=batch_n, seed_size=seed_n)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    seed_idx = perm[:seed_n].contiguous()
    rest = perm[seed_n:].contiguous()

    log.info(
        "streaming cover: N=%d seed=%d batch=%d L=%d knn_mode=%s RSS≈%.0f MiB",
        n,
        seed_n,
        batch_n,
        int(n_landmarks),
        knn_mode,
        rss_mb(),
    )
    BUILD_PROGRESS.set("streaming", f"seed N0={seed_n}")

    Xs = X[seed_idx]
    seed_graph, M_seed, _a1s, _acs = build_graph(
        Xs,
        metric,
        n_neighbors=n_neighbors,
        n_landmarks=min(int(n_landmarks), seed_n),
        c_buckets=c_buckets,
        epsilon=epsilon,
        delta=delta,
        dedup=dedup,
        local_connectivity=local_connectivity,
        beta_multiplicity=beta_multiplicity,
        lambda_backbone=lambda_backbone,
        knn_mode=knn_mode,
        c_search=c_search,
        seed=seed,
        r_band=r_band,
        alpha_guard=alpha_guard,
    )
    eps = float(seed_graph.stats.epsilon)
    net_radius = float(seed_graph.stats.delta) if seed_graph.stats.delta else eps
    if not dedup:
        eps = 0.0
        net_radius = 0.0

    M = M_seed.contiguous()
    _d_lm, i_lm = chunked_cdist(dist_fn, M, Xs, topk=1, out_device=X.device)
    landmark_idx = seed_idx[i_lm.reshape(-1)].contiguous()
    M = X[landmark_idx].contiguous()

    local_reps = seed_graph.reps
    rep_idx = seed_idx[local_reps.rep_idx].contiguous()
    member_of = torch.full((n,), -1, dtype=torch.int64, device=X.device)
    member_of[seed_idx] = local_reps.member_of.to(device=X.device)

    X_rep = X[rep_idx]
    rep_top1, _rep_topc = assign_buckets(
        X_rep, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
    )
    knn_idx = seed_graph.knn_idx.to(device=X.device).clone()
    knn_dist = torch.empty_like(knn_idx, dtype=torch.float32)
    for r in range(int(rep_idx.shape[0])):
        nb = knn_idx[r]
        if nb.numel() == 0:
            continue
        knn_dist[r] = dist_fn(X_rep[r : r + 1], X_rep[nb])[0]

    covered = torch.zeros(n, dtype=torch.bool, device=X.device)
    covered[seed_idx] = True

    stats = GraphStats(
        epsilon=eps,
        delta=float(net_radius),
        dedup=bool(dedup),
        knn_mode=str(seed_graph.stats.knn_mode),
    )
    stats.extra["delta_mode"] = seed_graph.stats.extra.get("delta_mode")
    stats.extra["seed_compression"] = float(seed_graph.stats.compression_ratio)

    offset = 0
    round_id = 0
    while offset < int(rest.numel()):
        br = rest[offset : offset + batch_n]
        offset += int(br.numel())
        round_id += 1
        BUILD_PROGRESS.set(
            "streaming",
            f"round {round_id} batch={int(br.numel())} R={int(rep_idx.shape[0])}",
        )
        novelty_r = _novelty_radius(M, dist_fn, net_radius)
        (
            rep_idx,
            member_of,
            rep_top1,
            _topc_b,
            n_abs,
            n_sp,
            dirty,
            novelty_rows,
        ) = _ingest_batch_absorb_spawn(
            X,
            br,
            M=M,
            rep_idx=rep_idx,
            member_of=member_of,
            rep_top1=rep_top1,
            dist_fn=dist_fn,
            delta=net_radius,
            c_buckets=c_buckets,
            novelty_radius=novelty_r,
        )
        covered[br] = True
        report.n_absorbed += int(n_abs)
        report.n_spawned += int(n_sp)

        landmark_idx, M, n_nov = _reconcile_landmark_indices(
            X,
            landmark_idx,
            novelty_rows,
            n_landmarks=int(n_landmarks),
            dist_fn=dist_fn,
            seed=seed + round_id,
        )
        report.n_novelty_landmarks += int(n_nov)
        if n_nov > 0:
            X_rep = X[rep_idx]
            rep_top1, _ = assign_buckets(
                X_rep, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
            )

        knn_idx, knn_dist, n_dirty = _refresh_dirty_knn(
            X,
            rep_idx,
            dirty,
            knn_idx,
            knn_dist,
            k=n_neighbors,
            dist_fn=dist_fn,
            knn_mode=knn_mode,
            M=M,
            metric=metric,
            c_buckets=c_buckets,
            c_search=c_search,
        )
        round_info = {
            "round": round_id,
            "batch": int(br.numel()),
            "absorbed": int(n_abs),
            "spawned": int(n_sp),
            "novelty": int(n_nov),
            "dirty_R": int(n_dirty),
            "R": int(rep_idx.shape[0]),
        }
        report.rounds.append(round_info)
        log.info(
            "streaming round %d: absorbed=%d spawned=%d novelty=%d dirty=%d R=%d",
            round_id,
            n_abs,
            n_sp,
            n_nov,
            n_dirty,
            int(rep_idx.shape[0]),
        )

    report.n_rounds = int(round_id)

    missing = torch.where(~covered)[0]
    if missing.numel() > 0:
        log.warning("streaming: %d uncovered rows; forcing spawn", int(missing.numel()))
        for gi in missing.tolist():
            r_new = int(rep_idx.shape[0])
            rep_idx = torch.cat(
                [rep_idx, torch.as_tensor([gi], dtype=torch.int64, device=X.device)]
            )
            member_of[gi] = r_new
            report.n_spawned += 1
        X_rep = X[rep_idx]
        rep_top1, _ = assign_buckets(
            X_rep, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
        )

    if (member_of < 0).any():
        orphan = torch.where(member_of < 0)[0]
        X_rep = X[rep_idx]
        for gi in orphan.tolist():
            d = dist_fn(X[gi : gi + 1], X_rep)[0]
            member_of[gi] = int(d.argmin().item())

    BUILD_PROGRESS.set("streaming", "finalize halo + knn")
    assign_top1, assign_topc = assign_buckets(
        X, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
    )
    reps = representatives_from_membership(rep_idx, member_of, n=n)
    if dedup and float(net_radius) > 0.0:
        reps, halo_info = _halo_merge(X, reps, assign_topc, dist_fn, net_radius)
        stats.extra.update(halo_info)
        rep_idx = reps.rep_idx
        member_of = reps.member_of

    X_rep = X[rep_idx]
    _rep_top1, rep_topc = assign_buckets(
        X_rep, M, dist_fn, c=min(int(c_buckets), int(M.shape[0]))
    )
    knn_dist_f, knn_idx_f, knn_info = knn_representatives(
        X_rep,
        dist_fn,
        n_neighbors,
        mode=knn_mode,
        landmarks=M,
        assign_topc=rep_topc,
        c_search=c_search,
        metric=metric,
    )
    stats.knn_mode = str(knn_info.get("mode", knn_mode))
    stats.knn_recall = knn_info.get("recall")
    stats.n_reps = int(rep_idx.shape[0])
    stats.compression_ratio = float(n) / max(stats.n_reps, 1)
    report.n_reps = stats.n_reps
    report.compression_ratio = stats.compression_ratio

    reps = representatives_from_membership(rep_idx, member_of, n=n)
    graph = assemble_graph_from_knn(
        X,
        metric,
        reps,
        knn_idx_f,
        knn_dist_f,
        M,
        stats=stats,
        local_connectivity=local_connectivity,
        beta_multiplicity=beta_multiplicity,
        lambda_backbone=lambda_backbone,
    )
    report.extra["epsilon"] = eps
    report.extra["delta"] = float(net_radius)
    graph.stats.extra["streaming"] = report.to_dict()

    if compute_knn_overlap and n <= 20_000:
        try:
            g_ref, _, _, _ = build_graph(
                X,
                metric,
                n_neighbors=n_neighbors,
                n_landmarks=n_landmarks,
                c_buckets=c_buckets,
                epsilon=eps,
                delta=net_radius,
                dedup=dedup,
                knn_mode=knn_mode,
                seed=seed,
            )
            report.knn_overlap = knn_overlap_jaccard(
                graph.knn_idx,
                g_ref.knn_idx,
                rep_idx_a=graph.reps.rep_idx,
                rep_idx_b=g_ref.reps.rep_idx,
                seed=seed,
            )
            graph.stats.extra["streaming"] = report.to_dict()
            log.info("streaming knn_overlap vs single-pass: %.4f", report.knn_overlap)
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.warning("streaming knn_overlap audit failed: %s", exc)

    log.info(
        "streaming done: R=%d compression=%.4f absorbed=%d spawned=%d "
        "novelty=%d rounds=%d RSS≈%.0f MiB",
        report.n_reps,
        report.compression_ratio,
        report.n_absorbed,
        report.n_spawned,
        report.n_novelty_landmarks,
        report.n_rounds,
        rss_mb(),
    )
    BUILD_PROGRESS.clear()
    return graph, M, assign_top1, assign_topc, report


def build_graph_pyramid_streaming(
    X: torch.Tensor,
    metric: MetricSpec,
    pyramid_scales: int = 3,
    pyramid_rep_ratio: float = PYRAMID_REP_RATIO,
    pyramid_min_reps: int = 256,
    pyramid_coarse_backbone: float = 1.0,
    pyramid_squash: str = "rational_q99",
    **streaming_kwargs: Any,
) -> Tuple[
    List[Graph],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StreamingBuildReport,
]:
    """Streaming fine graph + Galerkin pyramid (same coarsen path as local)."""
    seed = int(streaming_kwargs.get("seed", 0))
    graph0, M, assign_top1, assign_topc, report = build_graph_streaming(
        X, metric, **streaming_kwargs
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
    return graphs, M, assign_top1, assign_topc, report

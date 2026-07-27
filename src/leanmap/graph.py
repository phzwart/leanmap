"""Epsilon-net, kNN, smooth memberships, symmetrisation, and backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .distance import DistanceFn, EuclideanDistance, chunked_cdist
from .landmarks import (
    AnchorAffinity,
    assign_buckets,
    fps_init,
    fps_init_indices,
    fps_init_indices_geodesic,
    poisson_disk_indices_geodesic,
)
from .metrics import MetricSpec
from .utils import get_logger


@dataclass
class GraphStats:
    """Diagnostics emitted by each graph-construction stage."""

    epsilon: float = 0.0
    dedup: bool = True
    frac_exact_zero: float = 0.0
    nn1_deciles: List[float] = field(default_factory=list)
    compression_ratio: float = 1.0
    n_reps: int = 0
    knn_mode: str = "brute"
    knn_recall: Optional[float] = None
    n_no_bracket: int = 0
    n_hit_floor: int = 0
    n_degenerate: int = 0
    n_components_before_backbone: int = 1
    in_degree_deciles: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Representatives:
    """Epsilon-net representatives and CSR cell membership.

    Attributes
    ----------
    rep_idx : (R,) int64
        Index into ``X`` of each representative.
    member_of : (N,) int64
        Representative id (0..R-1) for each raw point.
    weight : (R,) float32
        Number of raw points in each cell.
    offsets : (R+1,) int64
        CSR offsets into ``values``.
    values : (N,) int64
        Raw indices per cell (CSR).
    """

    rep_idx: torch.Tensor
    member_of: torch.Tensor
    weight: torch.Tensor
    offsets: torch.Tensor
    values: torch.Tensor


@dataclass
class Graph:
    """Fuzzy neighbour graph over representatives (training only; not saved).

    Attributes
    ----------
    edges : (E, 2) int64
        Representative indices, upper triangle only.
    weights : (E,) float32
        Memberships in (0, 1].
    reps : Representatives
    knn_idx : (R, k) int64
    stats : GraphStats
    """

    edges: torch.Tensor
    weights: torch.Tensor
    reps: Representatives
    knn_idx: torch.Tensor
    stats: GraphStats


def estimate_epsilon(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    n_sample: int = 10_000,
    quantile: float = 0.01,
    seed: int = 0,
) -> Tuple[float, dict]:
    """Estimate duplicate scale from 1-NN distances on a subsample.

    Primary estimate is ``quantile(nn1, quantile)`` (§4.1). When that collapses
    to ≤0 because exact duplicates dominate the low tail, fall back to the
    median of *strictly positive* 1-NN distances (same idea as leanmap's cull
    radius) so the ε-net still merges near-ties.

    Parameters
    ----------
    X : (N, D) float32
    dist_fn : DistanceFn
    n_sample : int
    quantile : float
    seed : int

    Returns
    -------
    epsilon : float
    diagnostics : dict
    """
    log = get_logger()
    n = X.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    take = min(n_sample, n)
    idx = torch.randperm(n, generator=g)[:take]
    Xs = X[idx]
    Dmat = chunked_cdist(dist_fn, Xs, Xs, out_device=Xs.device)
    assert isinstance(Dmat, torch.Tensor)
    Dmat = Dmat.clone()
    Dmat.fill_diagonal_(float("inf"))
    nn1 = Dmat.min(dim=1).values
    eps = float(torch.quantile(nn1, quantile).item())
    frac_exact_zero = float((nn1 == 0).float().mean().item())
    deciles = [float(torch.quantile(nn1, q).item()) for q in [i / 10 for i in range(1, 10)]]
    used_fallback = False
    if eps <= 0.0:
        # Quantile hit the exact-duplicate mass; use positive 1-NN median.
        pos = nn1[nn1 > 0]
        if pos.numel() > 0:
            eps = float(pos.median().item())
            used_fallback = True
            log.info(
                "epsilon quantile=%.4f was ≤0 (frac_exact_zero=%.3f); "
                "using median positive 1-NN = %.6g",
                quantile,
                frac_exact_zero,
                eps,
            )
        else:
            # All sampled 1-NNs are exact zeros — tiny eps still collapses ties.
            eps = 1e-12
            used_fallback = True
            log.info(
                "all sampled 1-NN distances are 0; using epsilon=%.0e for exact-dup collapse",
                eps,
            )
    if frac_exact_zero > 0.5:
        log.warning(
            "frac_exact_zero=%.3f > 0.5: more than half the subsample are exact "
            "duplicates — check the data pipeline",
            frac_exact_zero,
        )
    return eps, {
        "frac_exact_zero": frac_exact_zero,
        "nn1_deciles": deciles,
        "epsilon": eps,
        "used_positive_median_fallback": used_fallback,
    }


def _epsilon_net_bucket(
    X: torch.Tensor,
    P: torch.Tensor,
    dist_fn: DistanceFn,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[List[int], Dict[int, int]]:
    """Greedy epsilon-net on index set ``P`` (indices into X)."""
    order = P.cpu().numpy().copy()
    rng.shuffle(order)
    reps: List[int] = []
    member_of: Dict[int, int] = {}
    for i in order:
        i = int(i)
        if not reps:
            reps.append(i)
            member_of[i] = 0
            continue
        xi = X[i : i + 1]
        Xr = X[torch.tensor(reps, dtype=torch.int64, device=X.device)]
        dists = dist_fn(xi, Xr)  # (1, |reps|)
        j = int(dists[0].argmin().item())
        dmin = float(dists[0, j].item())
        if dmin <= epsilon:
            member_of[i] = j
        else:
            reps.append(i)
            member_of[i] = len(reps) - 1
    return reps, member_of


def build_representatives(
    X: torch.Tensor,
    assign_top1: torch.Tensor,
    dist_fn: DistanceFn,
    epsilon: float,
    L: int,
    seed: int = 0,
    max_bucket: int = 20_000,
) -> Representatives:
    """Greedy epsilon-net within each top-1 landmark bucket.

    If a bucket exceeds ``max_bucket``, it is split into random sub-blocks,
    netted independently, then merged with one final pass over the union of
    sub-block representatives (approximate; ``epsilon`` is heuristic anyway).

    Parameters
    ----------
    X : (N, D) float32
    assign_top1 : (N,) int64
    dist_fn : DistanceFn
    epsilon : float
    L : int
        Number of landmarks (for logging).
    seed : int
    max_bucket : int

    Returns
    -------
    Representatives
    """
    log = get_logger()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    all_reps: List[int] = []
    member_of = torch.full((n,), -1, dtype=torch.int64)

    if epsilon == 0.0:
        # Fast path: every point is its own representative.
        rep_idx = torch.arange(n, dtype=torch.int64)
        member_of = torch.arange(n, dtype=torch.int64)
        weight = torch.ones(n, dtype=torch.float32)
        offsets = torch.arange(n + 1, dtype=torch.int64)
        values = torch.arange(n, dtype=torch.int64)
        log.info("epsilon=0: skipping deduplication (R == N)")
        return Representatives(rep_idx, member_of, weight, offsets, values)

    for b in range(L):
        P = torch.where(assign_top1 == b)[0]
        if P.numel() == 0:
            continue
        if P.numel() > max_bucket:
            # Approximate: sub-block nets then merge.
            perm = P.cpu().numpy().copy()
            rng.shuffle(perm)
            sub_reps: List[int] = []
            for start in range(0, len(perm), max_bucket):
                block = torch.as_tensor(perm[start : start + max_bucket], dtype=torch.int64)
                reps_b, _ = _epsilon_net_bucket(X, block, dist_fn, epsilon, rng)
                sub_reps.extend(reps_b)
            # Final merge over union of sub-reps, then assign all P to nearest rep
            reps_t = torch.tensor(sub_reps, dtype=torch.int64)
            reps, _ = _epsilon_net_bucket(X, reps_t, dist_fn, epsilon, rng)
            # Assign every point in P to nearest final rep
            Xr = X[torch.tensor(reps, dtype=torch.int64, device=X.device)]
            for s in range(0, P.numel(), 4096):
                e = min(P.numel(), s + 4096)
                d = dist_fn(X[P[s:e]], Xr)
                local = d.argmin(dim=1)
                for ii, j in enumerate(local.tolist()):
                    # Will remap to global rep ids below
                    member_of[int(P[s + ii])] = int(j)
            # Store temporary local ids; remap after collecting global reps
            base = len(all_reps)
            for j, r in enumerate(reps):
                all_reps.append(r)
            # Fix member_of for this bucket: currently local 0..len(reps)-1
            mask = torch.zeros(n, dtype=torch.bool)
            mask[P] = True
            # points in P have local ids; shift
            member_of[P] = member_of[P] + base
        else:
            reps, mem = _epsilon_net_bucket(X, P, dist_fn, epsilon, rng)
            base = len(all_reps)
            for r in reps:
                all_reps.append(r)
            for raw_i, local_j in mem.items():
                member_of[raw_i] = base + local_j

    if (member_of < 0).any():
        raise RuntimeError("build_representatives left unassigned points")

    # Compact: all_reps may have duplicates across buckets (shouldn't for top-1)
    rep_idx = torch.tensor(all_reps, dtype=torch.int64)
    R = rep_idx.shape[0]
    # Build CSR cell_members
    order = torch.argsort(member_of)
    values = order  # raw indices sorted by cell
    counts = torch.bincount(member_of, minlength=R)
    offsets = torch.zeros(R + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    # values must be grouped by cell — argsort gives that
    weight = counts.float()
    compression = float(n) / max(R, 1)
    log.info("compression_ratio = N/R = %.4f (N=%d, R=%d)", compression, n, R)
    if compression < 1.05:
        log.info("deduplication was a no-op (compression_ratio < 1.05)")
    return Representatives(rep_idx, member_of, weight, offsets, values)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _measure_knn_recall(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    knn_idx: torch.Tensor,
    k: int,
    n_probe: int = 2000,
    seed: int = 0,
) -> float:
    R = X_rep.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    take = min(n_probe, R)
    q = torch.randperm(R, generator=g)[:take]
    # brute true knn
    true_vals, true_idx = chunked_cdist(
        dist_fn, X_rep[q], X_rep, topk=k + 1, out_device=X_rep.device
    )
    # drop self
    true_sets = []
    for i, qi in enumerate(q.tolist()):
        row = [int(j) for j in true_idx[i].tolist() if int(j) != qi][:k]
        true_sets.append(set(row))
    overlaps = []
    for i, qi in enumerate(q.tolist()):
        pred = set(int(j) for j in knn_idx[qi].tolist() if int(j) != qi) 
        # knn_idx already excludes self
        pred = set(int(j) for j in knn_idx[qi].tolist())
        if not true_sets[i]:
            overlaps.append(1.0)
        else:
            overlaps.append(len(pred & true_sets[i]) / float(k))
    return float(np.mean(overlaps))


def knn_representatives(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    mode: str = "auto",
    landmarks: Optional[torch.Tensor] = None,
    assign_topc: Optional[torch.Tensor] = None,
    c_search: int = 8,
    metric: Optional[MetricSpec] = None,
    extra_assign_topc: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """kNN among representatives; self excluded; sorted ascending.

    Parameters
    ----------
    X_rep : (R, D) float32
    dist_fn : DistanceFn
        Exact distance (already scale-normalised if desired).
    k : int
    mode : {"auto", "brute", "ivf", "ann"}
    landmarks : (L, D), optional
        Required for ``ivf``.
    assign_topc : (R, c) int64, optional
        Landmark assignment of each representative for ``ivf``.
    c_search : int
    metric : MetricSpec, optional
        Used for ANN eligibility / logging.
    extra_assign_topc : list of (R, c) int64, optional
        Additional per-factor assignments; IVF candidate landmarks are the
        **union** of primary and extra top-c (factored conditioning).

    Returns
    -------
    knn_dist : (R, k) float32
    knn_idx : (R, k) int64
    info : dict
    """
    log = get_logger()
    R = X_rep.shape[0]
    info: dict = {}

    if mode == "auto":
        if R <= 200_000:
            mode = "brute"
        elif (
            metric is not None
            and metric.l2_transform is not None
            and _faiss_available()
        ):
            mode = "ann"
        else:
            mode = "ivf"
    info["mode"] = mode

    if mode == "brute":
        vals, idx = chunked_cdist(
            dist_fn, X_rep, X_rep, topk=k + 1, out_device=X_rep.device
        )
        arange = torch.arange(R, device=X_rep.device)
        self_mask = idx == arange.unsqueeze(1)
        knn_dist = torch.empty(R, k, dtype=torch.float32, device=X_rep.device)
        knn_idx = torch.empty(R, k, dtype=torch.int64, device=X_rep.device)
        for i in range(R):
            m = ~self_mask[i]
            kept_i = idx[i][m][:k]
            kept_v = vals[i][m][:k]
            if kept_i.numel() < k:
                d = dist_fn(X_rep[i : i + 1], X_rep)[0]
                d[i] = float("inf")
                kept_v, kept_i = torch.topk(d, k=k, largest=False)
            knn_idx[i] = kept_i
            knn_dist[i] = kept_v
        info["recall"] = 1.0
        return knn_dist, knn_idx, info

    if mode == "ivf":
        if landmarks is None or assign_topc is None:
            raise ValueError("ivf mode requires landmarks and assign_topc")
        return _knn_ivf(
            X_rep,
            dist_fn,
            k,
            landmarks,
            assign_topc,
            c_search,
            metric,
            info,
            extra_assign_topc=extra_assign_topc,
        )

    if mode == "ann":
        if metric is None or metric.l2_transform is None:
            raise ValueError("ann mode requires metric.l2_transform")
        if not _faiss_available():
            raise ImportError("faiss is required for ann mode")
        return _knn_ann(X_rep, dist_fn, k, metric, info)

    raise ValueError(f"Unknown knn mode {mode!r}")


def union_assign_topc(
    assigns: Sequence[torch.Tensor],
    c: int,
) -> torch.Tensor:
    """Row-wise union of per-factor top-c assignments (joint IVF shortlist).

    Parameters
    ----------
    assigns : list of (N, c_f) int64
    c : int
        Max landmarks kept per row after union (truncated in arbitrary order).

    Returns
    -------
    (N, c') int64 with c' <= c * n_factors, unique per row (padded with -1).
    """
    if not assigns:
        raise ValueError("assigns must be non-empty")
    n = assigns[0].shape[0]
    device = assigns[0].device
    out_rows = []
    for i in range(n):
        ids = torch.cat([a[i, :c] for a in assigns]).unique()
        if ids.numel() > c:
            ids = ids[:c]
        pad = torch.full((c,), -1, dtype=torch.int64, device=device)
        pad[: ids.numel()] = ids
        out_rows.append(pad)
    return torch.stack(out_rows, dim=0)


def _knn_ivf(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    landmarks: torch.Tensor,
    assign_topc: torch.Tensor,
    c_search: int,
    metric: Optional[MetricSpec],
    info: dict,
    extra_assign_topc: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    log = get_logger()
    R = X_rep.shape[0]
    L = landmarks.shape[0]
    # Single-factor: primary assign only (bit-compatible with pre-factor IVF).
    # Multi-factor: union candidates from each factor's own top-c buckets.
    use_joint = bool(extra_assign_topc)
    if use_joint:
        search_assign = assign_topc
        partitions: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        for assign in (assign_topc, *list(extra_assign_topc or ())):
            Lf = int(assign.max().item()) + 1 if assign.numel() else 0
            top1_f = assign[:, 0]
            buckets_f = [torch.where(top1_f == b)[0] for b in range(Lf)]
            partitions.append((assign, buckets_f))
    else:
        search_assign = assign_topc
        partitions = []
    c_search = min(c_search, L, search_assign.shape[1])
    knn_dist = torch.full((R, k), float("inf"), dtype=torch.float32, device=X_rep.device)
    knn_idx = torch.full((R, k), -1, dtype=torch.int64, device=X_rep.device)

    top1 = assign_topc[:, 0]
    buckets: List[torch.Tensor] = [torch.where(top1 == b)[0] for b in range(L)]

    def _cands_for(q_in_b: torch.Tensor, c_use: int) -> torch.Tensor:
        if not use_joint:
            land_ids = search_assign[q_in_b][:, :c_use].reshape(-1).unique()
            land_ids = land_ids[land_ids >= 0]
            cand_list = [
                buckets[int(li)]
                for li in land_ids.tolist()
                if 0 <= int(li) < len(buckets) and buckets[int(li)].numel()
            ]
        else:
            cand_list = []
            for assign_f, buckets_f in partitions:
                c_f = min(c_use, assign_f.shape[1])
                land_ids = assign_f[q_in_b][:, :c_f].reshape(-1).unique()
                land_ids = land_ids[land_ids >= 0]
                for li in land_ids.tolist():
                    if 0 <= int(li) < len(buckets_f) and buckets_f[int(li)].numel():
                        cand_list.append(buckets_f[int(li)])
        if not cand_list:
            return torch.empty(0, dtype=torch.int64, device=X_rep.device)
        return torch.cat(cand_list).unique()

    def run(c_use: int, rows: Optional[torch.Tensor] = None) -> None:
        nonlocal knn_dist, knn_idx
        query_ids = rows if rows is not None else torch.arange(R, device=X_rep.device)
        for b in range(L):
            q_in_b = query_ids[top1[query_ids] == b]
            if q_in_b.numel() == 0:
                continue
            cand = _cands_for(q_in_b, c_use)
            if cand.numel() == 0:
                continue
            d = dist_fn(X_rep[q_in_b], X_rep[cand])
            for qi, q in enumerate(q_in_b.tolist()):
                row = d[qi].clone()
                self_pos = (cand == q).nonzero(as_tuple=False)
                if self_pos.numel():
                    row[self_pos[0, 0]] = float("inf")
                take = min(k, int((row < float("inf")).sum().item()))
                if take <= 0:
                    continue
                vv, ii = torch.topk(row, k=min(k, row.numel()), largest=False)
                valid = vv < float("inf")
                vv, ii = vv[valid][:k], ii[valid][:k]
                knn_dist[q, : vv.numel()] = vv
                knn_idx[q, : ii.numel()] = cand[ii]

    run(c_search)
    need = (knn_idx < 0).any(dim=1) | (knn_dist == float("inf")).any(dim=1)
    n_retry = 0
    c_cur = c_search
    while need.any() and c_cur < L:
        c_cur = min(L, c_cur * 2)
        bad = torch.where(need)[0]
        n_retry += int(bad.numel())
        run(c_cur, bad)
        need = (knn_idx < 0).any(dim=1) | (knn_dist == float("inf")).any(dim=1)
    if need.any():
        bad = torch.where(need)[0]
        for q in bad.tolist():
            d = dist_fn(X_rep[q : q + 1], X_rep)[0]
            d[q] = float("inf")
            vv, ii = torch.topk(d, k=k, largest=False)
            knn_dist[q] = vv
            knn_idx[q] = ii
    assert not (knn_idx < 0).any(), "IVF left rows without k neighbours"
    if n_retry:
        log.info("IVF: %d rows needed c_search retry", n_retry)

    recall = _measure_knn_recall(X_rep, dist_fn, knn_idx, k)
    log.info(
        "IVF knn recall@%d = %.4f%s",
        k,
        recall,
        ""
        if metric is None or metric.l2_exact
        else " (metric.l2_exact=False: transform+index jointly)",
    )
    if recall < 0.9 and c_search < L:
        c2 = min(L, c_search * 2)
        log.info("recall < 0.9; retrying IVF with c_search=%d", c2)
        knn_dist = torch.full((R, k), float("inf"), dtype=torch.float32, device=X_rep.device)
        knn_idx = torch.full((R, k), -1, dtype=torch.int64, device=X_rep.device)
        run(c2)
        need = (knn_idx < 0).any(dim=1)
        if need.any():
            bad = torch.where(need)[0]
            for q in bad.tolist():
                d = dist_fn(X_rep[q : q + 1], X_rep)[0]
                d[q] = float("inf")
                vv, ii = torch.topk(d, k=k, largest=False)
                knn_dist[q] = vv
                knn_idx[q] = ii
        recall = _measure_knn_recall(X_rep, dist_fn, knn_idx, k)
        log.info("IVF knn recall@%d after retry = %.4f", k, recall)
    info["recall"] = recall
    info["n_retry_rows"] = n_retry
    return knn_dist, knn_idx, info



def _knn_ann(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    metric: MetricSpec,
    info: dict,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    import faiss

    log = get_logger()
    assert metric.l2_transform is not None
    Xt = metric.l2_transform(X_rep).detach().cpu().numpy().astype(np.float32)
    R, d = Xt.shape
    k_over = min(R, 4 * k)
    # IndexIVFFlat when R large enough, else HNSW
    if R >= 1000:
        nlist = min(int(np.sqrt(R)), R // 10)
        nlist = max(nlist, 1)
        quant = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, nlist)
        index.train(Xt)
        index.add(Xt)
        index.nprobe = min(32, nlist)
    else:
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(Xt)
        nlist = 1

    _, cand = index.search(Xt, k_over)
    knn_dist = torch.empty(R, k, dtype=torch.float32, device=X_rep.device)
    knn_idx = torch.empty(R, k, dtype=torch.int64, device=X_rep.device)
    for i in range(R):
        cidx = [int(j) for j in cand[i].tolist() if j >= 0 and j != i]
        if len(cidx) < k:
            # pad with brute
            d = dist_fn(X_rep[i : i + 1], X_rep)[0]
            d[i] = float("inf")
            vv, ii = torch.topk(d, k=k, largest=False)
            knn_dist[i] = vv
            knn_idx[i] = ii
            continue
        C = X_rep[torch.tensor(cidx, dtype=torch.int64, device=X_rep.device)]
        d = dist_fn(X_rep[i : i + 1], C)[0]
        vv, ii = torch.topk(d, k=min(k, d.numel()), largest=False)
        knn_dist[i, : vv.numel()] = vv
        knn_idx[i, : ii.numel()] = torch.tensor(
            [cidx[int(j)] for j in ii.tolist()], dtype=torch.int64, device=X_rep.device
        )

    recall = _measure_knn_recall(X_rep, dist_fn, knn_idx, k)
    log.info(
        "ANN knn recall@%d = %.4f%s",
        k,
        recall,
        ""
        if metric.l2_exact
        else " (metric.l2_exact=False: transform+index jointly)",
    )
    if recall < 0.9 and isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = min(nlist, index.nprobe * 4)
        log.info("recall < 0.9; retrying ANN with nprobe=%d", index.nprobe)
        _, cand = index.search(Xt, k_over)
        for i in range(R):
            cidx = [int(j) for j in cand[i].tolist() if j >= 0 and j != i]
            if len(cidx) < k:
                d = dist_fn(X_rep[i : i + 1], X_rep)[0]
                d[i] = float("inf")
                vv, ii = torch.topk(d, k=k, largest=False)
                knn_dist[i] = vv
                knn_idx[i] = ii
                continue
            C = X_rep[torch.tensor(cidx, dtype=torch.int64, device=X_rep.device)]
            d = dist_fn(X_rep[i : i + 1], C)[0]
            vv, ii = torch.topk(d, k=min(k, d.numel()), largest=False)
            knn_dist[i, : vv.numel()] = vv
            knn_idx[i, : ii.numel()] = torch.tensor(
                [cidx[int(j)] for j in ii.tolist()],
                dtype=torch.int64,
                device=X_rep.device,
            )
        recall = _measure_knn_recall(X_rep, dist_fn, knn_idx, k)
        log.info("ANN knn recall@%d after retry = %.4f", k, recall)
    info["recall"] = recall
    return knn_dist, knn_idx, info


def smooth_knn(
    knn_dist: torch.Tensor,
    local_connectivity: int = 1,
    target: Optional[float] = None,
    n_iter: int = 64,
    tol: float = 1e-5,
    min_sigma_frac: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Solve per-row bandwidth ``sigma`` and local connectivity ``rho``.

    Parameters
    ----------
    knn_dist : (R, k) float32
        Self already excluded.
    local_connectivity : int
    target : float | None
        Defaults to ``log2(k)``.
    n_iter, tol, min_sigma_frac : numeric

    Returns
    -------
    rho : (R,) float32
    sigma : (R,) float32
    diagnostics : dict
    """
    log = get_logger()
    R, k = knn_dist.shape
    if target is None:
        target = float(np.log2(k))
    # rho_i = local_connectivity-th smallest strictly positive entry
    rho = torch.zeros(R, dtype=torch.float32, device=knn_dist.device)
    for i in range(R):
        pos = knn_dist[i][knn_dist[i] > 0]
        if pos.numel() == 0:
            rho[i] = 0.0
        elif pos.numel() < local_connectivity:
            rho[i] = pos.max()
        else:
            # local_connectivity-th smallest (1-indexed) → index lc-1
            vals, _ = torch.sort(pos)
            rho[i] = vals[local_connectivity - 1]

    # Vectorised bisection for sigma
    delta = torch.clamp(knn_dist - rho.unsqueeze(1), min=0.0)
    lo = torch.full((R,), 1e-12, dtype=torch.float32, device=knn_dist.device)
    hi = torch.ones(R, dtype=torch.float32, device=knn_dist.device)

    def g(sig: torch.Tensor) -> torch.Tensor:
        return torch.exp(-delta / sig.unsqueeze(1)).sum(dim=1)

    n_no_bracket = 0
    for _ in range(64):
        need = g(hi) < target
        if not need.any():
            break
        hi = torch.where(need, hi * 2.0, hi)
    n_no_bracket = int((g(hi) < target).sum().item())

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        too_big = gm > target
        hi = torch.where(too_big, mid, hi)
        lo = torch.where(too_big, lo, mid)
        if float((gm - target).abs().max().item()) < tol:
            break
    sigma = 0.5 * (lo + hi)
    mean_rho = float(rho.mean().item()) if R else 0.0
    floor = min_sigma_frac * max(mean_rho, 1e-12)
    hit = sigma < floor
    n_hit_floor = int(hit.sum().item())
    sigma = torch.clamp(sigma, min=floor)

    # degenerate: all k neighbours within epsilon of rho
    # SPEC: "within epsilon of rho_i" — use a relative tolerance on rho
    # SPEC-AMBIGUITY: epsilon here means a small numerical margin, not §4.1 epsilon
    margin = 1e-6 + 1e-3 * rho.unsqueeze(1)
    n_degenerate = int(((knn_dist - rho.unsqueeze(1)).abs() <= margin).all(dim=1).sum().item())
    # Also count rows where all distances ≈ rho (no spread)
    n_degenerate = int(
        ((torch.clamp(knn_dist - rho.unsqueeze(1), min=0.0) <= margin).all(dim=1)).sum().item()
    )

    log.info(
        "smooth_knn: n_no_bracket=%d n_hit_floor=%d n_degenerate=%d",
        n_no_bracket,
        n_hit_floor,
        n_degenerate,
    )
    if R > 0 and n_degenerate / R > 0.05:
        log.warning(
            "n_degenerate/R = %.3f > 0.05 — raise local_connectivity or epsilon "
            "(duplicate-dominated data)",
            n_degenerate / R,
        )
    return rho, sigma, {
        "n_no_bracket": n_no_bracket,
        "n_hit_floor": n_hit_floor,
        "n_degenerate": n_degenerate,
    }


def landmark_backbone(
    M: torch.Tensor,
    dist_fn: DistanceFn,
    rep_idx: torch.Tensor,
    X: torch.Tensor,
    lambda_bb: float = 0.01,
) -> sparse.coo_matrix:
    """MST over landmarks, mapped to nearest representatives.

    Parameters
    ----------
    M : (L, D) float32
    dist_fn : DistanceFn
    rep_idx : (R,) int64
    X : (N, D) float32
    lambda_bb : float

    Returns
    -------
    scipy.sparse.coo_matrix of shape (R, R)
    """
    L = M.shape[0]
    R = rep_idx.shape[0]
    if L <= 1:
        return sparse.coo_matrix((R, R), dtype=np.float32)
    D_ll = dist_fn(M, M).detach().cpu().numpy()
    mst = minimum_spanning_tree(sparse.csr_matrix(D_ll)).tocoo()
    X_rep = X[rep_idx]
    # Map each landmark to nearest representative
    _, nearest = chunked_cdist(dist_fn, M, X_rep, topk=1, out_device=X.device)
    nearest = nearest[:, 0].cpu().numpy()
    rows, cols, data = [], [], []
    for i, j in zip(mst.row, mst.col):
        ri, rj = int(nearest[i]), int(nearest[j])
        if ri == rj:
            continue
        rows.append(ri)
        cols.append(rj)
        data.append(lambda_bb)
        rows.append(rj)
        cols.append(ri)
        data.append(lambda_bb)
    return sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)), shape=(R, R)
    )


def build_graph(
    X: torch.Tensor,
    metric: MetricSpec,
    n_neighbors: int = 15,
    n_landmarks: int = 256,
    c_buckets: int = 8,
    epsilon: Optional[float] = None,
    dedup: bool = True,
    local_connectivity: int = 1,
    beta_multiplicity: float = 0.5,
    hub_correction: bool = False,
    lambda_backbone: float = 0.01,
    knn_mode: str = "auto",
    c_search: int = 8,
    seed: int = 0,
    extra_ivf_anchors: Optional[
        Sequence[Tuple[Any, DistanceFn, torch.Tensor]]
    ] = None,
    fps_view: Optional[Any] = None,
    fps_view_metric: Optional[DistanceFn] = None,
    fps_geodesic: bool = False,
    fps_geodesic_k: Optional[int] = None,
    fps_poisson: bool = False,
) -> Tuple[Graph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full graph pipeline on the training split.

    Parameters
    ----------
    X : (N, D) float32
    metric : MetricSpec
        Should already have ``natural_scale`` fitted. Used for edge distances,
        ε-net radii, and (by default) landmark FPS.
    dedup : bool
        If True (default), collapse near-duplicates with an ε-net before kNN.
        If False, force ``epsilon=0`` and keep ``R == N``.
    extra_ivf_anchors : sequence of (view_fn, view_metric, M_f), optional
        Additional factor anchors for joint IVF shortlists (union of per-factor
        top-c buckets).
    fps_view : callable, optional
        If set, landmark FPS runs in this view (e.g. normalized roots). Selected
        rows of ambient ``X`` become graph landmarks; the IVF inverted index is
        built in the same view (knn tree). Exact kNN distances still use
        ``metric``.
    fps_view_metric : DistanceFn, optional
        Metric on ``fps_view`` outputs (default: Euclidean).

    Returns
    -------
    graph : Graph
    M : (L, D) landmarks
    assign_top1 : (N,) int64
    assign_topc : (N, c) int64
    """
    log = get_logger()
    dist_fn: DistanceFn = metric  # MetricSpec.__call__ applies natural_scale
    stats = GraphStats(dedup=bool(dedup))

    if not metric.is_true_metric:
        log.info(
            "metric %s is not a true metric: epsilon-net cell diameters may "
            "exceed 2ε — consider lowering epsilon",
            metric.name,
        )

    # 4.1 epsilon / dedup switch
    if not dedup:
        eps = 0.0
        stats.epsilon = 0.0
        log.info("dedup=False: skipping ε-net (R == N)")
    elif epsilon is None:
        eps, diag = estimate_epsilon(X, dist_fn, seed=seed)
        stats.epsilon = eps
        stats.frac_exact_zero = diag["frac_exact_zero"]
        stats.nn1_deciles = diag["nn1_deciles"]
        stats.extra["used_positive_median_fallback"] = diag.get(
            "used_positive_median_fallback", False
        )
    else:
        eps = float(epsilon)
        stats.epsilon = eps
        if eps == 0.0:
            log.info("epsilon=0: skipping ε-net (R == N)")

    # 4.2 landmarks — FPS in scoring space, or in fps_view (conditioning) space
    ivf_view_assign: Optional[Tuple[Any, DistanceFn, torch.Tensor]] = None
    if fps_view is not None:
        v_metric = fps_view_metric if fps_view_metric is not None else EuclideanDistance()
        V = fps_view(X)
        idx = fps_init_indices(V, v_metric, n_landmarks, seed=seed)
        M = X[idx].contiguous()
        M_view = V[idx].contiguous()
        # ε-net / multiplicity assignment still uses scoring metric on ambient rows
        assign_top1, assign_topc = assign_buckets(X, M, dist_fn, c=c_buckets)
        ivf_view_assign = (fps_view, v_metric, M_view)
        log.info(
            "landmark FPS in view space D_f=%d (IVF knn tree); "
            "edge distances use scoring metric %s",
            M_view.shape[1],
            getattr(metric, "name", metric),
        )
    else:
        if fps_poisson:
            gk = fps_geodesic_k if fps_geodesic_k is not None else n_neighbors
            idx = poisson_disk_indices_geodesic(
                X, dist_fn, n_landmarks, n_neighbors=gk, seed=seed
            )
            M = X[idx].contiguous()
            log.info(
                "landmarks: geodesic Poisson-disk (blue-noise, k=%d) -> %d picked",
                gk,
                M.shape[0],
            )
        elif fps_geodesic:
            gk = fps_geodesic_k if fps_geodesic_k is not None else n_neighbors
            idx = fps_init_indices_geodesic(
                X, dist_fn, n_landmarks, n_neighbors=gk, seed=seed
            )
            M = X[idx].contiguous()
            log.info("landmark FPS: geodesic (kNN shortest-path, k=%d)", gk)
        else:
            M = fps_init(X, dist_fn, n_landmarks, seed=seed)
        assign_top1, assign_topc = assign_buckets(X, M, dist_fn, c=c_buckets)
    L = M.shape[0]

    # 4.3 representatives
    reps = build_representatives(X, assign_top1, dist_fn, eps, L, seed=seed)
    stats.n_reps = int(reps.rep_idx.shape[0])
    stats.compression_ratio = float(X.shape[0]) / max(stats.n_reps, 1)
    X_rep = X[reps.rep_idx]

    # Assign landmarks for representatives (for IVF)
    if ivf_view_assign is not None:
        view_fn, v_metric, M_view = ivf_view_assign
        V_rep = view_fn(X_rep)
        rep_top1, rep_topc = assign_buckets(
            V_rep, M_view, v_metric, c=min(c_buckets, M_view.shape[0])
        )
        # Primary IVF buckets live in view/landmark index space of size L
        landmarks_for_ivf = M_view
    else:
        rep_top1, rep_topc = assign_buckets(X_rep, M, dist_fn, c=c_buckets)
        landmarks_for_ivf = M

    extra_assign_topc: Optional[List[torch.Tensor]] = None
    if extra_ivf_anchors:
        extra_assign_topc = []
        for view_fn, view_metric, M_f in extra_ivf_anchors:
            v_rep = view_fn(X_rep)
            _, topc_f = assign_buckets(
                v_rep,
                M_f.to(device=X_rep.device, dtype=X_rep.dtype),
                view_metric,
                c=min(c_buckets, M_f.shape[0]),
            )
            extra_assign_topc.append(topc_f)

    # Prefer IVF when we built a landmark tree in a conditioning view
    mode = knn_mode
    if mode == "auto" and ivf_view_assign is not None:
        mode = "ivf"

    # 4.4 kNN — exact distances via scoring metric; IVF shortlist via landmark tree
    knn_dist, knn_idx, knn_info = knn_representatives(
        X_rep,
        dist_fn,
        k=n_neighbors,
        mode=mode,
        landmarks=landmarks_for_ivf,
        assign_topc=rep_topc,
        c_search=c_search,
        metric=metric,
        extra_assign_topc=extra_assign_topc,
    )
    stats.knn_mode = knn_info.get("mode", knn_mode)
    stats.knn_recall = knn_info.get("recall")

    # 4.5 smooth
    rho, sigma, sdiag = smooth_knn(knn_dist, local_connectivity=local_connectivity)
    stats.n_no_bracket = sdiag["n_no_bracket"]
    stats.n_hit_floor = sdiag["n_hit_floor"]
    stats.n_degenerate = sdiag["n_degenerate"]

    # 4.6 memberships + fuzzy union (NOT mutual-kNN — orphans sparse regions)
    R = X_rep.shape[0]
    p = torch.exp(-torch.clamp(knn_dist - rho.unsqueeze(1), min=0.0) / sigma.unsqueeze(1))
    rows = torch.arange(R, device=X.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    cols = knn_idx.reshape(-1)
    vals = p.reshape(-1)
    P = sparse.coo_matrix(
        (
            vals.detach().cpu().numpy().astype(np.float32),
            (rows.cpu().numpy(), cols.cpu().numpy()),
        ),
        shape=(R, R),
    ).tocsr()
    # Fuzzy union: P + P.T - P.multiply(P.T). Do NOT use mutual-kNN (P.multiply(P.T)).
    PT = P.T.tocsr()
    P_sym = P + PT - P.multiply(PT)

    # 4.7 multiplicity
    w = reps.weight.cpu().numpy()
    P_sym = P_sym.tocoo()
    reweight = (w[P_sym.row] * w[P_sym.col]) ** beta_multiplicity
    P_sym.data = P_sym.data * reweight
    P_sym = P_sym.tocsr()
    if P_sym.data.size:
        P_sym.data /= P_sym.data.max()

    # degree deciles
    deg = np.asarray(P_sym.sum(axis=1)).ravel()
    stats.in_degree_deciles = [float(np.quantile(deg, q)) for q in [i / 10 for i in range(1, 10)]]
    log.info("in-degree deciles: %s", [f"{v:.4f}" for v in stats.in_degree_deciles])

    # 4.8 hub correction
    if hub_correction:
        deg = np.asarray(P_sym.sum(axis=1)).ravel()
        P_sym = P_sym.tocoo()
        P_sym.data = P_sym.data / np.sqrt(deg[P_sym.row] * deg[P_sym.col] + 1e-12)
        P_sym = P_sym.tocsr()
        if P_sym.data.size:
            P_sym.data /= P_sym.data.max()

    # 4.9 backbone
    n_comp, _ = connected_components(P_sym, directed=False)
    stats.n_components_before_backbone = int(n_comp)
    log.info("connected components before backbone: %d", n_comp)
    bb = landmark_backbone(M, dist_fn, reps.rep_idx, X, lambda_bb=lambda_backbone)
    # maximum merge
    P_sym = P_sym.tocsr()
    bb = bb.tocsr()
    # elementwise maximum of two sparse matrices
    diff = bb - P_sym
    diff.data = np.maximum(diff.data, 0)
    P_sym = P_sym + diff
    if P_sym.data.size:
        mx = P_sym.data.max()
        if mx > 0:
            P_sym.data /= mx

    # 4.10 edges upper triangle
    P_sym = P_sym.tocoo()
    mask = P_sym.row < P_sym.col
    edges = torch.stack(
        [
            torch.as_tensor(P_sym.row[mask], dtype=torch.int64),
            torch.as_tensor(P_sym.col[mask], dtype=torch.int64),
        ],
        dim=1,
    )
    weights = torch.as_tensor(P_sym.data[mask], dtype=torch.float32)
    # Drop zeros
    keep = weights > 0
    edges, weights = edges[keep], weights[keep]

    graph = Graph(edges=edges, weights=weights, reps=reps, knn_idx=knn_idx.cpu(), stats=stats)
    return graph, M, assign_top1, assign_topc


def _coarsen_graph(
    graph_l: Graph,
    X: torch.Tensor,
    dist_fn: DistanceFn,
    target_reps: int,
    seed: int = 0,
) -> Optional[Graph]:
    """Coarsen a fuzzy graph by Galerkin edge contraction (multiscale pyramid).

    Picks ``target_reps`` coarse representatives among ``graph_l``'s reps (FPS on
    rep coordinates), assigns every level-l rep to its nearest coarse rep, and
    aggregates level-l edges between coarse cells (coarse weight = sum of
    crossing fine-edge weights; self-loops dropped; normalized max->1). Composed
    CSR cells map each coarse rep to the union of its raw members.

    A coarse edge exists only where fine edges already cross, so connectivity is
    preserved (no Isomap short-circuits across manifold folds) and thin bridges
    are strengthened rather than invented.

    Returns ``None`` if coarsening is not possible (``target_reps >= R_l``, too
    few reps, or no cross-cell edges remain).
    """
    reps_l = graph_l.reps
    R_l = int(reps_l.rep_idx.shape[0])
    if target_reps >= R_l or R_l <= 2:
        return None

    X_rep = X[reps_l.rep_idx]
    # coarse reps: FPS among level-l reps (well-spread cover of the manifold)
    coarse_local = fps_init_indices(X_rep, dist_fn, target_reps, seed=seed).cpu()
    Rc = int(coarse_local.shape[0])
    # map each level-l rep -> nearest coarse rep (0..Rc-1); coarse reps map to self
    mapping, _ = assign_buckets(X_rep, X_rep[coarse_local], dist_fn, c=1)
    mapping = mapping.cpu().to(torch.int64)  # (R_l,)

    # --- composed CSR cells: raw point -> coarse rep ---
    member_of_l = reps_l.member_of.cpu().to(torch.int64)  # (N,) in 0..R_l-1
    member_of_c = mapping[member_of_l]  # (N,) in 0..Rc-1
    counts = torch.bincount(member_of_c, minlength=Rc)
    order = torch.argsort(member_of_c)
    values = order.to(torch.int64)  # raw indices grouped by coarse cell
    offsets = torch.zeros(Rc + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    weight = counts.float()
    rep_idx_c = reps_l.rep_idx.cpu()[coarse_local].to(torch.int64)
    reps_c = Representatives(rep_idx_c, member_of_c, weight, offsets, values)

    # --- Galerkin edge aggregation over coarse cells ---
    e = graph_l.edges.cpu().to(torch.int64)  # (E, 2) in 0..R_l-1
    w = graph_l.weights.cpu().to(torch.float64)
    a = mapping[e[:, 0]]
    b = mapping[e[:, 1]]
    keep = a != b  # drop self-loops (edges internal to a coarse cell)
    a, b, w = a[keep], b[keep], w[keep]
    if a.numel() == 0:
        return None
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    key = lo * Rc + hi
    uniq, inv = torch.unique(key, return_inverse=True)
    wsum = torch.zeros(uniq.shape[0], dtype=torch.float64)
    wsum.scatter_add_(0, inv, w)
    lo_u = torch.div(uniq, Rc, rounding_mode="floor").to(torch.int64)
    hi_u = (uniq % Rc).to(torch.int64)
    edges_c = torch.stack([lo_u, hi_u], dim=1)
    # Aggregated (summed) crossing weights are heavy-tailed: one wide bridge can
    # be orders of magnitude stronger than a thin one. Dividing by the global max
    # would squash the bulk toward 0, so most long-range edges would barely
    # attract. Normalize by a high quantile and clamp to 1 so typical coarse
    # edges keep a meaningful membership while the strongest still saturate.
    if wsum.numel() > 0:
        scale = float(torch.quantile(wsum, 0.99)) if wsum.numel() > 1 else float(wsum.max())
        scale = scale if scale > 0 else float(wsum.max())
        weights_c = torch.clamp(wsum / max(scale, 1e-12), max=1.0).to(torch.float32)
    else:
        weights_c = wsum.to(torch.float32)

    # diagnostics: connectivity + degree (unweighted degree = 2E/R; weighted
    # degree is the sum of memberships and is expected to be < 1 per node)
    A = sparse.coo_matrix(
        (
            np.ones(edges_c.shape[0] * 2, dtype=np.float64),
            (
                np.concatenate([lo_u.numpy(), hi_u.numpy()]),
                np.concatenate([hi_u.numpy(), lo_u.numpy()]),
            ),
        ),
        shape=(Rc, Rc),
    )
    n_comp = int(connected_components(A, directed=False)[0])
    deg = torch.zeros(Rc, dtype=torch.float64)
    deg.scatter_add_(0, lo_u, weights_c.to(torch.float64))
    deg.scatter_add_(0, hi_u, weights_c.to(torch.float64))
    stats = GraphStats(dedup=graph_l.stats.dedup)
    stats.n_reps = Rc
    stats.extra = {
        "n_edges": int(edges_c.shape[0]),
        "n_components": n_comp,
        "mean_unweighted_degree": float(2.0 * edges_c.shape[0] / max(Rc, 1)),
        "mean_weighted_degree": float(deg.mean()),
        "median_edge_weight": float(weights_c.median()) if weights_c.numel() else 0.0,
    }
    knn_idx_c = torch.empty((Rc, 0), dtype=torch.int64)
    return Graph(edges=edges_c, weights=weights_c, reps=reps_c, knn_idx=knn_idx_c, stats=stats)


def _add_coarse_backbone(
    graph: Graph,
    X: torch.Tensor,
    dist_fn: DistanceFn,
    weight: float,
) -> Graph:
    """Bridge disconnected regions of ``graph`` with strong spanning edges.

    Adds the minimal set of edges (Kruskal over the graph metric, ``n_comp - 1``
    of them) needed to tie the coarse level into one component, each at a fixed
    high ``weight``. If the level is already connected this is a no-op.

    Bridges only ever join *different* components, so no existing edge weight is
    ever modified. An earlier version laid a full ``R-1``-edge MST over the reps
    and max-merged it, which overwrote hundreds of already-present edges to the
    maximum weight (on a level whose median weight is ~0.02) and imposed an
    arbitrary tree geometry on the global layout -- badly distorting geodesic
    fidelity and density preservation even when nothing was disconnected.
    """
    log = get_logger()
    rep_idx = graph.reps.rep_idx
    R = int(rep_idx.shape[0])
    if R <= 1 or weight <= 0:
        return graph

    ex_e = graph.edges.cpu().numpy().astype(np.int64)
    ex_w = graph.weights.cpu().numpy().astype(np.float64)
    A_ex = sparse.coo_matrix(
        (
            np.ones(ex_e.shape[0] * 2),
            (
                np.concatenate([ex_e[:, 0], ex_e[:, 1]]),
                np.concatenate([ex_e[:, 1], ex_e[:, 0]]),
            ),
        ),
        shape=(R, R),
    )
    n_comp, labels = connected_components(A_ex, directed=False)
    if n_comp <= 1:
        stats = graph.stats
        stats.extra = {
            **(stats.extra or {}),
            "coarse_backbone_w": 0.0,
            "coarse_backbone_bridges": 0,
            "coarse_backbone_skipped": True,
        }
        log.info(
            "coarse backbone: level already connected (1 component); "
            "no bridges added"
        )
        return Graph(
            edges=graph.edges,
            weights=graph.weights,
            reps=graph.reps,
            knn_idx=graph.knn_idx,
            stats=stats,
        )

    Xr = X[rep_idx]
    D = dist_fn(Xr, Xr).detach().cpu().numpy().astype(np.float64)
    # Union-find seeded with the existing components; only merging edges are kept.
    parent = list(range(n_comp))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bridges: list[tuple[int, int]] = []
    # MST over the complete rep-distance graph supplies cheap candidates; any
    # residual components are closed by explicit nearest cross-set search.
    mst = minimum_spanning_tree(sparse.csr_matrix(D)).tocoo()
    cand = sorted(zip(mst.data, mst.row, mst.col), key=lambda t: t[0])
    for _, u, v in cand:
        ru, rv = find(int(labels[u])), find(int(labels[v]))
        if ru != rv:
            parent[ru] = rv
            bridges.append((int(u), int(v)))
    n_sets = len({find(c) for c in range(n_comp)})
    while n_sets > 1:
        set_of = np.array([find(int(labels[i])) for i in range(R)])
        Dm = np.where(set_of[:, None] == set_of[None, :], np.inf, D)
        u, v = np.unravel_index(int(np.argmin(Dm)), Dm.shape)
        if not np.isfinite(Dm[u, v]):
            break
        parent[find(int(labels[u]))] = find(int(labels[v]))
        bridges.append((int(u), int(v)))
        n_sets = len({find(c) for c in range(n_comp)})

    if not bridges:
        return graph
    bb = np.asarray(bridges, dtype=np.int64)
    bb_lo = np.minimum(bb[:, 0], bb[:, 1])
    bb_hi = np.maximum(bb[:, 0], bb[:, 1])
    lo = np.concatenate([ex_e[:, 0], bb_lo])
    hi = np.concatenate([ex_e[:, 1], bb_hi])
    w = np.concatenate([ex_w, np.full(bb_lo.shape[0], float(weight))])
    edges = torch.from_numpy(np.stack([lo, hi], axis=1)).to(torch.int64)
    weights = torch.from_numpy(w).to(torch.float32)
    A = sparse.coo_matrix(
        (
            np.ones(edges.shape[0] * 2),
            (np.concatenate([lo, hi]), np.concatenate([hi, lo])),
        ),
        shape=(R, R),
    )
    n_comp_after = int(connected_components(A, directed=False)[0])
    stats = graph.stats
    stats.extra = {
        **(stats.extra or {}),
        "n_edges": int(edges.shape[0]),
        "n_components": n_comp_after,
        "coarse_backbone_w": float(weight),
        "coarse_backbone_bridges": int(bb_lo.shape[0]),
        "coarse_backbone_skipped": False,
    }
    log.info(
        "coarse backbone: %d component(s) -> %d via %d bridge edge(s) at w=%.2f",
        n_comp,
        n_comp_after,
        int(bb_lo.shape[0]),
        weight,
    )
    return Graph(edges=edges, weights=weights, reps=graph.reps, knn_idx=graph.knn_idx, stats=stats)


def build_graph_pyramid(
    X: torch.Tensor,
    metric: MetricSpec,
    pyramid_scales: int = 3,
    pyramid_rep_ratio: float = 4.0,
    pyramid_min_reps: int = 256,
    pyramid_coarse_backbone: float = 1.0,
    **build_graph_kwargs: Any,
) -> Tuple[List[Graph], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-scale graph pyramid: fine graph + Galerkin-coarsened coarse levels.

    Level 0 is the standard :func:`build_graph` output. Each coarser level
    contracts the previous graph to ``~R0 / pyramid_rep_ratio**l`` reps (floored
    at ``pyramid_min_reps``) via :func:`_coarsen_graph`. Coarse levels supply the
    medium/long-range attraction that anchors global (geodesic) structure.

    Default ``pyramid_coarse_backbone=1.0`` bridges any disconnected regions of
    the coarsest level with strong spanning edges; it is a no-op when that level
    is already connected. Pass ``0`` to disable; pass ``pyramid_scales=0`` for a
    single-scale (legacy) graph.

    Note that depth is capped by ``pyramid_min_reps``: coarsening stops once a
    level reaches it, so the number of levels can be smaller than
    ``pyramid_scales + 1`` (with the defaults, 4 levels need ``N`` of roughly
    17k or more). ``pyramid_level_weights`` is matched to the levels actually
    built -- see :func:`leanmap.train.fit`.

    Returns ``(graphs, M, assign_top1, assign_topc)`` with ``graphs[0]`` finest.
    """
    log = get_logger()
    graph0, M, assign_top1, assign_topc = build_graph(X, metric, **build_graph_kwargs)
    graphs: List[Graph] = [graph0]
    dist_fn: DistanceFn = metric
    seed = int(build_graph_kwargs.get("seed", 0))
    R0 = int(graph0.reps.rep_idx.shape[0])
    log.info("pyramid level 0: R=%d edges=%d", R0, int(graph0.edges.shape[0]))
    prev = graph0
    for level in range(1, max(pyramid_scales, 0) + 1):
        target = max(int(round(R0 / (pyramid_rep_ratio ** level))), pyramid_min_reps)
        if target >= int(prev.reps.rep_idx.shape[0]):
            log.info("pyramid: stopping at level %d (target %d >= prev R)", level, target)
            break
        g = _coarsen_graph(prev, X, dist_fn, target, seed=seed + level)
        if g is None:
            log.info("pyramid: coarsening returned None at level %d; stopping", level)
            break
        graphs.append(g)
        log.info(
            "pyramid level %d: R=%d edges=%d components=%d avg_degree=%.1f "
            "median_w=%.3f mean_wdeg=%.3f",
            level,
            g.stats.n_reps,
            g.stats.extra.get("n_edges", 0),
            g.stats.extra.get("n_components", -1),
            g.stats.extra.get("mean_unweighted_degree", 0.0),
            g.stats.extra.get("median_edge_weight", 0.0),
            g.stats.extra.get("mean_weighted_degree", 0.0),
        )
        prev = g
        if g.stats.n_reps <= pyramid_min_reps:
            break
    if pyramid_coarse_backbone > 0 and len(graphs) > 1:
        graphs[-1] = _add_coarse_backbone(
            graphs[-1], X, metric, pyramid_coarse_backbone
        )
    log.info("pyramid built: %d level(s)", len(graphs))
    return graphs, M, assign_top1, assign_topc

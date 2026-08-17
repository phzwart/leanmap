"""Epsilon-net, kNN, smooth memberships, symmetrisation, and backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from ..distance import DistanceFn, EuclideanDistance, chunked_cdist
from ..landmarks import (
    AnchorAffinity,
    assign_buckets,
    fps_init,
    fps_init_indices,
    fps_init_indices_geodesic,
    poisson_disk_indices_geodesic,
)
from ..config import (
    BETA_MULTIPLICITY,
    C_BUCKETS,
    C_SEARCH,
    LAMBDA_BACKBONE,
    PYRAMID_REP_RATIO,
)
from ..metrics import MetricSpec
from ..utils import get_logger, rss_mb


@dataclass
class GraphStats:
    """Diagnostics emitted by each graph-construction stage."""

    epsilon: float = 0.0
    delta: float = 0.0
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


PrecomputedKNN = Tuple[torch.Tensor, torch.Tensor]  # (knn_idx, knn_dist) or see validate


def validate_precomputed_knn(
    knn_idx: torch.Tensor,
    knn_dist: torch.Tensor,
    n: int,
    *,
    n_neighbors: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Validate a caller-supplied kNN and optionally truncate to ``n_neighbors``.

    Parameters
    ----------
    knn_idx : (N, k) int64
        Neighbor indices into the same row space as the training matrix.
    knn_dist : (N, k) float
        Non-negative finite edge distances (any metric; need not match ambient).
    n : int
        Expected number of rows (``R`` = number of representatives; equals ``N``
        when ``dedup=False``).
    n_neighbors : int, optional
        If set and ``k > n_neighbors``, truncate columns to ``n_neighbors``.
        If ``k < n_neighbors``, the supplied ``k`` is kept.

    Returns
    -------
    knn_idx, knn_dist, k
        Contiguous tensors on CPU (float32 distances, int64 indices) and width.
    """
    if not isinstance(knn_idx, torch.Tensor):
        knn_idx = torch.as_tensor(knn_idx)
    if not isinstance(knn_dist, torch.Tensor):
        knn_dist = torch.as_tensor(knn_dist)
    if knn_idx.ndim != 2 or knn_dist.ndim != 2:
        raise ValueError(
            f"precomputed_knn requires 2-D (N, k) tensors; got idx={tuple(knn_idx.shape)} "
            f"dist={tuple(knn_dist.shape)}"
        )
    if knn_idx.shape != knn_dist.shape:
        raise ValueError(
            f"precomputed_knn idx/dist shape mismatch: {tuple(knn_idx.shape)} vs "
            f"{tuple(knn_dist.shape)}"
        )
    if knn_idx.shape[0] != n:
        raise ValueError(
            f"precomputed_knn has {knn_idx.shape[0]} rows but expected R={n}"
        )
    k = int(knn_idx.shape[1])
    if k < 1:
        raise ValueError("precomputed_knn must have k >= 1 neighbors")
    if n_neighbors is not None and k > int(n_neighbors):
        k = int(n_neighbors)
        knn_idx = knn_idx[:, :k]
        knn_dist = knn_dist[:, :k]
    knn_idx = knn_idx.detach().to(dtype=torch.int64, device="cpu").contiguous()
    knn_dist = knn_dist.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not torch.isfinite(knn_dist).all():
        raise ValueError("precomputed_knn distances must be finite")
    if (knn_dist < 0).any():
        raise ValueError("precomputed_knn distances must be non-negative")
    if (knn_idx < 0).any() or (knn_idx >= n).any():
        raise ValueError(f"precomputed_knn indices must lie in [0, {n})")
    rows = torch.arange(n, dtype=torch.int64).unsqueeze(1).expand_as(knn_idx)
    if (knn_idx == rows).any():
        raise ValueError("precomputed_knn must not include self-neighbors")
    return knn_idx, knn_dist, k


def _intrinsic_dim_levina_bickel(
    X: torch.Tensor, dist_fn: DistanceFn, k: int = 10, seed: int = 0
) -> float:
    """Levina-Bickel MLE intrinsic dimension on a subsample (clamped to [1, D])."""
    n, D = X.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    take = min(2000, n)
    Xs = X[torch.randperm(n, generator=g)[:take]]
    kk = min(k, max(2, take - 1))
    vals, _ = chunked_cdist(dist_fn, Xs, Xs, topk=kk + 1, out_device=Xs.device)
    r = vals[:, 1 : kk + 1].clamp_min(1e-12).double()
    ratio = torch.log(r[:, -1:] / r[:, :-1]).mean(dim=1)
    ratio = ratio[torch.isfinite(ratio) & (ratio > 0)]
    if ratio.numel() == 0:
        return float(D)
    m = float(1.0 / ratio.mean().clamp_min(1e-12))
    return float(min(max(m, 1.0), float(D)))


def _one_nn_all(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    metric: Optional[MetricSpec] = None,
    max_dense: int = 20_000,
) -> Optional[torch.Tensor]:
    """Exact 1-NN distance for **every** row, or None if it would be too costly.

    Dense for small ``N``. For large ``N`` an ANN index supplies the candidate
    neighbour and the returned distance is still evaluated with ``dist_fn``, so
    the value is exact whenever the candidate is correct.
    """
    n = X.shape[0]
    if n < 2:
        return None
    if n <= max_dense:
        vals, _ = chunked_cdist(dist_fn, X, X, topk=2, out_device=X.device)
        return vals[:, 1].contiguous()

    if metric is None or metric.l2_transform is None or not _faiss_available():
        return None
    import faiss

    Xt = metric.l2_transform(X).detach().cpu().numpy().astype(np.float32)
    d = Xt.shape[1]
    nlist = max(1, min(int(np.sqrt(n)), n // 10))
    quant = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quant, d, nlist)
    index.train(Xt)
    index.add(Xt)
    index.nprobe = min(32, nlist)
    # k=8 over-fetch: the true 1-NN can be missed at nprobe=32, and exact ties
    # to self must be skipped.
    _, cand = index.search(Xt, min(8, n))
    out = torch.full((n,), float("inf"), dtype=torch.float32)
    chunk = 4096
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        for i in range(s, e):
            cidx = [int(j) for j in cand[i].tolist() if j >= 0 and j != i]
            if not cidx:
                continue
            C = X[torch.tensor(cidx, dtype=torch.int64, device=X.device)]
            out[i] = float(dist_fn(X[i : i + 1], C)[0].min().item())
    finite = torch.isfinite(out)
    if not bool(finite.all()):
        out[~finite] = out[finite].max() if bool(finite.any()) else 0.0
    return out


def estimate_epsilon(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    n_sample: int = 10_000,
    quantile: float = 0.01,
    seed: int = 0,
    metric: Optional[MetricSpec] = None,
) -> Tuple[float, dict]:
    """Estimate the duplicate scale as the ``quantile`` of 1-NN distances.

    The quantile is taken over **all** ``N`` rows whenever that is affordable,
    because a 1-NN distance shrinks like ``n^{-1/m}``: reading it off a fixed
    subsample makes ε a function of dataset size. A 10^4 subsample of a 10^6
    point set inflates ε by ``100^{1/m}`` (a factor of ~2 at m = 6), and it does
    so precisely at the scale where deduplication matters.

    When the full pass is unaffordable (no ANN backend, or a metric with no
    Euclidean transform), the subsample estimate is kept but rescaled by
    ``(n_sample / N)^(1/m)`` with ``m`` the Levina-Bickel intrinsic dimension,
    which removes the leading size dependence.

    If the quantile collapses to <= 0 because exact duplicates dominate the low
    tail, fall back to the median of *strictly positive* 1-NN distances so the
    ε-net still merges near-ties.

    Parameters
    ----------
    X : (N, D) float32
    dist_fn : DistanceFn
    n_sample : int
        Subsample size for the fallback path.
    quantile : float
    seed : int
    metric : MetricSpec, optional
        Enables the ANN path for the full-N pass.

    Returns
    -------
    epsilon : float
    diagnostics : dict
    """
    log = get_logger()
    n = X.shape[0]
    nn1 = _one_nn_all(X, dist_fn, metric=metric)
    scope = "full"
    subsample_correction = 1.0
    intrinsic_dim = float("nan")

    if nn1 is None:
        scope = "subsample"
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
        if take < n:
            intrinsic_dim = _intrinsic_dim_levina_bickel(X, dist_fn, seed=seed)
            subsample_correction = float((take / n) ** (1.0 / max(intrinsic_dim, 1.0)))
            log.info(
                "epsilon from a %d/%d subsample; rescaling by (n_sub/N)^(1/m)=%.4f "
                "with Levina-Bickel m=%.2f so epsilon does not drift with N",
                take,
                n,
                subsample_correction,
                intrinsic_dim,
            )

    eps = float(torch.quantile(nn1, quantile).item()) * subsample_correction
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
            "frac_exact_zero=%.3f > 0.5: more than half the %s 1-NN distances are "
            "exact duplicates — check the data pipeline",
            frac_exact_zero,
            scope,
        )
    log.info(
        "epsilon = %.6g (%s 1-NN quantile=%.3f over %d value(s))",
        eps,
        scope,
        quantile,
        int(nn1.numel()),
    )
    return eps, {
        "frac_exact_zero": frac_exact_zero,
        "nn1_deciles": deciles,
        "epsilon": eps,
        "used_positive_median_fallback": used_fallback,
        "scope": scope,
        "subsample_correction": subsample_correction,
        "intrinsic_dim": intrinsic_dim,
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


def _halo_merge(
    X: torch.Tensor,
    reps: Representatives,
    assign_topc: torch.Tensor,
    dist_fn: DistanceFn,
    epsilon: float,
) -> Tuple[Representatives, dict]:
    """Merge ε-close representatives that landed in different landmark buckets.

    The greedy net runs inside top-1 buckets, so a near-duplicate pair straddling
    a Voronoi boundary survives as two cells — and that boundary is exactly where
    the conditioning code switches, so those are the duplicates most worth
    collapsing.

    Candidate pairs are read off the **pre-merge** representative set and then
    collapsed by union-find, so the result is the set of connected components of
    the "within ε and in different buckets" graph. That makes the outcome
    independent of visit order; a chain A-B-C collapses to one cell rather than
    to whichever pair was seen first. Roots are the lowest representative index,
    so the labelling is deterministic.

    Returns the rebuilt ``Representatives`` and a diagnostics dict.
    """
    R = int(reps.rep_idx.shape[0])
    info = {"halo_pairs": 0, "halo_merged": 0, "R_before": R}
    if R < 2 or epsilon <= 0.0:
        return reps, info

    rep_idx = reps.rep_idx
    X_rep = X[rep_idx]
    rep_top1 = assign_topc[rep_idx][:, 0]
    # Group representatives by every bucket they are shortlisted for; two reps
    # can only be compared if they share one, which is what makes this cheap.
    shortlist: Dict[int, List[int]] = {}
    topc_rep = assign_topc[rep_idx]
    for r in range(R):
        for b in topc_rep[r].tolist():
            b = int(b)
            if b >= 0:
                shortlist.setdefault(b, []).append(r)

    parent = list(range(R))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Lowest index wins so the labelling does not depend on visit order.
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    n_pairs = 0
    for members in shortlist.values():
        if len(members) < 2:
            continue
        idx_t = torch.as_tensor(sorted(members), dtype=torch.int64)
        D = dist_fn(X_rep[idx_t], X_rep[idx_t])
        close = (D <= epsilon).nonzero(as_tuple=False)
        for a_pos, b_pos in close.tolist():
            if a_pos >= b_pos:
                continue
            ra, rb = int(idx_t[a_pos]), int(idx_t[b_pos])
            # Same-bucket pairs are already handled by the greedy net.
            if int(rep_top1[ra]) == int(rep_top1[rb]):
                continue
            n_pairs += 1
            union(ra, rb)

    roots = [find(r) for r in range(R)]
    uniq = sorted(set(roots))
    if len(uniq) == R:
        info["halo_pairs"] = n_pairs
        info["R_after"] = R
        return reps, info

    remap = {old: new for new, old in enumerate(uniq)}
    new_of_old = torch.as_tensor([remap[roots[r]] for r in range(R)], dtype=torch.int64)
    R_new = len(uniq)

    member_of = new_of_old[reps.member_of]
    rep_idx_new = rep_idx[torch.as_tensor(uniq, dtype=torch.int64)]
    order = torch.argsort(member_of, stable=True)
    counts = torch.bincount(member_of, minlength=R_new)
    offsets = torch.zeros(R_new + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)

    info.update(
        {
            "halo_pairs": n_pairs,
            "halo_merged": R - R_new,
            "R_after": R_new,
        }
    )
    get_logger().info(
        "halo pass: %d cross-bucket pair(s) within epsilon merged %d representative(s) "
        "(R %d -> %d)",
        n_pairs,
        R - R_new,
        R,
        R_new,
    )
    return (
        Representatives(rep_idx_new, member_of, counts.float(), offsets, order),
        info,
    )


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


def _knn_spill_to_stages(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    mode: str,
    metric: Optional[MetricSpec],
    stages_root: Path,
    landmarks: Optional[torch.Tensor] = None,
    assign_topc: Optional[torch.Tensor] = None,
    c_search: int = C_SEARCH,
    extra_assign_topc: Optional[List[torch.Tensor]] = None,
    batch: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute representative kNN in row batches and spill to Zarr stages."""
    from . import graph_stages as stages

    log = get_logger()
    R = int(X_rep.shape[0])
    info: dict = {"mode": f"spill_{mode}", "spill": True}
    g = stages.create_knn_store(stages_root, R, k)
    # Prefer ANN when possible — IVF shortlists on full R are the OOM risk.
    use_mode = mode
    if use_mode in ("auto", "ivf") and metric is not None and metric.l2_transform is not None:
        if _faiss_available():
            use_mode = "ann"
    if use_mode == "ann" and (metric is None or metric.l2_transform is None or not _faiss_available()):
        use_mode = "brute"

    if use_mode == "ann":
        import faiss

        assert metric is not None and metric.l2_transform is not None
        Xt = metric.l2_transform(X_rep).detach().cpu().numpy().astype(np.float32)
        d = Xt.shape[1]
        nlist = min(int(np.sqrt(R)), max(R // 39, 1))
        nlist = max(nlist, 1)
        quant = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, nlist)
        index.train(Xt)
        index.add(Xt)
        index.nprobe = min(32, nlist)
        idx_path = stages_root / "faiss_ivf.index"
        faiss.write_index(index, str(idx_path))
        del index
        try:
            index = faiss.read_index(str(idx_path), faiss.IO_FLAG_MMAP)
        except Exception:  # noqa: BLE001
            index = faiss.read_index(str(idx_path))
        index.nprobe = min(32, nlist)
        k_over = min(R, max(4 * k, k + 1))
        log.info(
            "knn spill ANN: R=%d nlist=%d nprobe=%d batch=%d -> %s",
            R,
            nlist,
            index.nprobe,
            batch,
            idx_path,
        )
        for s in range(0, R, batch):
            e = min(R, s + batch)
            _, cand = index.search(Xt[s:e], k_over)
            batch_idx = np.empty((e - s, k), dtype=np.int64)
            batch_dist = np.empty((e - s, k), dtype=np.float32)
            for bi, i in enumerate(range(s, e)):
                cidx = [int(j) for j in cand[bi].tolist() if j >= 0 and j != i]
                if len(cidx) < k:
                    drow = dist_fn(X_rep[i : i + 1], X_rep)[0]
                    drow[i] = float("inf")
                    vv, ii = torch.topk(drow, k=k, largest=False)
                    batch_dist[bi] = vv.detach().cpu().numpy()
                    batch_idx[bi] = ii.detach().cpu().numpy()
                    continue
                C = X_rep[torch.as_tensor(cidx, dtype=torch.int64, device=X_rep.device)]
                drow = dist_fn(X_rep[i : i + 1], C)[0]
                vv, ii = torch.topk(drow, k=min(k, int(drow.numel())), largest=False)
                out_i = np.array([cidx[int(j)] for j in ii.tolist()], dtype=np.int64)
                out_v = vv.detach().cpu().numpy().astype(np.float32)
                if out_i.size < k:
                    pad_i = np.full(k - out_i.size, out_i[0] if out_i.size else 0, dtype=np.int64)
                    pad_v = np.full(
                        k - out_v.size, float(out_v[-1]) if out_v.size else 0.0, dtype=np.float32
                    )
                    out_i = np.concatenate([out_i, pad_i])
                    out_v = np.concatenate([out_v, pad_v])
                batch_idx[bi] = out_i[:k]
                batch_dist[bi] = out_v[:k]
            g.idx[s:e] = batch_idx
            g.dist[s:e] = batch_dist
            if (e // batch) % 20 == 0 or e == R:
                log.info(
                    "knn spill progress %d / %d RSS≈%.0f MiB",
                    e,
                    R,
                    rss_mb(),
                )
        info["mode"] = "spill_ann"
    else:
        log.info("knn spill brute tiles: R=%d batch=%d", R, batch)
        chunk_b = 16384
        for s in range(0, R, batch):
            e = min(R, s + batch)
            vals, idx = chunked_cdist(
                dist_fn,
                X_rep[s:e],
                X_rep,
                topk=k + 1,
                out_device=X_rep.device,
                chunk_a=min(1024, e - s),
                chunk_b=chunk_b,
            )
            batch_idx = np.empty((e - s, k), dtype=np.int64)
            batch_dist = np.empty((e - s, k), dtype=np.float32)
            for bi, i in enumerate(range(s, e)):
                row_i = idx[bi]
                row_v = vals[bi]
                m = row_i != i
                kept_i = row_i[m][:k]
                kept_v = row_v[m][:k]
                if kept_i.numel() < k:
                    drow = dist_fn(X_rep[i : i + 1], X_rep)[0]
                    drow[i] = float("inf")
                    kept_v, kept_i = torch.topk(drow, k=k, largest=False)
                batch_idx[bi] = kept_i.detach().cpu().numpy().astype(np.int64)
                batch_dist[bi] = kept_v.detach().cpu().numpy().astype(np.float32)
            g.idx[s:e] = batch_idx
            g.dist[s:e] = batch_dist
            if (e // batch) % 20 == 0 or e == R:
                log.info(
                    "knn spill progress %d / %d RSS≈%.0f MiB",
                    e,
                    R,
                    rss_mb(),
                )
        info["mode"] = "spill_brute"

    stages.mark_knn_complete(stages_root)
    log.info("knn spill complete: R=%d k=%d RSS≈%.0f MiB", R, k, rss_mb())
    loaded = stages.load_knn(stages_root)
    assert loaded is not None
    knn_idx, knn_dist = loaded
    knn_idx = knn_idx.to(device=X_rep.device)
    knn_dist = knn_dist.to(device=X_rep.device)
    try:
        info["recall"] = _measure_knn_recall(X_rep, dist_fn, knn_idx, k)
    except Exception:  # noqa: BLE001
        info["recall"] = None
    return knn_dist, knn_idx, info


def knn_representatives(
    X_rep: torch.Tensor,
    dist_fn: DistanceFn,
    k: int,
    mode: str = "auto",
    landmarks: Optional[torch.Tensor] = None,
    assign_topc: Optional[torch.Tensor] = None,
    c_search: int = C_SEARCH,
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
        # Prefer ANN/IVF well before R=200k so full-rep graphs stay off the
        # brute tile path (peak RAM). Exactness is recovered by rescoring.
        if R <= 50_000:
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
        # Shrink B-tile when R is large so peak distance tiles stay bounded.
        chunk_b = 65536 if R <= 40_000 else 16384
        chunk_a = 4096 if R <= 40_000 else 1024
        vals, idx = chunked_cdist(
            dist_fn,
            X_rep,
            X_rep,
            topk=k + 1,
            out_device=X_rep.device,
            chunk_a=chunk_a,
            chunk_b=chunk_b,
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


def _resolve_net_radius(
    X: torch.Tensor,
    dist_fn: DistanceFn,
    eps: float,
    delta: Optional[Union[float, str]],
    *,
    seed: int = 0,
    r_band: Tuple[float, float] = (1e5, 1e6),
    alpha_guard: float = 0.95,
    n_probe: int = 2048,
) -> Tuple[float, Dict[str, Any]]:
    """Resolve the net radius δ from ``delta`` config (None/\"eps\" → ε)."""
    if delta is None or delta == "eps":
        return float(eps), {
            "delta": float(eps),
            "eps_ref": float(eps),
            "r_est": float(X.shape[0]),
            "r_band": (float(r_band[0]), float(r_band[1])),
            "alpha_guard": float(alpha_guard),
            "guard_ok": True,
            "mode": "eps",
        }
    if isinstance(delta, str) and delta == "auto":
        from .resolution import solve_delta

        n = int(X.shape[0])
        take = min(int(n_probe), n)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:take]
        Xs = X[idx]
        Dmat = chunked_cdist(dist_fn, Xs, Xs, out_device=Xs.device)
        assert isinstance(Dmat, torch.Tensor)
        dlt, report = solve_delta(Dmat, r_band=r_band, alpha_guard=alpha_guard, n_rows=n)
        # Def-1 ε is authoritative: never net finer than ε.
        dlt = max(float(dlt), float(eps))
        report = dict(report)
        report["delta"] = float(dlt)
        report["eps_ref"] = float(eps)
        if abs(dlt - float(eps)) <= 1e-15 * (1.0 + abs(float(eps))):
            if report.get("mode") == "calibrated":
                report["mode"] = "eps"
        return float(dlt), report
    dlt = float(delta)
    if dlt < float(eps):
        get_logger().warning(
            "delta=%.6g < epsilon=%.6g; clamping delta to epsilon", dlt, eps
        )
        dlt = float(eps)
    mode = "eps" if abs(dlt - float(eps)) <= 1e-15 * (1.0 + abs(float(eps))) else "calibrated"
    return dlt, {
        "delta": float(dlt),
        "eps_ref": float(eps),
        "r_est": float(X.shape[0]),
        "r_band": (float(r_band[0]), float(r_band[1])),
        "alpha_guard": float(alpha_guard),
        "guard_ok": True,
        "mode": mode,
    }


def build_graph(
    X: torch.Tensor,
    metric: MetricSpec,
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
    extra_ivf_anchors: Optional[
        Sequence[Tuple[Any, DistanceFn, torch.Tensor]]
    ] = None,
    fps_view: Optional[Any] = None,
    fps_view_metric: Optional[DistanceFn] = None,
    fps_geodesic: bool = False,
    fps_geodesic_k: Optional[int] = None,
    fps_poisson: bool = False,
    precomputed_knn: Optional[Tuple[Any, Any]] = None,
    stages_dir: Optional[Union[str, Path]] = None,
    r_band: Tuple[float, float] = (1e5, 1e6),
    alpha_guard: float = 0.95,
) -> Tuple[Graph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full graph pipeline on the training split.

    Parameters
    ----------
    X : (N, D) float32
    metric : MetricSpec
        Should already have ``natural_scale`` fitted. Used for landmark FPS,
        ε-net radii, and (when ``precomputed_knn`` is None) kNN edge distances.
    dedup : bool
        If True (default), collapse near-duplicates with an ε-net before kNN.
        If False, force ``epsilon=0`` and keep ``R == N``.
    delta : float | \"eps\" | \"auto\" | None
        Net radius. ``None`` / ``\"eps\"`` / ``delta == epsilon`` keep today's
        ε-net (bit-compat). ``\"auto\"`` calibrates δ via :func:`solve_delta`.
        A float ``> epsilon`` nets at δ while still recording ε in stats.
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
    precomputed_knn : (knn_idx, knn_dist) or None
        Optional caller-supplied kNN over **representatives**. Arrays are
        ``(R, k)`` with indices in ``[0, R)``. When ``dedup=False`` (so
        ``R == N``) this is the ambient row space. When ``dedup=True``, supply
        neighbors among the ε-net reps (after a prior build or external ANN).
    stages_dir : path, optional
        Zarr stage directory (landmarks → ε-net → knn). Resume if fingerprint
        matches; requires the optional ``zarr`` dependency.

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
    stages_p = Path(stages_dir) if stages_dir is not None else None
    stages = None
    if stages_p is not None:
        from . import graph_stages as stages

        if stages.read_meta(stages_p) is None or not stages.fingerprint_matches(
            stages_p, X
        ):
            stages.init_meta(
                stages_p,
                X,
                epsilon=epsilon,
                n_landmarks=n_landmarks,
                n_neighbors=n_neighbors,
                seed=seed,
                dedup=bool(dedup),
                knn_mode=knn_mode,
            )
        else:
            log.info("stages: fingerprint match under %s", stages_p)

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
        stats.delta = 0.0
        log.info("dedup=False: skipping ε-net (R == N)")
    elif epsilon is None:
        eps, diag = estimate_epsilon(X, dist_fn, seed=seed, metric=metric)
        stats.epsilon = eps
        stats.frac_exact_zero = diag["frac_exact_zero"]
        stats.nn1_deciles = diag["nn1_deciles"]
        stats.extra["used_positive_median_fallback"] = diag.get(
            "used_positive_median_fallback", False
        )
        stats.extra["epsilon_scope"] = diag.get("scope")
        stats.extra["epsilon_subsample_correction"] = diag.get("subsample_correction")
        stats.extra["epsilon_intrinsic_dim"] = diag.get("intrinsic_dim")
    else:
        eps = float(epsilon)
        stats.epsilon = eps
        if eps == 0.0:
            log.info("epsilon=0: skipping ε-net (R == N)")

    # 4.1b δ resolution (default δ = ε for bit-compat)
    if not dedup:
        net_radius = 0.0
        delta_report: Dict[str, Any] = {
            "delta": 0.0,
            "eps_ref": 0.0,
            "r_est": float(X.shape[0]),
            "r_band": (float(r_band[0]), float(r_band[1])),
            "alpha_guard": float(alpha_guard),
            "guard_ok": True,
            "mode": "eps",
        }
    else:
        net_radius, delta_report = _resolve_net_radius(
            X,
            dist_fn,
            eps,
            delta,
            seed=seed,
            r_band=r_band,
            alpha_guard=alpha_guard,
        )
        stats.delta = float(net_radius)
        stats.extra["delta"] = float(net_radius)
        stats.extra["epsilon"] = float(eps)
        stats.extra["delta_mode"] = delta_report.get("mode")
        stats.extra["delta_guard_ok"] = delta_report.get("guard_ok")
        stats.extra["delta_r_est"] = delta_report.get("r_est")
        stats.extra["delta_report"] = delta_report
        if float(net_radius) > float(eps) + 1e-15 * (1.0 + abs(float(eps))):
            log.info(
                "net radius delta=%.6g > epsilon=%.6g (mode=%s, r_est=%s)",
                net_radius,
                eps,
                delta_report.get("mode"),
                delta_report.get("r_est"),
            )

    # 4.2 landmarks — FPS in scoring space, or in fps_view (conditioning) space
    ivf_view_assign: Optional[Tuple[Any, DistanceFn, torch.Tensor]] = None
    loaded_landmarks = (
        stages.load_landmarks(stages_p) if stages is not None else None
    )
    if loaded_landmarks is not None:
        M, assign_top1, assign_topc = loaded_landmarks
        M = M.to(device=X.device, dtype=X.dtype)
        assign_top1 = assign_top1.to(device=X.device)
        assign_topc = assign_topc.to(device=X.device)
        if int(M.shape[0]) != int(n_landmarks):
            log.warning(
                "stages landmarks L=%d != config n_landmarks=%d; keeping staged L",
                int(M.shape[0]),
                int(n_landmarks),
            )
    elif fps_view is not None:
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
    if stages is not None and loaded_landmarks is None:
        stages.save_landmarks(stages_p, M, assign_top1, assign_topc)
    log.info("landmarks done: L=%d RSS≈%.0f MiB", L, rss_mb())

    # 4.3 representatives (+ halo pass across Voronoi boundaries)
    # Net at δ (defaults to ε). ε remains in stats for diagnostics / nesting.
    loaded_enet = stages.load_enet(stages_p) if stages is not None else None
    if loaded_enet is not None:
        reps, staged_radius = loaded_enet
        reps = Representatives(
            rep_idx=reps.rep_idx.to(device=X.device),
            member_of=reps.member_of.to(device=X.device),
            weight=reps.weight.to(device=X.device),
            offsets=reps.offsets.to(device=X.device),
            values=reps.values.to(device=X.device),
        )
        if abs(float(staged_radius) - float(net_radius)) > 1e-12:
            log.warning(
                "stages net radius=%.6g != config delta=%.6g; using staged",
                staged_radius,
                net_radius,
            )
        stats.delta = float(staged_radius)
        stats.extra["halo_pairs"] = stats.extra.get("halo_pairs", 0)
        stats.extra["halo_merged"] = stats.extra.get("halo_merged", 0)
        stats.extra["stages_enet"] = True
    else:
        reps = build_representatives(
            X, assign_top1, dist_fn, net_radius, L, seed=seed
        )
        reps, halo_info = _halo_merge(X, reps, assign_topc, dist_fn, net_radius)
        stats.extra.update(halo_info)
        if stages is not None:
            stages.save_enet(stages_p, reps, float(net_radius))
    stats.n_reps = int(reps.rep_idx.shape[0])
    stats.compression_ratio = float(X.shape[0]) / max(stats.n_reps, 1)
    if stats.delta == 0.0 and float(net_radius) != 0.0:
        stats.delta = float(net_radius)
    log.info(
        "δ-net done: R=%d compression=%.4f delta=%.6g epsilon=%.6g RSS≈%.0f MiB",
        stats.n_reps,
        stats.compression_ratio,
        float(stats.delta),
        float(stats.epsilon),
        rss_mb(),
    )
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

    # 4.4 kNN — staged / caller-supplied / computed
    R = int(X_rep.shape[0])
    staged_knn = stages.load_knn(stages_p) if stages is not None else None
    if precomputed_knn is not None:
        raw_idx, raw_dist = precomputed_knn
        knn_idx, knn_dist, k_eff = validate_precomputed_knn(
            raw_idx, raw_dist, R, n_neighbors=n_neighbors
        )
        knn_idx = knn_idx.to(device=X.device)
        knn_dist = knn_dist.to(device=X.device)
        if k_eff != int(n_neighbors):
            log.info(
                "precomputed_knn: using k=%d (config n_neighbors=%d)",
                k_eff,
                int(n_neighbors),
            )
        stats.knn_mode = "precomputed"
        stats.knn_recall = None
        stats.extra["precomputed_k"] = int(k_eff)
    elif staged_knn is not None:
        knn_idx, knn_dist = staged_knn
        knn_idx, knn_dist, k_eff = validate_precomputed_knn(
            knn_idx, knn_dist, R, n_neighbors=n_neighbors
        )
        knn_idx = knn_idx.to(device=X.device)
        knn_dist = knn_dist.to(device=X.device)
        stats.knn_mode = "staged"
        stats.knn_recall = None
        stats.extra["precomputed_k"] = int(k_eff)
    else:
        use_spill = stages is not None and R > 50_000
        if use_spill:
            knn_dist, knn_idx, knn_info = _knn_spill_to_stages(
                X_rep,
                dist_fn,
                k=n_neighbors,
                mode=mode,
                metric=metric,
                stages_root=stages_p,
                landmarks=landmarks_for_ivf,
                assign_topc=rep_topc,
                c_search=c_search,
                extra_assign_topc=extra_assign_topc,
            )
        else:
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
            if stages is not None:
                g = stages.create_knn_store(stages_p, R, int(n_neighbors))
                g.idx[:] = knn_idx.detach().cpu().numpy().astype(np.int64)
                g.dist[:] = knn_dist.detach().cpu().numpy().astype(np.float32)
                stages.mark_knn_complete(stages_p)
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

    # 4.8 backbone
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


def _squash_coarse_weights(wsum: torch.Tensor, mode: str = "rational_q99") -> torch.Tensor:
    """Map heavy-tailed aggregated crossing weights into a membership in (0, 1).

    Summed crossing weights are heavy-tailed: one wide bridge can be orders of
    magnitude stronger than a thin one.

    ``"rational_q99"`` (default)
        ``w / (w + q99)``. Strictly monotone and unsaturating, so the strongest
        coarse edges — the long-range structure a ``(1, 2, 8)`` pyramid exists
        to exploit — keep their ordering, while the *selectivity* of the old
        clamp is preserved: anchoring at the 0.99 quantile keeps typical coarse
        edges weak and only the strong bridges near the ceiling.

    ``"rational"``
        ``w / (w + q50)``, anchored at the median. Also monotone, but it lifts
        the mean membership by ~6x and makes coarse attraction diffuse rather
        than selective. Measurably worse density correspondence on clustered
        data; kept for ablation.

    ``"quantile_clamp"`` (previous behaviour, kept for ablation)
        ``min(w / q99, 1)``. Everything above the 0.99 quantile is flattened to
        a common weight of exactly 1, discarding the ranking among the very
        edges the pyramid is for.

    The modes differ in *magnitude* as well as shape, so
    ``pyramid_level_weights`` does not transfer between them; ``rational_q99``
    is the one that is magnitude-comparable to ``quantile_clamp``.
    """
    if wsum.numel() == 0:
        return wsum.to(torch.float32)

    def _q(p: float) -> float:
        v = float(torch.quantile(wsum, p)) if wsum.numel() > 1 else float(wsum.max())
        if v > 0:
            return v
        pos = wsum[wsum > 0]
        return float(pos.median()) if pos.numel() else 1.0

    if mode == "quantile_clamp":
        scale = _q(0.99)
        return torch.clamp(wsum / max(scale, 1e-12), max=1.0).to(torch.float32)
    if mode == "rational":
        anchor = _q(0.5)
    elif mode == "rational_q99":
        anchor = _q(0.99)
    else:
        raise ValueError(f"unknown pyramid_squash={mode!r}")
    return (wsum / (wsum + max(anchor, 1e-12))).to(torch.float32)


def _coarsen_graph(
    graph_l: Graph,
    X: torch.Tensor,
    dist_fn: DistanceFn,
    target_reps: int,
    seed: int = 0,
    squash: str = "rational_q99",
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
    weights_c = _squash_coarse_weights(wsum, mode=squash)

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
    pyramid_rep_ratio: float = PYRAMID_REP_RATIO,
    pyramid_min_reps: int = 256,
    pyramid_coarse_backbone: float = 1.0,
    pyramid_squash: str = "rational_q99",
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
        g = _coarsen_graph(
            prev, X, dist_fn, target, seed=seed + level, squash=pyramid_squash
        )
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


GRAPH_PYRAMID_VERSION = 1


def _cpu_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()


def tensor_fingerprint(X: torch.Tensor) -> Dict[str, Any]:
    """Cheap identity check so a cached graph cannot be reused on the wrong X."""
    x = X.detach().cpu().reshape(-1)
    n = int(x.numel())
    return {
        "shape": [int(s) for s in X.shape],
        "mean": float(X.mean().cpu()),
        "head": x[:8].tolist() if n else [],
        "tail": x[-8:].tolist() if n else [],
    }


def check_tensor_fingerprint(
    X: torch.Tensor, fingerprint: Dict[str, Any], *, what: str = "X_train"
) -> None:
    got = tensor_fingerprint(X)
    want_shape = list(fingerprint["shape"])
    if got["shape"] != want_shape:
        raise ValueError(f"{what} shape {got['shape']} != cached {want_shape}")
    atol = 1e-4 * (1.0 + abs(float(fingerprint["mean"])))
    if abs(got["mean"] - float(fingerprint["mean"])) > atol:
        raise ValueError(
            f"{what} mean {got['mean']:.6g} != cached {fingerprint['mean']:.6g}; "
            "rebuild the graph"
        )
    for key in ("head", "tail"):
        a = [float(v) for v in got[key]]
        b = [float(v) for v in fingerprint[key]]
        if len(a) != len(b) or any(abs(x - y) > 1e-5 for x, y in zip(a, b)):
            raise ValueError(f"{what} {key} does not match the cached graph; rebuild")


def graph_to_state(graph: Graph) -> Dict[str, Any]:
    reps = graph.reps
    return {
        "edges": _cpu_tensor(graph.edges),
        "weights": _cpu_tensor(graph.weights),
        "knn_idx": _cpu_tensor(graph.knn_idx),
        "reps": {
            "rep_idx": _cpu_tensor(reps.rep_idx),
            "member_of": _cpu_tensor(reps.member_of),
            "weight": _cpu_tensor(reps.weight),
            "offsets": _cpu_tensor(reps.offsets),
            "values": _cpu_tensor(reps.values),
        },
        "stats": asdict(graph.stats),
    }


def graph_from_state(state: Dict[str, Any]) -> Graph:
    allowed = {f.name for f in fields(GraphStats)}
    stats_raw = dict(state["stats"])
    if "delta" not in stats_raw and "epsilon" in stats_raw:
        stats_raw["delta"] = stats_raw["epsilon"]
    stats = GraphStats(**{k: v for k, v in stats_raw.items() if k in allowed})
    reps_raw = state["reps"]
    reps = Representatives(
        rep_idx=torch.as_tensor(reps_raw["rep_idx"]),
        member_of=torch.as_tensor(reps_raw["member_of"]),
        weight=torch.as_tensor(reps_raw["weight"]),
        offsets=torch.as_tensor(reps_raw["offsets"]),
        values=torch.as_tensor(reps_raw["values"]),
    )
    return Graph(
        edges=torch.as_tensor(state["edges"]),
        weights=torch.as_tensor(state["weights"]),
        reps=reps,
        knn_idx=torch.as_tensor(state["knn_idx"]),
        stats=stats,
    )


def save_graph_pyramid(
    path: Union[str, Path],
    *,
    graphs: Sequence[Graph],
    M: torch.Tensor,
    assign_top1: torch.Tensor,
    assign_topc: torch.Tensor,
    train_idx: torch.Tensor,
    calib_idx: torch.Tensor,
    fingerprint: Dict[str, Any],
    metric_name: str,
    n_all: int,
    n_neighbors: int,
    epsilon: float,
    seed: int,
    dedup: bool,
) -> Path:
    """Write the training graph pyramid plus the split it was built on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": GRAPH_PYRAMID_VERSION,
        "metric_name": str(metric_name),
        "n_all": int(n_all),
        "n_landmarks": int(M.shape[0]),
        "n_neighbors": int(n_neighbors),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "dedup": bool(dedup),
        "fingerprint": fingerprint,
        "train_idx": _cpu_tensor(torch.as_tensor(train_idx)),
        "calib_idx": _cpu_tensor(torch.as_tensor(calib_idx)),
        "graphs": [graph_to_state(g) for g in graphs],
        "M": _cpu_tensor(M),
        "assign_top1": _cpu_tensor(assign_top1),
        "assign_topc": _cpu_tensor(assign_topc),
    }
    torch.save(payload, path)
    log = get_logger()
    log.info(
        "saved graph pyramid %s (%d level(s), R=%d, L=%d)",
        path,
        len(graphs),
        int(graphs[0].reps.rep_idx.shape[0]) if graphs else 0,
        int(M.shape[0]),
    )
    return path


def load_graph_pyramid(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a pyramid written by :func:`save_graph_pyramid`."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = int(payload.get("version", 0))
    if version != GRAPH_PYRAMID_VERSION:
        raise ValueError(
            f"graph cache version {version} != {GRAPH_PYRAMID_VERSION} ({path})"
        )
    payload["graphs"] = [graph_from_state(g) for g in payload["graphs"]]
    payload["train_idx"] = torch.as_tensor(payload["train_idx"], dtype=torch.int64)
    payload["calib_idx"] = torch.as_tensor(payload["calib_idx"], dtype=torch.int64)
    payload["M"] = torch.as_tensor(payload["M"])
    payload["assign_top1"] = torch.as_tensor(payload["assign_top1"])
    payload["assign_topc"] = torch.as_tensor(payload["assign_topc"])
    return payload

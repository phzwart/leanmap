"""Evaluation metrics and benchmarking for PLANE."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .conformal import ConformalCalibrator, geometry_consistency_score
from .distance import DistanceFn, chunked_cdist
from .model import PLANE
from .utils import ensure_2d_float32, get_logger


def _optional_sklearn():
    try:
        from sklearn.manifold import trustworthiness as sk_trust

        return sk_trust
    except Exception:  # noqa: BLE001
        return None


def shepard_stats(
    d_orig: np.ndarray | torch.Tensor,
    d_embed: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """Spearman + scale-invariant stress for a Shepard pair cloud.

    Fits isotropic scale ``alpha`` with ``alpha * d_embed ≈ d_orig`` (detached
    least squares), then reports normalized residual stress.
    """
    from scipy.stats import spearmanr

    nan = float("nan")
    g = np.asarray(d_orig, dtype=np.float64).ravel()
    e = np.asarray(d_embed, dtype=np.float64).ravel()
    if g.size == 0 or e.size != g.size:
        return {
            "spearman": nan,
            "stress": nan,
            "alpha": nan,
            "n_pairs": int(g.size),
        }
    rho = float(spearmanr(g, e).correlation)
    alpha = float((g * e).sum() / max(float((e * e).sum()), 1e-12))
    stress = float(
        np.sqrt(((alpha * e - g) ** 2).sum() / max(float((g * g).sum()), 1e-12))
    )
    return {
        "spearman": rho,
        "stress": stress,
        "alpha": alpha,
        "n_pairs": int(g.size),
    }


def shepard_pairs_ambient(
    X: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    n_pairs: int = 32768,
    seed: int = 0,
    dist_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ambient vs embedding distances (Shepard).

    Parameters
    ----------
    dist_fn : callable, optional
        Ambient distance, called row-block-wise as ``dist_fn(A, B) -> (n, m)``.
        Defaults to Euclidean. Pass the metric the map was *trained* under;
        scoring an L1-trained map with an L2 ruler measures transfer across
        metrics, not fidelity. The embedding side stays Euclidean because the
        low-dimensional kernel is.

    Returns
    -------
    d_orig, d_embed : (P,) float64
    """
    Xnp = np.asarray(X, dtype=np.float64)
    Znp = np.asarray(Z, dtype=np.float64)
    n = Xnp.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    p = int(min(max(1, n_pairs), n * (n - 1) // 2))
    i = rng.integers(0, n, size=p)
    j = rng.integers(0, n, size=p)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    if dist_fn is None:
        d_orig = np.linalg.norm(Xnp[i] - Xnp[j], axis=1)
    else:
        A = torch.as_tensor(Xnp[i], dtype=torch.float32)
        B = torch.as_tensor(Xnp[j], dtype=torch.float32)
        d_orig = _pairwise_rowwise(dist_fn, A, B)
    d_embed = np.linalg.norm(Znp[i] - Znp[j], axis=1)
    return d_orig, d_embed


def _pairwise_rowwise(dist_fn: Callable, A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """d(A[k], B[k]) for each k, via the diagonal of blocked (n, m) calls."""
    out = np.empty(A.shape[0], dtype=np.float64)
    block = 1024
    with torch.no_grad():
        for s in range(0, A.shape[0], block):
            e = min(s + block, A.shape[0])
            D = dist_fn(A[s:e], B[s:e])
            out[s:e] = torch.diagonal(D).detach().cpu().numpy().astype(np.float64)
    return out


def shepard_pairs_geodesic(
    edges: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    n_sources: int = 64,
    max_targets: int = 512,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample graph-geodesic vs embedding Euclidean distances (Shepard).

    Builds a weighted graph from fuzzy edges with hop length ``1/membership``,
    runs Dijkstra from ``n_sources`` nodes, and returns reachable pair distances.

    Returns
    -------
    d_geo, d_embed : (P,) float64
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra

    e = np.asarray(edges, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    Znp = np.asarray(Z, dtype=np.float64)
    R = Znp.shape[0]
    empty = (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    if e.size == 0 or R < 3:
        return empty

    length = 1.0 / np.clip(w, 1e-6, None)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    data = np.concatenate([length, length])
    A = sparse.coo_matrix((data, (rows, cols)), shape=(R, R)).tocsr()

    rng = np.random.default_rng(seed)
    n_src = min(n_sources, R)
    src = rng.choice(R, size=n_src, replace=False)
    D = dijkstra(A, directed=False, indices=src)  # (n_src, R)

    gd_parts: list[np.ndarray] = []
    ed_parts: list[np.ndarray] = []
    for si, s in enumerate(src):
        d = D[si]
        finite = np.isfinite(d)
        finite[s] = False
        tgt = np.where(finite)[0]
        if tgt.size == 0:
            continue
        if tgt.size > max_targets:
            tgt = rng.choice(tgt, size=max_targets, replace=False)
        gd_parts.append(d[tgt])
        ed_parts.append(np.linalg.norm(Znp[tgt] - Znp[s], axis=1))

    if not gd_parts:
        return empty
    return np.concatenate(gd_parts), np.concatenate(ed_parts)


def geodesic_fidelity(
    edges: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    n_sources: int = 64,
    max_targets: int = 512,
    seed: int = 0,
) -> Dict[str, float]:
    """Correlate graph shortest-path (geodesic) distance vs embedding distance.

    Quantifies whether the embedding preserves *global* manifold structure, not
    just local neighbourhoods. See :func:`shepard_pairs_geodesic`.

    Returns
    -------
    dict with ``geodesic_spearman`` (higher is better), ``geodesic_stress``
    (scale-invariant, lower is better), and ``geodesic_pairs`` (count).
    """
    nan = float("nan")
    gd, ed = shepard_pairs_geodesic(
        edges,
        weights,
        Z,
        n_sources=n_sources,
        max_targets=max_targets,
        seed=seed,
    )
    if gd.size == 0:
        return {"geodesic_spearman": nan, "geodesic_stress": nan, "geodesic_pairs": 0}
    st = shepard_stats(gd, ed)
    return {
        "geodesic_spearman": st["spearman"],
        "geodesic_stress": st["stress"],
        "geodesic_pairs": st["n_pairs"],
    }


def trustworthiness_continuity(
    X: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    dist_fn: DistanceFn,
    k_list: Tuple[int, ...] = (5, 15, 50),
    n_sample: int = 5000,
    seed: int = 0,
) -> dict:
    """Trustworthiness / continuity at several neighbourhood sizes.

    Parameters
    ----------
    X : (N, D)
    Z : (N, d)
    dist_fn : DistanceFn on X
    k_list : neighbourhood sizes
    n_sample : subsample size

    Returns
    -------
    dict with keys like ``trust_15``, ``cont_15``
    """
    X = ensure_2d_float32(X)
    Z = ensure_2d_float32(Z)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n > n_sample:
        idx = rng.choice(n, size=n_sample, replace=False)
        X, Z = X[idx], Z[idx]
        n = n_sample
    Xt = torch.as_tensor(X)
    Zt = torch.as_tensor(Z)
    # High-D neighbours via dist_fn; low-D via Euclidean
    from .distance import EuclideanDistance

    out = {}
    sk_trust = _optional_sklearn()
    for k in k_list:
        kk = min(k, n - 1)
        if sk_trust is not None:
            # sklearn trustworthiness assumes Euclidean on X; we approximate
            # with our embedding Euclidean which is what sklearn expects for Z
            try:
                t = float(sk_trust(X, Z, n_neighbors=kk))
            except Exception:  # noqa: BLE001
                t = float("nan")
        else:
            t = float("nan")
        # Continuity: swap roles with Euclidean on Z as "input" — approximate
        if sk_trust is not None:
            try:
                c = float(sk_trust(Z, X, n_neighbors=kk))
            except Exception:  # noqa: BLE001
                c = float("nan")
        else:
            c = float("nan")
        out[f"trust_{k}"] = t
        out[f"cont_{k}"] = c
        # Also compute a dist_fn-aware trustworthiness overlap
        _, hd = chunked_cdist(dist_fn, Xt, Xt, topk=kk + 1)
        _, ld = chunked_cdist(EuclideanDistance(), Zt, Zt, topk=kk + 1)
        overlaps = []
        for i in range(n):
            hi = set(int(j) for j in hd[i].tolist() if j != i) 
            # take kk
            hi = set(list(set(hd[i].tolist()) - {i}) )
            # properly:
            row = [int(j) for j in hd[i].tolist() if int(j) != i][:kk]
            low = [int(j) for j in ld[i].tolist() if int(j) != i][:kk]
            overlaps.append(len(set(row) & set(low)) / float(kk))
        out[f"knn_overlap_{k}"] = float(np.mean(overlaps))
    return out


def knn_local_density(
    X: np.ndarray | torch.Tensor,
    k: int = 15,
    *,
    dist_fn: Optional[DistanceFn] = None,
) -> np.ndarray:
    """Local density proxy: ``1 / mean distance to k nearest neighbours``.

    Parameters
    ----------
    X : (N, D)
    k : neighbourhood size (excludes self)
    dist_fn : optional ambient metric; default Euclidean via sklearn/torch

    Returns
    -------
    dens : (N,) float64
    """
    Xnp = np.asarray(ensure_2d_float32(X), dtype=np.float64)
    n = Xnp.shape[0]
    kk = int(min(max(1, k), max(1, n - 1)))
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    if dist_fn is None:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=kk + 1, algorithm="auto").fit(Xnp)
        d, _ = nn.kneighbors(Xnp)
        mean_d = np.maximum(d[:, 1:].mean(axis=1), 1e-12)
        return (1.0 / mean_d).astype(np.float64)
    Xt = torch.as_tensor(Xnp, dtype=torch.float32)
    d, _ = chunked_cdist(dist_fn, Xt, Xt, topk=kk + 1)
    # first neighbour can be self at ~0; drop the nearest
    dnp = d.detach().cpu().numpy().astype(np.float64)
    # sort each row and skip the zero/self entry
    dnp.sort(axis=1)
    mean_d = np.maximum(dnp[:, 1 : kk + 1].mean(axis=1), 1e-12)
    return (1.0 / mean_d).astype(np.float64)


def density_correspondence(
    X: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    k: int = 15,
    *,
    dist_fn: Optional[DistanceFn] = None,
) -> Dict[str, np.ndarray | float]:
    """Compare local density in ambient ``X`` vs embedding ``Z``.

    Returns densities, log-residuals of embed dens vs a linear prediction from
    ambient dens, and Spearman / Pearson(log) correlations.
    """
    from scipy.stats import pearsonr, spearmanr

    dens_a = knn_local_density(X, k=k, dist_fn=dist_fn)
    dens_z = knn_local_density(Z, k=k, dist_fn=None)
    la = np.log10(np.maximum(dens_a, 1e-12))
    lz = np.log10(np.maximum(dens_z, 1e-12))
    # least-squares: lz ≈ a + b * la
    b = float(np.cov(la, lz, ddof=0)[0, 1] / max(float(np.var(la)), 1e-12))
    a = float(lz.mean() - b * la.mean())
    resid = lz - (a + b * la)
    rho = float(spearmanr(dens_a, dens_z).correlation)
    try:
        pear = float(pearsonr(la, lz).statistic)  # type: ignore[attr-defined]
    except AttributeError:
        pear = float(pearsonr(la, lz)[0])
    return {
        "dens_ambient": dens_a.astype(np.float32),
        "dens_embed": dens_z.astype(np.float32),
        "mean_knn_ambient": (1.0 / np.maximum(dens_a, 1e-12)).astype(np.float32),
        "mean_knn_embed": (1.0 / np.maximum(dens_z, 1e-12)).astype(np.float32),
        "log_resid_embed_vs_ambient": resid.astype(np.float32),
        "fit_intercept": a,
        "fit_slope": b,
        "spearman": rho,
        "pearson_log": pear,
        "k": int(k),
    }


def knn_recall_out_of_sample(
    X_train: np.ndarray | torch.Tensor,
    Z_train: np.ndarray | torch.Tensor,
    X_test: np.ndarray | torch.Tensor,
    Z_test: np.ndarray | torch.Tensor,
    dist_fn: DistanceFn,
    k: int = 15,
) -> float:
    """Mean overlap of true HD kNN (train) vs embedding Euclidean kNN.

    Parameters
    ----------
    X_train : (N, D), Z_train : (N, d)
    X_test : (M, D), Z_test : (M, d)
    dist_fn : DistanceFn
    k : int

    Returns
    -------
    float in [0, 1]
    """
    from .distance import EuclideanDistance

    Xtr = torch.as_tensor(ensure_2d_float32(X_train))
    Xte = torch.as_tensor(ensure_2d_float32(X_test))
    Ztr = torch.as_tensor(ensure_2d_float32(Z_train))
    Zte = torch.as_tensor(ensure_2d_float32(Z_test))
    k = min(k, Xtr.shape[0])
    _, hd = chunked_cdist(dist_fn, Xte, Xtr, topk=k)
    _, ld = chunked_cdist(EuclideanDistance(), Zte, Ztr, topk=k)
    overlaps = []
    for i in range(Xte.shape[0]):
        overlaps.append(
            len(set(hd[i].tolist()) & set(ld[i].tolist())) / float(k)
        )
    return float(np.mean(overlaps))


def _procrustes_fit(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Similarity transform (rotation, translation, uniform scale) taking A to B.

    Returns ``(R, t, s)`` such that ``s * A @ R + t`` best matches ``B``.
    """
    Ac = A - A.mean(dim=0, keepdim=True)
    Bc = B - B.mean(dim=0, keepdim=True)
    M = Ac.transpose(0, 1) @ Bc
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] = -U[:, -1]
        R = U @ Vh
        S = S.clone()
        S[-1] = -S[-1]
    denom = float((Ac * Ac).sum().clamp_min(eps))
    s = float(S.sum() / denom)
    t = B.mean(dim=0) - s * (A.mean(dim=0) @ R)
    return R, t, s


def procrustes_disagreement(
    Z_a: np.ndarray | torch.Tensor,
    Z_b: np.ndarray | torch.Tensor,
    anchor_idx: Sequence[int],
    eval_idx: Sequence[int],
) -> Dict[str, float]:
    """Median coordinate disagreement between two maps of the same points.

    The similarity transform is fitted on ``anchor_idx`` and scored on
    ``eval_idx``. Fitting and scoring on the same points understates
    disagreement, because the alignment absorbs part of what is being measured.

    Disagreement is reported relative to the spread of the reference map, so it
    does not inherit the arbitrary overall scale of an embedding.

    Parameters
    ----------
    Z_a, Z_b : (N, d)
        Two embeddings of the *same* rows, in the same order.
    anchor_idx : indices used to fit the alignment
    eval_idx : disjoint indices used to score it

    Returns
    -------
    dict with ``median``, ``p90``, and ``scale`` (the fitted similarity scale)
    """
    A = torch.as_tensor(ensure_2d_float32(Z_a))
    B = torch.as_tensor(ensure_2d_float32(Z_b))
    if A.shape != B.shape:
        raise ValueError(f"embeddings must match in shape; got {A.shape} vs {B.shape}")
    a_idx = torch.as_tensor(list(anchor_idx), dtype=torch.int64)
    e_idx = torch.as_tensor(list(eval_idx), dtype=torch.int64)
    if a_idx.numel() < 3 or e_idx.numel() < 1:
        return {"median": float("nan"), "p90": float("nan"), "scale": float("nan")}

    R, t, s = _procrustes_fit(A[a_idx], B[a_idx])
    aligned = s * (A[e_idx] @ R) + t
    resid = (aligned - B[e_idx]).norm(dim=1)
    # Normalise by the reference spread so the number is scale-free.
    ref = B[e_idx] - B[e_idx].mean(dim=0, keepdim=True)
    spread = float(ref.norm(dim=1).median().clamp_min(1e-12))
    return {
        "median": float(resid.median()) / spread,
        "p90": float(torch.quantile(resid, 0.9)) / spread,
        "scale": s,
    }


def neighborhood_rank_agreement(
    Z_a: np.ndarray | torch.Tensor,
    Z_b: np.ndarray | torch.Tensor,
    k: int = 15,
    n_sample: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Do two independently fitted maps agree on *who is near whom*?

    Reports Spearman correlation between neighbour ranks and the Jaccard
    overlap of the top-``k`` neighbour sets, both on a shared probe sample.

    This is the primary persistence number. Procrustes disagreement cannot
    repair genuine topological ambiguity — which arm of a branching structure
    lands on which side is not a rigid motion — so a large coordinate
    disagreement combined with high rank agreement is a gauge artefact, not
    instability. Read the two together.
    """
    from .distance import EuclideanDistance

    A = torch.as_tensor(ensure_2d_float32(Z_a))
    B = torch.as_tensor(ensure_2d_float32(Z_b))
    if A.shape[0] != B.shape[0]:
        raise ValueError("embeddings must cover the same rows")
    n = A.shape[0]
    k = int(min(k, n - 1))
    if n < 3 or k < 1:
        return {"spearman": float("nan"), "jaccard": float("nan"), "n_probe": 0}

    g = torch.Generator().manual_seed(seed)
    take = min(n_sample, n)
    probe = torch.randperm(n, generator=g)[:take]

    euclid = EuclideanDistance()
    dA = euclid(A[probe], A)
    dB = euclid(B[probe], B)
    # Exclude self so a trivially shared zero does not inflate agreement.
    ar = torch.arange(take)
    dA[ar, probe] = float("inf")
    dB[ar, probe] = float("inf")

    rank_rho = []
    jac = []
    for i in range(take):
        ra = torch.argsort(torch.argsort(dA[i])).double()
        rb = torch.argsort(torch.argsort(dB[i])).double()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = float((ra.norm() * rb.norm()).clamp_min(1e-12))
        rank_rho.append(float((ra @ rb) / denom))
        sa = set(torch.topk(dA[i], k, largest=False).indices.tolist())
        sb = set(torch.topk(dB[i], k, largest=False).indices.tolist())
        jac.append(len(sa & sb) / float(len(sa | sb)))
    return {
        "spearman": float(np.mean(rank_rho)),
        "jaccard": float(np.mean(jac)),
        "n_probe": int(take),
    }


def persistence_summary(
    embeddings: Sequence[np.ndarray | torch.Tensor],
    k: int = 15,
    anchor_frac: float = 0.5,
    n_sample: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Aggregate pairwise persistence over >= 2 refits of the same points.

    ``embeddings`` must be maps of an identical, identically-ordered probe set —
    produced by refitting at different seeds, or on different training
    subsamples and then embedding a common holdout.

    Returns the mean over all unordered pairs of
    :func:`procrustes_disagreement` and :func:`neighborhood_rank_agreement`,
    plus the worst pair, which is what a reproducibility claim has to survive.
    """
    Zs = [torch.as_tensor(ensure_2d_float32(Z)) for Z in embeddings]
    if len(Zs) < 2:
        raise ValueError("persistence needs at least two embeddings")
    n = Zs[0].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_anchor = max(3, int(anchor_frac * n))
    anchor_idx = perm[:n_anchor].tolist()
    eval_idx = perm[n_anchor:].tolist() or anchor_idx

    coord, rho, jac = [], [], []
    for i in range(len(Zs)):
        for j in range(i + 1, len(Zs)):
            p = procrustes_disagreement(Zs[i], Zs[j], anchor_idx, eval_idx)
            r = neighborhood_rank_agreement(
                Zs[i], Zs[j], k=k, n_sample=n_sample, seed=seed
            )
            coord.append(p["median"])
            rho.append(r["spearman"])
            jac.append(r["jaccard"])
    return {
        "n_maps": len(Zs),
        "n_pairs": len(coord),
        "coord_disagreement_median": float(np.mean(coord)),
        "coord_disagreement_worst": float(np.max(coord)),
        "rank_spearman_mean": float(np.mean(rho)),
        "rank_spearman_worst": float(np.min(rho)),
        f"rank_jaccard_{k}_mean": float(np.mean(jac)),
        f"rank_jaccard_{k}_worst": float(np.min(jac)),
    }


def benchmark_inference(
    model: PLANE,
    X: np.ndarray | torch.Tensor,
    umap_model=None,
) -> dict:
    """Wall-clock inference vs optional ``umap.transform``."""
    import time

    X = ensure_2d_float32(X)
    Xt = torch.as_tensor(X)
    t0 = time.perf_counter()
    _ = model.embed(Xt, return_score=False)
    t_plane = time.perf_counter() - t0
    out = {"plane_seconds": t_plane, "n": X.shape[0]}
    if umap_model is not None:
        try:
            t0 = time.perf_counter()
            _ = umap_model.transform(X)
            out["umap_seconds"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            out["umap_error"] = str(exc)
    return out


def uniformity_of_pvalues(
    calibrator: ConformalCalibrator,
    X_holdout: np.ndarray | torch.Tensor,
    model: PLANE,
) -> dict:
    """KS test of conformal p-values against Uniform[0,1] on exchangeable holdout.

    A KS p-value below 0.01 suggests calibration leakage.
    """
    from scipy.stats import kstest

    X = torch.as_tensor(ensure_2d_float32(X_holdout))
    device = next(model.parameters()).device
    assert calibrator.tau_embed is not None
    scores, _ = geometry_consistency_score(
        model, X.to(device), tau_embed=calibrator.tau_embed
    )
    p = calibrator.p_value(scores, model=model).cpu().numpy()
    stat = kstest(p, "uniform")
    return {
        "ks_statistic": float(stat.statistic),
        "ks_pvalue": float(stat.pvalue),
        "mean_p": float(p.mean()),
    }

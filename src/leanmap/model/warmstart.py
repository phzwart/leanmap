"""Starting the layout from the landmark geodesics instead of from PCA or zero.

Classical MDS of the landmark geodesic matrix is already computed before training
-- :mod:`leanmap.train` builds it for the Procrustes anchor term -- and it is a
coarse but *topologically* informed layout of the manifold. Until now it was only
ever used as a running per-step penalty, so training began either from a linear
PCA projection (which on digits caps 5-NN label accuracy at ~0.65 no matter how
the rest is tuned, see ``PLANEConfig.pca_skip``) or, with the skip off, from a
near-zero head that has to build a layout from scratch at a 20x learning rate.

Both are worse starting points than something we already have. Placing the points
*between* the landmarks needs no optimisation, only interpolation: each point sits
at the weighted barycentre of its nearest landmarks' coarse coordinates, which is
the Nystrom out-of-sample extension of landmark MDS.

The interpolation kernel has to be *local*, and this is worth stating because the
obvious choice fails. The FiLM affinity ``a(x)`` looks like the natural kernel --
it is already a softmax partition of unity over exactly these landmarks -- but at
the default temperature it is nearly uniform (on digits, 124 of 128 landmarks are
effective for the median point), so ``a(x) @ Z_mds`` collapses towards the
centroid and carries almost no local structure: trust_5 0.823 against PCA's 0.830,
i.e. no better than the init it was meant to replace. Inverse-distance weights
over the ``c`` nearest landmarks instead beat PCA everywhere measured -- trust_5
0.995 vs 0.977 and knn_overlap_5 0.552 vs 0.311 on swiss_roll, 0.999 vs 0.981 and
0.718 vs 0.279 on s_curve, 0.848 vs 0.830 and 0.101 vs 0.078 on digits. On s_curve
that interpolation alone reaches the *fully trained* neighbour overlap (0.718
against 0.725) before a single gradient step.

Fitting the encoder to those targets is then plain regression -- one forward per
point, no negatives, no triplets, no graph sampling -- which costs roughly a ninth
of a full training step at the same batch size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW

from ..distance import DistanceFn, chunked_cdist
from ..utils import chunk_ranges, get_logger

# Subsample used to measure the target layout's neighbour spacing. The median is
# stable well below the full set and this keeps the scaling O(1) in N.
SPACING_SAMPLE: int = 4096

# How many nearest landmarks a point interpolates between, and how sharply. The
# kernel has to stay local to carry local structure (see the module docstring);
# c=8 matches the graph's own ``C_BUCKETS`` shortlist width.
NEIGHBOUR_LANDMARKS: int = 8
INVERSE_DISTANCE_POWER: float = 2.0
# A point sitting exactly on a landmark would otherwise divide by zero; this
# floor makes it take that landmark's coordinate, which is the right answer.
NEAR_ZERO: float = 1e-8

# Accepted ``warm_start_layout`` values. "auto" ranks the rest and takes the best;
# the others name one directly.
LAYOUTS: Tuple[str, ...] = ("auto", "isomap", "spectral", "pca")


def _median_nn_dist(Z: torch.Tensor, sample: int = SPACING_SAMPLE, seed: int = 0) -> float:
    """Median nearest-neighbour distance of a layout, on a subsample."""
    n = Z.shape[0]
    if n < 2:
        return 0.0
    if n > sample:
        g = torch.Generator().manual_seed(seed)
        Z = Z[torch.randperm(n, generator=g)[:sample]]
    d = torch.cdist(Z, Z)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


@torch.no_grad()
def spectral_layout(
    edges: torch.Tensor, weights: torch.Tensor, n: int, d: int, seed: int = 0
) -> torch.Tensor:
    """Leading eigenvectors of the fuzzy graph's normalised affinity.

    This is what UMAP initialises from, and it is the candidate that wins exactly
    where Isomap fails: when the graph geodesics are not realisable in ``d``
    dimensions the MDS spectrum goes negative and its layout is a lossy projection,
    whereas eigenvectors of a diffusion operator are defined whatever the shape.

    The trade runs the other way on a plain sheet. On an elongated domain the
    leading eigenvectors are successive harmonics of the *same* direction, so the
    second axis comes out as a function of the first and the layout is a curve
    rather than a surface -- measurably worse than PCA on swiss_roll. Neither
    method dominates, which is why :func:`rank_inits` exists.

    Parameters
    ----------
    edges : (E, 2) int64 -- upper-triangle representative index pairs.
    weights : (E,) float32 -- memberships.
    n : number of nodes.
    d : embedding dimension.
    seed : fixes ARPACK's starting vector. Without it the layout -- and so the
        whole warm start -- is not reproducible across runs of the same config.
    """
    import numpy as np
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    e = edges.cpu().numpy()
    w = weights.cpu().numpy().astype(np.float64)
    W = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-30)))
    # Shifted so the wanted end of the spectrum is the largest in magnitude, which
    # is the end ARPACK converges on reliably.
    S = inv_sqrt @ W @ inv_sqrt + eye(n) * 2.0
    v0 = np.random.default_rng(seed).standard_normal(n)
    vals, vecs = eigsh(S, k=min(d + 1, n - 1), which="LM", v0=v0)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    # Column 0 is the stationary vector: constant, carries no geometry.
    coords = np.asarray(inv_sqrt @ vecs[:, 1 : d + 1])
    # Eigenvector signs are arbitrary; pin them so the layout does not mirror
    # between runs. Keyed on the largest-magnitude entry, not the column sum: these
    # eigenvectors are essentially mean-zero, so a sum's sign is numerical noise.
    pivot = np.argmax(np.abs(coords), axis=0)
    flip = np.where(coords[pivot, np.arange(coords.shape[1])] < 0, -1.0, 1.0)
    coords = coords * flip[None, :]
    # Cast before the final pin: float32 can reorder near-ties on |max|, and the
    # torch tensor is what callers (and tests) see.
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    pivot = np.argmax(np.abs(coords), axis=0)
    flip = np.where(coords[pivot, np.arange(coords.shape[1])] < 0, -1.0, 1.0)
    coords = coords * flip[None, :]
    return torch.as_tensor(coords, dtype=torch.float32)


def _knn_indices(Z: torch.Tensor, k: int):
    """Indices of the ``k`` nearest other rows, via a KD-tree (d is small here)."""
    from scipy.spatial import cKDTree

    A = Z.detach().cpu().numpy()
    return cKDTree(A).query(A, k=k + 1)[1][:, 1:]


@torch.no_grad()
def rank_inits(
    candidates: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    X_ref: torch.Tensor,
    reference_knn: torch.Tensor,
    dist_fn: DistanceFn,
    min_dist: float,
    seed: int = 0,
) -> "List[Tuple[str, float]]":
    """Score each candidate layout by neighbour agreement, best first.

    Every candidate is interpolated onto the same reference points and judged
    against neighbours the graph build already computed, so the comparison is at
    one resolution and costs a KD-tree query per candidate. This measures the
    quantity of interest directly rather than predicting it from a diagnostic:
    across s_curve, swiss_roll, swiss_cone, digits, pdb and smoke these scores
    matched the same layouts' full-resolution neighbour overlap to three decimals,
    so there is no threshold to calibrate and nothing to transfer between datasets.

    Parameters
    ----------
    candidates : name -> ``(anchors, anchor_layout)``.
    X_ref : (R, D) points to score on -- the graph's representatives.
    reference_knn : (R, k) int64 -- ambient neighbours of those points.
    """
    k = int(reference_knn.shape[1])
    if k == 0 or reference_knn.shape[0] == 0:
        raise ValueError(
            "ranking init layouts needs reference neighbours; got an empty "
            "reference_knn. Name a layout explicitly instead of 'auto'."
        )
    truth = [set(row) for row in reference_knn.cpu().numpy().tolist()]
    scored: List[Tuple[str, float]] = []
    for name, (anchors, Z_anchor) in candidates.items():
        Z_ref = nystrom_targets(
            X_ref, anchors, Z_anchor, dist_fn, min_dist=min_dist, seed=seed
        )
        nb = _knn_indices(Z_ref, k)
        overlap = sum(len(truth[i] & set(nb[i])) for i in range(len(truth)))
        scored.append((name, overlap / (len(truth) * k)))
    return sorted(scored, key=lambda t: -t[1])


@torch.no_grad()
def nystrom_targets(
    X: torch.Tensor,
    X_lm: torch.Tensor,
    Z_mds: torch.Tensor,
    dist_fn: DistanceFn,
    min_dist: float,
    c: int = NEIGHBOUR_LANDMARKS,
    power: float = INVERSE_DISTANCE_POWER,
    seed: int = 0,
) -> torch.Tensor:
    """Landmark MDS extended to every point, rescaled to the layout's own units.

    Parameters
    ----------
    X : (N, D) points to place.
    X_lm : (L, D) landmark coordinates, row-aligned with ``Z_mds``.
    Z_mds : (L, d) classical-MDS coordinates of those landmarks.
    dist_fn : the ambient metric, so "nearest" means what the graph meant by it.
    min_dist : neighbour spacing the geometric term treats as "close enough".
    c : how many nearest landmarks each point interpolates between.
    power : exponent on inverse distance; higher is more local.

    Returns
    -------
    (N, d) float32

    Notes
    -----
    The rescaling matters: ``Z_mds`` carries geodesic distance units, which have
    nothing to do with the scale the attraction/repulsion equilibrium settles at,
    so regressing onto raw MDS coordinates would hand the layout a correct shape
    at the wrong size and make the first phase of training an argument about
    scale. Setting the *median nearest-neighbour distance* to ``min_dist`` starts
    neighbours exactly where the geometric term wants them and leaves the shape
    untouched, since a uniform scaling is the only thing applied.

    Implemented via :func:`nystrom_targets_streaming` so the in-memory and
    memmap paths share one numeric kernel.
    """
    return nystrom_targets_streaming(
        X,
        X_lm,
        Z_mds,
        dist_fn,
        min_dist,
        seed=seed,
        chunk=8192,
        shortlist_idx=None,
        c=c,
        power=power,
    )


def _rows_as_tensor(X, s: int, e: int, device: torch.device) -> torch.Tensor:
    """Slice ``X[s:e]`` into a float32 tensor on ``device`` (tensor or memmap)."""
    rows = X[s:e]
    if isinstance(rows, torch.Tensor):
        return rows.to(device=device, dtype=torch.float32)
    # Copy so memmap views become writable torch tensors without a warning.
    return torch.as_tensor(np.array(rows, dtype=np.float32, copy=True), device=device)


def _shortlist_as_tensor(shortlist_idx, s: int, e: int, device: torch.device) -> torch.Tensor:
    rows = shortlist_idx[s:e]
    if isinstance(rows, torch.Tensor):
        return rows.to(device=device, dtype=torch.int64)
    return torch.as_tensor(np.array(rows, dtype=np.int64, copy=True), device=device)


def _topk_within_shortlist(
    dist_fn: DistanceFn,
    Xc: torch.Tensor,
    anchors: torch.Tensor,
    shortlist: torch.Tensor,
    c: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest-``c`` among each row's shortlisted anchors (paired distances)."""
    b, c_sl = int(shortlist.shape[0]), int(shortlist.shape[1])
    take = min(c, c_sl, anchors.shape[0])
    # Distances to shortlisted anchors without materialising (b, L).
    vals_all = torch.empty(b, c_sl, dtype=torch.float32, device=Xc.device)
    tile = 512
    for j in range(c_sl):
        Bj = anchors[shortlist[:, j]]
        for ss, ee in chunk_ranges(b, tile):
            block = dist_fn(Xc[ss:ee], Bj[ss:ee])
            vals_all[ss:ee, j] = block.diag()
    if take >= c_sl:
        return vals_all, shortlist
    vals, order = torch.topk(vals_all, k=take, dim=1, largest=False)
    idx = torch.gather(shortlist, 1, order)
    return vals, idx


@torch.no_grad()
def nystrom_targets_streaming(
    X_memmap_or_tensor,
    anchors: torch.Tensor,
    Z_anchor: torch.Tensor,
    dist_fn: DistanceFn,
    min_dist: float,
    seed: int = 0,
    chunk: int = 8192,
    shortlist_idx=None,
    c: int = NEIGHBOUR_LANDMARKS,
    power: float = INVERSE_DISTANCE_POWER,
) -> torch.Tensor:
    """Nyström extension over ``X`` in streaming chunks (tensor or memmap).

    Matches :func:`nystrom_targets` exactly (shared implementation). With a
    top-c ``shortlist_idx``, interpolation is restricted to those landmark
    columns (distances only among the shortlist).

    Parameters
    ----------
    X_memmap_or_tensor : (N, D) torch tensor, numpy array, or memmap.
    anchors : (L, D) landmark coordinates, row-aligned with ``Z_anchor``.
    Z_anchor : (L, d) coarse layout of those landmarks.
    shortlist_idx : (N, c_sl) int64, optional
        Per-row landmark indices (e.g. graph ``assign_topc``). When set, each
        point interpolates only among these columns.
    chunk : rows of ``X`` loaded per pass.
    """
    anchors = torch.as_tensor(anchors, dtype=torch.float32)
    Z_anchor = torch.as_tensor(Z_anchor, dtype=torch.float32)
    device = anchors.device
    n = int(X_memmap_or_tensor.shape[0])
    d_out = int(Z_anchor.shape[1])
    c = int(min(c, anchors.shape[0]))
    Z = torch.empty(n, d_out, dtype=torch.float32, device=device)
    for s, e in chunk_ranges(n, int(chunk)):
        Xc = _rows_as_tensor(X_memmap_or_tensor, s, e, device)
        if shortlist_idx is None:
            vals, idx = chunked_cdist(
                dist_fn, Xc, anchors, topk=c, out_device=device
            )
        else:
            sl = _shortlist_as_tensor(shortlist_idx, s, e, device)
            vals, idx = _topk_within_shortlist(dist_fn, Xc, anchors, sl, c)
        w = 1.0 / vals.clamp_min(NEAR_ZERO) ** float(power)
        w = w / w.sum(dim=1, keepdim=True)
        Z[s:e] = (w.unsqueeze(-1) * Z_anchor.to(device)[idx]).sum(dim=1)
    spacing = _median_nn_dist(Z, seed=seed)
    if spacing > 0:
        Z = Z * (float(min_dist) / spacing)
    return Z.contiguous()


def save_shortlist(path: Union[str, Path], idx) -> Path:
    """Persist a top-c shortlist index array as ``.npy`` (int64)."""
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    if isinstance(idx, torch.Tensor):
        arr = idx.detach().cpu().numpy()
    else:
        arr = np.asarray(idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=np.int64))
    return path


def load_shortlist(path: Union[str, Path]) -> torch.Tensor:
    """Load a shortlist saved by :func:`save_shortlist` as int64 ``(N, c)``."""
    arr = np.load(Path(path))
    return torch.as_tensor(arr, dtype=torch.int64)


def pretrain_to_targets(
    model: torch.nn.Module,
    X: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    batch: int,
    lr: float,
    seed: int = 0,
) -> float:
    """Regress the encoder onto ``targets``; returns the final mean batch loss.

    Only encoder parameters are updated. The landmark coordinates and
    temperatures stay frozen: the targets were computed from them, so letting
    them move here would make the objective chase its own tail, and the main loop
    is where they are supposed to adapt anyway.
    """
    log = get_logger()
    device = next(model.parameters()).device
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if not enc_params or steps <= 0:
        return float("nan")
    opt = AdamW(enc_params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    was_training = model.training
    model.train()
    last = float("nan")
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        z, _, _ = model(X[idx].to(device))
        loss = ((z - targets[idx].to(device)) ** 2).mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            log.warning("warm start: non-finite loss at step %d — stopping", step)
            break
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if step % max(1, steps // 4) == 0:
            log.info("warm start: step %d/%d mse=%.4g", step, steps, last)
    if not was_training:
        model.eval()
    return last


def warm_start(
    model: torch.nn.Module,
    X: torch.Tensor,
    candidates: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    dist_fn: DistanceFn,
    *,
    layout: str,
    X_ref: torch.Tensor,
    reference_knn: torch.Tensor,
    steps: int,
    batch: int,
    lr: float,
    min_dist: float,
    seed: int = 0,
) -> dict:
    """Pick a coarse layout, extend it over ``X``, and fit the encoder to it.

    ``layout="auto"`` ranks the candidates by neighbour agreement and takes the
    best; anything else names one directly. ``"pca"`` winning is a real outcome and
    means the coarse layouts had nothing to add on this data, so the regression is
    skipped rather than spent teaching the encoder something PCA already gives it.

    Returns a diagnostics dict recorded on the fit result.
    """
    log = get_logger()
    ranking: List[Tuple[str, float]] = []
    if layout == "auto":
        if len(candidates) > 1:
            ranking = rank_inits(
                candidates, X_ref, reference_knn, dist_fn, min_dist, seed=seed
            )
            log.info(
                "warm start: candidate layouts by %d-neighbour agreement on %d "
                "representatives: %s",
                int(reference_knn.shape[1]),
                X_ref.shape[0],
                ", ".join(f"{n}={s:.3f}" for n, s in ranking),
            )
        chosen = ranking[0][0] if ranking else next(iter(candidates))
    elif layout in candidates:
        chosen = layout
    else:
        log.warning(
            "warm_start_layout=%r is unavailable (have %s); skipping the warm start",
            layout,
            sorted(candidates),
        )
        return {}

    info: dict = {
        "warm_start_layout": chosen,
        "warm_start_ranking": {n: round(s, 4) for n, s in ranking},
    }
    if chosen == "pca":
        log.info(
            "warm start: PCA scored best, so the coarse layouts add nothing here "
            "and the encoder is left at its PCA seed"
        )
        return {**info, "warm_start_steps": 0}

    anchors, Z_anchor = candidates[chosen]
    targets = nystrom_targets(
        X, anchors, Z_anchor, dist_fn, min_dist=min_dist, seed=seed
    )
    spacing = _median_nn_dist(targets, seed=seed)
    log.info(
        "warm start: %s layout, %d anchors interpolated over %d points on the %d "
        "nearest, spacing=%.4g (min_dist=%.4g), %d steps at lr=%.3g",
        chosen,
        anchors.shape[0],
        X.shape[0],
        min(NEIGHBOUR_LANDMARKS, anchors.shape[0]),
        spacing,
        min_dist,
        steps,
        lr,
    )
    mse = pretrain_to_targets(
        model, X, targets, steps=steps, batch=batch, lr=lr, seed=seed
    )
    log.info("warm start: final mse=%.4g", mse)
    return {
        **info,
        "warm_start_steps": int(steps),
        "warm_start_mse": mse,
        "warm_start_target_spacing": spacing,
    }

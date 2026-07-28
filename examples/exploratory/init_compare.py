#!/usr/bin/env python
"""Which coarse layout is the best warm start?

``leanmap.warmstart`` fits the encoder to a coarse layout of the landmarks before
training begins. Classical MDS of the landmark geodesic matrix -- i.e. landmark
Isomap -- is what it uses, but that is a choice, not a necessity, and Isomap has a
known weakness the fit already reports: when the geodesics are not embeddable in
``d_out`` dimensions the MDS spectrum goes negative and the layout is a lossy
projection of something that does not fit.

Spectral methods do not have that failure mode, because they never try to realise
a distance matrix -- they take leading eigenvectors of a diffusion operator, which
are defined whatever the manifold's shape. That is also what UMAP initialises
from. They have their own failure mode instead: leading eigenvectors of an
elongated domain can be successive harmonics of the *same* direction, giving a
folded layout.

So this scores each candidate as a layout, against the PCA init it would replace,
holding the landmarks and the interpolation fixed so only the algorithm varies.

Usage::

    python examples/exploratory/init_compare.py --datasets s_curve swiss_roll digits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data"

# Zelnik-Manor/Perona self-tuning bandwidth: the k-th neighbour distance per node,
# so one kernel width does not have to fit every part of the manifold.
SELF_TUNE_K = 7


def affinity_from_distances(G: np.ndarray, k: int = SELF_TUNE_K) -> np.ndarray:
    """Self-tuning Gaussian affinity from a dense distance matrix."""
    G = np.asarray(G, dtype=np.float64).copy()
    np.fill_diagonal(G, 0.0)
    finite = np.isfinite(G)
    G[~finite] = np.nanmax(G[finite]) if finite.any() else 1.0
    order = np.sort(G, axis=1)
    sigma = np.maximum(order[:, min(k, G.shape[1] - 1)], 1e-12)
    W = np.exp(-(G**2) / np.outer(sigma, sigma))
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


def diffusion_map(W: np.ndarray, d: int = 2, alpha: float = 1.0, t: float = 1.0):
    """Coifman-Lafon diffusion coordinates from a symmetric affinity.

    ``alpha=1`` divides the sampling density out, so the operator approaches the
    Laplace-Beltrami operator of the manifold and the layout reflects geometry
    rather than where the points happen to be dense. ``t`` is the diffusion time,
    which only reweights the axes.
    """
    deg = W.sum(axis=1)
    Wa = W / np.outer(deg**alpha, deg**alpha)
    da = Wa.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(da, 1e-30))
    S = Wa * np.outer(inv_sqrt, inv_sqrt)
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # vals[0] / vecs[:, 0] are the trivial stationary pair: constant, no geometry.
    psi = vecs[:, 1 : d + 1] * inv_sqrt[:, None]
    return psi * (np.abs(vals[1 : d + 1]) ** t)[None, :], vals


def laplacian_eigenmap(W: np.ndarray, d: int = 2):
    """Normalised-Laplacian eigenmap: the same axes with no diffusion reweighting."""
    coords, _ = diffusion_map(W, d=d, alpha=0.0, t=0.0)
    return coords


def harmonic_score(coords: np.ndarray) -> float:
    """R^2 of axis 2 regressed on a spline of axis 1; high means a folded layout.

    The repeated-eigendirection failure looks exactly like this: the second axis is
    a function of the first rather than an independent direction, so the layout
    curls along one coordinate instead of spanning two.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    x, y = coords[:, :1], coords[:, 1]
    if np.std(y) < 1e-12:
        return 1.0
    m = make_pipeline(
        SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-4, 2, 9))
    )
    pred = m.fit(x, y).predict(x)
    return float(max(0.0, 1.0 - np.var(y - pred) / np.var(y)))


def spectral_sparse(edges: np.ndarray, weights: np.ndarray, n: int, d: int = 2):
    """Normalised-affinity spectral embedding of a sparse symmetric graph.

    Same axes as :func:`diffusion_map` with ``alpha=0``, but via ARPACK so the
    representative-resolution graph does not need a dense factorisation.
    """
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    W = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-30)))
    S = inv_sqrt @ W @ inv_sqrt
    # shift to keep ARPACK on the well-separated end of the spectrum
    vals, vecs = eigsh(S + eye(n) * 2.0, k=d + 1, which="LM")
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    return np.asarray(inv_sqrt @ vecs[:, 1 : d + 1])


def overlap(A_nb: np.ndarray, B_nb: np.ndarray) -> float:
    """Mean fraction of shared neighbours between two neighbour-index arrays."""
    k = A_nb.shape[1]
    return float(np.mean([len(set(p) & set(q)) for p, q in zip(A_nb, B_nb)]) / k)


def _knn_from_distance_matrix(G: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` nearest entries per row of a distance matrix."""
    G = np.asarray(G, dtype=np.float64).copy()
    np.fill_diagonal(G, np.inf)
    G[~np.isfinite(G)] = np.inf
    return np.argsort(G, axis=1)[:, :k]


def score(X: np.ndarray, Z: np.ndarray, nb_x: np.ndarray, k: int) -> Tuple[float, float]:
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    nb_z = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    ov = np.mean([len(set(p) & set(q)) for p, q in zip(nb_x, nb_z)]) / k
    return trustworthiness(X, Z, n_neighbors=k), ov


def candidates(X: np.ndarray, cfg, k: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Every warm-start layout, all interpolated to the full point set the same way."""
    from sklearn.decomposition import PCA

    from leanmap.graph import build_graph_pyramid
    from leanmap.landmarks import classical_mds, landmark_geodesic_matrix
    from leanmap.metrics import fit_natural_scale, get_metric
    from leanmap.warmstart import nystrom_targets

    Xt = torch.as_tensor(X)
    metric = fit_natural_scale(get_metric("l2"), Xt)
    graphs, M, *_ = build_graph_pyramid(
        Xt,
        metric,
        pyramid_scales=cfg.pyramid_scales,
        n_neighbors=cfg.n_neighbors,
        n_landmarks=cfg.n_landmarks,
        seed=0,
    )
    X_lm, G, finite = landmark_geodesic_matrix(Xt, M, metric, n_neighbors=cfg.n_neighbors)
    Z_mds, diag = classical_mds(G, d=2, finite=finite, return_diagnostics=True)
    W = affinity_from_distances(G.numpy())
    dmap, vals = diffusion_map(W, d=2)
    lap = laplacian_eigenmap(W, d=2)

    def extend(Z_lm: np.ndarray) -> np.ndarray:
        return nystrom_targets(
            Xt,
            X_lm,
            torch.as_tensor(np.ascontiguousarray(Z_lm), dtype=torch.float32),
            metric,
            min_dist=cfg.min_dist,
        ).numpy()

    # UMAP's own init for reference: spectral on the fuzzy graph itself, over the
    # epsilon-net representatives, then interpolated like the rest.
    rep_idx = graphs[0].reps.rep_idx.numpy()
    e = graphs[0].edges.numpy()
    w = graphs[0].weights.numpy()
    R = len(rep_idx)
    Wf = np.zeros((R, R))
    Wf[e[:, 0], e[:, 1]] = w
    Wf = Wf + Wf.T
    spec_rep, _ = diffusion_map(Wf, d=2)
    fuzzy = nystrom_targets(
        Xt,
        Xt[torch.as_tensor(rep_idx)],
        torch.as_tensor(np.ascontiguousarray(spec_rep), dtype=torch.float32),
        metric,
        min_dist=cfg.min_dist,
    ).numpy()

    # Resolution control. The fuzzy-graph spectral gets one anchor per
    # representative and the landmark layouts get L, so comparing them directly
    # would confound "spectral vs MDS" with "1786 anchors vs 128". Isomap at the
    # same resolution -- geodesics among all reps -- separates the two.
    X_rep = Xt[torch.as_tensor(rep_idx)]
    _, G_rep, fin_rep = landmark_geodesic_matrix(
        Xt, X_rep, metric, n_neighbors=cfg.n_neighbors
    )
    Z_mds_rep, diag_rep = classical_mds(
        G_rep, d=2, finite=fin_rep, return_diagnostics=True
    )
    isomap_full = nystrom_targets(
        Xt, X_rep, Z_mds_rep, metric, min_dist=cfg.min_dist
    ).numpy()

    info = {
        "L": int(M.shape[0]),
        "mds_neg_eigen": float(diag["mds_neg_eigen_ratio"]),
        "mds_neg_eigen_rep": float(diag_rep["mds_neg_eigen_ratio"]),
        "spectral_gap": float(vals[1] / max(vals[2], 1e-30)),
        "n_reps": R,
    }
    return {
        "PCA-2D (current init)": PCA(n_components=2).fit_transform(X),
        f"landmark Isomap (L={M.shape[0]}, in use)": extend(Z_mds.numpy()),
        f"landmark diffusion map (L={M.shape[0]})": extend(dmap),
        f"landmark Laplacian eigenmap (L={M.shape[0]})": extend(lap),
        f"Isomap at rep resolution ({R})": isomap_full,
        f"fuzzy-graph spectral ({R}, UMAP-like)": fuzzy,
    }, info


def probe(X: np.ndarray, cfg, k: int) -> Tuple[Dict[str, Dict[str, float]], Dict]:
    """Rank PCA / Isomap / diffusion map from the landmarks, then from all points.

    The landmark-only columns cost one Dijkstra per landmark and a few ``L x L``
    decompositions -- everything the warm start already builds. The full-resolution
    column is the thing they would have to predict. Whether the cheap ranking picks
    the same winner is the whole question.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    from leanmap.graph import build_graph_pyramid
    from leanmap.landmarks import classical_mds, landmark_geodesic_matrix
    from leanmap.metrics import fit_natural_scale, get_metric
    from leanmap.warmstart import nystrom_targets

    Xt = torch.as_tensor(X)
    metric = fit_natural_scale(get_metric("l2"), Xt)
    graphs, M, *_ = build_graph_pyramid(
        Xt,
        metric,
        pyramid_scales=cfg.pyramid_scales,
        n_neighbors=cfg.n_neighbors,
        n_landmarks=cfg.n_landmarks,
        seed=0,
    )
    X_lm, G, finite = landmark_geodesic_matrix(
        Xt, M, metric, n_neighbors=cfg.n_neighbors
    )
    Gn = G.numpy()
    Z_mds, diag = classical_mds(G, d=2, finite=finite, return_diagnostics=True)
    W = affinity_from_distances(Gn)
    dmap, _ = diffusion_map(W, d=2)

    lm_layouts = {
        "PCA": PCA(n_components=2).fit_transform(X_lm.numpy()),
        "Isomap": Z_mds.numpy(),
        "diffusion": dmap,
    }
    # Ground truth at landmark resolution, two ways: geodesic neighbours (the
    # manifold's own notion, but also Isomap's objective, so it may flatter it) and
    # ambient neighbours (independent of every candidate).
    gt_geo = _knn_from_distance_matrix(Gn, k)
    Xl = X_lm.numpy()
    gt_amb = NearestNeighbors(n_neighbors=k + 1).fit(Xl).kneighbors(Xl)[1][:, 1:]

    # Rep-resolution scoring: every candidate interpolated onto the same points and
    # judged against the graph's own ambient kNN, which construction already
    # computed. This is the only comparison that is fair across candidates living
    # at different resolutions, and it costs one 2-D kNN each.
    R = int(graphs[0].reps.rep_idx.shape[0])
    rep_idx = graphs[0].reps.rep_idx.numpy()
    X_rep = Xt[torch.as_tensor(rep_idx)]
    gt_rep = graphs[0].knn_idx.numpy()[:, :k]

    def to_reps(anchors: torch.Tensor, Z_anchor: np.ndarray) -> np.ndarray:
        return nystrom_targets(
            X_rep,
            anchors,
            torch.as_tensor(np.ascontiguousarray(Z_anchor), dtype=torch.float32),
            metric,
            min_dist=cfg.min_dist,
        ).numpy()

    def rep_score(Z_rep: np.ndarray) -> float:
        nb = NearestNeighbors(n_neighbors=k + 1).fit(Z_rep).kneighbors(Z_rep)[1][:, 1:]
        return overlap(nb, gt_rep)

    nb_x = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    out: Dict[str, Dict[str, float]] = {}
    for name, Z_lm in lm_layouts.items():
        nb_lm = NearestNeighbors(n_neighbors=k + 1).fit(Z_lm).kneighbors(Z_lm)[1][:, 1:]
        if name == "PCA":
            # The real PCA init is a linear map applied to every point, not an
            # interpolation of a landmark layout, so score it that way.
            pca = PCA(n_components=2).fit(X)
            Z_full = pca.transform(X)
            Z_rep = pca.transform(X_rep.numpy())
        else:
            Z_full = nystrom_targets(
                Xt,
                X_lm,
                torch.as_tensor(np.ascontiguousarray(Z_lm), dtype=torch.float32),
                metric,
                min_dist=cfg.min_dist,
            ).numpy()
            Z_rep = to_reps(X_lm, Z_lm)
        out[name] = {
            "lm_geo": overlap(nb_lm, gt_geo),
            "lm_amb": overlap(nb_lm, gt_amb),
            "rep": rep_score(Z_rep),
            "folded": harmonic_score(Z_lm),
            "full": score(X, Z_full, nb_x, k)[1],
        }

    # Not landmark-only: needs the representative graph. Included because it is the
    # candidate that wins where Isomap's diagnostic says Isomap is failing.
    spec = spectral_sparse(
        graphs[0].edges.numpy(), graphs[0].weights.numpy(), R, d=2
    )
    Z_spec = nystrom_targets(
        Xt,
        X_rep,
        torch.as_tensor(np.ascontiguousarray(spec), dtype=torch.float32),
        metric,
        min_dist=cfg.min_dist,
    ).numpy()
    out["fuzzy-spectral"] = {
        "lm_geo": float("nan"),
        "lm_amb": float("nan"),
        "rep": rep_score(spec),
        "folded": harmonic_score(Z_spec),
        "full": score(X, Z_spec, nb_x, k)[1],
    }
    return out, {
        "L": int(M.shape[0]),
        "n_reps": R,
        "mds_neg_eigen": float(diag["mds_neg_eigen_ratio"]),
    }


def cmd_probe(args) -> None:
    from leanmap import PLANEConfig

    agree_geo = agree_amb = agree_rep = total = 0
    rows = []
    for ds in args.datasets:
        X = np.load(DATA / f"{ds}_X.npy").astype(np.float32)
        cfg = PLANEConfig.for_scale(len(X))
        res, info = probe(X, cfg, args.k)
        lm_only = {n: v for n, v in res.items() if n != "fuzzy-spectral"}
        best_geo = max(lm_only, key=lambda n: lm_only[n]["lm_geo"])
        best_amb = max(lm_only, key=lambda n: lm_only[n]["lm_amb"])
        best_of_3 = max(lm_only, key=lambda n: lm_only[n]["full"])
        # the selector that could actually ship: all candidates, one fair scale
        pick_rep = max(res, key=lambda n: res[n]["rep"])
        best_any = max(res, key=lambda n: res[n]["full"])
        total += 1
        agree_geo += best_geo == best_of_3
        agree_amb += best_amb == best_of_3
        agree_rep += pick_rep == best_any
        print(
            f"\n=== {ds}: N={len(X)} L={info['L']} reps={info['n_reps']} "
            f"MDS negative-eigenvalue mass={info['mds_neg_eigen']:.3f}"
        )
        print(
            f"    {'candidate':16s} {'lm(geo)':>8s} {'lm(amb)':>8s} {'rep':>7s} "
            f"{'folded':>7s} {'FULL':>7s}"
        )
        for n, v in res.items():
            mark = " <- best" if n == best_any else ""
            print(
                f"    {n:16s} {v['lm_geo']:8.3f} {v['lm_amb']:8.3f} {v['rep']:7.3f} "
                f"{v['folded']:7.2f} {v['full']:7.3f}{mark}"
            )
        print(
            f"    landmark-only pick of 3: geo->{best_geo}, amb->{best_amb} "
            f"(truth {best_of_3});  rep-resolution pick of all: {pick_rep} "
            f"(truth {best_any})"
        )
        rows.append((ds, info["mds_neg_eigen"], best_of_3, pick_rep, best_any))

    print(f"\n{'=' * 78}\nhow often does a cheap ranking name the right winner?")
    print(f"  landmark-only, geodesic truth, 3 candidates: {agree_geo}/{total}")
    print(f"  landmark-only, ambient truth,  3 candidates: {agree_amb}/{total}")
    print(f"  rep-resolution, all candidates:              {agree_rep}/{total}")
    print(
        f"\n  {'dataset':12s} {'neg_eigen':>10s} {'best of 3':>12s} "
        f"{'rep pick':>15s} {'best overall':>15s}"
    )
    for ds, ne, b3, pr, ba in rows:
        flag = "" if pr == ba else "   MISMATCH"
        print(f"  {ds:12s} {ne:10.3f} {b3:>12s} {pr:>15s} {ba:>15s}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["s_curve", "swiss_roll", "digits"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="rank candidates from landmarks only and check against full resolution",
    )
    args = ap.parse_args()
    if args.probe:
        cmd_probe(args)
        return

    from sklearn.neighbors import NearestNeighbors

    from leanmap import PLANEConfig

    for ds in args.datasets:
        X = np.load(DATA / f"{ds}_X.npy").astype(np.float32)
        cfg = PLANEConfig.for_scale(len(X))
        cands, info = candidates(X, cfg, args.k)
        nb_x = (
            NearestNeighbors(n_neighbors=args.k + 1).fit(X).kneighbors(X)[1][:, 1:]
        )
        print(
            f"\n=== {ds}: N={len(X)}  MDS negative-eigenvalue mass="
            f"{info['mds_neg_eigen']:.3f} at L={info['L']}, "
            f"{info['mds_neg_eigen_rep']:.3f} at rep resolution"
        )
        print(f"    {'layout':38s} {'trust_5':>8s} {'knn_ov_5':>9s} {'folded':>7s}")
        for name, Z in cands.items():
            t, ov = score(X, Z, nb_x, args.k)
            print(f"    {name:38s} {t:8.4f} {ov:9.4f} {harmonic_score(Z):7.2f}")


if __name__ == "__main__":
    main()

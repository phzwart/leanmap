#!/usr/bin/env python
"""Stage B/C hyperparameter selection for leanmap: held-out geometry scoring.

Design decisions that make this a selection procedure rather than a sweep:

1. **tau is set by perplexity, not by value.** Every candidate ``n_landmarks``
   gets its own ``tau_scale``, calibrated so the median affinity perplexity
   matches ``--target-perp``. Without this, changing L silently changes the
   effective conditioning bandwidth and no two configs are comparable.

2. **Scoring happens out of sample.** leanmap is parametric, so we fit on a
   train split and score held-out points. At N~5k the model can otherwise
   memorise the training graph and every in-sample metric looks better than the
   embedding deserves.

3. **The objective never rewards clusteredness.** Silhouette in Z is reported
   only as a *diagnostic pair* with silhouette in X: high in Z and low in X is
   the signature of decoder-manufactured islands, not real groups.

Usage
-----
    python hparam_select.py --search landmarks
    python hparam_select.py --search perp
    python hparam_select.py --search weights
    python hparam_select.py --search seeds --n-landmarks 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _demo import OUT_DIR, fit_embed
from hparam_probe import affinity_stats, default_tau, euclid, solve_tau_scale
from leanmap.landmarks import fps_init_indices
from pdb_validation import DATA, load_table

# 3-tuples: at N=5000 only 3 pyramid levels are built (coarsening floors at
# pyramid_min_reps=256), so a 4th entry never reaches a level.
WEIGHT_GRID = [
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 2.0),
    (1.0, 2.0, 8.0),
    (1.0, 4.0, 16.0),
    (1.0, 8.0, 32.0),
    (8.0, 1.0, 1.0),
]


def calibrate_tau_scale(X: torch.Tensor, L: int, target_perp: float, seed: int) -> float:
    idx = fps_init_indices(X, euclid, L, seed=seed)
    M = X[idx].contiguous()
    tau0 = default_tau(M)
    Dm = torch.cdist(X, M)
    return solve_tau_scale(Dm, tau0, target_perp)


def knn_indices(A: np.ndarray, k: int) -> np.ndarray:
    d = torch.cdist(torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32))
    d.fill_diagonal_(float("inf"))
    return torch.topk(d, k, dim=1, largest=False).indices.numpy()


def knn_recall(Xh: np.ndarray, Zh: np.ndarray, k: int = 10) -> float:
    ax, az = knn_indices(Xh, k), knn_indices(Zh, k)
    hits = [len(set(ax[i]) & set(az[i])) for i in range(len(ax))]
    return float(np.mean(hits) / k)


def log_density(A: np.ndarray, k: int = 10) -> np.ndarray:
    d = torch.cdist(torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32))
    d.fill_diagonal_(float("inf"))
    r = torch.topk(d, k, dim=1, largest=False).values[:, -1].numpy()
    return -np.log(np.clip(r, 1e-12, None))


def shepard_banded(Xh: np.ndarray, Zh: np.ndarray, n_pairs: int, seed: int) -> dict:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(Xh)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dx = np.linalg.norm(Xh[i] - Xh[j], axis=1)
    dz = np.linalg.norm(Zh[i] - Zh[j], axis=1)
    out = {"shepard_rho": float(spearmanr(dx, dz).correlation)}
    edges = np.quantile(dx, [0.0, 1 / 3, 2 / 3, 1.0])
    for b, name in enumerate(["local", "mid", "global"]):
        m = (dx >= edges[b]) & (dx <= edges[b + 1])
        out[f"shepard_{name}"] = (
            float(spearmanr(dx[m], dz[m]).correlation) if m.sum() > 32 else float("nan")
        )
    return out


def geodesic_shepard(
    Xh: np.ndarray,
    Zh: np.ndarray,
    k: int,
    seed: int,
    n_sources: int = 96,
    max_targets: int = 384,
) -> dict:
    """Graph-geodesic vs embedded distance, banded by geodesic range.

    This is the yardstick the graph pyramid actually targets: coarse levels
    supply long-range attraction that anchors *geodesic* structure, not ambient
    Euclidean structure. Also reports the unreachable-pair fraction, which is
    the "flew off to infinity" symptom the coarse backbone exists to prevent.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.stats import spearmanr

    n = len(Xh)
    d = torch.cdist(torch.as_tensor(Xh, dtype=torch.float32), torch.as_tensor(Xh, dtype=torch.float32))
    d.fill_diagonal_(float("inf"))
    vals, idx = torch.topk(d, k, dim=1, largest=False)
    rows = np.repeat(np.arange(n), k)
    cols = idx.numpy().ravel()
    w = vals.numpy().ravel()
    A = sparse.coo_matrix((w, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T).tocsr()  # symmetrize
    n_comp, _ = connected_components(A, directed=False)

    rng = np.random.default_rng(seed)
    src = rng.choice(n, size=min(n_sources, n), replace=False)
    D = dijkstra(A, directed=False, indices=src)
    gd, ed, n_inf, n_tot = [], [], 0, 0
    for si, s in enumerate(src):
        row = D[si]
        n_tot += n - 1
        n_inf += int(np.sum(~np.isfinite(row))) - 0
        finite = np.isfinite(row)
        finite[s] = False
        tgt = np.where(finite)[0]
        if tgt.size == 0:
            continue
        if tgt.size > max_targets:
            tgt = rng.choice(tgt, size=max_targets, replace=False)
        gd.append(row[tgt])
        ed.append(np.linalg.norm(Zh[tgt] - Zh[s], axis=1))
    if not gd:
        return {"geo_rho": float("nan")}
    gd = np.concatenate(gd)
    ed = np.concatenate(ed)
    out = {
        "geo_rho": float(spearmanr(gd, ed).correlation),
        "geo_unreachable": n_inf / max(n_tot, 1),
        "geo_components": int(n_comp),
    }
    edges = np.quantile(gd, [0.0, 1 / 3, 2 / 3, 1.0])
    for b, name in enumerate(["near", "mid", "far"]):
        m = (gd >= edges[b]) & (gd <= edges[b + 1])
        out[f"geo_{name}"] = (
            float(spearmanr(gd[m], ed[m]).correlation) if m.sum() > 32 else float("nan")
        )
    return out


def cluster_diagnostics(Xh: np.ndarray, Zh: np.ndarray, k: int, seed: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Zh)
    return {
        "sil_Z": float(silhouette_score(Zh, lab)),
        "sil_X": float(silhouette_score(Xh, lab)),
        "labels": lab,
    }


def score_config(
    X: np.ndarray,
    *,
    n_landmarks: int,
    tau_scale: float,
    weights: tuple,
    epochs: int,
    lr: float,
    seed: int,
    holdout: float,
    k: int,
    n_clusters: int,
    eval_n: int,
    pyramid_scales: int | None = None,
    pyramid_coarse_backbone: float | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_hold = int(holdout * n)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]

    t0 = time.time()
    result, _, _ = fit_embed(
        X[train_idx],
        epochs=epochs,
        seed=seed,
        n_landmarks=n_landmarks,
        tau_scale=tau_scale,
        learn_tau=False,
        pyramid_level_weights=weights,
        pyramid_scales=pyramid_scales,
        pyramid_coarse_backbone=pyramid_coarse_backbone,
        lr=lr,
    )
    with torch.no_grad():
        Zh, _ = result.embed(X[hold_idx])
    Zh = Zh.detach().cpu().numpy()
    Xh = X[hold_idx]

    # Subsample for the O(n^2) metrics.
    if len(Xh) > eval_n:
        sub = rng.choice(len(Xh), eval_n, replace=False)
        Xh, Zh = Xh[sub], Zh[sub]

    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness

    trust = float(trustworthiness(Xh, Zh, n_neighbors=k))
    rec = knn_recall(Xh, Zh, k=k)
    shep = shepard_banded(Xh, Zh, n_pairs=32768, seed=seed)
    geo = geodesic_shepard(Xh, Zh, k, seed)
    dens = float(spearmanr(log_density(Xh, k), log_density(Zh, k)).correlation)
    clus = cluster_diagnostics(Xh, Zh, n_clusters, seed)

    parts = [trust, rec, max(shep["shepard_rho"], 0.0), max(dens, 0.0)]
    return {
        "n_landmarks": n_landmarks,
        "tau_scale": tau_scale,
        "weights": list(weights) if weights is not None else None,
        "pyramid_scales": pyramid_scales,
        "pyramid_coarse_backbone": pyramid_coarse_backbone,
        "seed": seed,
        "epochs": epochs,
        **{kk: vv for kk, vv in geo.items()},
        "trust": trust,
        "knn_recall": rec,
        **{kk: vv for kk, vv in shep.items()},
        "density_rho": dens,
        "sil_Z": clus["sil_Z"],
        "sil_X": clus["sil_X"],
        "sil_gap": clus["sil_Z"] - clus["sil_X"],
        "score": float(np.mean(parts)),
        "worst_band": float(
            np.nanmin([shep["shepard_local"], shep["shepard_mid"], shep["shepard_global"]])
        ),
        "secs": time.time() - t0,
        "_labels": clus["labels"],
        "_hold_idx": hold_idx,
    }


def print_row(r: dict, label: str) -> None:
    print(
        f"{label:>24} | trust {r['trust']:.3f} rec {r['knn_recall']:.3f} "
        f"| GEO rho {r['geo_rho']:.3f} "
        f"near/mid/far {r['geo_near']:.2f}/{r['geo_mid']:.2f}/{r['geo_far']:.2f} "
        f"unreach {r['geo_unreachable']:.3f} "
        f"| euc {r['shepard_rho']:.3f} (glob {r['shepard_global']:.2f}) "
        f"| dens {r['density_rho']:+.3f} "
        f"| silZ {r['sil_Z']:.2f} silX {r['sil_X']:.2f} | {r['secs']:.0f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DATA)
    ap.add_argument(
        "--search",
        choices=["landmarks", "perp", "weights", "seeds", "pyramid"],
        default="landmarks",
    )
    ap.add_argument("--landmarks", type=int, nargs="+", default=[32, 128, 250, 500, 1000])
    ap.add_argument("--perps", type=float, nargs="+", default=[1.2, 4.0, 8.0, 16.0, 64.0])
    ap.add_argument("--target-perp", type=float, default=8.0)
    ap.add_argument("--n-landmarks", type=int, default=500)
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    ap.add_argument("--epochs", type=int, default=40, help="screening budget")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-clusters", type=int, default=13)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument(
        "--null",
        choices=["none", "shuffle", "gauss"],
        default="none",
        help=(
            "null calibration. 'shuffle' permutes each feature independently, "
            "destroying all dependence between columns while keeping every "
            "marginal exactly; 'gauss' uses matched-covariance Gaussian noise. "
            "Any structure the embedding shows under a null is manufactured by "
            "the method, so these runs give the chance level for every metric."
        ),
    )
    ap.add_argument(
        "--only",
        default=None,
        help="substring filter on --search pyramid variant names",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    X, _, _ = load_table(args.csv)
    if args.null == "shuffle":
        rng0 = np.random.default_rng(12345)
        X = np.column_stack([rng0.permutation(X[:, j]) for j in range(X.shape[1])])
        X = np.ascontiguousarray(X, dtype=np.float32)
        print("NULL: each feature permuted independently (marginals kept, joint destroyed)")
    elif args.null == "gauss":
        rng0 = np.random.default_rng(12345)
        X = rng0.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=len(X))
        X = np.ascontiguousarray(X, dtype=np.float32)
        print("NULL: matched-covariance Gaussian (no clusters, no manifold)")
    Xt = torch.as_tensor(X)
    print(f"N={len(X)} d={X.shape[1]}  holdout={args.holdout:.0%}  epochs={args.epochs}")
    print("score = mean(trust, knn_recall, shepard_rho, density_rho), all held out")
    print("silZ >> silX means the islands are decoder artifacts, not real groups\n")

    results = []
    if args.search == "landmarks":
        for L in args.landmarks:
            ts = calibrate_tau_scale(Xt, L, args.target_perp, seed=0)
            r = score_config(
                X,
                n_landmarks=L,
                tau_scale=ts,
                weights=tuple(args.weights),
                epochs=args.epochs,
                lr=args.lr,
                seed=0,
                holdout=args.holdout,
                k=args.k,
                n_clusters=args.n_clusters,
                eval_n=args.eval_n,
            )
            results.append(r)
            print_row(r, f"L={L} ts={ts:.3f}")

    elif args.search == "perp":
        L = args.n_landmarks
        for p in args.perps:
            ts = calibrate_tau_scale(Xt, L, p, seed=0)
            r = score_config(
                X,
                n_landmarks=L,
                tau_scale=ts,
                weights=tuple(args.weights),
                epochs=args.epochs,
                lr=args.lr,
                seed=0,
                holdout=args.holdout,
                k=args.k,
                n_clusters=args.n_clusters,
                eval_n=args.eval_n,
            )
            r["target_perp"] = p
            results.append(r)
            print_row(r, f"perp={p:g} ts={ts:.3f}")

    elif args.search == "pyramid":
        # Ablate the multiresolution graph. NOTE: at N=5000 only 3 levels are
        # built (R0=3784 -> 946 -> 256, floored by pyramid_min_reps), so
        # pyramid_level_weights is truncated to its first 3 entries. Coarse
        # emphasis must therefore be written as a 3-tuple; a 4th entry is dead.
        L = args.n_landmarks
        ts = calibrate_tau_scale(Xt, L, args.target_perp, seed=0)
        variants = [
            ("pyramid OFF", dict(pyramid_scales=0, weights=None)),
            ("3 lvl w=1/1/2 (default)", dict(pyramid_scales=3, weights=(1.0, 1.0, 2.0))),
            ("3 lvl w=1/2/8 coarse", dict(pyramid_scales=3, weights=(1.0, 2.0, 8.0))),
            ("3 lvl w=8/1/1 fine", dict(pyramid_scales=3, weights=(8.0, 1.0, 1.0))),
            (
                "3 lvl w=1/1/2, MST off",
                dict(pyramid_scales=3, weights=(1.0, 1.0, 2.0), backbone=0.0),
            ),
            (
                "3 lvl w=1/2/8, MST off",
                dict(pyramid_scales=3, weights=(1.0, 2.0, 8.0), backbone=0.0),
            ),
            (
                "3 lvl w=1/4/16, MST off",
                dict(pyramid_scales=3, weights=(1.0, 4.0, 16.0), backbone=0.0),
            ),
        ]
        if args.only:
            variants = [(n, v) for n, v in variants if args.only in n]
        for name, v in variants:
            r = score_config(
                X,
                n_landmarks=L,
                tau_scale=ts,
                weights=v["weights"],
                pyramid_scales=v["pyramid_scales"],
                pyramid_coarse_backbone=v.get("backbone"),
                epochs=args.epochs,
                lr=args.lr,
                seed=0,
                holdout=args.holdout,
                k=args.k,
                n_clusters=args.n_clusters,
                eval_n=args.eval_n,
            )
            r["variant"] = name
            results.append(r)
            print_row(r, name)

    elif args.search == "weights":
        L = args.n_landmarks
        ts = calibrate_tau_scale(Xt, L, args.target_perp, seed=0)
        for w in WEIGHT_GRID:
            r = score_config(
                X,
                n_landmarks=L,
                tau_scale=ts,
                weights=w,
                epochs=args.epochs,
                lr=args.lr,
                seed=0,
                holdout=args.holdout,
                k=args.k,
                n_clusters=args.n_clusters,
                eval_n=args.eval_n,
            )
            results.append(r)
            print_row(r, "w=" + "/".join(f"{x:g}" for x in w))

    else:  # seeds -> stability / "realism" of cluster membership
        from sklearn.metrics import adjusted_rand_score

        L = args.n_landmarks
        ts = calibrate_tau_scale(Xt, L, args.target_perp, seed=0)
        for s in args.seeds:
            r = score_config(
                X,
                n_landmarks=L,
                tau_scale=ts,
                weights=tuple(args.weights),
                epochs=args.epochs,
                lr=args.lr,
                seed=s,
                holdout=args.holdout,
                k=args.k,
                n_clusters=args.n_clusters,
                eval_n=args.eval_n,
            )
            results.append(r)
            print_row(r, f"seed={s}")
        # Cross-seed label agreement on the intersection of holdout sets.
        print("\ncross-seed ARI of cluster membership (on shared holdout points):")
        for a in range(len(results)):
            for b in range(a + 1, len(results)):
                ra, rb = results[a], results[b]
                ia = {int(v): m for m, v in enumerate(ra["_hold_idx"])}
                shared = [v for v in rb["_hold_idx"] if int(v) in ia]
                if len(shared) < 50:
                    continue
                la = ra["_labels"]
                lb = rb["_labels"]
                nmin = min(len(la), len(lb))
                print(
                    f"  seed {ra['seed']} vs {rb['seed']}: "
                    f"ARI={adjusted_rand_score(la[:nmin], lb[:nmin]):.3f} "
                    f"(n={nmin}, order-matched subsample)"
                )
        sc = np.array([r["score"] for r in results])
        print(f"\nscore across seeds: {sc.mean():.3f} +/- {sc.std():.3f}")

    if results:
        best = max(results, key=lambda r: r["score"])
        print(
            f"\nbest by held-out score: L={best['n_landmarks']} "
            f"tau_scale={best['tau_scale']:.3f} w={best['weights']} "
            f"score={best['score']:.3f}"
        )

    out = args.json_out or OUT_DIR / f"hparam_select_{args.search}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    out.write_text(json.dumps(clean, indent=2))
    print(f"json -> {out}")


if __name__ == "__main__":
    main()

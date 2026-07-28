#!/usr/bin/env python
"""Find island centroids that are close in ambient space but far along the manifold.

Two P(r) profiles can be a short straight line apart in the 100-dimensional
ambient space and still be far apart along the data, if no sequence of observed
profiles interpolates between them. That is what folding means: the sheet passes
near itself without the data connecting across the gap.

Each island from ``pr_centroids`` is represented by its medoid, the real profile
closest to the island's median curve, so it can serve as a node in the ambient
kNN graph. For every pair the graph shortest path (geodesic) is compared with
the direct chord, and the detour ratio geodesic/chord ranks the folds. A ratio
near 1 means the data fills in the straight line; a large ratio on a short chord
means two lobes of the manifold are adjacent in ambient space but only reachable
from one another the long way round.

The ratio depends on how densely the graph is wired, so it is recomputed across
several neighbour counts; a fold that survives all of them is not a wiring
artifact.

Usage::

    python examples/exploratory/pr_geodesic.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_geodesic.py --run runs/sasbdb_pr_l1_frozen --k 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pr_islands import find_islands

_ROOT = Path(__file__).resolve().parents[2]


def knn_graph(X: np.ndarray, k: int, metric: str):
    from scipy.sparse import coo_matrix
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(len(X)), k)
    A = coo_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())),
                   shape=(len(X), len(X))).tocsr()
    return A.maximum(A.T)


def medoids(X: np.ndarray, lab: np.ndarray, ids: np.ndarray, sp_metric: str):
    """Real profile nearest each island's median curve."""
    from scipy.spatial.distance import cdist

    out, cent = [], []
    for c in ids:
        m = np.flatnonzero(lab == c)
        med = np.median(X[m], axis=0)
        out.append(int(m[cdist(med[None], X[m], metric=sp_metric)[0].argmin()]))
        cent.append(med)
    return np.array(out), np.stack(cent)


def _excess(chord: np.ndarray, ratio: np.ndarray, frac: float = 0.15):
    """Local baseline and robust z of the detour ratio at a given chord length."""
    o = np.argsort(chord)
    w = max(31, int(frac * len(chord)) | 1)
    s = pd.Series(ratio[o])
    med = s.rolling(w, center=True, min_periods=w // 3).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=w // 3).median()
    base = np.empty_like(ratio)
    z = np.empty_like(ratio)
    base[o] = med.to_numpy()
    # 1.4826 puts the MAD on a standard-deviation footing for Gaussian noise.
    z[o] = (s.to_numpy() - med.to_numpy()) / np.maximum(1.4826 * mad.to_numpy(), 1e-6)
    return base, z


def path_between(pred: np.ndarray, src_row: int, dst: int) -> list[int]:
    path, node = [dst], dst
    while node >= 0 and pred[src_row, node] >= 0:
        node = int(pred[src_row, node])
        path.append(node)
    return path[::-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--eps-scale", type=float, default=3.0)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--k", type=int, default=15, help="neighbours in the geodesic graph")
    ap.add_argument("--k-check", type=int, nargs="*", default=[10, 15, 30, 50],
                    help="neighbour counts for the robustness check")
    ap.add_argument("--top", type=int, default=8, help="folds reported")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")
    sp_metric = "cityblock" if args.metric == "manhattan" else args.metric

    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial.distance import cdist, squareform

    lab, eps = find_islands(Z, args.eps_scale, args.min_samples)
    ids = np.array([c for c in np.unique(lab) if c >= 0])
    med_idx, cent = medoids(X, lab, ids, sp_metric)
    n_isl = len(ids)
    print(f"{run.name}: {len(X)} profiles, {n_isl} islands (DBSCAN eps={eps:.3g})")

    A = knn_graph(X, args.k, args.metric)
    ncomp, comp = connected_components(A, directed=False)
    if ncomp > 1:
        print(f"  WARNING: kNN(k={args.k}) graph has {ncomp} components; "
              "cross-component pairs are unreachable and are dropped")

    G = dijkstra(A, directed=False, indices=med_idx, return_predecessors=True)
    geo_all, pred = G[0], G[1]
    geo = geo_all[:, med_idx]
    chord = squareform(cdist(cent, cent, metric=sp_metric), checks=False)
    chord = cdist(cent, cent, metric=sp_metric)
    chord_l2 = cdist(cent, cent, metric="euclidean")

    iu = np.triu_indices(n_isl, 1)
    ok = np.isfinite(geo[iu]) & (chord[iu] > 0)
    ratio = np.full(geo.shape, np.nan)
    ratio[iu] = np.where(ok, geo[iu] / np.maximum(chord[iu], 1e-12), np.nan)
    ratio.T[iu] = ratio[iu]

    rr, cc = iu[0][ok], iu[1][ok]
    rv = ratio[rr, cc]
    print(f"  detour ratio over {len(rv)} island pairs: median {np.median(rv):.2f}, "
          f"90th pct {np.percentile(rv, 90):.2f}, max {rv.max():.2f}")

    # A path through discrete samples always zigzags, and proportionally more so
    # over short chords, so the raw ratio is biased toward exactly the pairs the
    # question is about. Score each pair against other pairs of similar chord.
    base, zsc = _excess(chord[rr, cc], rv)
    top_ex = np.argsort(-zsc)[: args.top]
    print(f"\n  excess detour, after removing the chord-length trend "
          f"(z against pairs of similar chord):")
    print("     pair        chord    ratio   expected   excess    z")
    for t in top_ex:
        a, b = rr[t], cc[t]
        print(f"    {int(ids[a]):3d} <-> {int(ids[b]):3d}   {chord[a, b]:.4f}  "
              f"{rv[t]:6.2f}   {base[t]:7.2f}   {rv[t] / base[t]:6.2f}  {zsc[t]:5.1f}")
    verdict = ("a genuine fold" if zsc.max() > 4 and rv[np.argmax(zsc)] / base[np.argmax(zsc)] > 1.25
               else "nothing beyond the sampling-zigzag trend")
    print(f"  -> {verdict}")

    # A fold is a short chord with a long path, so rank by ratio but keep the
    # chord short: pairs already far apart trivially have long paths.
    near = chord[rr, cc] <= np.percentile(chord[rr, cc], 40)
    order = np.argsort(-rv)
    print(f"\n  strongest detours overall (ratio = geodesic / straight-line):")
    print("     pair        chord L1  chord L2   geodesic   ratio   hops   near?")
    for t in order[: args.top]:
        a, b = rr[t], cc[t]
        hops = len(path_between(pred, a, med_idx[b])) - 1
        print(f"    {int(ids[a]):3d} <-> {int(ids[b]):3d}   {chord[a, b]:7.4f}  "
              f"{chord_l2[a, b]:7.4f}   {geo[a, b]:8.4f}  {ratio[a, b]:6.2f}  "
              f"{hops:5d}   {'yes' if near[t] else '-'}")

    on = order[near[order]]
    print(f"\n  the answer to the question -- short chord, long way round "
          f"(chord in the closest 40% of pairs):")
    for t in on[: args.top]:
        a, b = rr[t], cc[t]
        codes = meta['sasbdb_code'].to_numpy()
        print(f"    islands {int(ids[a]):3d} <-> {int(ids[b]):3d}  "
              f"({codes[med_idx[a]]} / {codes[med_idx[b]]}):  chord {chord[a, b]:.4f} "
              f"(rank {int((chord[rr, cc] < chord[a, b]).sum())}/{len(rv)}), "
              f"geodesic {geo[a, b]:.4f}, ratio {ratio[a, b]:.2f}")

    # A real fold should not depend on how densely the graph is wired.
    print("\n  robustness across neighbour counts:")
    keep = on[: min(3, len(on))]
    hist = {}
    for kk in args.k_check:
        Ak = knn_graph(X, kk, args.metric)
        gk = dijkstra(Ak, directed=False, indices=med_idx)[:, med_idx]
        rk = gk / np.maximum(chord, 1e-12)
        hist[kk] = rk
        txt = "  ".join(f"{int(ids[rr[t]])}-{int(ids[cc[t]])}: {rk[rr[t], cc[t]]:.2f}"
                        for t in keep)
        print(f"    k={kk:3d}  median ratio {np.median(rk[iu][np.isfinite(rk[iu])]):.2f}"
              f"   top folds  {txt}")

    out = args.out or (run / "geodesic.png")
    pd.DataFrame({
        "island_a": ids[rr], "island_b": ids[cc],
        "code_a": meta["sasbdb_code"].to_numpy()[med_idx[rr]],
        "code_b": meta["sasbdb_code"].to_numpy()[med_idx[cc]],
        "chord_l1": chord[rr, cc], "chord_l2": chord_l2[rr, cc],
        "geodesic": geo[rr, cc], "ratio": rv,
    }).sort_values("ratio", ascending=False).to_csv(
        out.with_name(out.stem + "_pairs.csv"), index=False)

    _plot(run, X, Z, lab, ids, cent, med_idx, chord, geo, ratio, rr, cc, rv, near,
          pred, hist, out, args)


def _plot(run, X, Z, lab, ids, cent, med_idx, chord, geo, ratio, rr, cc, rv, near,
          pred, hist, out, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = X.shape[1]
    r = np.linspace(0.0, 1.0, nb)
    order = np.argsort(-rv)
    on = order[near[order]]
    best = on[0] if len(on) else order[0]
    a, b = rr[best], cc[best]
    path = path_between(pred, a, med_idx[b])

    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))

    ax = axs[0, 0]
    sc = ax.scatter(chord[rr, cc], geo[rr, cc], c=rv, cmap="viridis", s=18,
                    linewidths=0.3, edgecolors="0.4")
    lim = [0, 1.05 * np.nanmax(geo[rr, cc])]
    ax.plot(lim, lim, color="crimson", ls="--", lw=1, label="geodesic = chord")
    ax.scatter(chord[a, b], geo[a, b], s=150, facecolors="none", edgecolors="crimson",
               linewidths=2, label=f"islands {int(ids[a])}-{int(ids[b])}")
    ax.set_xlabel(f"straight-line {args.metric} distance between centroids")
    ax.set_ylabel("geodesic distance through the kNN graph")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="detour ratio")
    ax.set_title("chord vs geodesic, every island pair")

    ax = axs[0, 1]
    ax.plot(r, cent[a], color="crimson", lw=2.4, label=f"island {int(ids[a])}")
    ax.plot(r, cent[b], color="navy", lw=2.4, label=f"island {int(ids[b])}")
    ax.fill_between(r, np.minimum(cent[a], cent[b]), np.maximum(cent[a], cent[b]),
                    color="0.75", alpha=0.5, label=f"L1 gap {chord[a, b]:.3f}")
    ax.set_xlabel("r / Dmax")
    ax.set_ylabel("P(r), unit sum")
    ax.legend(fontsize=8)
    ax.set_title(f"the two centroids: close in ambient space\n"
                 f"chord {chord[a, b]:.3f}, geodesic {geo[a, b]:.3f}, "
                 f"ratio {ratio[a, b]:.2f}")

    ax = axs[0, 2]
    take = path if len(path) <= 14 else [path[int(t)] for t in
                                         np.linspace(0, len(path) - 1, 14)]
    step = 0.55 * float(X[take].max())
    cm = plt.get_cmap("coolwarm")
    for j, i in enumerate(take):
        ax.plot(r, X[i] + j * step, color=cm(j / max(len(take) - 1, 1)), lw=1.4)
    ax.set_yticks([])
    ax.set_xlabel("r / Dmax")
    ax.set_title(f"the {len(path) - 1}-hop path between them\n"
                 f"(every profile the data has to pass through)")

    ax = axs[1, 0]
    ax.scatter(Z[:, 0], Z[:, 1], s=4, c="0.85", linewidths=0)
    cols = ["crimson", "darkorange", "seagreen"]
    for j, t in enumerate(on[:3] if len(on) else order[:3]):
        aa, bb = rr[t], cc[t]
        p = path_between(pred, aa, med_idx[bb])
        ax.plot(Z[p, 0], Z[p, 1], color=cols[j], lw=1.2, alpha=0.9,
                label=f"{int(ids[aa])}-{int(ids[bb])} (ratio {ratio[aa, bb]:.1f})")
        ax.plot(Z[[med_idx[aa], med_idx[bb]], 0], Z[[med_idx[aa], med_idx[bb]], 1],
                color=cols[j], lw=1.0, ls=":", alpha=0.8)
    lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
    mid, half = 0.5 * (lo + hi), 0.55 * float((hi - lo).max())
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8)
    ax.set_title("the detours on the embedding\n(dotted = the straight line they avoid)")

    ax = axs[1, 1]
    for kk, rk in hist.items():
        v = rk[np.triu_indices(len(ids), 1)]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=40, histtype="step", lw=1.6, label=f"k={kk}")
    ax.axvline(ratio[a, b], color="crimson", ls="--", lw=1.5,
               label=f"islands {int(ids[a])}-{int(ids[b])}")
    ax.set_xlabel("detour ratio (geodesic / chord)")
    ax.set_ylabel("island pairs")
    ax.legend(fontsize=8)
    ax.set_title("how much of this is graph wiring?")

    ax = axs[1, 2]
    base, zsc = _excess(chord[rr, cc], rv)
    o = np.argsort(chord[rr, cc])
    sc2 = ax.scatter(chord[rr, cc], rv, c=zsc, cmap="coolwarm", vmin=-4, vmax=4,
                     s=16, linewidths=0.3, edgecolors="0.4")
    ax.plot(chord[rr, cc][o], base[o], color="k", lw=2,
            label="expected zigzag at this chord")
    hot = np.argsort(-zsc)[:3]
    for t in hot:
        ax.annotate(f"{int(ids[rr[t]])}-{int(ids[cc[t]])} (z={zsc[t]:.1f})",
                    (chord[rr[t], cc[t]], rv[t]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(f"straight-line {args.metric} distance")
    ax.set_ylabel("detour ratio")
    ax.legend(fontsize=8)
    fig.colorbar(sc2, ax=ax, fraction=0.046, label="excess detour (robust z)")
    ax.set_title("a fold would sit far above the black line\n"
                 f"largest excess: z = {zsc.max():.1f} "
                 f"({rv[np.argmax(zsc)] / base[np.argmax(zsc)]:.2f}x expected)")

    fig.suptitle(f"{run.name}: centroids close in ambient space, far along the manifold",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"\nsaved {out} and {out.with_name(out.stem + '_pairs.csv')}")


if __name__ == "__main__":
    main()

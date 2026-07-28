#!/usr/bin/env python
"""Reduce each DBSCAN island to one median P(r) curve and compare those only.

An island is summarised by the elementwise median of its member profiles -- a
centroid curve that is robust to the outliers a density-based cluster tends to
pick up at its rim. Everything downstream then works on the handful of centroid
curves rather than the thousands of individual profiles.

The comparison that matters is not whether two centroids differ, but whether
they differ by more than the scatter inside their own islands. Every distance is
therefore reported both raw and divided by the within-island dispersion; a pair
below 1 is two islands the ambient data cannot tell apart.

Distances are L1, matching the metric the embedding was fitted with, and are
bounded by 2 because the profiles are unit-sum.

Usage::

    python examples/exploratory/pr_centroids.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_centroids.py --run runs/sasbdb_pr_umap --eps-scale 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pr_islands import find_islands

_ROOT = Path(__file__).resolve().parents[2]


def centroid_table(X: np.ndarray, Z: np.ndarray, lab: np.ndarray, metric: str):
    """Median curve, embedding position and internal dispersion per island."""
    ids = np.array([c for c in np.unique(lab) if c >= 0])
    bins = np.linspace(0.0, 1.0, X.shape[1])
    curves = np.empty((len(ids), X.shape[1]))
    rows = []
    for i, c in enumerate(ids):
        m = lab == c
        curves[i] = np.median(X[m], axis=0)
        if metric == "manhattan":
            spread = np.abs(X[m] - curves[i]).sum(axis=1)
        else:
            spread = np.linalg.norm(X[m] - curves[i], axis=1)
        w = curves[i] / curves[i].sum()
        rows.append(
            {
                "island": int(c),
                "n": int(m.sum()),
                "peak_pos": float(bins[np.argmax(curves[i])]),
                "mean_pos": float(w @ bins),
                "z0": float(Z[m, 0].mean()),
                "z1": float(Z[m, 1].mean()),
                "dispersion": float(np.median(spread)),
            }
        )
    return ids, curves, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--eps-scale", type=float, default=8.0,
                    help="DBSCAN eps as a multiple of the median embedding NN distance")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--metric", default="manhattan", help="metric on the curves")
    ap.add_argument("--min-size", type=int, default=10,
                    help="drop islands smaller than this before comparing")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)

    lab, eps = find_islands(Z, args.eps_scale, args.min_samples)
    ids, curves, tab = centroid_table(X, Z, lab, args.metric)
    keep = tab["n"].to_numpy() >= args.min_size
    ids, curves, tab = ids[keep], curves[keep], tab[keep].reset_index(drop=True)

    print(f"{run.name}: N={len(Z)}  DBSCAN eps={eps:.3g}  "
          f"islands={len(ids)} (>= {args.min_size} members)  noise={int((lab < 0).sum())}")

    from scipy.spatial.distance import pdist, squareform

    d = squareform(pdist(curves, metric="cityblock" if args.metric == "manhattan" else "euclidean"))
    np.fill_diagonal(d, np.inf)
    disp = tab["dispersion"].to_numpy()
    # Two islands are only distinguishable if their centroids are further apart
    # than the typical member-to-centroid distance inside them.
    ratio = d / (0.5 * (disp[:, None] + disp[None, :]))
    tab["nearest"] = [int(ids[j]) for j in d.argmin(axis=1)]
    tab["d_nearest"] = d.min(axis=1)
    tab["ratio_nearest"] = ratio[np.arange(len(ids)), d.argmin(axis=1)]
    np.fill_diagonal(d, 0.0)

    order = np.argsort(-tab["n"].to_numpy())
    print("\n  island   n   peak  <r>    dispersion   nearest  L1     L1/dispersion")
    for i in order:
        r = tab.iloc[i]
        print(f"  {int(r['island']):5d} {int(r['n']):5d}  {r['peak_pos']:.2f}  "
              f"{r['mean_pos']:.2f}   {r['dispersion']:.4f}     "
              f"{int(r['nearest']):5d}  {r['d_nearest']:.4f}   {r['ratio_nearest']:.2f}")

    iu = np.triu_indices(len(ids), 1)
    pr = ratio[iu]
    print(f"\n  centroid separation over {len(pr)} pairs: "
          f"L1 median {np.median(d[iu]):.4f}, max {d[iu].max():.4f}")
    print(f"  pairs closer than their own internal scatter (ratio < 1): "
          f"{int((pr < 1).sum())} of {len(pr)}")
    far = np.argsort(-d[iu])[:5]
    print("  most distinct pairs:")
    for t in far:
        a, b = iu[0][t], iu[1][t]
        print(f"    {int(ids[a]):4d} <-> {int(ids[b]):4d}:  L1 {d[a, b]:.4f}  "
              f"ratio {ratio[a, b]:.2f}")

    out = args.out or (run / "centroids.png")
    out_csv = out.with_name(out.stem + "_curves.csv")
    cols = {f"b{j:03d}": curves[:, j] for j in range(curves.shape[1])}
    pd.concat([tab, pd.DataFrame(cols)], axis=1).to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    _plot(run, Z, lab, ids, curves, tab, d, np.median(X, axis=0), out, args)


def _plot(run, Z, lab, ids, curves, tab, d, pop_med, out, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    from scipy.spatial.distance import squareform

    nb = curves.shape[1]
    r = np.linspace(0.0, 1.0, nb)
    cmap = plt.get_cmap("turbo")
    # Colour by peak position so the same hue means the same shape everywhere.
    pk = tab["peak_pos"].to_numpy()
    norm = plt.Normalize(pk.min(), pk.max())
    col = cmap(norm(pk))

    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))

    ax = axs[0, 0]
    ax.scatter(Z[:, 0], Z[:, 1], s=4, c="0.85", linewidths=0)
    for i, c in enumerate(ids):
        m = lab == c
        ax.scatter(Z[m, 0], Z[m, 1], s=5, color=col[i], linewidths=0)
        ax.annotate(str(int(c)), (tab["z0"][i], tab["z1"][i]), fontsize=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="0.6", alpha=0.8))
    # A handful of far-flung noise points would otherwise squash the bulk.
    lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
    pad = 0.08 * float((hi - lo).max())
    mid, half = 0.5 * (lo + hi), 0.5 * float((hi - lo).max()) + pad
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{len(ids)} islands (grey = noise / dropped)")

    ax = axs[0, 1]
    ax.plot(r, pop_med, color="0.5", lw=2.5, ls="--", label="population median", zorder=1)
    for i in range(len(ids)):
        ax.plot(r, curves[i], color=col[i], lw=1.6)
    ax.set_xlabel("r / Dmax")
    ax.set_ylabel("P(r), unit sum")
    ax.legend(fontsize=8)
    ax.set_title("centroid curves (island medians) only")

    # Offsetting separates curves that overlap almost everywhere.
    ax = axs[0, 2]
    step = 0.6 * float(curves.max())
    for k, i in enumerate(np.argsort(pk)):
        ax.plot(r, curves[i] + k * step, color=col[i], lw=1.4)
        ax.annotate(f"{int(ids[i])} (n={int(tab['n'][i])})", (1.005, k * step),
                    fontsize=7, va="bottom", color=col[i])
    ax.set_xlim(0, 1.16)
    ax.set_yticks([])
    ax.set_xlabel("r / Dmax")
    ax.set_title("same curves, offset and ordered by peak position")

    ax = axs[1, 0]
    Zl = linkage(squareform(d, checks=False), method="average")
    dn = dendrogram(Zl, labels=[int(c) for c in ids], ax=ax, color_threshold=0,
                    link_color_func=lambda _: "0.4")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylabel(f"{args.metric} distance between centroids")
    ax.set_title("centroid curve dendrogram (average linkage)")

    ax = axs[1, 1]
    o = leaves_list(Zl)
    im = ax.imshow(d[np.ix_(o, o)], cmap="magma_r")
    ax.set_xticks(range(len(o)), [int(ids[i]) for i in o], fontsize=6, rotation=90)
    ax.set_yticks(range(len(o)), [int(ids[i]) for i in o], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="L1 between centroids")
    ax.set_title("pairwise centroid distance (dendrogram order)")

    ax = axs[1, 2]
    disp = tab["dispersion"].to_numpy()
    dn_ = tab["d_nearest"].to_numpy()
    ax.scatter(disp, dn_, s=30 + 3 * np.sqrt(tab["n"].to_numpy()), c=col,
               edgecolors="0.3", linewidths=0.5)
    for i in range(len(ids)):
        ax.annotate(str(int(ids[i])), (disp[i], dn_[i]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    lim = [0, 1.05 * max(disp.max(), dn_.max())]
    ax.plot(lim, lim, color="crimson", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("within-island dispersion (median member -> centroid)")
    ax.set_ylabel("distance to nearest other centroid")
    ax.set_title("above the line = island is distinct from its neighbour")

    fig.suptitle(f"{run.name}: island centroid curves compared", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Full pairwise distance matrix over the P(r) profiles, seriated to show structure.

A distance matrix only shows structure if the rows are in a sensible order, so
this builds two orderings of the same matrix and renders both:

``blocks``    DBSCAN islands, the islands themselves ordered by a dendrogram over
              their centroid curves and each island internally leaf-ordered, so
              genuine clusters appear as dark squares on the diagonal
``seriation`` a single global ordering from the Fiedler vector of the kNN graph,
              which ignores any partition; a continuum shows up as a smooth band
              along the diagonal rather than as blocks

Comparing the two answers the question the block picture alone cannot: whether
the diagonal squares are real clusters or arbitrary cuts through a gradient.

Noise points are assigned to their nearest centroid so the matrix stays fully
ordered, and are marked in grey on the membership strips.

Usage::

    python examples/exploratory/pr_distmatrix.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_distmatrix.py --run runs/sasbdb_pr_l1_frozen \\
        --smooth --eps-scale 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pr_islands import find_islands

_ROOT = Path(__file__).resolve().parents[2]

# Optimal leaf ordering is O(n^3); above this a block is ordered by plain
# dendrogram leaf order instead.
_OLO_MAX = 600


def leaf_order(D: np.ndarray, olo: bool = True) -> np.ndarray:
    """Dendrogram leaf order for a small square distance matrix."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    n = len(D)
    if n < 3:
        return np.arange(n)
    cond = squareform(D, checks=False)
    L = linkage(cond, method="average")
    if olo and n <= _OLO_MAX:
        L = optimal_leaf_ordering(L, cond)
    return leaves_list(L)


def block_order(D: np.ndarray, lab: np.ndarray, cent_d: np.ndarray, ids: np.ndarray):
    """Islands ordered by centroid dendrogram, points leaf-ordered inside each."""
    order, bounds = [], [0]
    for c in ids[leaf_order(cent_d)]:
        m = np.flatnonzero(lab == c)
        order.extend(m[leaf_order(D[np.ix_(m, m)])])
        bounds.append(len(order))
    return np.array(order), np.array(bounds)


def fiedler_order(X: np.ndarray, k: int, metric: str):
    """Global seriation: sort by the Fiedler vector of a self-tuning kNN graph.

    A single global bandwidth underflows the weights of outlying points, which
    detaches them numerically and makes the low end of the spectrum a pile of
    near-zero eigenvalues with spike-shaped eigenvectors. The Zelnik-Manor and
    Perona local scale (distance to each point's k-th neighbour) keeps every
    edge weight O(1) and leaves a clean spectral gap.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    d, j = dist[:, 1:], idx[:, 1:]
    sigma = np.maximum(d[:, -1], 1e-12)
    rows = np.repeat(np.arange(len(X)), k)
    w = np.exp(-(d.ravel() ** 2) / (sigma[rows] * sigma[j.ravel()]))
    A = csr_matrix((w, (rows, j.ravel())), shape=(len(X), len(X)))
    A = A.maximum(A.T)
    L = laplacian(A, normed=True).astype(np.float64)
    # Shift-invert converges far faster than 'SM' on the near-null end.
    vals, vecs = eigsh(L, k=3, sigma=-1e-6, which="LM")
    o = np.argsort(vals)
    f = vecs[:, o[1]]
    print(f"  spectral gap: lambda1={vals[o][1]:.4g}, lambda2={vals[o][2]:.4g}; "
          f"5 largest entries hold {np.sort(f**2)[-5:].sum() * 100:.1f}% of the "
          "Fiedler norm (low = delocalised, so the ordering is global)")
    return np.argsort(f), f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--eps-scale", type=float, default=3.0)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--k", type=int, default=15, help="neighbours for the seriation graph")
    ap.add_argument("--smooth", action="store_true",
                    help="use the ringing-filtered X_smooth.npy if present")
    ap.add_argument("--clip", type=float, default=98.0,
                    help="colour scale upper percentile; lower it for more contrast")
    ap.add_argument("--dpi", type=int, default=340, help="giant matrix resolution")
    ap.add_argument("--save-matrix", action="store_true", help="also write the ordered .npy")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    src = run / ("X_smooth.npy" if args.smooth else "X.npy")
    if args.smooth and not src.exists():
        raise SystemExit(f"{src} not found; run pr_ringing.py first")
    X = np.load(src).astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")

    from scipy.spatial.distance import pdist, squareform

    sp_metric = "cityblock" if args.metric == "manhattan" else args.metric
    print(f"{run.name}: {X.shape[0]} profiles from {src.name}, {args.metric} distances")
    D = squareform(pdist(X, metric=sp_metric))
    print(f"  matrix {D.shape[0]}x{D.shape[1]}  "
          f"({D.nbytes / 1e6:.0f} MB)  median off-diagonal {np.median(D[np.triu_indices(len(D), 1)]):.4f}")

    lab, eps = find_islands(Z, args.eps_scale, args.min_samples)
    ids = np.array([c for c in np.unique(lab) if c >= 0])
    cent = np.stack([np.median(X[lab == c], axis=0) for c in ids])
    noise = lab < 0
    if noise.any():
        # Nearest centroid in curve space, so the assignment respects the metric
        # the matrix is built from rather than the embedding.
        from scipy.spatial.distance import cdist

        lab = lab.copy()
        lab[noise] = ids[cdist(X[noise], cent, metric=sp_metric).argmin(axis=1)]
    print(f"  DBSCAN eps={eps:.3g}: {len(ids)} islands, "
          f"{int(noise.sum())} noise points folded into the nearest centroid")

    cent_d = squareform(pdist(cent, metric=sp_metric))
    ob, bounds = block_order(D, lab, cent_d, ids)
    print("  block ordering done; computing global seriation")
    os_, fvec = fiedler_order(X, args.k, args.metric)

    from scipy.stats import spearmanr

    bins = np.linspace(0.0, 1.0, X.shape[1])
    mean_pos = (X / X.sum(axis=1, keepdims=True)) @ bins
    print(f"  seriation axis vs peak position: rho={spearmanr(fvec, bins[X.argmax(1)]).statistic:+.3f}"
          f"   vs mean position: rho={spearmanr(fvec, mean_pos).statistic:+.3f}")

    # Does the partition actually explain the matrix?
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(D), 1)
    win, bet = D[iu][same[iu]], D[iu][~same[iu]]
    print(f"  within-island pairs: mean {win.mean():.4f}   "
          f"between-island: mean {bet.mean():.4f}   ratio {bet.mean() / win.mean():.2f}")

    out = args.out or (run / ("distmatrix_smooth.png" if args.smooth else "distmatrix.png"))
    pd.DataFrame({
        "row": np.arange(len(ob)),
        "index": ob,
        "sasbdb_code": meta["sasbdb_code"].to_numpy()[ob],
        "island": lab[ob],
        "was_noise": noise[ob],
        "seriation_rank": np.argsort(os_)[ob],
        "fiedler": fvec[ob],
    }).to_csv(out.with_name(out.stem + "_order.csv"), index=False)
    if args.save_matrix:
        np.save(out.with_name(out.stem + "_ordered.npy"), D[np.ix_(ob, ob)].astype(np.float32))

    vmax = float(np.percentile(D[iu], args.clip))
    _giant(D, ob, bounds, ids, lab, noise, vmax, out, args)
    _panels(D, ob, bounds, os_, cent_d, ids, lab, vmax, out, args)


def _strip_colors(lab_ord, noise_ord, ids):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    pos = {c: i for i, c in enumerate(ids)}
    col = np.array([cmap(pos[c] % 20) for c in lab_ord])
    col[noise_ord] = (0.75, 0.75, 0.75, 1.0)
    return col


def _giant(D, ob, bounds, ids, lab, noise, vmax, out, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    M = D[np.ix_(ob, ob)]
    col = _strip_colors(lab[ob], noise[ob], ids)

    fig = plt.figure(figsize=(13.5, 13))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.02, 1, 0.03],
                  height_ratios=[0.02, 1], wspace=0.012, hspace=0.012)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=vmax, interpolation="nearest",
                   origin="upper")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="#00bfff", lw=0.35, alpha=0.75)
        ax.axvline(b - 0.5, color="#00bfff", lw=0.35, alpha=0.75)
    ax.set_xticks([])
    ax.set_yticks([])

    mid = 0.5 * (bounds[:-1] + bounds[1:])
    order_ids = [int(lab[ob][int(m)]) for m in mid]
    for m, c, b0, b1 in zip(mid, order_ids, bounds[:-1], bounds[1:]):
        if b1 - b0 >= 0.006 * len(M):
            ax.annotate(f"{c}\n{b1 - b0}", (m, len(M) + 0.006 * len(M)), fontsize=6,
                        ha="center", va="top", annotation_clip=False, color="0.25",
                        linespacing=0.95)

    top = fig.add_subplot(gs[0, 1], sharex=ax)
    top.imshow(col[None, :, :], aspect="auto", interpolation="nearest")
    top.set_xticks([])
    top.set_yticks([])
    left = fig.add_subplot(gs[1, 0], sharey=ax)
    left.imshow(col[:, None, :], aspect="auto", interpolation="nearest")
    left.set_xticks([])
    left.set_yticks([])

    cax = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=cax, label=f"{args.metric} distance between P(r) profiles")
    fig.suptitle(f"{out.parent.name}: {len(M)} P(r) profiles, "
                 f"ordered by DBSCAN island then leaf order\n"
                 f"{len(ids)} islands; grey on the strips = folded-in noise; "
                 f"colour clipped at the {args.clip:g}th percentile",
                 fontsize=11, y=0.965)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}  ({out.stat().st_size / 1e6:.1f} MB, {args.dpi} dpi)")


def _panels(D, ob, bounds, os_, cent_d, ids, lab, vmax, out, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 7))

    ax = axs[0]
    ax.imshow(D[np.ix_(ob, ob)], cmap="magma_r", vmin=0, vmax=vmax, interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="#00bfff", lw=0.3, alpha=0.7)
        ax.axvline(b - 0.5, color="#00bfff", lw=0.3, alpha=0.7)
    ax.set_title(f"DBSCAN blocks ({len(ids)} islands, leaf-ordered)")

    ax = axs[1]
    im = ax.imshow(D[np.ix_(os_, os_)], cmap="magma_r", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_title("global seriation (Fiedler vector), no partition\n"
                 "smooth band = continuum, sharp squares = clusters")
    fig.colorbar(im, ax=axs[:2], fraction=0.025, label=f"{args.metric} distance")

    ax = axs[2]
    o = leaf_order(cent_d)
    im2 = ax.imshow(cent_d[np.ix_(o, o)], cmap="magma_r", interpolation="nearest")
    ax.set_xticks(range(len(o)), [int(ids[i]) for i in o], fontsize=5, rotation=90)
    ax.set_yticks(range(len(o)), [int(ids[i]) for i in o], fontsize=5)
    fig.colorbar(im2, ax=ax, fraction=0.046, label="centroid distance")
    ax.set_title("island centroids only (same ordering)")

    for a in axs[:2]:
        a.set_xticks([])
        a.set_yticks([])
    out2 = out.with_name(out.stem + "_panels.png")
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Validate apparent islands and holes in a P(r) embedding.

A 2-D embedding of an intrinsically higher-dimensional manifold has to tear it,
so visual islands come in two kinds: genuine ambient separation, and folds where
the projection pulled apart points that are neighbours in the data. This tells
them apart by carrying the ambient kNN graph onto the embedding.

Reported per run:

* connected components of the ambient kNN graph -- the only true separations
* islands found in the embedding (DBSCAN), and how many ambient edges bridge
  each pair; a pair with many bridges is a projection tear, not a real gap
* tear edges: ambient neighbours placed far apart in the embedding
* holes: whether ambient edges pass straight through the empty regions

Usage::

    python examples/exploratory/pr_islands.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_islands.py --run runs/sasbdb_pr_umap --k 15
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def ambient_knn(X: np.ndarray, k: int, metric: str):
    """Symmetric kNN graph on the ambient profiles."""
    from scipy.sparse import coo_matrix
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(len(X)), k)
    cols = idx[:, 1:].ravel()
    vals = dist[:, 1:].ravel()
    A = coo_matrix((vals, (rows, cols)), shape=(len(X), len(X))).tocsr()
    return A.maximum(A.T), np.column_stack([rows, cols])


def find_islands(Z: np.ndarray, eps_scale: float, min_samples: int):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    eps = eps_scale * float(np.median(d[:, 1]))
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    return lab, eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--k", type=int, default=15, help="ambient kNN neighbours")
    ap.add_argument(
        "--metric",
        default="manhattan",
        help="ambient metric; match the one the run was fitted with",
    )
    # At eps_scale=3 DBSCAN shatters this data into ~83 fragments, which is far
    # finer than the islands the eye picks out; 8 recovers the visible lobes.
    ap.add_argument("--eps-scale", type=float, default=8.0,
                    help="DBSCAN eps as a multiple of the median embedding NN distance")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--top-islands", type=int, default=8,
                    help="report pairs among the N largest islands only")
    ap.add_argument("--max-edges", type=int, default=40000, help="edges drawn")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")

    from scipy.sparse.csgraph import connected_components

    A, edges = ambient_knn(X, args.k, args.metric)
    n_comp, comp = connected_components(A, directed=False)
    print(f"{run.name}: N={len(Z)}  ambient {args.metric} kNN(k={args.k})")
    print(f"  ambient connected components: {n_comp}"
          f"{'  -> every visual island is ambient-connected' if n_comp == 1 else ''}")
    if n_comp > 1:
        sizes = np.bincount(comp)
        print(f"  component sizes: {np.sort(sizes)[::-1][:10]}")

    lab, eps = find_islands(Z, args.eps_scale, args.min_samples)
    ids = [c for c in np.unique(lab) if c >= 0]
    noise = int((lab < 0).sum())
    print(f"  embedding islands (DBSCAN eps={eps:.3g}): {len(ids)}"
          f"  sizes={[int((lab == c).sum()) for c in ids]}  noise={noise}")

    # Bridges: ambient edges whose endpoints sit in different embedding islands.
    e0, e1 = edges[:, 0], edges[:, 1]
    l0, l1 = lab[e0], lab[e1]
    cross = (l0 != l1) & (l0 >= 0) & (l1 >= 0)
    print(f"  ambient edges: {len(edges)}   crossing islands: {int(cross.sum())}")
    if len(ids) > 1:
        big = sorted(ids, key=lambda c: -int((lab == c).sum()))[: args.top_islands]
        rows = []
        for a, b in itertools.combinations(big, 2):
            m = ((l0 == a) & (l1 == b)) | ((l0 == b) & (l1 == a))
            rows.append((int(m.sum()), a, b))
        rows.sort(reverse=True)
        print(f"\n  bridges among the {len(big)} largest islands "
              "(edges > 0 means the gap is a projection tear):")
        for n_edge, a, b in rows:
            na, nb = int((lab == a).sum()), int((lab == b).sum())
            verdict = "TEAR" if n_edge else "no direct ambient edge"
            print(f"    {a} (n={na:4d}) <-> {b} (n={nb:4d}): {n_edge:5d} edges   {verdict}")

    # Tears: ambient neighbours that the embedding placed far apart.
    d_emb = np.linalg.norm(Z[e0] - Z[e1], axis=1)
    med = float(np.median(d_emb))
    for mult in (5, 10, 20):
        frac = float((d_emb > mult * med).mean())
        print(f"  ambient edges stretched >{mult:2d}x median in embedding: "
              f"{frac * 100:.2f}%")

    _plot(run, Z, lab, edges, d_emb, med, cross, args)


def _plot(run, Z, lab, edges, d_emb, med, cross, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    ids = [c for c in np.unique(lab) if c >= 0]
    cmap = plt.get_cmap("tab20")
    col = np.array([(0.75, 0.75, 0.75, 1.0)] * len(Z))
    for i, c in enumerate(ids):
        col[lab == c] = cmap(i % 20)
    axs[0].scatter(Z[:, 0], Z[:, 1], c=col, s=6, linewidths=0)
    axs[0].set_title(f"embedding islands (DBSCAN): {len(ids)}  grey = noise")

    # Ambient graph carried onto the embedding.
    take = np.arange(len(edges))
    if len(take) > args.max_edges:
        take = np.random.default_rng(0).choice(len(take), args.max_edges, replace=False)
    segs = np.stack([Z[edges[take, 0]], Z[edges[take, 1]]], axis=1)
    axs[1].add_collection(
        LineCollection(segs, colors="0.6", linewidths=0.15, alpha=0.35, zorder=1)
    )
    br = take[cross[take]]
    if len(br):
        axs[1].add_collection(
            LineCollection(
                np.stack([Z[edges[br, 0]], Z[edges[br, 1]]], axis=1),
                colors="crimson", linewidths=0.5, alpha=0.8, zorder=2,
            )
        )
    axs[1].scatter(Z[:, 0], Z[:, 1], s=3, c="k", alpha=0.5, linewidths=0, zorder=3)
    axs[1].autoscale_view()
    axs[1].set_title(f"ambient kNN edges on the embedding\nred = bridges between islands "
                     f"({int(cross.sum())} of {len(edges)})")

    # Only the stretched edges: these trace where the projection tore.
    tear = take[d_emb[take] > 5 * med]
    axs[2].scatter(Z[:, 0], Z[:, 1], s=3, c="0.8", linewidths=0, zorder=1)
    if len(tear):
        axs[2].add_collection(
            LineCollection(
                np.stack([Z[edges[tear, 0]], Z[edges[tear, 1]]], axis=1),
                colors="darkorange", linewidths=0.4, alpha=0.7, zorder=2,
            )
        )
    axs[2].autoscale_view()
    axs[2].set_title(f"tear edges: ambient neighbours >5x median apart\n"
                     f"({len(tear)} shown) — these cross the 'holes'")

    for a in axs:
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{run.name}: are the islands real?", fontsize=12)
    fig.tight_layout()
    out = args.out or (run / "islands.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

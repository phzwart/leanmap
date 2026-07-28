#!/usr/bin/env python
"""Density as a heatmap under the points: what the map shows against what the data has.

A scatter plot is a poor instrument for judging density. Overplotting saturates
wherever it matters most, marker size sets an apparent floor on how tight a group
can look, and the eye reads outline rather than concentration. So the question
"does this map's density reflect the data's" gets argued from summary statistics
that nobody can see.

This paints it instead. Every panel is the same layout; only the field underneath
changes:

``embedded``  local density measured in the map itself -- what the picture shows
``ambient``   each point's ambient graph density, painted at its map position --
              what the data has, carried into the map's coordinates
``residual``  embedded minus ambient after matching scales, i.e. where the map is
              denser or sparser than the data warrants

Reading the third panel is the point. Blue means the map has spread apart
something the data holds together, red means it has crowded something the data
keeps apart, and white means the map is telling the truth there. Because ambient
and embedded densities live in different dimensions and cannot be compared raw,
both are converted to within-dataset z-scores of log density first, which compares
*orderings* -- the same thing the density term in training optimises, and the only
comparison that is meaningful across a dimension change.

Usage::

    python examples/exploratory/pr_density_map.py --runs runs/sasbdb_pr_density
    python examples/exploratory/pr_density_map.py \
        --runs runs/digits_off:lambda=0 runs/digits_both:lambda=1 \
        --metric euclidean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pr_clumpiness import density, intrinsic_dim, knn_dist  # noqa: E402


def z(v: np.ndarray) -> np.ndarray:
    """Standardise, so fields from different dimensions can share a colour scale."""
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def fields(X: np.ndarray, Z: np.ndarray, k: int, metric: str):
    """Ambient and embedded log-density as comparable z-scores, plus the residual."""
    dim = intrinsic_dim(knn_dist(X, 10, metric))
    la = z(np.log10(density(knn_dist(X, k, metric), dim)))
    lz = z(np.log10(density(knn_dist(Z, k, "euclidean"), float(Z.shape[1]))))
    return la, lz, lz - la


def paint(ax, Z: np.ndarray, v: np.ndarray, grid: int, smooth: float, cmap: str,
          vlim: float, lo, hi):
    """Nearest-neighbour interpolation of ``v`` onto a grid, blurred, then shown.

    Nearest neighbour rather than a fitted surface because the field is only
    defined where there are points; blurring afterwards keeps cell boundaries from
    reading as structure. Regions far from any point are masked so the panel never
    implies knowledge of empty space.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    gx = np.linspace(lo[0], hi[0], grid)
    gy = np.linspace(lo[1], hi[1], grid)
    mx, my = np.meshgrid(gx, gy)
    tree = cKDTree(Z[:, :2])
    dist, idx = tree.query(np.column_stack([mx.ravel(), my.ravel()]))
    img = v[idx].reshape(grid, grid)
    img = gaussian_filter(img, smooth)
    # Mask cells further from data than the typical point spacing allows.
    reach = np.percentile(tree.query(Z[:, :2], k=2)[0][:, 1], 95) * 3.0
    img = np.ma.masked_where(dist.reshape(grid, grid) > reach, img)
    im = ax.imshow(
        img,
        origin="lower",
        extent=(lo[0], hi[0], lo[1], hi[1]),
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        interpolation="bilinear",
        aspect="equal",
    )
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["runs/sasbdb_pr_density"])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--grid", type=int, default=320)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--pct", type=float, default=1.0)
    ap.add_argument("--vlim", type=float, default=2.0)
    ap.add_argument("--points", action="store_true", help="overlay the points too")
    ap.add_argument("--out", default="runs/density_heatmap.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for spec in args.runs:
        path, _, label = spec.rpartition(":")
        run = Path(path) if Path(path).is_absolute() else _ROOT / path
        X = np.load(run / "X.npy").astype(np.float64)
        Z = np.load(run / "Z.npy").astype(np.float64)
        la, lz, res = fields(X, Z, args.k, args.metric)
        rows.append((label or run.name, Z, la, lz, res))

    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(13.5, 4.6 * n), squeeze=False)
    for i, (label, Z, la, lz, res) in enumerate(rows):
        lo = np.percentile(Z[:, :2], args.pct, axis=0)
        hi = np.percentile(Z[:, :2], 100 - args.pct, axis=0)
        c = 0.5 * (lo + hi)
        half = 0.5 * max(hi - lo) * 1.05
        lo, hi = c - half, c + half
        panels = (
            ("embedded: what the map shows", lz, "viridis"),
            ("ambient: what the data has", la, "viridis"),
            ("residual: map minus data", res, "coolwarm"),
        )
        for j, (title, v, cmap) in enumerate(panels):
            a = ax[i][j]
            im = paint(a, Z, v, args.grid, args.smooth, cmap, args.vlim, lo, hi)
            if args.points:
                a.scatter(Z[:, 0], Z[:, 1], s=1.0, c="k", alpha=0.22, linewidths=0)
            # After scatter, which would otherwise stretch the axes out to the
            # outliers the percentile window exists to exclude.
            a.set_xlim(lo[0], hi[0])
            a.set_ylim(lo[1], hi[1])
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(title, fontsize=9.5)
            if j == 0:
                a.set_ylabel(label, fontsize=11)
            fig.colorbar(im, ax=a, fraction=0.046, label="z of log density")
        # A number to hold the picture against: how much of the embedded density
        # ordering the ambient field accounts for.
        r = float(np.corrcoef(la, lz)[0, 1])
        ax[i][2].set_xlabel(
            f"corr(ambient, embedded) = {r:+.3f};  residual sd = {res.std():.2f}",
            fontsize=9,
        )

    fig.suptitle(
        "Density under the points: blue = map spread out what the data holds "
        "together, red = map crowded what the data keeps apart",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out) if Path(args.out).is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    for label, _, la, lz, res in rows:
        print(
            f"{label:<22} corr(ambient, embedded) = {np.corrcoef(la, lz)[0, 1]:+.3f}"
            f"   residual sd = {res.std():.3f}"
        )


if __name__ == "__main__":
    main()

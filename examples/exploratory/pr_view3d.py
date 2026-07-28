#!/usr/bin/env python
"""Viewers for a 3-D embedding: a corner plot and a slab montage.

Two different jobs, and it matters which one you read for what.

The **corner plot** is the triangle of pairwise projections with marginals on the
diagonal. It is the right way to read shape -- arms, gaps, how the metadata
gradients run -- but it is the wrong way to read density, because collapsing the
third axis superimposes points that are far apart and inflates apparent density
in exactly the manner the density budget exists to control.

The **slab montage** cuts the out-of-plane axis into equal-count sections and
draws each separately, the way a volume is read as a stack of sections. Within a
slab there is little superposition left, so what you see is close to the real
local density. Equal *count* rather than equal thickness keeps the same number of
points behind every panel, so panels are comparable by eye; the thickness of each
is printed on it, and thin slabs are themselves a sign of dense regions.

Both work on 2-D runs too (the corner plot degenerates to one panel), so the same
command can be pointed at any run.

Usage::

    python examples/exploratory/pr_view3d.py --run runs/sasbdb_pr_density3
    python examples/exploratory/pr_view3d.py --run runs/sasbdb_pr_density3 \
        --color dmax --slabs 9
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

CMAP = {
    "mode_pos": "plasma",
    "dmax": "viridis",
    "rg_over_dmax": "coolwarm",
    "skew": "cividis",
}
LABELS = {
    "mode_pos": "peak position r/Dmax",
    "dmax": "log10 Dmax",
    "rg_over_dmax": "Rg / Dmax",
    "skew": "P(r) skewness",
}


def color_panels(X: np.ndarray, meta) -> dict:
    dmax = meta["dmax"].to_numpy(dtype=np.float64)
    rg = meta["rg_pr"].to_numpy(dtype=np.float64)
    bins = np.linspace(0.0, 1.0, X.shape[1])
    w = X / X.sum(axis=1, keepdims=True)
    mean_pos = w @ bins
    return {
        "mode_pos": bins[np.argmax(X, axis=1)],
        "dmax": np.log10(dmax),
        "rg_over_dmax": rg / dmax,
        "skew": w @ (bins**3) - 3 * mean_pos * (w @ (bins**2)) + 2 * mean_pos**3,
    }


def _limits(Z: np.ndarray, pad: float = 0.04):
    """Shared, equal-aspect limits so every panel is on one scale."""
    lo, hi = np.percentile(Z, [0.3, 99.7], axis=0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (1.0 + pad) * float((hi - lo).max())
    return [(m - half, m + half) for m in mid]


def corner(Z, c, key, out: Path, title: str, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Z.shape[1]
    lim = _limits(Z)
    vmin, vmax = np.percentile(c, [2, 98])
    fig, axs = plt.subplots(d, d, figsize=(4.4 * d, 4.2 * d), squeeze=False)
    sc = None
    for r in range(d):
        for q in range(d):
            ax = axs[r][q]
            if q > r:
                ax.axis("off")
                continue
            if q == r:
                ax.hist(Z[:, q], bins=80, color="0.55", histtype="stepfilled")
                ax.set_xlim(*lim[q])
                ax.set_yticks([])
                ax.set_ylabel("count" if q == 0 else "")
            else:
                sc = ax.scatter(
                    Z[:, q], Z[:, r], c=c, cmap=CMAP[key], s=3.5, linewidths=0,
                    vmin=vmin, vmax=vmax,
                )
                ax.set_xlim(*lim[q])
                ax.set_ylim(*lim[r])
                ax.set_aspect("equal")
            if r == d - 1:
                ax.set_xlabel(f"dim {q + 1}")
            if q == 0 and r > 0:
                ax.set_ylabel(f"dim {r + 1}")
    if sc is not None:
        fig.colorbar(
            sc, ax=axs[0][d - 1] if d > 1 else axs[0][0],
            fraction=0.4, label=LABELS[key], location="left",
        )
    fig.suptitle(
        f"{title} — pairwise projections ({LABELS[key]})\n"
        "shape only: collapsing an axis superimposes points and overstates density",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def slabs(Z, c, key, out: Path, title: str, n_slabs: int, axis: int, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [i for i in range(Z.shape[1]) if i != axis]
    lim = _limits(Z)
    vmin, vmax = np.percentile(c, [2, 98])
    order = np.argsort(Z[:, axis])
    parts = np.array_split(order, n_slabs)

    ncol = min(3, n_slabs)
    nrow = int(np.ceil(n_slabs / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(5.0 * ncol, 5.0 * nrow), squeeze=False
    )
    sc = None
    for k, part in enumerate(parts):
        ax = axs[k // ncol][k % ncol]
        v = Z[part, axis]
        sc = ax.scatter(
            Z[part, keep[0]], Z[part, keep[1]], c=c[part], cmap=CMAP[key],
            s=5, linewidths=0, vmin=vmin, vmax=vmax,
        )
        ax.set_xlim(*lim[keep[0]])
        ax.set_ylim(*lim[keep[1]])
        ax.set_aspect("equal")
        ax.set_xlabel(f"dim {keep[0] + 1}")
        ax.set_ylabel(f"dim {keep[1] + 1}")
        ax.set_title(
            f"slab {k + 1}/{n_slabs}   n={len(part)}\n"
            f"dim {axis + 1} in [{v.min():.3g}, {v.max():.3g}]"
            f"   thickness {v.max() - v.min():.3g}",
            fontsize=9,
        )
    for k in range(n_slabs, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axs, fraction=0.02, label=LABELS[key])
    fig.suptitle(
        f"{title} — equal-count slabs along dim {axis + 1} ({LABELS[key]})\n"
        "density is readable here: little superposition survives within a slab",
        fontsize=12,
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_density3")
    ap.add_argument(
        "--color", default="all", choices=("all", *CMAP), help="'all' writes one per key"
    )
    ap.add_argument("--slabs", type=int, default=6)
    # Default: slab along the axis with the least spread, which is the one whose
    # collapse would otherwise do the most damage per unit of screen area.
    ap.add_argument("--slab-axis", type=int, default=-1, help="-1 = narrowest axis")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    import pandas as pd

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")
    cols = color_panels(X, meta)
    keys = list(CMAP) if args.color == "all" else [args.color]
    axis = int(np.argmin(Z.std(axis=0))) if args.slab_axis < 0 else args.slab_axis

    print(f"{run.name}: N={len(Z)}  d_out={Z.shape[1]}")
    print(f"  per-axis sd: {np.array2string(Z.std(axis=0), precision=3)}")
    for i, j in combinations(range(Z.shape[1]), 2):
        r = float(np.corrcoef(Z[:, i], Z[:, j])[0, 1])
        print(f"  corr(dim {i + 1}, dim {j + 1}) = {r:+.3f}")

    for key in keys:
        suffix = "" if args.color != "all" else f"_{key}"
        print(f"  wrote {corner(Z, cols[key], key, run / f'corner{suffix}.png', run.name, args.dpi)}")
    if Z.shape[1] > 2:
        key = keys[0]
        print(
            "  wrote "
            f"{slabs(Z, cols[key], key, run / 'slabs.png', run.name, args.slabs, axis, args.dpi)}"
        )


if __name__ == "__main__":
    main()

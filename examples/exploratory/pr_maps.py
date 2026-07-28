#!/usr/bin/env python
"""Side-by-side embedding panels, drawn twice: full extent and robust extent.

Comparing maps by eye is unreliable when a handful of far-flung points set the
axis limits. One outlier three cloud-widths away squeezes the bulk into a corner
and makes an ordinary layout look compact and structureless, while a map without
that outlier fills the frame and shows every detail it has. The two rows here are
the same coordinates: the top on full limits, the bottom clipped to a central
percentile window, with the number of hidden points stated. Any impression that
survives the bottom row is about the map; any impression that does not was about
the axes.

Clark-Evans R is printed on each panel as the numeric check -- below 1 means
clumped, near 1 means Poisson-like -- so the visual reading can be held against a
statistic computed on the same coordinates.

Usage::

    python examples/exploratory/pr_maps.py \
        --runs runs/sasbdb_pr_lm32:32 runs/sasbdb_pr_density:128 runs/sasbdb_pr_lm512:512
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

from pr_clumpiness import clark_evans  # noqa: E402

DEFAULT_RUNS = (
    "runs/sasbdb_pr_lm32:L=32",
    "runs/sasbdb_pr_density:L=128",
    "runs/sasbdb_pr_lm512:L=512",
)


def peak_position(X: np.ndarray) -> np.ndarray:
    """Location of each profile's maximum, as a fraction of Dmax."""
    return X.argmax(axis=1) / float(X.shape[1])


def window(Z: np.ndarray, pct: float):
    """Central percentile box, widened to equal aspect so shapes stay honest."""
    lo = np.percentile(Z, pct, axis=0)
    hi = np.percentile(Z, 100 - pct, axis=0)
    c = 0.5 * (lo + hi)
    half = 0.5 * max(hi - lo) * 1.05
    return c - half, c + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--pct", type=float, default=1.0)
    ap.add_argument(
        "--color",
        default=None,
        help="path to an .npy of per-point values; default is P(r) peak position",
    )
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--out", default="runs/maps_side_by_side.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = None
    if args.color is not None:
        p = Path(args.color)
        ext = np.load(p if p.is_absolute() else _ROOT / p).astype(np.float64)

    rows = []
    for spec in args.runs:
        path, _, label = spec.rpartition(":")
        run = Path(path) if Path(path).is_absolute() else _ROOT / path
        Z = np.load(run / "Z.npy").astype(np.float64)
        X = np.load(run / "X.npy").astype(np.float64)
        c = ext if ext is not None else peak_position(X)
        rows.append((label or run.name, Z, c, clark_evans(Z)))

    n = len(rows)
    cmap = args.cmap or ("tab10" if ext is not None and len(np.unique(ext)) <= 12 else "plasma")
    vmin, vmax = (None, None) if ext is not None else (0.0, 0.8)
    fig, ax = plt.subplots(2, n, figsize=(4.3 * n, 8.8))
    ax = np.atleast_2d(ax)
    for j, (label, Z, c, ce) in enumerate(rows):
        lo, hi = window(Z, args.pct)
        hidden = int(((Z < lo) | (Z > hi)).any(axis=1).sum())
        for row in (0, 1):
            a = ax[row, j]
            a.scatter(Z[:, 0], Z[:, 1], c=c, s=5, cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            if row == 1:
                a.set_xlim(lo[0], hi[0])
                a.set_ylim(lo[1], hi[1])
                a.set_title(
                    f"central {100 - 2 * args.pct:g}%  ({hidden} points off-frame)",
                    fontsize=9,
                )
            else:
                a.set_title(f"{label}   full extent, Clark-Evans R={ce:.3f}", fontsize=10)
    fig.suptitle(
        "Same embeddings, full extent (top) versus robust extent (bottom)\n"
        f"colour = {'supplied labels' if ext is not None else 'peak position r/Dmax'}",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out) if Path(args.out).is_absolute() else _ROOT / args.out
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    for label, Z, _, ce in rows:
        lo, hi = window(Z, args.pct)
        span = (Z.max(axis=0) - Z.min(axis=0)).max()
        core = (hi - lo).max()
        print(
            f"{label:<8} Clark-Evans R={ce:.3f}   full span/core span = "
            f"{span / max(core, 1e-12):.1f}x"
        )


if __name__ == "__main__":
    main()

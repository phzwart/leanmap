#!/usr/bin/env python
"""Does the leftover density structure live at the size of a landmark cell?

``pr_licensed.py`` established that the residual -- the part of the embedded
density that ambient density, curvature and local dimension all fail to explain
-- is organised at short range and dies off much faster than the true ambient
field. Short range is suspicious because leanmap lays points out through a soft
mixture of experts over ``n_landmarks`` anchors, and such a layout can only
fabricate density at the size of one expert's territory, roughly ``N /
n_landmarks`` points. But "short range" on its own is not proof: plenty of real
structure is also small.

The way to settle it is to move the suspect and see whether the effect follows.
If the tessellation is the source, then changing ``n_landmarks`` rescales the
cell and the residual's correlation length must ride along with it -- so plotted
against ``k`` divided by the cell size, curves from different landmark counts
collapse onto one another. If instead the residual reflects real fine structure
in the data, its correlation length is a property of the data and must sit still
while ``n_landmarks`` moves.

The ambient field is carried through every panel as a control: it is the same
data in every run, so it must *not* collapse under cell-size rescaling, and it
must *not* shift with ``n_landmarks``. If the control moves too, the comparison
is broken and nothing should be read from it.

Usage::

    python examples/exploratory/pr_landmark_scale.py \
        --runs runs/sasbdb_pr_lm32:32 runs/sasbdb_pr_density:128 \
               runs/sasbdb_pr_lm512:512
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

from pr_clumpiness import density, intrinsic_dim, knn_dist, morans_i  # noqa: E402
from pr_licensed import cv_residual, local_dim  # noqa: E402

DEFAULT_RUNS = (
    "runs/sasbdb_pr_lm32:32",
    "runs/sasbdb_pr_density:128",
    "runs/sasbdb_pr_lm512:512",
)
K_GRID = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


def half_scale(ks: np.ndarray, mi: np.ndarray) -> float:
    """Neighbourhood size at which Moran's I has fallen to half its short-range value.

    A single number for "how far does this field stay correlated". Interpolated in
    log-k because the grid is geometric. Returns NaN when the curve never halves
    inside the grid, which is itself informative -- that field is broad.
    """
    if mi[0] <= 0:
        return float("nan")
    target = 0.5 * mi[0]
    below = np.nonzero(mi <= target)[0]
    if not len(below):
        return float("nan")
    j = int(below[0])
    if j == 0:
        return float(ks[0])
    lo, hi = mi[j - 1], mi[j]
    w = 0.0 if lo == hi else (lo - target) / (lo - hi)
    return float(np.exp(np.log(ks[j - 1]) + w * (np.log(ks[j]) - np.log(ks[j - 1]))))


def curves(run: Path, k: int, metric: str, seed: int):
    """Moran's I of the unexplained residual and of the ambient field, versus scale."""
    from sklearn.neighbors import NearestNeighbors

    X = np.load(run / "X.npy").astype(np.float64)
    Z = np.load(run / "Z.npy").astype(np.float64)
    d_amb = knn_dist(X, k, metric)
    dim = intrinsic_dim(knn_dist(X, 10, metric))
    la = np.log10(density(d_amb, dim))
    lz = np.log10(density(knn_dist(Z, k, "euclidean"), float(Z.shape[1])))
    ldim = local_dim(d_amb)
    feats = np.column_stack([la, np.log10(np.clip(ldim, 1e-6, None))])
    # The most forgiving fit from pr_licensed: curved, plus local dimension. What
    # survives this is what needs explaining.
    _, resid = cv_residual("spline", feats, lz, seed=seed)

    ks = np.array([kk for kk in K_GRID if kk < len(Z) - 1], dtype=float)
    mi_r, mi_a = [], []
    for kk in ks:
        nb = NearestNeighbors(n_neighbors=int(kk) + 1).fit(Z).kneighbors(Z)[1][:, 1:]
        mi_r.append(morans_i(resid, nb))
        mi_a.append(morans_i(la, nb))
    return len(Z), ks, np.array(mi_r), np.array(mi_a)


def _plot(rows, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))

    for (name, lm, n, ks, mi_r, mi_a), c in zip(rows, cols):
        cell = n / lm
        lab = f"L={lm} (cell≈{cell:.0f})"
        ax[0, 0].plot(ks, mi_r, "o-", color=c, label=lab, ms=4)
        ax[0, 1].plot(ks / cell, mi_r, "o-", color=c, label=lab, ms=4)
        ax[1, 0].plot(ks, mi_a, "o-", color=c, label=lab, ms=4)
        ax[1, 1].plot(ks / cell, mi_a, "o-", color=c, label=lab, ms=4)

    for a in ax.ravel():
        a.set_xscale("log")
        a.axhline(0, color="0.6", lw=0.8)
        a.legend(fontsize=8)
        a.grid(alpha=0.25)
    ax[0, 0].set_title("residual vs raw scale\n(tessellation ⇒ curves separate)", fontsize=10)
    ax[0, 1].set_title(
        "residual vs scale / cell size\n(tessellation ⇒ curves collapse)", fontsize=10
    )
    ax[1, 0].set_title("CONTROL ambient field, raw scale\n(must not move with L)", fontsize=10)
    ax[1, 1].set_title(
        "CONTROL ambient field, rescaled\n(must NOT collapse)", fontsize=10
    )
    for a in ax[1]:
        a.set_xlabel("neighbourhood size k" if a is ax[1, 0] else "k / (N / n_landmarks)")
    for a in ax[:, 0]:
        a.set_ylabel("Moran's I")
    fig.suptitle(
        "Is the unexplained density organised at the landmark cell scale?", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS), help="path:n_landmarks")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/landmark_scale.png")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        name, _, lm = spec.rpartition(":")
        lm = int(lm)
        run = Path(name) if Path(name).is_absolute() else _ROOT / name
        n, ks, mi_r, mi_a = curves(run, args.k, args.metric, args.seed)
        rows.append((run.name, lm, n, ks, mi_r, mi_a))

    hdr = f"{'run':<24}{'L':>5}{'cell':>7}{'k_half res':>12}{'/cell':>8}{'k_half amb':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, lm, n, ks, mi_r, mi_a in rows:
        cell = n / lm
        hr, ha = half_scale(ks, mi_r), half_scale(ks, mi_a)
        print(
            f"{name:<24}{lm:>5}{cell:>7.0f}{hr:>12.1f}{hr / cell:>8.2f}{ha:>12.1f}"
        )
    print()
    print("read: 'k_half res' tracking 'cell' (so '/cell' roughly constant) means the")
    print("leftover is the tessellation; 'k_half res' sitting still while cell moves")
    print("16-fold means it is in the data. 'k_half amb' is the control and must be")
    print("flat across rows -- it is the same ambient data every time.")

    out = Path(args.out) if Path(args.out).is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot(rows, out)


if __name__ == "__main__":
    main()

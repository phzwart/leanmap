#!/usr/bin/env python
"""How evenly does an embedding spread the points, and is the spread faithful?

Two different questions the usual battery does not separate:

``spacing_cv``
    Coefficient of variation of the kNN radius *in Z*. Pure uniformity of the
    layout -- low means points are evenly spaced on the page.
``area_sd``
    Standard deviation of ``log(r_Z / r_X)``, the local magnification the map
    applies. Zero means every neighbourhood is blown up by the same factor, i.e.
    the map is area-preserving up to a global scale. This is the honest target:
    a layout can be perfectly uniform on the page while badly misrepresenting a
    non-uniform sample.

For a uniformly sampled feed the two coincide, which is why the s-curve is a
clean test case. Boundary points have one-sided neighbourhoods and inflate both
numbers, so an ``--interior`` mask on the intrinsic coordinates is supported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def knn_radius(A: np.ndarray, k: int = 10) -> np.ndarray:
    """Distance to the k-th nearest neighbour of every row."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
    d, _ = nn.kneighbors(A)
    return d[:, -1]


def interior_mask(coords: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Keep points inside the central ``1 - 2*frac`` quantile of every column."""
    keep = np.ones(len(coords), dtype=bool)
    for j in range(coords.shape[1]):
        lo, hi = np.quantile(coords[:, j], [frac, 1.0 - frac])
        keep &= (coords[:, j] >= lo) & (coords[:, j] <= hi)
    return keep


def uniformity(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    k: int = 10,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return ``(spacing_cv, area_sd)``; radii use all points, stats use ``mask``."""
    rx = knn_radius(np.asarray(X, dtype=np.float64), k)
    rz = knn_radius(np.asarray(Z, dtype=np.float64), k)
    sel = np.ones(len(rx), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    spacing_cv = float(rz[sel].std() / rz[sel].mean())
    ratio = np.log(np.clip(rz[sel], 1e-12, None) / np.clip(rx[sel], 1e-12, None))
    return spacing_cv, float(ratio.std())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument(
        "--Z", nargs="+", required=True, help="one or more Z.npy; label:path also works"
    )
    ap.add_argument(
        "--intrinsic",
        nargs="+",
        default=None,
        help=".npy files holding the true manifold coordinates, for the interior mask",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--edge-frac", type=float, default=0.1)
    args = ap.parse_args()

    X = np.load(args.X)
    mask = None
    if args.intrinsic:
        cols = [np.load(p).reshape(len(X), -1) for p in args.intrinsic]
        mask = interior_mask(np.hstack(cols), args.edge_frac)

    print(f"N={len(X)}  k={args.k}", end="")
    if mask is not None:
        print(f"  interior={int(mask.sum())} ({mask.mean():.0%}, edge_frac={args.edge_frac})")
    else:
        print()
    print(f"{'run':28s}{'spacing_cv':>12}{'area_sd':>10}", end="")
    print(f"{'spacing_cv_int':>16}{'area_sd_int':>13}" if mask is not None else "")
    print("-" * (78 if mask is not None else 50))
    for spec in args.Z:
        label, _, path = spec.partition(":")
        if not path:
            label, path = Path(spec).parent.name, spec
        Z = np.load(path)
        cv, sd = uniformity(X, Z, k=args.k)
        row = f"{label[:28]:28s}{cv:>12.3f}{sd:>10.3f}"
        if mask is not None:
            cv_i, sd_i = uniformity(X, Z, k=args.k, mask=mask)
            row += f"{cv_i:>16.3f}{sd_i:>13.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

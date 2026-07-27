#!/usr/bin/env python
"""Side-by-side panels: a leanmap run against the reference methods.

Recomputes UMAP / PCA on the same features so all panels share one color scale
and one point set, which ``bar.json`` alone cannot show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ingest import load_array, load_features  # noqa: E402


def _umap(X: np.ndarray, *, n_neighbors: int, seed: int, densmap: bool = False):
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        densmap=densmap,
        random_state=seed,
    ).fit_transform(X)


def _pca(X: np.ndarray):
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(X)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", "--color", dest="y", default=None)
    ap.add_argument("--Z", required=True, help="leanmap Z.npy to compare")
    ap.add_argument("--label", default="leanmap (matched)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmap", default="tab10")
    ap.add_argument("--colorbar-label", default="")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = load_features(args.X)
    y = None if args.y is None else load_array(args.y)
    Z_lm = np.load(args.Z)
    if len(Z_lm) != len(X):
        raise SystemExit(
            f"--Z has {len(Z_lm)} rows but --X has {len(X)}; pass the full-set Z"
        )

    panels = [
        (args.label, Z_lm),
        (f"UMAP (n_neighbors={args.n_neighbors})", _umap(X, n_neighbors=args.n_neighbors, seed=args.seed)),
        ("densMAP", _umap(X, n_neighbors=args.n_neighbors, seed=args.seed, densmap=True)),
        ("PCA-2D", _pca(X)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, Z) in zip(np.atleast_1d(axes), panels):
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], c=y, s=4, cmap=args.cmap, linewidths=0, alpha=0.85
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_alpha(0.25)
    if y is not None and args.colorbar_label:
        fig.colorbar(sc, ax=list(np.atleast_1d(axes)), label=args.colorbar_label, shrink=0.85)
    fig.suptitle(
        "Same points, same colors -- leanmap against the reference bar", fontsize=12
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

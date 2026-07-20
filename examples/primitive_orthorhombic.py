"""Leanmap of primitive-orthorhombic PDB unit cells.

Sample dataset: 58,025 primitive (P-centered) orthorhombic unit cells drawn
from the PDB, described by the volume-normalized Kurlin cell roots
(``rn0..rn5`` = ``r_i / V**(1/3)``). These are a dimensionless, scale-invariant
shape descriptor of the Delaunay-reduced lattice, so the embedding captures
lattice *shape* only.

This reproduces the plain (no-inducing-points) leanmap that was run on the full
cell set, restricted to the clean primitive-orthorhombic island. Because the
orthorhombic root space is a smooth low-dimensional manifold, the graph-loss
leanmap draws the continuous shape transitions as clean folds.

Run:
    python examples/primitive_orthorhombic.py
"""
from __future__ import annotations

import pathlib

import numpy as np

from leanmap import LeanMap

HERE = pathlib.Path(__file__).parent
CSV = HERE / "data" / "primitive_orthorhombic_cells.csv"


def load_roots(csv_path: pathlib.Path):
    """Read the CSV and return (volume-normalized roots, space-group numbers)."""
    import csv

    roots, sg = [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            roots.append([float(row[f"rn{i}"]) for i in range(6)])
            sg.append(int(row["sg_number"]))
    return np.asarray(roots, dtype=np.float32), np.asarray(sg, dtype=int)


def main() -> None:
    X, sg = load_roots(CSV)
    print(f"loaded {len(X):,} primitive-orthorhombic cells, {X.shape[1]}-D roots")

    # Plain leanmap: graph loss on the 6-D roots, no inducing points, no UMAP
    # prior. center-only scaling keeps the roots on their native scale.
    model = LeanMap(
        n_neighbors=15,
        epochs=30,
        scale_mode="center",
        distance_kernel="linear",
        device="mps",
        verbose=True,
    )
    emb = model.fit_transform(X)
    print(f"embedding: {emb.shape}, spread {emb.std(0).round(2)}")

    out = HERE / "primitive_orthorhombic_embedding.npy"
    np.save(out, emb.astype(np.float32))
    print(f"saved {out}")

    # Optional scatter if matplotlib is available.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=1.5, c=sg, cmap="viridis",
                        lw=0, alpha=0.5, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("leanmap of primitive-orthorhombic PDB cells "
                     "(volume-normalized Kurlin roots)")
        fig.colorbar(sc, ax=ax, label="space-group number")
        png = HERE / "primitive_orthorhombic_embedding.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        print(f"saved {png}")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()

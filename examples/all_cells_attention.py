"""Attention leanmap of all 206,184 PDB unit cells (the `cells_leanmap` recipe).

AttentionMapper with 2000 learned inducing points, linear distance kernel, Gram
anchoring, and sparse top-P attention — the configuration behind
`cells_leanmap_latest`. Tuned here for a *pleasant* run: it converges by ~epoch
6 (loss ~0.28), so the default is a short run, the loss prints per epoch, and it
plots once at the end (plotting 206k points every epoch is what makes it crawl).

Runs on GPU automatically when available (device=None -> CUDA -> MPS -> CPU).
On CPU expect ~40-60 s/epoch; on a CUDA box it is far faster — this heavy
attention-over-2000-landmarks step is exactly what the GPU is for.

    python examples/all_cells_attention.py                 # short converged run
    python examples/all_cells_attention.py --epochs 20     # longer
    python examples/all_cells_attention.py --dense         # disable sparse attn
    python examples/all_cells_attention.py --device cuda    # force GPU

Speed levers (fastest first):
    * run on a GPU (--device cuda)
    * keep --plot-every 0 (plot only at the end)  <- biggest CPU win
    * keep sparse attention on (default; --dense turns it off)
    * fewer --epochs (it has converged by ~6)
    * fewer --inducing (500-1000 is often enough for a smooth manifold)
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from leanmap import LeanMap
from leanmap._model import pca_components
from leanmap._graph import standardize

HERE = pathlib.Path(__file__).parent
CSV = HERE / "data" / "all_pdb_cells.csv"

SYS_COLORS = {
    "orthorhombic": "#4daf4a", "monoclinic": "#ff7f00", "tetragonal": "#e41a1c",
    "trigonal": "#984ea3", "hexagonal": "#8c6d5c", "triclinic": "#4c92c3",
    "cubic": "#f781bf",
}
SYS_ORDER = ["orthorhombic", "monoclinic", "tetragonal", "trigonal",
             "hexagonal", "triclinic", "cubic"]


def load(csv_path: pathlib.Path):
    """Return (volume-normalized roots, crystal-system labels)."""
    import csv

    roots, syslab = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            roots.append([float(row[f"rn{i}"]) for i in range(6)])
            syslab.append(row["crystal_system"])
    return np.asarray(roots, dtype=np.float32), np.asarray(syslab)


def plot(emb, syslab, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 7.5))
    for s in SYS_ORDER:
        m = syslab == s
        ax.scatter(emb[m, 0], emb[m, 1], s=1.0, c=SYS_COLORS[s], lw=0,
                   alpha=0.45, rasterized=True)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=SYS_COLORS[s],
                      mec="none", ms=6, label=s) for s in SYS_ORDER]
    ax.legend(handles=handles, title="crystal system", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--inducing", type=int, default=2000)
    ap.add_argument("--attend-top-p", type=int, default=32,
                    help="sparse: attend to the P nearest landmarks (0/dense = all)")
    ap.add_argument("--dense", action="store_true", help="disable sparse attention")
    ap.add_argument("--gram", type=float, default=0.1, help="Gram-anchor weight")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--plot-every", type=int, default=0,
                    help="also plot every N epochs (0 = only at the end)")
    args = ap.parse_args()

    X, syslab = load(CSV)
    print(f"loaded {len(X):,} cells, {X.shape[1]}-D volume-normalized roots")

    # Landmark 2D initialization = PCA-2 of the standardized roots (a geometric
    # init, NOT a UMAP prior). The API needs reference_coords for all points in
    # attention mode; it selects landmarks by FPS in data space and takes their
    # reference coords from here.
    xs, _, _ = standardize(X, mode="center")
    ref = (xs @ pca_components(xs, 2).T).astype(np.float32)

    model = LeanMap(
        device=args.device,
        conditioning="attention",
        n_inducing=args.inducing,
        learn_landmarks=True,
        distance_kernel="linear",
        gram_anchor_weight=args.gram,
        attend_top_p=None if args.dense else args.attend_top_p,
        n_neighbors=15,
        scale_mode="center",
        hidden_dims=(128, 128),
        negative_sample_rate=5,
        learning_rate=2e-3,
        epochs=args.epochs,
        verbose=True,
    )

    cb = None
    if args.plot_every > 0:
        def cb(ep, emb, loss, enc):
            if ep % args.plot_every == 0:
                plot(emb, syslab, HERE / f"all_cells_ep{ep:02d}.png",
                     f"leanmap all cells — epoch {ep} (loss {loss:.3f})")

    t = time.time()
    model.fit(X, reference_coords=ref, on_epoch=cb)
    emb = model.transform(X)
    print(f"fit {time.time() - t:.0f}s | embedding {emb.shape} "
          f"spread {emb.std(0).round(2)}")

    np.save(HERE / "all_cells_embedding.npy", emb.astype(np.float32))
    model.save(HERE / "all_cells_leanmap.pt")
    print("saved all_cells_embedding.npy, all_cells_leanmap.pt")
    plot(emb, syslab, HERE / "all_cells_embedding.png",
         "leanmap of all PDB unit cells (attention + Gram anchor, "
         f"{args.inducing} landmarks)")


if __name__ == "__main__":
    main()

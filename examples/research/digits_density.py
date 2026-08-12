#!/usr/bin/env python
"""Digits density preservation demo.

Fits leanmap on sklearn 8×8 digits with the recommended recipe and reports
ambient↔embedding local-density Spearman (a scorecard axis where leanmap
beats UMAP under L2). Writes a label scatter, a density-coloured map, and a
hexbin density panel under ``examples/out/research/digits_density/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _demo import fit_embed, save_density, save_scatter  # noqa: E402

DEFAULT_OUT = _EXAMPLES / "out" / "research" / "digits_density"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=15, help="k for local density")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    n = len(X)

    print(f"fitting leanmap on digits N={n} D={X.shape[1]} ...", flush=True)
    _, Z, score = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        pca_skip=False,
        lr=2e-2,
        lambda_geo=0.15,
        min_dist=0.5,
        pyramid_level_weights=(1, 2, 8),
    )

    from leanmap import density_correspondence, knn_local_density

    dc = density_correspondence(X, Z, k=args.k)
    dens_a = np.asarray(dc["dens_ambient"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Z.npy", Z)

    save_scatter(
        Z,
        y,
        title=f"leanmap — digits labels (N={n})",
        path=out / "scatter_labels.png",
        cmap="tab10",
        colorbar_label="digit",
    )
    save_scatter(
        Z,
        dens_a,
        title=f"leanmap — digits coloured by ambient density (k={args.k})",
        path=out / "scatter_ambient_density.png",
        cmap="magma",
        colorbar_label="ambient 1/mean-kNN",
    )
    save_density(
        Z,
        title=f"leanmap — digits embedding density (N={n})",
        path=out / "density.png",
    )

    metrics = {
        "n": n,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "k": int(args.k),
        "density_spearman": float(dc["spearman"]),
        "density_pearson_log": float(dc["pearson_log"]),
        "density_fit_slope": float(dc["fit_slope"]),
        "score_mean": float(np.mean(score)),
        "ambient_density_mean": float(knn_local_density(X, k=args.k).mean()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(
        f"density Spearman={metrics['density_spearman']:.3f} "
        f"(higher = better ambient↔embed density match)"
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

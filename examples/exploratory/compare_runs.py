#!/usr/bin/env python
"""Score several runs of the same data side by side on one metric battery.

Every run must have been fitted on the same ``X.npy``; the first run supplies
it and the rest are checked against it, so a mismatch is caught rather than
quietly producing an unfair comparison. The battery is the one from
``umap_coerce``, which includes Clark-Evans R -- the direct measure of how
clumped a layout is.

Usage::

    python examples/exploratory/compare_runs.py \\
        runs/sasbdb_pr_l1_frozen runs/sasbdb_pr_perp16
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from umap_coerce import score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.neighbors import NearestNeighbors

    X = None
    rows = []
    for run in args.runs:
        Xr = np.load(run / "X.npy").astype(np.float64)
        Z = np.load(run / "Z.npy").astype(np.float64)
        if X is None:
            X = Xr
            nn = NearestNeighbors(n_neighbors=args.k + 1, metric="manhattan").fit(X)
            amb_nb = nn.kneighbors(X)[1][:, 1:]
            amb_edges = np.column_stack(
                [np.repeat(np.arange(len(X)), args.k), amb_nb.ravel()]
            )
            bins = np.linspace(0.0, 1.0, X.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grad = (X / X.sum(axis=1, keepdims=True)) @ bins
        elif Xr.shape != X.shape or not np.allclose(Xr, X, atol=1e-6):
            raise SystemExit(f"{run} was fitted on different data; cannot compare")
        rows.append(dict(run=run.name,
                         **score(X, Z, amb_nb, amb_edges, grad, args.k, args.seed)))
        print(f"  scored {run.name}", flush=True)

    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        tab.to_csv(args.out, index=False)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Turn a continuous color channel into a discrete grouping for the label battery.

``label_metrics`` needs classes, but manifold feeds carry a continuous parameter
(s-curve ``t``, PDB resolution). Quantile bins give the battery something to
score while keeping the groups balanced, so ``label_acc_X`` still reads as the
ceiling the raw features support and chance stays at ``1 / bins``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quantile_bins(values: np.ndarray, bins: int = 5) -> np.ndarray:
    """Assign ``values`` to ``bins`` roughly equal-sized quantile groups."""
    v = np.asarray(values, dtype=np.float64).ravel()
    edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(v, edges).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values", required=True, help=".npy of the continuous channel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    v = np.load(args.values)
    y = quantile_bins(v, args.bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, y)
    print(f"{args.out}: {args.bins} bins, counts={np.bincount(y).tolist()}, chance={1/args.bins:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

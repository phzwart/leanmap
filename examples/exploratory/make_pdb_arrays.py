#!/usr/bin/env python
"""Materialize the PDB X-ray validation table as harness arrays.

Reuses ``pdb_validation.load_table`` so the feature set, median fill and
per-channel min-max scaling stay identical to the standalone PDB script:
resolution is written out as a color channel only, never as a coordinate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from pdb_validation import FEATURE_COLS, load_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=_EXAMPLES / "data" / "pdb_xray_reslt2_5k.csv",
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--tag", default="pdb")
    ap.add_argument(
        "--bins",
        type=int,
        default=5,
        help="quantile bins of resolution, written as a proxy grouping for the label battery",
    )
    args = ap.parse_args()

    X, resolution, pdb_ids = load_table(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    x_path = args.out / f"{args.tag}_X.npy"
    c_path = args.out / f"{args.tag}_res.npy"
    np.save(x_path, X.astype(np.float32))
    np.save(c_path, resolution.astype(np.float32))
    np.save(args.out / f"{args.tag}_ids.npy", pdb_ids)

    # PDB has no ground truth, so the label battery gets resolution in quantile
    # bins instead. label_acc_X then reports the ceiling: how much of this
    # grouping the validation features support before any embedding.
    edges = np.quantile(resolution, np.linspace(0, 1, args.bins + 1)[1:-1])
    y_bin = np.digitize(resolution, edges).astype(np.int64)
    b_path = args.out / f"{args.tag}_resbin.npy"
    np.save(b_path, y_bin)

    print(f"{x_path}: N={X.shape[0]} D={X.shape[1]}")
    print(f"features: {FEATURE_COLS}")
    print(f"{c_path}: resolution {resolution.min():.2f}-{resolution.max():.2f} A")
    counts = np.bincount(y_bin)
    print(f"{b_path}: {args.bins} quantile bins, counts={counts.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

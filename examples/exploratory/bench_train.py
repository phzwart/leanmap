#!/usr/bin/env python
"""Where does training time actually go?

Times a short fit under configurations that each remove one cost, so the
difference attributes wall time to that cost rather than guessing from the code.
Absolute numbers are only comparable within one invocation -- MPS timings move
with whatever else is on the GPU -- so always read the deltas, not the totals.

Usage::

    python examples/exploratory/bench_train.py --dataset digits --epochs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data"


def arms(epochs: int):
    """(label, config overrides). Each removes exactly one thing from the default."""
    return [
        ("default", {}),
        ("n_negatives=1 (edge batch -55% forwards)", {"n_negatives": 1}),
        ("lambda_geo=0 (no all-landmark embed, no Procrustes SVD)", {"lambda_geo": 0.0}),
        ("lambda_density=0 (no density stars)", {"lambda_density": 0.0}),
        ("batch_edges=1024 (4x more, 4x cheaper steps)", {"batch_edges": 1024}),
        ("pyramid_scales=0 (single scale)", {"pyramid_scales": 0}),
        ("device=cpu", {"device": "cpu"}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--only", default=None, help="substring filter on arm labels")
    args = ap.parse_args()

    from leanmap import PLANEConfig, fit
    from leanmap.utils import get_logger
    import logging

    get_logger().setLevel(logging.WARNING)  # timings only; drop the epoch chatter

    X = np.load(DATA / f"{args.dataset}_X.npy").astype(np.float32)
    print(f"{args.dataset}: N={len(X)} D={X.shape[1]} epochs={args.epochs}\n")

    rows = []
    for label, over in arms(args.epochs):
        if args.only and args.only not in label:
            continue
        cfg = replace(
            PLANEConfig.for_scale(len(X)), epochs=args.epochs, seed=0, **over
        )
        t0 = time.perf_counter()
        fit(X, args.metric, cfg)
        dt = time.perf_counter() - t0
        rows.append((label, dt))
        print(f"  {dt:7.1f}s  {label}")

    if len(rows) > 1:
        base = rows[0][1]
        print(f"\n  deltas against '{rows[0][0]}' ({base:.1f}s):")
        for label, dt in rows[1:]:
            print(f"    {100 * (dt - base) / base:+6.1f}%  {label}")


if __name__ == "__main__":
    main()

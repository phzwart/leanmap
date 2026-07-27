#!/usr/bin/env python
"""leanmap on 8x8 handwritten digits (sklearn MNIST-like)."""

from __future__ import annotations

import argparse

from sklearn.datasets import load_digits

from _demo import OUT_DIR, fit_embed, save_scatter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-class",
        type=int,
        default=None,
        help="optional cap on samples per digit class",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = load_digits()
    X = data.data.astype("float32")  # (1797, 64) — flattened 8x8
    y = data.target.astype("int64")
    if args.n_class is not None:
        import numpy as np

        rng = np.random.default_rng(args.seed)
        keep = []
        for c in range(10):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            keep.append(idx[: args.n_class])
        keep = np.concatenate(keep)
        X, y = X[keep], y[keep]

    result, Z, _ = fit_embed(
        X, epochs=args.epochs, seed=args.seed, device=args.device
    )
    out = save_scatter(
        Z,
        y,
        title="leanmap — digits (8×8)",
        path=OUT_DIR / "digits.png",
        cmap="tab10",
        colorbar_label="digit",
    )
    print(f"N={len(X)} d={X.shape[1]} -> embedding {Z.shape}")
    print(f"pyramid_scales={result.config.pyramid_scales} "
          f"level_weights={result.config.pyramid_level_weights} "
          f"coarse_backbone={result.config.pyramid_coarse_backbone}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

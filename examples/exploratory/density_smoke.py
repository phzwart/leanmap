#!/usr/bin/env python
"""Smoke test for the density correlation term.

Two cases with known answers. A uniformly sampled s-curve has no density ordering
to reproduce, so raising ``lambda_density`` must leave it uniform -- if the term
manufactures contrast here it is broken. A two-blob mixture with a 10:1 sampling
ratio does have an ordering, and the correlation between ambient and embedded log
radius should climb with the weight rather than being flattened the way a
uniformity-enforcing objective would flatten it.

Note what is *not* checked: the amount of contrast. The term is scale free by
design, so a layout that reproduces the ordering with half the spread is a
success, not a failure. Magnitude was the previous design and could not be met at
all when the intrinsic dimension greatly exceeded ``d_out``.

Usage::

    python examples/exploratory/density_smoke.py
    python examples/exploratory/density_smoke.py --lambdas 0 1 --epochs 60
"""

from __future__ import annotations

import argparse

import numpy as np


def make_scurve(n: int, seed: int) -> np.ndarray:
    from sklearn.datasets import make_s_curve

    return make_s_curve(n, noise=0.02, random_state=seed)[0].astype(np.float32)


def make_blobs(n: int, seed: int) -> np.ndarray:
    """Two Gaussians of equal spread but 10:1 sampling, i.e. real 10x contrast."""
    rng = np.random.default_rng(seed)
    n_dense = int(n * 10 / 11)
    a = rng.normal(0, 1, (n_dense, 8))
    b = rng.normal(0, 1, (n - n_dense, 8)) + 12.0
    return np.vstack([a, b]).astype(np.float32)


def log_radius(A: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    d = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, -1]
    return np.log(np.maximum(d, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    from dataclasses import replace

    from scipy.stats import spearmanr

    from leanmap import PLANEConfig, fit

    for name, X in (("s_curve (uniform)", make_scurve(args.n, 0)),
                    ("blobs (10x contrast)", make_blobs(args.n, 0))):
        lr_amb = log_radius(X, args.k)
        print(f"\n=== {name}  N={len(X)}  ambient log-r sd={lr_amb.std():.3f} ===")
        for lam in args.lambdas:
            cfg = replace(
                PLANEConfig.for_scale(len(X)),
                epochs=args.epochs,
                lambda_density=lam,
                seed=0,
            )
            res = fit(X, "l2", cfg)
            Z = np.asarray(res.embed(X, return_score=False)[0])
            lr_emb = log_radius(Z, args.k)
            info = getattr(res, "density", {})
            print(
                f"  lambda={lam:<5g} layout log-r sd={lr_emb.std():.3f}"
                f"  spearman(ambient, embedded)={spearmanr(lr_amb, lr_emb).statistic:+.3f}"
                f"  dim={info.get('intrinsic_dim', float('nan')):.2f}"
            )


if __name__ == "__main__":
    main()

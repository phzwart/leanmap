#!/usr/bin/env python
"""leanmap on the classic Swiss-roll manifold."""

from __future__ import annotations

import argparse

from sklearn.datasets import make_swiss_roll

from _demo import OUT_DIR, fit_embed, save_scatter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="number of points")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--lambda-geo",
        type=float,
        default=0.5,
        help="weight for the coarse geodesic backbone: classical MDS of "
        "landmark geodesics + Procrustes pull (pins/untwists the global "
        "metric gauge). 0 = off.",
    )
    ap.add_argument(
        "--landmark-poisson",
        dest="landmark_poisson",
        action="store_true",
        help="pick landmarks by geodesic Poisson-disk (blue-noise) sampling",
    )
    ap.set_defaults(landmark_poisson=True)
    args = ap.parse_args()

    X, t = make_swiss_roll(n_samples=args.n, noise=args.noise, random_state=args.seed)
    X = X.astype("float32")
    result, Z, _ = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        landmark_poisson=args.landmark_poisson,
        lambda_geo=args.lambda_geo,
        # Mild delay so local affinity establishes topology before the
        # MDS gauge locks the untwisted layout.
        geo_ramp=(0.2, 0.45),
        learn_landmarks=False,
    )
    out = save_scatter(
        Z,
        t,
        title="leanmap — Swiss roll",
        path=OUT_DIR / "swiss_roll.png",
        colorbar_label="manifold parameter",
    )
    print(f"N={len(X)} d={X.shape[1]} -> embedding {Z.shape}")
    print(f"pyramid_scales={result.config.pyramid_scales} "
          f"level_weights={result.config.pyramid_level_weights} "
          f"coarse_backbone={result.config.pyramid_coarse_backbone} "
          f"lambda_geo={result.config.lambda_geo}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""leanmap on a Swiss cone with a punched hole (Isomap-style stress test).

The strip flares with the spiral (cone-like generators) and a rectangle is
removed in parameter space so the intrinsic domain is multiply connected —
a classic check that local affinities do not bridge the gap.
"""

from __future__ import annotations

import argparse

import numpy as np

from _demo import OUT_DIR, fit_embed, save_scatter


def make_swiss_cone(
    n_samples: int = 5000,
    *,
    noise: float = 0.05,
    hole: bool = True,
    n_turns: float = 1.5,
    width0: float = 4.0,
    width1: float = 14.0,
    hole_t: tuple[float, float] = (0.35, 0.65),
    hole_h: tuple[float, float] = (0.25, 0.75),
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a flaring Swiss-roll ribbon (cone) with an optional hole.

    Parameters live on ``(t, h)`` with ``t`` the spiral angle and ``h`` the
    cross-ribbon coordinate in ``[0, 1]``. Ambient map::

        width(t) = width0 + (width1 - width0) * (t - t_min) / (t_max - t_min)
        x = t cos(t),  y = (h - 0.5) * width(t),  z = t sin(t)

    When ``hole`` is True, points whose normalised ``(t, h)`` fall in the
    axis-aligned box ``hole_t × hole_h`` are rejected (rejection sampling).
    """
    rng = np.random.default_rng(random_state)
    t_min = 1.5 * np.pi
    t_max = t_min + n_turns * 2.0 * np.pi

    pts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    # Oversample; hole rejects ~ fraction of the box area.
    batch = max(n_samples * 3, 1024)
    while sum(map(len, pts)) < n_samples:
        t = rng.uniform(t_min, t_max, size=batch)
        h = rng.uniform(0.0, 1.0, size=batch)
        if hole:
            t_n = (t - t_min) / (t_max - t_min)
            in_hole = (
                (t_n >= hole_t[0])
                & (t_n <= hole_t[1])
                & (h >= hole_h[0])
                & (h <= hole_h[1])
            )
            keep = ~in_hole
            t, h = t[keep], h[keep]
        if t.size == 0:
            continue
        frac = (t - t_min) / (t_max - t_min)
        width = width0 + (width1 - width0) * frac
        x = t * np.cos(t)
        y = (h - 0.5) * width
        z = t * np.sin(t)
        pts.append(np.column_stack([x, y, z]))
        colors.append(t)

    X = np.concatenate(pts, axis=0)[:n_samples]
    t = np.concatenate(colors, axis=0)[:n_samples]
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32), t.astype(np.float64)


def _save_ambient_3d(X: np.ndarray, t: np.ndarray, path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, s=2, cmap="viridis", linewidths=0)
    ax.set_title("Swiss cone + hole (ambient)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08, label="spiral parameter t")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000, help="number of points")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--no-hole", action="store_true", help="keep the full ribbon")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--lambda-geo",
        type=float,
        default=0.5,
        help="geodesic MDS + Procrustes backbone weight (0 = off)",
    )
    args = ap.parse_args()

    X, t = make_swiss_cone(
        n_samples=args.n,
        noise=args.noise,
        hole=not args.no_hole,
        random_state=args.seed,
    )
    ambient = OUT_DIR / "swiss_cone_ambient.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_ambient_3d(X, t, ambient)
    print(f"saved ambient {ambient}")

    result, Z, _ = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        landmark_poisson=True,
        lambda_geo=args.lambda_geo,
        geo_ramp=(0.2, 0.45),
        learn_landmarks=False,
    )
    out = save_scatter(
        Z,
        t,
        title="leanmap — Swiss cone + hole",
        path=OUT_DIR / "swiss_cone.png",
        colorbar_label="spiral parameter t",
    )
    print(f"N={len(X)} d={X.shape[1]} -> embedding {Z.shape}")
    print(
        f"pyramid_scales={result.config.pyramid_scales} "
        f"level_weights={result.config.pyramid_level_weights} "
        f"coarse_backbone={result.config.pyramid_coarse_backbone} "
        f"lambda_geo={result.config.lambda_geo}"
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

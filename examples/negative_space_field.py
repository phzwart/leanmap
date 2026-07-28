#!/usr/bin/env python
"""Probe the negative-space estimator with a uniform field over the S-curve box.

We push a uniform 3-D field of points through the trained leanmap encoder and:

1. compute the *true* algebraic minimal distance to the continuous S-manifold
   (not the empirical distance-to-support the head was trained on),
2. plot that true distance against the predicted lower / median / upper score
   (a calibration view — the median should hug the diagonal, and the band
   should bracket the truth ~(1-alpha) of the time),
3. show where the field lands in the 2-D embedding, coloured by an on-manifold
   weight exp(-score / lambda) so near-manifold points glow and far points fade.

The S-curve (sklearn ``make_s_curve``) is the extrusion along y in [0, 2] of the
planar curve C(t) = (sin t, sign(t)(cos t - 1)) for t in [-1.5pi, 1.5pi], so the
true distance factorises into an in-plane curve distance and a y-slab distance.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch
from sklearn.datasets import make_s_curve

from _demo import OUT_DIR

from leanmap import ALL_FEATURES, PLANEConfig, fit, fit_negative_space
from leanmap.negative_space import PerturbationConfig, _median_nn_scale
from leanmap.distance import EuclideanDistance

PERTURB = PerturbationConfig(n_base=3000, radii_per_base=6, n_uniform_far=1500)


def base_config(n, epochs, device, seed):
    cfg = PLANEConfig.for_scale(n)
    return replace(
        cfg,
        epochs=int(epochs),
        seed=int(seed),
        device=device,
        batch_edges=512,
        n_neighbors=10,
        lr=1e-2,
        lr_after=5e-3,
        lr_switch_epochs=5,
        min_dist=0.3,
        n_negatives=15,
        tau_scale=2.0,
        learn_landmarks=False,
        learn_tau=False,
    )


def true_distance_to_scurve(P: torch.Tensor, n_curve: int = 4000) -> torch.Tensor:
    """Algebraic min distance from points ``P`` (N,3) to the S-curve surface."""
    P = torch.as_tensor(P).float()
    t = torch.linspace(-1.5 * np.pi, 1.5 * np.pi, n_curve)
    Cxz = torch.stack([torch.sin(t), torch.sign(t) * (torch.cos(t) - 1.0)], dim=1)  # (T,2)
    pxz = P[:, [0, 2]]
    # y-slab distance to [0, 2] (extrusion axis).
    py = P[:, 1]
    dy = torch.clamp(torch.maximum(0.0 - py, py - 2.0), min=0.0)
    out = torch.empty(P.shape[0])
    for s in range(0, P.shape[0], 1024):
        chunk = pxz[s : s + 1024]
        dxz = torch.cdist(chunk, Cxz).min(dim=1).values  # (chunk,)
        out[s : s + 1024] = torch.sqrt(dxz**2 + dy[s : s + 1024] ** 2)
    return out


def uniform_field(X_train: np.ndarray, n: int, pad: float, seed: int) -> torch.Tensor:
    """Uniform points over the (padded) bounding box of the training data."""
    Xt = torch.as_tensor(X_train).float()
    lo = Xt.min(dim=0).values
    hi = Xt.max(dim=0).values
    rng = (hi - lo).clamp_min(1e-9)
    lo_p, hi_p = lo - pad * rng, hi + pad * rng
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, Xt.shape[1], generator=g)
    return lo_p + u * (hi_p - lo_p)


def calibration_panel(ax, pred, y_true, key, color):
    x = pred[key].numpy()
    y = y_true.numpy()
    hi_lim = float(np.percentile(np.concatenate([x, y]), 99.5))
    ax.scatter(x, y, s=4, alpha=0.25, c=color, linewidths=0)
    ax.plot([0, hi_lim], [0, hi_lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, hi_lim)
    ax.set_ylim(0, hi_lim)
    ax.set_xlabel(f"predicted {key}")
    ax.set_ylabel("true algebraic distance")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500, help="S-curve training points")
    ap.add_argument("--n-field", type=int, default=6000, help="uniform field points")
    ap.add_argument("--pad", type=float, default=0.35, help="bbox padding fraction")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=None, help="exp(-score/lam) scale (auto if unset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, t = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed)
    X = X.astype("float32")

    cfg = base_config(args.n, args.epochs, args.device, args.seed)
    print("training leanmap ...", flush=True)
    res = fit(X, dist_fn="l2", config=cfg)

    print("fitting negative-space head ...", flush=True)
    ns, _ = fit_negative_space(
        res.model, X, feature_groups=ALL_FEATURES, alpha=args.alpha,
        perturb=replace(PERTURB, seed=args.seed), epochs=args.head_epochs, seed=args.seed,
    )

    # Uniform field -> true distance, predicted quantiles, embedding.
    Xf = uniform_field(X, args.n_field, args.pad, args.seed + 321)
    y_true = true_distance_to_scurve(Xf)
    pred = ns.predict(Xf)
    lo, med, hi = pred["lo"], pred["med"], pred["hi"]
    with torch.no_grad():
        Zf, _ = res.model.embed(Xf, return_score=False)
        Zt, _ = res.model.embed(torch.as_tensor(X), return_score=False)
    Zf, Zt = Zf.cpu().numpy(), Zt.cpu().numpy()

    coverage = float(((y_true >= lo) & (y_true <= hi)).float().mean())
    lam = args.lam if args.lam is not None else float(8.0 * _median_nn_scale(torch.as_tensor(X).float(), EuclideanDistance()))
    weight = torch.exp(-med / lam).numpy()
    print(
        f"field: N={args.n_field} true_dist[min/med/max]="
        f"{float(y_true.min()):.3f}/{float(y_true.median()):.3f}/{float(y_true.max()):.3f} "
        f"| bracket coverage of TRUE dist={coverage:.3f} (target={1 - args.alpha:.2f}) | lam={lam:.4f}",
        flush=True,
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
    calibration_panel(axes[0, 0], pred, y_true, "lo", "#2c7fb8")
    calibration_panel(axes[0, 1], pred, y_true, "med", "#d95f0e")
    calibration_panel(axes[0, 2], pred, y_true, "hi", "#31a354")
    axes[0, 1].set_title(
        f"true distance vs predicted score   (bracket coverage={coverage:.3f}, "
        f"target={1 - args.alpha:.2f})"
    )

    # Interval width grows with true distance (uncertainty off-manifold).
    ax = axes[1, 0]
    ax.scatter(y_true.numpy(), (hi - lo).numpy(), s=4, alpha=0.25, c="#756bb1", linewidths=0)
    ax.set_xlabel("true algebraic distance")
    ax.set_ylabel("predicted interval width (hi - lo)")
    ax.set_title("uncertainty vs true distance")

    # Embedding: field coloured by on-manifold weight exp(-median/lam).
    ax = axes[1, 1]
    ax.scatter(Zt[:, 0], Zt[:, 1], s=6, c="0.82", linewidths=0, label="train manifold")
    order = np.argsort(weight)  # draw bright (on-manifold) points last
    s = ax.scatter(
        Zf[order, 0], Zf[order, 1], c=weight[order], s=9, cmap="magma",
        vmin=0.0, vmax=1.0, linewidths=0,
    )
    ax.set_title(f"field in embedding — colour = exp(-median / {lam:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)

    # Embedding: field coloured by TRUE distance (ground-truth reference).
    ax = axes[1, 2]
    ax.scatter(Zt[:, 0], Zt[:, 1], s=6, c="0.82", linewidths=0)
    order2 = np.argsort(-y_true.numpy())  # draw near-manifold last
    s2 = ax.scatter(
        Zf[order2, 0], Zf[order2, 1], c=y_true.numpy()[order2], s=9, cmap="viridis",
        linewidths=0,
    )
    ax.set_title("field in embedding — colour = true distance")
    fig.colorbar(s2, ax=ax, fraction=0.046, pad=0.04)

    for a in (axes[1, 1], axes[1, 2]):
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Negative-space field probe on the S-curve (frozen encoder)")
    fig.tight_layout()
    out = OUT_DIR / "negative_space_field.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One-class conformal novelty on the S-curve: calibrate the NULL, not negatives.

The distance-to-manifold *interval* can't be honestly calibrated against
negatives — their distribution is unknowable and coverage silently depends on
it (we saw 0.90 -> 0.73 across negative mixes). So we flip it: calibrate the
score distribution of held-out ON-MANIFOLD points and turn the median score into
a conformal p-value. Flagging ``p <= alpha`` then controls the on-manifold
false-alarm rate at ``alpha`` — independent of whatever negatives arrive.

This script shows that guarantee holds: the empirical on-manifold flag rate
tracks the nominal ``alpha`` (diagonal), while detection power on three very
different negative sets varies — but the null calibration never moves.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch
from sklearn.datasets import make_s_curve

from _demo import OUT_DIR
from negative_space_cotrain import PERTURB, base_config
from negative_space_field import uniform_field

from leanmap import ALL_FEATURES, calibrate_novelty, fit, fit_negative_space
from leanmap.distance import EuclideanDistance
from leanmap.negative_space import _median_nn_scale


def gen_negatives(X, seed, n=4000):
    Xt = torch.as_tensor(X).float()
    nn = float(_median_nn_scale(Xt, EuclideanDistance()))
    g = torch.Generator().manual_seed(seed)
    base = Xt[torch.randint(0, Xt.shape[0], (n,), generator=g)]
    d = torch.randn(n, 3, generator=g)
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    near = base + (6.0 * nn) * d
    uni = uniform_field(X, n, 0.4, seed + 1)
    center = Xt.mean(0) + torch.tensor([3.0, 0.0, 3.0])
    far = center + 0.5 * torch.randn(n, 3, generator=g)
    return {"near shell (6·nn)": near, "uniform box": uni, "far gaussian": far}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed)
    X = X.astype("float32")

    print("training leanmap + negative-space head ...", flush=True)
    cfg = base_config(args.n, args.epochs, args.device, args.seed)
    res = fit(X, dist_fn="l2", config=cfg)
    ns, _ = fit_negative_space(
        res.model, X, feature_groups=ALL_FEATURES, alpha=args.alpha,
        perturb=replace(PERTURB, seed=args.seed), epochs=args.head_epochs, seed=args.seed,
    )

    # Calibrate the NULL on held-out on-manifold points; test on a fresh draw.
    X_cal, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed + 100)
    X_test_on, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed + 200)
    det = calibrate_novelty(ns, X_cal.astype("float32"))

    p_on = det.pvalue(torch.as_tensor(X_test_on).float()).numpy()
    negatives = gen_negatives(X, args.seed + 300)
    p_neg = {name: det.pvalue(Xn).numpy() for name, Xn in negatives.items()}

    a = args.alpha
    print("\n=========== one-class conformal novelty (alpha = %.2f) ===========" % a)
    print(f"ON-MANIFOLD false-alarm rate (target <= {a:.2f}): {float((p_on <= a).mean()):.3f}")
    for name, p in p_neg.items():
        print(f"detection power on {name:18s}: {float((p <= a).mean()):.3f}")
    print("(the null calibration/threshold is identical in every case above)")

    # --- figure ---
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 101)
    fpr = np.array([(p_on <= g).mean() for g in grid])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    ax = axes[0, 0]
    ax.hist(p_on, bins=25, range=(0, 1), color="#4575b4", alpha=0.85,
            weights=np.full_like(p_on, 1.0 / len(p_on)))
    ax.axhline(1.0 / 25, color="k", ls=":", lw=1, label="uniform")
    ax.axvline(a, color="crimson", ls="--", lw=1.2, label=f"alpha={a}")
    ax.set_title(f"on-manifold p-values (empirical FPR@{a}={float((p_on <= a).mean()):.3f})")
    ax.set_xlabel("conformal p-value"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (guarantee)")
    ax.plot(grid, fpr, color="#4575b4", lw=2, label="on-manifold flag rate")
    ax.set_title("null calibration: nominal alpha vs empirical FPR")
    ax.set_xlabel("nominal alpha"); ax.set_ylabel("empirical on-manifold flag rate")
    ax.set_aspect("equal", adjustable="box"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(grid, fpr, "k--", lw=1.5, label="on-manifold (=FPR)")
    for name, p in p_neg.items():
        power = np.array([(p <= g).mean() for g in grid])
        ax.plot(grid, power, lw=2, label=name)
    ax.axvline(a, color="crimson", ls=":", lw=1)
    ax.set_title("detection power vs alpha (same null threshold for all)")
    ax.set_xlabel("nominal alpha"); ax.set_ylabel("fraction flagged novel")
    ax.legend(fontsize=8)

    # Embedding: uniform field coloured by conformal p-value (projection story).
    ax = axes[1, 1]
    Xfield = negatives["uniform box"]
    p_field = det.pvalue(Xfield).numpy()
    with torch.no_grad():
        Zf, _ = res.model.embed(Xfield, return_score=False)
        Zt, _ = res.model.embed(torch.as_tensor(X), return_score=False)
    Zf, Zt = Zf.cpu().numpy(), Zt.cpu().numpy()
    ax.scatter(Zt[:, 0], Zt[:, 1], s=6, c="0.82", linewidths=0, label="train manifold")
    order = np.argsort(p_field)  # draw high-p (in-distribution) last
    s = ax.scatter(Zf[order, 0], Zf[order, 1], c=p_field[order], s=10, cmap="magma",
                   vmin=0.0, vmax=1.0, linewidths=0)
    ax.set_title("uniform field in embedding — colour = conformal p-value")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Positives-only conformal novelty on the S-curve "
                 "(null calibrated once; stable across negatives)")
    fig.tight_layout()
    out = OUT_DIR / "negative_space_novelty.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()

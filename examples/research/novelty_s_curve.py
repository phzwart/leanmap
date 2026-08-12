#!/usr/bin/env python
"""Positives-only conformal novelty on the S-curve.

Calibrate the score distribution of held-out ON-MANIFOLD points and turn the
median score into a conformal p-value. Flagging ``p <= alpha`` controls the
on-manifold false-alarm rate at ``alpha``, independent of whatever negatives
arrive. This demo shows that the empirical on-manifold flag rate tracks the
nominal alpha, while detection power varies across negative sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_s_curve

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from negative_space_field import PERTURB, base_config, uniform_field  # noqa: E402

from leanmap import ALL_FEATURES, calibrate_novelty, fit, fit_negative_space
from leanmap.distance import EuclideanDistance
from leanmap.negative_space import _median_nn_scale

DEFAULT_OUT = _EXAMPLES / "out" / "research" / "novelty_s_curve"


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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed)
    X = X.astype("float32")

    print("training leanmap + negative-space head ...", flush=True)
    cfg = base_config(args.n, args.epochs, args.device, args.seed)
    res = fit(X, dist_fn="l2", config=cfg)
    ns, _ = fit_negative_space(
        res.model,
        X,
        feature_groups=ALL_FEATURES,
        alpha=args.alpha,
        perturb=replace(PERTURB, seed=args.seed),
        epochs=args.head_epochs,
        seed=args.seed,
    )

    X_cal, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed + 100)
    X_test_on, _ = make_s_curve(n_samples=args.n, noise=0.0, random_state=args.seed + 200)
    det = calibrate_novelty(ns, X_cal.astype("float32"))

    p_on = det.pvalue(torch.as_tensor(X_test_on).float()).numpy()
    negatives = gen_negatives(X, args.seed + 300)
    p_neg = {name: det.pvalue(Xn).numpy() for name, Xn in negatives.items()}

    a = args.alpha
    fpr = float((p_on <= a).mean())
    power = {name: float((p <= a).mean()) for name, p in p_neg.items()}
    print(f"\n=========== one-class conformal novelty (alpha = {a:.2f}) ===========")
    print(f"ON-MANIFOLD false-alarm rate (target <= {a:.2f}): {fpr:.3f}")
    for name, pw in power.items():
        print(f"detection power on {name:18s}: {pw:.3f}")

    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 101)
    fpr_curve = np.array([(p_on <= g).mean() for g in grid])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    ax = axes[0, 0]
    ax.hist(
        p_on,
        bins=25,
        range=(0, 1),
        color="#4575b4",
        alpha=0.85,
        weights=np.full_like(p_on, 1.0 / len(p_on)),
    )
    ax.axhline(1.0 / 25, color="k", ls=":", lw=1, label="uniform")
    ax.axvline(a, color="crimson", ls="--", lw=1.2, label=f"alpha={a}")
    ax.set_title(f"on-manifold p-values (empirical FPR@{a}={fpr:.3f})")
    ax.set_xlabel("conformal p-value")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (guarantee)")
    ax.plot(grid, fpr_curve, color="#4575b4", lw=2, label="on-manifold flag rate")
    ax.set_title("null calibration: nominal alpha vs empirical FPR")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("empirical on-manifold flag rate")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(grid, fpr_curve, "k--", lw=1.5, label="on-manifold (=FPR)")
    for name, p in p_neg.items():
        pwr = np.array([(p <= g).mean() for g in grid])
        ax.plot(grid, pwr, lw=2, label=name)
    ax.axvline(a, color="crimson", ls=":", lw=1)
    ax.set_title("detection power vs alpha (same null threshold for all)")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("fraction flagged novel")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    Xfield = negatives["uniform box"]
    p_field = det.pvalue(Xfield).numpy()
    with torch.no_grad():
        Zf, _ = res.model.embed(Xfield, return_score=False)
        Zt, _ = res.model.embed(torch.as_tensor(X), return_score=False)
    Zf, Zt = Zf.cpu().numpy(), Zt.cpu().numpy()
    ax.scatter(Zt[:, 0], Zt[:, 1], s=6, c="0.82", linewidths=0, label="train manifold")
    order = np.argsort(p_field)
    s = ax.scatter(
        Zf[order, 0],
        Zf[order, 1],
        c=p_field[order],
        s=10,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
    )
    ax.set_title("uniform field in embedding — colour = conformal p-value")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Positives-only conformal novelty on the S-curve "
        "(null calibrated once; stable across negatives)"
    )
    fig.tight_layout()
    fig_path = out / "novelty_s_curve.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    metrics = {
        "n": int(args.n),
        "epochs": int(args.epochs),
        "head_epochs": int(args.head_epochs),
        "alpha": float(a),
        "seed": int(args.seed),
        "on_manifold_fpr": fpr,
        "detection_power": power,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nsaved {fig_path}", flush=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

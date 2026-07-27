#!/usr/bin/env python
"""Amortized, calibrated distance-to-manifold from leanmap's internal state.

Trains leanmap on the Swiss cone, then fits a quantile-regression head on the
frozen encoder's internal features (landmark distances/affinities, FiLM
gamma/beta, gamma-clamp hit, hidden activations, embedding). The head predicts
a calibrated (CQR) lower/median/upper bound on the empirical distance-to-support
``min_j ||x_tilde - x_j||`` — a map of the "negative space" around the manifold.

Compares the full internal-state features against a Dm-only ablation baseline
(does the network's internal geometry sharpen the intervals beyond raw landmark
distances?) and visualizes the median field + interval width in the embedding.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _demo import OUT_DIR, fit_embed
from swiss_cone import make_swiss_cone

from leanmap import (
    ALL_FEATURES,
    DM_ONLY_FEATURES,
    PerturbationConfig,
    distance_to_support,
    fit_negative_space,
)
from leanmap.negative_space import _coverage


def _eval_holdout(ns, X_pert, y_true):
    pred = ns.predict(X_pert)
    lo, med, hi = pred["lo"], pred["med"], pred["hi"]
    cov = _coverage(y_true, lo, hi)
    width = float((hi - lo).median())
    mae = float((med - y_true).abs().median())
    return {"coverage": cov, "median_width": width, "median_abs_err": mae}


def _make_holdout_perturbations(X_train, cfg, seed=123):
    """Independent perturbation set for honest coverage evaluation."""
    from leanmap.distance import EuclideanDistance
    from leanmap.negative_space import _median_nn_scale

    Xt = torch.as_tensor(X_train).float()
    n, D = Xt.shape
    g = torch.Generator().manual_seed(seed)
    dist_fn = EuclideanDistance()
    nn_scale = _median_nn_scale(Xt, dist_fn)
    radii = torch.logspace(
        np.log10(cfg.r_min_mult * nn_scale),
        np.log10(cfg.r_max_mult * nn_scale),
        cfg.radii_per_base,
    )
    base = Xt[torch.randint(0, n, (2000,), generator=g)]
    parts = [base.clone()]
    for r in radii.tolist():
        d = torch.randn(base.shape[0], D, generator=g)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        parts.append(base + r * d)
    X_pert = torch.cat(parts, dim=0)
    y = distance_to_support(X_pert, Xt, dist_fn=dist_fn)
    return X_pert, y


def _plot_field(Z_train, t, ns, X_train, path, title):
    import matplotlib.pyplot as plt

    # A cloud of perturbed points spanning the negative space, embedded + scored.
    cfg = PerturbationConfig(n_base=1500, radii_per_base=5, n_uniform_far=1500, seed=7)
    X_field, y_field = _make_holdout_perturbations(X_train, cfg, seed=7)
    pred = ns.predict(X_field)
    med = pred["med"].numpy()
    width = (pred["hi"] - pred["lo"]).numpy()
    with torch.no_grad():
        Z_field, _ = ns.model.embed(X_field, return_score=False)
    Z_field = Z_field.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axes[0]
    ax.scatter(Z_train[:, 0], Z_train[:, 1], c=t, s=5, cmap="viridis", linewidths=0)
    ax.set_title("leanmap embedding (train)")
    for a in axes:
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal", adjustable="datalim")

    sc1 = axes[1].scatter(Z_field[:, 0], Z_field[:, 1], c=med, s=8, cmap="magma", linewidths=0)
    axes[1].set_title("predicted median distance-to-manifold")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label="median d")

    sc2 = axes[2].scatter(Z_field[:, 0], Z_field[:, 1], c=width, s=8, cmap="cividis", linewidths=0)
    axes[2].set_title("CQR interval width (uncertainty)")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04, label="hi - lo")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=60, help="leanmap training epochs")
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage (target=1-alpha)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, t = make_swiss_cone(n_samples=args.n, noise=args.noise, random_state=args.seed)

    print("== fitting leanmap on swiss cone ==")
    result, Z, _ = fit_embed(
        X, epochs=args.epochs, seed=args.seed, device=args.device,
        landmark_poisson=True, lambda_geo=0.5, geo_ramp=(0.2, 0.45),
        learn_landmarks=False,
    )
    model = result.model

    perturb = PerturbationConfig(n_base=4000, radii_per_base=6, n_uniform_far=2000, seed=args.seed)
    X_hold, y_hold = _make_holdout_perturbations(X, perturb, seed=args.seed + 999)

    print("\n== fitting FULL internal-state quantile head ==")
    ns_full, stats_full = fit_negative_space(
        model, X, feature_groups=ALL_FEATURES, alpha=args.alpha,
        perturb=perturb, epochs=args.head_epochs, seed=args.seed,
    )
    hold_full = _eval_holdout(ns_full, X_hold, y_hold)

    print("\n== fitting Dm-only ablation baseline ==")
    ns_dm, stats_dm = fit_negative_space(
        model, X, feature_groups=DM_ONLY_FEATURES, alpha=args.alpha,
        perturb=perturb, epochs=args.head_epochs, seed=args.seed,
    )
    hold_dm = _eval_holdout(ns_dm, X_hold, y_hold)

    print("\n================ RESULTS (holdout perturbations) ================")
    print(f"target coverage = {1 - args.alpha:.2f}")
    print(
        f"FULL   : coverage={hold_full['coverage']:.3f} "
        f"median_width={hold_full['median_width']:.4g} "
        f"median|err|={hold_full['median_abs_err']:.4g}"
    )
    print(
        f"Dm-only: coverage={hold_dm['coverage']:.3f} "
        f"median_width={hold_dm['median_width']:.4g} "
        f"median|err|={hold_dm['median_abs_err']:.4g}"
    )
    sharper = hold_dm["median_width"] - hold_full["median_width"]
    print(
        f"\ninternal states {'SHARPEN' if sharper > 0 else 'do NOT sharpen'} "
        f"the interval by {sharper:.4g} (median width) at matched coverage"
    )

    out = _plot_field(
        Z, t, ns_full, X, OUT_DIR / "negative_space_field.png",
        title="Negative-space map (full internal-state quantile head, CQR)",
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Co-trained vs frozen negative-space distance estimator on the S-curve.

Two arms, identical config except the co-training weight:

* **frozen**  : train leanmap (lambda_dist=0), freeze, then fit the quantile
  head as a post-hoc probe on the frozen internal states.
* **cotrain** : train leanmap with lambda_dist>0 so the auxiliary
  distance-to-support quantile head is optimised *jointly* — its pinball loss
  also back-props into the encoder — then recalibrate (CQR) on the frozen model.

Reports, on an independent holdout perturbation set: CQR coverage, median
interval width (sharpness), median |error| of the median predictor; plus the
embedding-fidelity cost (geodesic Spearman / stress) so we can see whether
letting the encoder reshape its internals sharpens the intervals and at what
price to the on-manifold chart.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch
from sklearn.datasets import make_s_curve

from _demo import OUT_DIR

from leanmap import (
    ALL_FEATURES,
    PLANEConfig,
    fit,
    fit_negative_space,
    sample_perturbations,
)
from leanmap.negative_space import PerturbationConfig, _coverage

# Shared perturbation distribution: calibration and holdout MUST be exchangeable
# for the CQR coverage guarantee to hold, so both use this config (fresh seeds).
PERTURB = PerturbationConfig(n_base=3000, radii_per_base=6, n_uniform_far=1500)


def eval_ns(ns, X_pert, y):
    pred = ns.predict(X_pert)
    lo, med, hi = pred["lo"], pred["med"], pred["hi"]
    return {
        "coverage": _coverage(y, lo, hi),
        "median_width": float((hi - lo).median()),
        "median_abs_err": float((med - y).abs().median()),
    }


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


def plot_fields(Z_base, Z_co, t, ns_base, ns_co, X_train, path):
    import matplotlib.pyplot as plt

    Xf, _ = sample_perturbations(
        torch.as_tensor(X_train).float(), replace(PERTURB, n_base=1500, seed=7)
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))
    for (name, Z, ns) in [("frozen", Z_base, ns_base), ("cotrain", Z_co, ns_co)]:
        row = 0 if name == "frozen" else 1
        pred = ns.predict(Xf)
        med = pred["med"].numpy()
        width = (pred["hi"] - pred["lo"]).numpy()
        with torch.no_grad():
            Zf, _ = ns.model.embed(Xf, return_score=False)
        Zf = Zf.cpu().numpy()
        axes[row, 0].scatter(Z[:, 0], Z[:, 1], c=t, s=5, cmap="viridis", linewidths=0)
        axes[row, 0].set_ylabel(name, fontsize=12)
        axes[row, 0].set_title(f"{name}: embedding")
        s1 = axes[row, 1].scatter(Zf[:, 0], Zf[:, 1], c=med, s=7, cmap="magma", linewidths=0)
        axes[row, 1].set_title(f"{name}: median distance-to-manifold")
        fig.colorbar(s1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        s2 = axes[row, 2].scatter(Zf[:, 0], Zf[:, 1], c=width, s=7, cmap="cividis", linewidths=0)
        axes[row, 2].set_title(f"{name}: CQR interval width")
        fig.colorbar(s2, ax=axes[row, 2], fraction=0.046, pad=0.04)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Negative-space distance estimator: frozen probe vs co-trained")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--lambda-dist", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, t = make_s_curve(n_samples=args.n, noise=args.noise, random_state=args.seed)
    X = X.astype("float32")

    # Holdout drawn from the SAME perturbation distribution (fresh seed) so it
    # is exchangeable with each arm's CQR calibration set.
    X_hold, y_hold = sample_perturbations(
        torch.as_tensor(X).float(), replace(PERTURB, seed=args.seed + 999)
    )

    # ---- Arm 1: frozen two-stage ----
    print("== ARM 1: train leanmap (frozen), then fit head as a probe ==")
    cfg = base_config(args.n, args.epochs, args.device, args.seed)
    res_base = fit(X, dist_fn="l2", config=cfg)
    with torch.no_grad():
        Z_base, _ = res_base.model.embed(torch.as_tensor(X), return_score=False)
    Z_base = Z_base.cpu().numpy()
    ns_base, _ = fit_negative_space(
        res_base.model, X, feature_groups=ALL_FEATURES, alpha=args.alpha,
        perturb=replace(PERTURB, seed=args.seed), epochs=args.head_epochs, seed=args.seed,
    )
    m_base = eval_ns(ns_base, X_hold, y_hold)

    # ---- Arm 2: co-trained (identical config + lambda_dist) ----
    print("\n== ARM 2: co-train leanmap + distance head jointly ==")
    cfg_co = replace(
        cfg, lambda_dist=args.lambda_dist, dist_ramp=(0.4, 0.7),
        dist_alpha=args.alpha, dist_perturb_per_step=256,
    )
    res_co = fit(X, dist_fn="l2", config=cfg_co)
    with torch.no_grad():
        Z_co, _ = res_co.model.embed(torch.as_tensor(X), return_score=False)
    Z_co = Z_co.cpu().numpy()
    # (2a) the head co-trained jointly with the encoder (light regression budget)
    ns_co = res_co.negative_space
    m_co = eval_ns(ns_co, X_hold, y_hold)
    # (2b) fair feature ablation: fresh, fully-trained head on the *co-trained*
    # (now frozen) encoder — isolates feature quality from the head's budget.
    print("\n== ARM 2b: fresh head on the co-trained (frozen) encoder ==")
    ns_cofeat, _ = fit_negative_space(
        res_co.model, X, feature_groups=ALL_FEATURES, alpha=args.alpha,
        perturb=replace(PERTURB, seed=args.seed), epochs=args.head_epochs, seed=args.seed,
    )
    m_cofeat = eval_ns(ns_cofeat, X_hold, y_hold)

    def gfid(r):
        gs = r.graph_stats
        return gs.get("geodesic_spearman", float("nan")), gs.get("geodesic_stress", float("nan"))

    sp_b, st_b = gfid(res_base)
    sp_c, st_c = gfid(res_co)

    print("\n================ RESULTS (holdout perturbations) ================")
    print(f"target coverage = {1 - args.alpha:.2f}")

    def row(tag, m, sp, st):
        return (
            f"{tag:16s} coverage={m['coverage']:.3f} width={m['median_width']:.4g} "
            f"|err|={m['median_abs_err']:.4g} | geo_spearman={sp:.3f} stress={st:.3f}"
        )

    print(row("FROZEN-feat+head", m_base, sp_b, st_b))
    print(row("COTRAIN head", m_co, sp_c, st_c))
    print(row("COTRAIN-feat+head", m_cofeat, sp_c, st_c))
    # Feature-quality comparison uses fresh, equally-trained heads on both encoders.
    dw = m_base["median_width"] - m_cofeat["median_width"]
    print(
        f"\nfeature ablation (fresh head both): co-training "
        f"{'SHARPENS' if dw > 0 else 'does NOT sharpen'} the interval by {dw:.4g} "
        f"(median width, +=better); embedding stress change = {st_c - st_b:+.4f} "
        f"(negative = better/unchanged chart)"
    )

    out = plot_fields(Z_base, Z_co, t, ns_base, ns_cofeat, X, OUT_DIR / "negative_space_cotrain.png")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

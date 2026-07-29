#!/usr/bin/env python
"""Digits fit with an OOD basin loss (no junk edges in the primary graph).

Real digits build the neighbour graph as usual. Each epoch, after the geometric
step, an auxiliary update pulls OOD copies (pixel-shuffled and/or μ/σ-matched
Gaussian) toward a learned junk anchor and gently pins real points to their
current embedding so the manifold does not follow the junk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch.optim import AdamW

from _demo import OUT_DIR, fit_embed
from leanmap.probes import structured_probes


def pixel_shuffle_expand(X: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_dec = int(round(frac * len(X)))
    parents = rng.integers(0, len(X), size=n_dec)
    out = np.empty((n_dec, X.shape[1]), dtype=np.float32)
    for i, p in enumerate(parents):
        out[i] = rng.permutation(X[p])
    return out


def gaussian_expand(X: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Independent Gaussian noise matched to per-feature mean/std of X."""
    rng = np.random.default_rng(seed)
    n = int(round(frac * len(X)))
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return rng.normal(mu, sigma, size=(n, X.shape[1])).astype(np.float32)


def make_basin_callback(
    X_real: np.ndarray,
    ood_pools: dict[str, np.ndarray],
    *,
    lambda_basin: float = 1.0,
    lambda_stay: float = 0.5,
    lambda_repel: float = 1.0,
    repel_margin: float | None = None,
    aux_steps: int = 30,
    aux_lr: float = 5e-3,
    batch: int = 256,
    seed: int = 0,
):
    """Park OOD at a junk anchor and hinge-repel them from real embeddings.

    Attraction alone can leave Gaussian junk near the digit cloud if the cover
    head still scores it as in-distribution. The hinge on min-distance to the
    real batch actively forbids that overlap in Z.
    """
    state: dict = {}
    rng = np.random.default_rng(seed + 17)
    names = list(ood_pools.keys())
    if not names:
        raise ValueError("need at least one OOD pool")

    def cb(epoch: int, model, metrics: dict) -> None:
        device = next(model.parameters()).device
        if "opt" not in state:
            with torch.no_grad():
                Z0, _ = model.embed(torch.as_tensor(X_real, dtype=torch.float32))
            Z0 = Z0.to(device)
            c = Z0.mean(dim=0)
            spread = (Z0 - c).pow(2).sum(dim=1).sqrt().median().clamp_min(0.5)
            # Park junk well outside the current cloud.
            anchor = torch.nn.Parameter(c + torch.tensor([4.0 * float(spread), 0.0], device=device))
            state["anchor"] = anchor
            state["Z_target"] = Z0.detach()
            state["spread"] = float(spread)
            state["margin"] = float(repel_margin) if repel_margin is not None else 2.5 * float(spread)
            state["opt"] = AdamW(
                list(model.parameters()) + [anchor],
                lr=aux_lr,
                weight_decay=0.0,
            )
            print(
                f"  [basin] init anchor={anchor.detach().cpu().tolist()} "
                f"spread={float(spread):.3f}  margin={state['margin']:.3f}  "
                f"pools={names}  λ_repel={lambda_repel}",
                flush=True,
            )

        # Soft-refresh the stay target so we track the evolving digit map.
        with torch.no_grad():
            Z_now, _ = model.embed(torch.as_tensor(X_real, dtype=torch.float32))
            state["Z_target"] = (
                0.9 * state["Z_target"] + 0.1 * Z_now.to(device).detach()
            )

        opt = state["opt"]
        anchor = state["anchor"]
        Z_target = state["Z_target"]
        margin = state["margin"]
        model.train()
        n_real = len(X_real)
        totals = {"basin": 0.0, "stay": 0.0, "repel": 0.0}
        per_pool = {nm: 0.0 for nm in names}
        for _ in range(aux_steps):
            ir = rng.integers(0, n_real, size=batch)
            xr = torch.as_tensor(X_real[ir], device=device)
            zr, _, _ = model(xr)
            L_stay = F.mse_loss(zr, Z_target[ir])
            # Detach reals for repulsion so we push junk away, not the manifold.
            z_struct = zr.detach()

            basin_terms = []
            repel_terms = []
            for nm in names:
                pool = ood_pools[nm]
                io = rng.integers(0, len(pool), size=batch)
                xo = torch.as_tensor(pool[io], device=device)
                zo, _, _ = model(xo)
                Lp = F.mse_loss(zo, anchor.expand_as(zo))
                # Hinge: each OOD must stay ≥ margin from nearest structured point.
                min_d = torch.cdist(zo, z_struct).min(dim=1).values
                Lr = F.relu(margin - min_d).mean()
                basin_terms.append(Lp)
                repel_terms.append(Lr)
                per_pool[nm] += float(Lp.item())
            L_basin = torch.stack(basin_terms).mean()
            L_repel = torch.stack(repel_terms).mean()

            loss = (
                lambda_stay * L_stay
                + lambda_basin * L_basin
                + lambda_repel * L_repel
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["basin"] += float(L_basin.item())
            totals["stay"] += float(L_stay.item())
            totals["repel"] += float(L_repel.item())
        if epoch % 10 == 0 or epoch == 1:
            pool_str = " ".join(f"{nm}={per_pool[nm]/aux_steps:.4f}" for nm in names)
            print(
                f"  [basin] epoch {epoch}: "
                f"basin={totals['basin']/aux_steps:.4f} ({pool_str}) "
                f"repel={totals['repel']/aux_steps:.4f} "
                f"stay={totals['stay']/aux_steps:.4f} "
                f"anchor={anchor.detach().cpu().tolist()}",
                flush=True,
            )

    return cb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    ap.add_argument("--frac-dec", type=float, default=0.5, help="shuffled pool size / N")
    ap.add_argument("--frac-gauss", type=float, default=0.5, help="Gaussian pool size / N")
    ap.add_argument("--lambda-basin", type=float, default=1.0)
    ap.add_argument("--lambda-stay", type=float, default=0.5)
    ap.add_argument(
        "--lambda-repel",
        type=float,
        default=1.0,
        help="hinge repulsion of OOD embeddings from real/structured Z",
    )
    ap.add_argument(
        "--repel-margin",
        type=float,
        default=None,
        help="min allowed OOD↔real distance (default: 2.5 × embedding spread)",
    )
    ap.add_argument("--aux-steps", type=int, default=30)
    ap.add_argument(
        "--ood",
        default="shuffle,gauss",
        help="comma-separated OOD kinds: shuffle,gauss (default both)",
    )
    args = ap.parse_args()

    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    kinds = [k.strip() for k in args.ood.split(",") if k.strip()]
    ood_pools: dict[str, np.ndarray] = {}
    if "shuffle" in kinds:
        ood_pools["shuffle"] = pixel_shuffle_expand(X, args.frac_dec, args.seed)
    if "gauss" in kinds:
        ood_pools["gauss"] = gaussian_expand(X, args.frac_gauss, args.seed + 1)
    unknown = set(kinds) - {"shuffle", "gauss"}
    if unknown:
        raise SystemExit(f"unknown --ood kinds: {sorted(unknown)}")
    if not ood_pools:
        raise SystemExit("need at least one of --ood shuffle,gauss")

    pool_desc = "  ".join(f"{k}={len(v)}" for k, v in ood_pools.items())
    print(f"real N={len(X)}  OOD pools: {pool_desc}  (graph uses real only)")

    cb = make_basin_callback(
        X,
        ood_pools,
        lambda_basin=args.lambda_basin,
        lambda_stay=args.lambda_stay,
        lambda_repel=args.lambda_repel,
        repel_margin=args.repel_margin,
        aux_steps=args.aux_steps,
        seed=args.seed,
    )
    result, Z_real, S_real = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        min_dist=args.min_dist,
        callbacks=[cb],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "dual" if set(ood_pools) == {"shuffle", "gauss"} else "_".join(ood_pools)
    if args.lambda_repel > 0:
        tag = f"{tag}_repel"
    model_path = OUT_DIR / f"digits_ood_basin_{tag}.pt"
    result.save(str(model_path))

    model = result.model
    cal = result.calibrator
    embeds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for nm, Xp in ood_pools.items():
            Z, S = model.embed(torch.as_tensor(Xp))
            p = cal.p_value(S, model=model).numpy()
            embeds[nm] = (Z.numpy(), S.numpy(), p)
        mass = float(np.median(X.sum(1)))
        P, pkinds = structured_probes((8, 8), n_variants=8, seed=args.seed, mass_match=mass)
        Z_p, S_p = model.embed(torch.as_tensor(P))
        p_p = cal.p_value(S_p, model=model).numpy()
        Z_p, S_p = Z_p.numpy(), S_p.numpy()

        # Fresh 10k holdouts (not the train pools) for conformal eval.
        rng = np.random.default_rng(args.seed + 99)
        mu, sigma = X.mean(0), X.std(0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        X_gauss_hold = rng.normal(mu, sigma, size=(10_000, X.shape[1])).astype(np.float32)
        Z_gh, S_gh = model.embed(torch.as_tensor(X_gauss_hold))
        p_gh = cal.p_value(S_gh, model=model).numpy()
        Z_gh, S_gh = Z_gh.numpy(), S_gh.numpy()

        parents = rng.integers(0, len(X), size=10_000)
        X_shuf_hold = np.empty((10_000, X.shape[1]), dtype=np.float32)
        for i, p in enumerate(parents):
            X_shuf_hold[i] = rng.permutation(X[p])
        Z_sh, S_sh = model.embed(torch.as_tensor(X_shuf_hold))
        p_sh = cal.p_value(S_sh, model=model).numpy()
        Z_sh, S_sh = Z_sh.numpy(), S_sh.numpy()

    p_real = cal.p_value(torch.as_tensor(S_real), model=model).numpy()
    print(f"\n{'set':14} {'n':>6} {'cover med':>10} {'p50':>8} {'p<0.05':>8}")
    print(
        f"{'real':14} {len(S_real):6d} {np.median(S_real):10.3f} "
        f"{np.median(p_real):8.3f} {(p_real < 0.05).mean():8.3f}"
    )
    for nm, (Z, S, p) in embeds.items():
        print(
            f"{'train_'+nm:14} {len(S):6d} {np.median(S):10.3f} "
            f"{np.median(p):8.3f} {(p < 0.05).mean():8.3f}"
        )
    print(
        f"{'hold_shuffle':14} {len(S_sh):6d} {np.median(S_sh):10.3f} "
        f"{np.median(p_sh):8.3f} {(p_sh < 0.05).mean():8.3f}"
    )
    print(
        f"{'hold_gauss':14} {len(S_gh):6d} {np.median(S_gh):10.3f} "
        f"{np.median(p_gh):8.3f} {(p_gh < 0.05).mean():8.3f}"
    )
    print(
        f"{'structured':14} {len(S_p):6d} {np.median(S_p):10.3f} "
        f"{np.median(p_p):8.3f} {(p_p < 0.05).mean():8.3f}"
    )
    print(f"saved {model_path}")

    # Overlay
    colors = {"shuffle": "crimson", "gauss": "darkorange"}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(Z_real[:, 0], Z_real[:, 1], c=y, cmap="tab10", s=8, alpha=0.55, linewidths=0, zorder=1)
    for nm, (Z, S, p) in embeds.items():
        ax.scatter(
            Z[:, 0], Z[:, 1], c=colors.get(nm, "gray"), s=14, alpha=0.7,
            edgecolors="k", linewidths=0.15, label=f"{nm} train (n={len(Z)})", zorder=2,
        )
    ax.scatter(
        Z_p[:, 0], Z_p[:, 1], c="lime", s=28, marker="x", linewidths=1.2,
        label="structured", zorder=3,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"digits + OOD basin ({'+'.join(ood_pools)}) + repel\n"
        f"min_dist={args.min_dist}  λ_basin={args.lambda_basin}  "
        f"λ_stay={args.lambda_stay}  λ_repel={args.lambda_repel}"
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    p1 = OUT_DIR / f"digits_ood_basin_{tag}_overlay.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    print(f"saved {p1}")

    # Binary-ish: real vs both OOD train pools
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(Z_real[:, 0], Z_real[:, 1], c="steelblue", s=8, alpha=0.5, linewidths=0, label="real", zorder=1)
    for nm, (Z, S, p) in embeds.items():
        ax.scatter(
            Z[:, 0], Z[:, 1], c=colors.get(nm, "gray"), s=12, alpha=0.8,
            linewidths=0, label=nm, zorder=2,
        )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("real vs OOD pools after basin+repel training")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    p2 = OUT_DIR / f"digits_ood_basin_{tag}_binary.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print(f"saved {p2}")

    # Histograms: real / hold shuffle / hold gauss
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(S_real, bins=40, alpha=0.55, label="real", color="steelblue", density=True)
    axes[0].hist(S_sh, bins=40, alpha=0.55, label="hold shuffle", color="crimson", density=True)
    axes[0].hist(S_gh, bins=40, alpha=0.55, label="hold gauss", color="darkorange", density=True)
    axes[0].set_xlabel("cover score"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
    axes[0].set_title("cover score")
    axes[1].hist(p_real, bins=40, alpha=0.55, label="real", color="steelblue", density=True)
    axes[1].hist(p_sh, bins=40, alpha=0.55, label="hold shuffle", color="crimson", density=True)
    axes[1].hist(p_gh, bins=40, alpha=0.55, label="hold gauss", color="darkorange", density=True)
    axes[1].axvline(0.05, color="k", ls="--", lw=1, label="α=0.05")
    axes[1].set_xlabel("conformal p-value"); axes[1].legend(fontsize=8)
    axes[1].set_title("p-value")
    fig.suptitle("dual-basin: 10k holdout shuffle vs Gaussian", y=1.02)
    fig.tight_layout()
    p3 = OUT_DIR / f"digits_ood_basin_{tag}_hist.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p3}")

    np.savez(
        OUT_DIR / f"digits_ood_basin_{tag}_split.npz",
        Z_real=Z_real, S_real=S_real, p_real=p_real, y=y,
        Z_probes=Z_p, S_probes=S_p, p_probes=p_p, kinds=pkinds,
        Z_hold_shuffle=Z_sh, S_hold_shuffle=S_sh, p_hold_shuffle=p_sh,
        Z_hold_gauss=Z_gh, S_hold_gauss=S_gh, p_hold_gauss=p_gh,
        **{f"Z_{nm}": Z for nm, (Z, _, _) in embeds.items()},
        **{f"S_{nm}": S for nm, (_, S, _) in embeds.items()},
        **{f"p_{nm}": p for nm, (_, _, p) in embeds.items()},
    )


if __name__ == "__main__":
    main()

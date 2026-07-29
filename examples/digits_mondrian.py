#!/usr/bin/env python
"""Digits Mondrian conformal demo (digit / gauss / shuffle levels).

Loads ``out/digits.pt`` (or fits one), calibrates Mondrian thresholds with the
default affinity-entropy score, evaluates holdout digits + fresh noise pools,
and writes plots / tables under ``out/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits

from _demo import OUT_DIR, fit_embed, save_scatter
from leanmap import MondrianCalibrator, load_plane, make_mondrian_groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=OUT_DIR / "digits.pt")
    ap.add_argument(
        "--tag",
        default=None,
        help="output name prefix (default: derived from --model stem)",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--score", default="affinity_entropy")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-ood", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.stem
    if tag == "digits":
        tag = "digits_mondrian"
    elif not tag.endswith("_mondrian"):
        tag = f"{tag}_mondrian"
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    i_cal = perm[: args.n_calib]
    i_te = perm[args.n_calib : args.n_calib + args.n_test]
    X_cal, y_cal = X[i_cal], y[i_cal]
    X_te, y_te = X[i_te], y[i_te]

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model.is_file():
        print(f"loading model {args.model}")
        model = load_plane(str(args.model), device=device)
    else:
        print(f"fitting digits model → {args.model}")
        result, Z_all, _ = fit_embed(
            X, epochs=args.epochs, seed=args.seed, device=device, min_dist=args.min_dist
        )
        result.save(str(args.model))
        model = result.model
        save_scatter(
            Z_all, y, title="leanmap — digits", path=OUT_DIR / "digits.png", cmap="tab10"
        )

    cal = MondrianCalibrator(score=args.score)
    cal.fit_from_digits(
        model,
        torch.as_tensor(X_cal),
        n_gauss=args.n_calib,
        n_shuffle=args.n_calib,
        seed=args.seed,
    )
    mondrian_path = OUT_DIR / f"{tag}.pt"
    torch.save(cal.state_dict(), str(mondrian_path))

    alphas = (0.01, 0.05, 0.1)
    levels = cal.levels(alphas=alphas)
    print(f"\nmodel={args.model.name}  tag={tag}")
    print(f"score={cal.score_name}  α grid={alphas}")
    print(f"{'group':10} " + " ".join(f"{'α='+str(a):>10}" for a in alphas) + f" {'n':>6}")
    for g, d in levels.items():
        cells = " ".join(f"{d[a]:10.4f}" for a in alphas)
        print(f"{g:10} {cells} {len(cal.s_calib[g]):6d}")

    # Fresh eval pools (not the calib noise draws)
    groups = make_mondrian_groups(
        torch.as_tensor(X_te[: args.n_ood]),
        n_gauss=args.n_ood,
        n_shuffle=args.n_ood,
        seed=args.seed + 7,
    )
    groups["digit"] = torch.as_tensor(X_te)  # full digit holdout

    with torch.no_grad():
        Z_te, _ = model.embed(torch.as_tensor(X_te))
        Z_te = Z_te.numpy()

    print(f"\neval @ α={args.alpha} (two-sided prediction sets)")
    print(f"{'pool':10} {'n':>5} {'med score':>10} {'p_digit>α':>10} {'top sets'}")
    embed_z = {"digit": Z_te}
    scores = {}
    sets_by = {}
    for name, Xg in groups.items():
        s = cal.score_points(model, Xg)
        p2 = cal.p_values(s, model=model, sided="two")
        sets = cal.prediction_set(s, alpha=args.alpha, model=model)
        scores[name] = s.numpy()
        sets_by[name] = sets
        top = Counter(sets).most_common(3)
        top_s = ", ".join(f"{t or '{}'}×{c}" for t, c in top)
        print(
            f"{name:10} {len(s):5d} {float(np.median(s)):10.3f} "
            f"{float((p2['digit'] > args.alpha).float().mean()):10.3f}  {top_s}"
        )
        if name != "digit":
            with torch.no_grad():
                Zg, _ = model.embed(Xg)
            embed_z[name] = Zg.numpy()

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {"digit": "steelblue", "gauss": "darkorange", "shuffle": "crimson"}
    for name, color in colors.items():
        axes[0].hist(scores[name], bins=40, alpha=0.55, density=True, label=name, color=color)
        thr = levels[name][args.alpha]
        axes[0].axvline(thr, color=color, ls="--", lw=1.0, alpha=0.9)
    axes[0].set_xlabel(f"{cal.score_name} score")
    axes[0].set_ylabel("density")
    axes[0].set_title("scores + α thresholds (dashed)")
    axes[0].legend(fontsize=8)

    # stacked bar of prediction-set labels
    labels = ["digit", "gauss", "shuffle", "digit|shuffle", "gauss|shuffle", "other/empty"]
    def bucket(t):
        if t == ("digit",):
            return "digit"
        if t == ("gauss",):
            return "gauss"
        if t == ("shuffle",):
            return "shuffle"
        if t == ("digit", "shuffle"):
            return "digit|shuffle"
        if t == ("gauss", "shuffle"):
            return "gauss|shuffle"
        return "other/empty"

    x = np.arange(3)
    width = 0.12
    pools = ["digit", "gauss", "shuffle"]
    for i, lab in enumerate(labels):
        vals = []
        for pool in pools:
            c = Counter(bucket(t) for t in sets_by[pool])
            vals.append(c.get(lab, 0) / max(len(sets_by[pool]), 1))
        axes[1].bar(x + (i - 2.5) * width, vals, width=width, label=lab)
    axes[1].set_xticks(x, pools)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title(f"prediction sets @ α={args.alpha}")
    axes[1].legend(fontsize=7, ncols=2)
    fig.suptitle(f"Mondrian conformal — {args.model.name}", y=1.02)
    fig.tight_layout()
    hist_path = OUT_DIR / f"{tag}_hist.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Embedding overlay: digits + OOD colored by whether digit set accepts them
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(Z_te[:, 0], Z_te[:, 1], c=y_te, cmap="tab10", s=10, alpha=0.55, linewidths=0, zorder=1)
    for name, marker, s in (("gauss", "o", 18), ("shuffle", "x", 28)):
        Zg = embed_z[name]
        accept_digit = np.array([("digit" in t) for t in sets_by[name]])
        ax.scatter(
            Zg[~accept_digit, 0], Zg[~accept_digit, 1],
            c="crimson" if name == "shuffle" else "darkorange",
            s=s, marker=marker, alpha=0.75, linewidths=0.8 if marker == "x" else 0,
            label=f"{name} (not digit)", zorder=2,
        )
        if accept_digit.any():
            ax.scatter(
                Zg[accept_digit, 0], Zg[accept_digit, 1],
                c="0.5", s=s, marker=marker, alpha=0.5,
                label=f"{name} (accepted as digit)", zorder=3,
            )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"{args.model.name}  Mondrian @ α={args.alpha}  score={cal.score_name}\n"
        f"digit thr={levels['digit'][args.alpha]:.3f}  "
        f"gauss={levels['gauss'][args.alpha]:.3f}  "
        f"shuffle={levels['shuffle'][args.alpha]:.3f}"
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    overlay_path = OUT_DIR / f"{tag}_overlay.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    eval_path = OUT_DIR / f"{tag}_eval.npz"
    np.savez(
        eval_path,
        y_te=y_te,
        Z_te=Z_te,
        score_digit=scores["digit"],
        score_gauss=scores["gauss"],
        score_shuffle=scores["shuffle"],
        levels_digit=np.array([levels["digit"][a] for a in alphas]),
        levels_gauss=np.array([levels["gauss"][a] for a in alphas]),
        levels_shuffle=np.array([levels["shuffle"][a] for a in alphas]),
        alphas=np.array(alphas),
        score_name=cal.score_name,
        alpha_eval=args.alpha,
        model=str(args.model),
    )

    print(f"\nsaved {mondrian_path}")
    print(f"saved {hist_path}")
    print(f"saved {overlay_path}")
    print(f"saved {eval_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reusability / OOD demo: a fitted leanmap is a *model*, not just a picture.

Trains one leanmap on a Swiss cone, then reuses it on fresh (in-distribution)
and uniform-ambient (out-of-distribution) points. Compares against a UMAP
mapper fit on the same data. Highlights three properties UMAP `transform`
does not give you:

  1. Fast amortized inference (single forward pass, graph discarded).
  2. A calibrated OOD score (landmark cover + conformal p-values).
  3. Honest rejection of off-manifold points instead of painting them onto
     the chart.

Run:  python examples/reusability.py
Writes plots + a metrics summary to examples/out/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _demo import OUT_DIR
from swiss_cone import make_swiss_cone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-umap", action="store_true")
    args = ap.parse_args()

    import torch
    from _demo import fit_embed
    from leanmap import ConformalCalibrator, load_plane

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data -------------------------------------------------------------
    X_train, t_train = make_swiss_cone(args.n_train, noise=0.05, random_state=args.seed)
    X_cal, _ = make_swiss_cone(2000, noise=0.05, random_state=args.seed + 1)
    X_fresh, t_fresh = make_swiss_cone(args.n_test, noise=0.05, random_state=99)
    lo, hi = X_train.min(0), X_train.max(0)
    pad = 0.05 * (hi - lo)
    rng = np.random.default_rng(42)
    X_uni = rng.uniform(lo - pad, hi + pad, size=(args.n_test, 3)).astype(np.float32)

    # --- fit leanmap once, then treat it as a saved model -----------------
    result, _, _ = fit_embed(
        X_train,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        n_landmarks=100,
        landmark_poisson=True,
        lambda_geo=1.0,
        geo_ramp=(0.25, 0.5),
        learn_landmarks=False,
    )
    lm_path = OUT_DIR / "reusability_leanmap.pt"
    result.save(lm_path)
    model = load_plane(lm_path)
    model.eval()

    # Calibrate the OOD score (landmark cover) on fresh in-distribution data.
    cal = ConformalCalibrator()
    cal.fit(model, torch.as_tensor(X_cal))

    # --- amortized inference + OOD score ----------------------------------
    def embed_scored(X):
        with torch.no_grad():
            Z, cover = model.embed(torch.as_tensor(X), return_score=True)
        Z = Z.cpu().numpy()
        p = cal.p_value(cover, model=model).cpu().numpy()
        return Z, cover.cpu().numpy(), p

    t0 = time.perf_counter()
    Z_fresh, cover_fresh, p_fresh = embed_scored(X_fresh)
    t_lm = time.perf_counter() - t0
    Z_uni, cover_uni, p_uni = embed_scored(X_uni)
    with torch.no_grad():
        Z_train, _ = model.embed(torch.as_tensor(X_train), return_score=False)
    Z_train = Z_train.cpu().numpy()

    alpha = 0.05
    rej_fresh = float((p_fresh <= alpha).mean())
    rej_uni = float((p_uni <= alpha).mean())
    print("=== leanmap ===")
    print(f"inference {args.n_test} pts: {t_lm * 1000:.0f} ms "
          f"({args.n_test / t_lm:.0f} pts/s)")
    print(f"cover  fresh med={np.median(cover_fresh):.3f}  "
          f"uniform med={np.median(cover_uni):.3f}")
    print(f"reject@alpha={alpha}: fresh(FPR)={rej_fresh:.3f}  "
          f"uniform(TPR)={rej_uni:.3f}")

    _save_green_red(
        Z_train, Z_fresh, Z_uni,
        "leanmap — fresh cone (green) vs uniform (red)",
        OUT_DIR / "reuse_leanmap_green_red.png",
    )
    _save_p(
        Z_train, np.vstack([Z_fresh, Z_uni]),
        np.concatenate([p_fresh, p_uni]),
        "leanmap — conformal p (cover OOD)",
        OUT_DIR / "reuse_leanmap_pvalue.png",
    )
    _save_p_filters(
        Z_train,
        np.vstack([Z_fresh, Z_uni]),
        np.concatenate([p_fresh, p_uni]),
        thresholds=(0.01, 0.05, 0.1),
        path=OUT_DIR / "reuse_leanmap_pvalue_filters.png",
    )

    # --- UMAP comparison ---------------------------------------------------
    if not args.skip_umap:
        import joblib
        import umap

        um = umap.UMAP(n_components=2, n_neighbors=30, random_state=args.seed)
        um.fit(X_train)
        joblib.dump(um, OUT_DIR / "reusability_umap30.joblib")

        t0 = time.perf_counter()
        Zf_um = um.transform(X_fresh)
        t_um = time.perf_counter() - t0
        Zu_um = um.transform(X_uni)
        Zt_um = um.transform(X_train)
        print("\n=== UMAP nn=30 ===")
        print(f"transform {args.n_test} pts: {t_um * 1000:.0f} ms "
              f"({args.n_test / t_um:.0f} pts/s)")
        print(f"speedup leanmap/UMAP: {t_um / t_lm:.1f}x")
        print("no calibrated OOD score available from transform()")
        _save_green_red(
            Zt_um, Zf_um, Zu_um,
            "UMAP nn=30 — fresh cone (green) vs uniform (red)",
            OUT_DIR / "reuse_umap_green_red.png",
        )

    print(f"\nsaved plots + {lm_path.name} in {OUT_DIR}")


def _save_green_red(Z_bg, Z_green, Z_red, title, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(Z_bg[:, 0], Z_bg[:, 1], c="0.82", s=1, linewidths=0, zorder=1,
               label="train (grey)")
    ax.scatter(Z_red[:, 0], Z_red[:, 1], c="red", s=2, linewidths=0, alpha=0.45,
               zorder=2, label="uniform / OOD (red)")
    ax.scatter(Z_green[:, 0], Z_green[:, 1], c="limegreen", s=2, linewidths=0,
               alpha=0.55, zorder=3, label="fresh in-dist (green)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def _save_p(Z_bg, Z_pts, p, title, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(Z_bg[:, 0], Z_bg[:, 1], c="0.82", s=1, linewidths=0, zorder=1)
    sc = ax.scatter(Z_pts[:, 0], Z_pts[:, 1], c=p, s=2, cmap="viridis",
                    linewidths=0, vmin=0, vmax=1, zorder=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("conformal p (↑ more in-support)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def _save_p_filters(Z_bg, Z_pts, p, *, thresholds, path):
    """One panel per p-value threshold: keep only points with p > threshold."""
    import matplotlib.pyplot as plt

    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    for ax, thr in zip(axes[0], thresholds):
        keep = p > thr
        ax.scatter(Z_bg[:, 0], Z_bg[:, 1], c="0.85", s=1, linewidths=0, zorder=1)
        ax.scatter(Z_pts[~keep, 0], Z_pts[~keep, 1], c="0.7", s=1.5,
                   linewidths=0, alpha=0.25, zorder=2)
        ax.scatter(Z_pts[keep, 0], Z_pts[keep, 1], c="crimson", s=2,
                   linewidths=0, alpha=0.7, zorder=3)
        ax.set_title(f"p > {thr:g}   (kept {100 * keep.mean():.1f}%)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(
        "leanmap — conformal p filter (red = retained as in-support)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

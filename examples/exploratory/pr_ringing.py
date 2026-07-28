#!/usr/bin/env python
"""Quantify IFT/GNOM ringing in the P(r) profiles and its effect on the embedding.

Indirect Fourier transform recovers P(r) from scattering data under a smoothness
regularizer, and the solution rings: high-frequency oscillations about the true
shape, worst when the data are sparse or Dmax was over-estimated. Those wiggles
are reconstruction artifacts, but the embedding metric cannot tell them from
shape, so they contribute to every distance it sees.

Residuals are taken about the island median from ``pr_centroids``, which removes
the shape and leaves the per-profile perturbation. Then:

* DCT power spectra of residuals against those of the centroid curves, giving
  the frequency where artifact overtakes signal
* where in r the perturbation lives, and how it scales with ``n_source_points``
  and Dmax -- the two things GNOM stability actually depends on
* how much of the pairwise L1 distance is carried by the ringing band, and
  whether low-passing the profiles changes who is a nearest neighbour

Writes a low-passed copy of the feature matrix (``X_smooth.npy``) that can be
re-embedded to test whether any map structure was ringing-driven.

Usage::

    python examples/exploratory/pr_ringing.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_ringing.py --run runs/sasbdb_pr_l1_frozen --keep 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pr_islands import find_islands

_ROOT = Path(__file__).resolve().parents[2]
_PARQUET = Path.home() / "Projects" / "SASDBD" / "data" / "catalog" / "pr_profiles.parquet"


def lowpass(X: np.ndarray, keep: int) -> np.ndarray:
    """Keep the first ``keep`` DCT modes, then restore positivity and unit sum."""
    from scipy.fft import dct, idct

    C = dct(X, type=2, norm="ortho", axis=1)
    C[:, keep:] = 0.0
    Xs = idct(C, type=2, norm="ortho", axis=1)
    np.clip(Xs, 0.0, None, out=Xs)
    return Xs / Xs.sum(axis=1, keepdims=True)


def residuals(X: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Each profile minus the median of the island it belongs to."""
    R = np.empty_like(X)
    for c in np.unique(lab):
        m = lab == c
        R[m] = X[m] - np.median(X[m], axis=0)
    return R


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int, metric: str) -> float:
    from sklearn.neighbors import NearestNeighbors

    ia = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(A).kneighbors(A)[1][:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(B).kneighbors(B)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) for a, b in zip(ia, ib)]) / k)


def source_info(codes) -> pd.DataFrame | None:
    """Per-entry IFT provenance: input points, and negative P(r) bins.

    P(r) is a distance distribution and cannot be negative; GNOM returning
    negative bins is a direct sign the regularised solution is oscillating.
    """
    if not _PARQUET.exists():
        return None
    import pyarrow.parquet as pq

    t = pq.read_table(
        _PARQUET, columns=["sasbdb_code", "n_source_points", "pr"]
    ).to_pandas()
    t["n_neg"] = t["pr"].map(lambda a: int((np.asarray(a, dtype=float) < 0).sum()))
    t["neg_depth"] = t["pr"].map(
        lambda a: float(-min(np.asarray(a, dtype=float).min(), 0.0))
        / max(float(np.abs(np.asarray(a, dtype=float)).max()), 1e-30)
    )
    t = t.drop(columns=["pr"]).set_index("sasbdb_code")
    return t.reindex(pd.Index(codes)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--eps-scale", type=float, default=3.0,
                    help="DBSCAN eps multiple; finer islands give a tighter local median")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--keep", type=int, default=12, help="DCT modes kept by the low-pass")
    ap.add_argument("--k", type=int, default=15, help="neighbours for the overlap test")
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")
    nb = X.shape[1]

    lab, eps = find_islands(Z, args.eps_scale, args.min_samples)
    R = residuals(X, lab)
    Xs = lowpass(X, args.keep)
    hf = X - Xs

    from scipy.fft import dct

    ids = np.array([c for c in np.unique(lab) if c >= 0])
    cent = np.stack([np.median(X[lab == c], axis=0) for c in ids])
    p_res = (dct(R, type=2, norm="ortho", axis=1) ** 2).mean(axis=0)
    p_cen = (dct(cent, type=2, norm="ortho", axis=1) ** 2).mean(axis=0)
    ratio = p_res / p_cen
    cross = int(np.argmax(ratio[1:] > 1.0) + 1) if (ratio[1:] > 1.0).any() else nb

    print(f"{run.name}: N={len(X)}  islands={len(ids)} (eps={eps:.3g})  bins={nb}")
    print(f"  residual power exceeds centroid power from DCT mode {cross} on "
          f"(~{nb / max(cross, 1):.0f} bins per oscillation)")

    # The island residual is dominated by smooth shape differences, so oscillation
    # has to be measured on the high-frequency component, not on the residual.
    for name, S in (("island residual", R), ("high-frequency part", hf)):
        flips_ = (np.diff(np.sign(S), axis=1) != 0).sum(axis=1)
        a, b = S[:, :-1], S[:, 1:]
        ac = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1) + 1e-30)
        print(f"  {name:20s}: {np.median(flips_):5.0f} sign changes of {nb - 1}, "
              f"lag-1 autocorr {np.median(ac):+.3f}")
    flips = (np.diff(np.sign(hf), axis=1) != 0).sum(axis=1)
    a, b = hf[:, :-1], hf[:, 1:]
    ac1 = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1) + 1e-30)

    l1_hf = np.abs(hf).sum(axis=1)
    print(f"\n  low-pass keeping {args.keep} DCT modes removes L1 mass per profile: "
          f"median {np.median(l1_hf):.4f}, 90th pct {np.percentile(l1_hf, 90):.4f}")
    disp = np.median([np.abs(X[lab == c] - np.median(X[lab == c], axis=0)).sum(axis=1).mean()
                      for c in ids])
    print(f"  for scale: median within-island dispersion is {disp:.4f}, so ringing is "
          f"{100 * np.median(l1_hf) / disp:.0f}% of the local spread")

    ov = knn_overlap(X, Xs, args.k, args.metric)
    print(f"  kNN(k={args.k}) overlap between raw and low-passed profiles: {ov:.3f}")

    src = source_info(meta["sasbdb_code"].astype(str).tolist())
    dmax = meta["dmax"].to_numpy(dtype=np.float64)
    n_src = None
    if src is not None:
        from scipy.stats import spearmanr

        n_src = src["n_source_points"].to_numpy(dtype=np.float64)
        n_neg = src["n_neg"].to_numpy(dtype=np.float64)
        ok = np.isfinite(n_src)
        print(f"\n  Spearman(ringing amplitude, n_source_points) = "
              f"{spearmanr(n_src[ok], l1_hf[ok]).statistic:+.3f}   [n={int(ok.sum())}]")
        print(f"  Spearman(ringing amplitude, Dmax)             = "
              f"{spearmanr(dmax, l1_hf).statistic:+.3f}")
        print(f"  Spearman(ringing amplitude, negative bins)    = "
              f"{spearmanr(n_neg, l1_hf).statistic:+.3f}")
        neg = n_neg > 0
        print(f"  profiles with negative P(r) bins (IFT went unstable): "
              f"{int(neg.sum())} of {len(neg)} ({100 * neg.mean():.1f}%)")
        print(f"    their ringing amplitude: median {np.median(l1_hf[neg]):.4f}  "
              f"vs {np.median(l1_hf[~neg]):.4f} for the rest "
              f"({np.median(l1_hf[neg]) / max(np.median(l1_hf[~neg]), 1e-12):.1f}x)")

    np.save(run / "X_smooth.npy", Xs)
    print(f"\nsaved {run / 'X_smooth.npy'}  (low-passed, unit-sum, non-negative)")

    _plot(run, X, Xs, R, lab, ids, meta, p_res, p_cen, cross, l1_hf, n_src, dmax,
          flips, ac1, ov, args)


def _plot(run, X, Xs, R, lab, ids, meta, p_res, p_cen, cross, l1_hf, n_src, dmax,
          flips, ac1, ov, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = X.shape[1]
    r = np.linspace(0.0, 1.0, nb)
    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))

    # Biggest island, so the median is well determined and the wiggles are visible.
    c = ids[np.argmax([(lab == i).sum() for i in ids])]
    m = np.flatnonzero(lab == c)
    med = np.median(X[m], axis=0)
    ax = axs[0, 0]
    for i in m[:120]:
        ax.plot(r, X[i], color="0.7", lw=0.5, alpha=0.7)
    ax.plot(r, med, color="crimson", lw=2.5, label=f"island {int(c)} median")
    ax.plot(r, np.median(Xs[m], axis=0), color="navy", lw=1.5, ls="--",
            label=f"low-pass ({args.keep} modes)")
    ax.set_xlabel("r / Dmax")
    ax.set_ylabel("P(r), unit sum")
    ax.legend(fontsize=8)
    ax.set_title(f"island {int(c)}: members about their median (n={len(m)})")

    ax = axs[0, 1]
    for i in m[:120]:
        ax.plot(r, R[i], color="0.6", lw=0.5, alpha=0.7)
    ax.plot(r, R[m].std(axis=0), color="crimson", lw=2, label="residual sd")
    ax.plot(r, -R[m].std(axis=0), color="crimson", lw=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("r / Dmax")
    ax.set_ylabel("residual")
    ax.legend(fontsize=8)
    ax.set_title("residuals about the island median")

    ax = axs[0, 2]
    k = np.arange(nb)
    ax.loglog(k[1:], p_cen[1:], color="navy", lw=1.8, label="centroid curves")
    ax.loglog(k[1:], p_res[1:], color="crimson", lw=1.8, label="residuals")
    ax.axvline(cross, color="0.4", ls=":", lw=1.5)
    ax.axvline(args.keep, color="green", ls="--", lw=1.5, label=f"low-pass cut ({args.keep})")
    ax.annotate(f"residual > signal\nfrom mode {cross}", (cross, p_res[1] * 0.02),
                fontsize=8, color="0.3", ha="left")
    ax.set_xlabel("DCT mode (half-oscillations across 0..Dmax)")
    ax.set_ylabel("mean power")
    ax.legend(fontsize=8)
    ax.set_title("power spectrum: shape vs perturbation")

    ax = axs[1, 0]
    ax.plot(r, np.abs(R).mean(axis=0), color="crimson", lw=1.8, label="|residual|")
    ax.plot(r, np.abs(X - Xs).mean(axis=0), color="green", lw=1.8,
            label="removed by low-pass")
    ax.plot(r, X.mean(axis=0) * 0.1, color="0.5", lw=1.2, ls="--",
            label="mean P(r) x 0.1")
    ax.set_xlabel("r / Dmax")
    ax.set_ylabel("mean magnitude")
    ax.legend(fontsize=8)
    ax.set_title("where along r the perturbation sits")

    ax = axs[1, 1]
    if n_src is not None and np.isfinite(n_src).any():
        ok = np.isfinite(n_src)
        ax.scatter(n_src[ok], l1_hf[ok], s=5, c=np.log10(dmax[ok]), cmap="viridis",
                   alpha=0.5, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        b = np.geomspace(max(np.nanmin(n_src[ok]), 1), np.nanmax(n_src[ok]), 14)
        w = np.digitize(n_src[ok], b)
        mu = [np.median(l1_hf[ok][w == j]) if (w == j).any() else np.nan
              for j in range(1, len(b))]
        ax.plot(np.sqrt(b[:-1] * b[1:]), mu, color="crimson", lw=2, label="median")
        ax.set_xlabel("n_source_points in the SASBDB entry")
        ax.legend(fontsize=8)
        cb = fig.colorbar(ax.collections[0], ax=ax, fraction=0.046)
        cb.set_label("log10 Dmax")
    else:
        ax.text(0.5, 0.5, "n_source_points unavailable", ha="center", transform=ax.transAxes)
    ax.set_ylabel("ringing amplitude (L1 removed)")
    ax.set_title("ringing vs how much data the IFT had")

    # The tail of the amplitude distribution is where ringing is visible by eye.
    ax = axs[1, 2]
    worst = np.argsort(-l1_hf)[:6]
    step = 1.15 * float(X[worst].max())
    codes = meta["sasbdb_code"].astype(str).to_numpy()
    for j, i in enumerate(worst):
        ax.plot(r, X[i] + j * step, color="0.35", lw=1.1)
        ax.plot(r, Xs[i] + j * step, color="crimson", lw=1.4, alpha=0.85)
        ax.axhline(j * step, color="0.85", lw=0.6)
        ax.annotate(f"{codes[i]}  L1={l1_hf[i]:.3f}, {int(flips[i])} flips",
                    (0.015, j * step + 0.78 * step), fontsize=7, color="0.25")
    ax.set_yticks([])
    ax.set_xlabel("r / Dmax")
    ax.set_title(f"6 worst offenders (grey) vs low-pass (red)\n"
                 f"HF lag-1 autocorr {np.median(ac1):+.2f}, "
                 f"kNN overlap raw vs smoothed {ov:.2f}")

    fig.suptitle(f"{run.name}: IFT/GNOM ringing about the island medians", fontsize=13)
    fig.tight_layout()
    out = args.out or (run / "ringing.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

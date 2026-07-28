#!/usr/bin/env python
"""Audit whether a map's clumping is licensed by the ambient graph.

A layout is entitled to show density contrast only to the extent the data has
it. This measures both sides on the same footing and splits the embedding's
density variation into the part the ambient graph accounts for and the part the
optimiser invented.

Densities are dimension-corrected, ``rho = k / r_k**d``, with ``d`` the
Levina-Bickel intrinsic dimension for the ambient side and exactly 2 for the
embedding. Without that correction the two sides are not comparable, since a
kNN radius means something different in 100 dimensions than in the plane.

Reported:

* contrast, the 95th/5th percentile ratio of local density, on both sides
* the log-log regression of embedded on ambient density: slope 1 means the map
  reproduces the contrast the data has, above 1 means it exaggerates it, and
  R-squared is the share of the map's density structure the data licenses
* Moran's I of the unexplained residual over the embedding's own graph. This is
  the decisive one: unlicensed density that is also spatially organised is
  fabricated clumping, not noise
* Clark-Evans R, for continuity with the other diagnostics

Usage::

    python examples/exploratory/pr_clumpiness.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_clumpiness.py --run runs/sasbdb_pr_umap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def knn_dist(A: np.ndarray, k: int, metric: str) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(A).kneighbors(A)[0][:, 1:]


def intrinsic_dim(d: np.ndarray) -> float:
    """Levina-Bickel MLE from a matrix of sorted kNN distances."""
    k = d.shape[1]
    r = np.maximum(d, 1e-12)
    ratio = np.log(r[:, -1:] / r[:, :-1]).mean(axis=1)
    return float(1.0 / np.maximum(ratio, 1e-12).mean())


def density(d: np.ndarray, dim: float) -> np.ndarray:
    """Dimension-corrected local density from the k-th neighbour radius."""
    return d.shape[1] / np.maximum(d[:, -1], 1e-12) ** dim


def morans_i(v: np.ndarray, nb: np.ndarray) -> float:
    """Spatial autocorrelation of ``v`` over a kNN neighbour-index array."""
    x = v - v.mean()
    num = float((x[:, None] * x[nb]).sum())
    return num / (nb.shape[1] * float((x**2).sum()) + 1e-30)


def clark_evans(Z: np.ndarray) -> float:
    """Observed mean nearest-neighbour distance over the Poisson expectation.

    In ``d`` dimensions that expectation is ``Gamma(1 + 1/d) / (lambda w_d)**(1/d)``
    with ``w_d`` the unit-ball volume, which reduces to the familiar
    ``0.5 * sqrt(area / N)`` at ``d = 2``.
    """
    from math import gamma, pi

    from scipy.spatial import ConvexHull

    d = Z.shape[1]
    lam = len(Z) / float(ConvexHull(Z).volume)
    w_d = pi ** (d / 2) / gamma(d / 2 + 1)
    d1 = knn_dist(Z, 1, "euclidean")[:, 0]
    return float(d1.mean() / (gamma(1 + 1 / d) / (lam * w_d) ** (1 / d)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    X = np.load(run / "X.npy").astype(np.float64)
    Z = np.load(run / "Z.npy").astype(np.float64)

    from sklearn.neighbors import NearestNeighbors

    d_out = Z.shape[1]
    d_amb = knn_dist(X, args.k, args.metric)
    dim = intrinsic_dim(knn_dist(X, 10, args.metric))
    rho_a = density(d_amb, dim)
    rho_z = density(knn_dist(Z, args.k, "euclidean"), float(d_out))

    nb_a = NearestNeighbors(n_neighbors=args.k + 1, metric=args.metric).fit(X)
    nb_a = nb_a.kneighbors(X)[1][:, 1:]
    nb_z = NearestNeighbors(n_neighbors=args.k + 1).fit(Z).kneighbors(Z)[1][:, 1:]

    la, lz = np.log10(rho_a), np.log10(rho_z)
    # Standardise the ambient axis so the slope is in units of "one standard
    # deviation of ambient log-density", which is comparable across runs.
    las = (la - la.mean()) / la.std()
    b = float(np.cov(las, lz, ddof=0)[0, 1] / np.var(las))
    pred = lz.mean() + b * las
    resid = lz - pred
    r2 = float(1.0 - resid.var() / lz.var())

    c_amb = float(np.percentile(rho_a, 95) / np.percentile(rho_a, 5))
    c_emb = float(np.percentile(rho_z, 95) / np.percentile(rho_z, 5))

    print(f"{run.name}: N={len(X)}  k={args.k}  d_out={d_out}")
    print(f"  ambient intrinsic dimension (Levina-Bickel) = {dim:.2f}")
    print(f"  density contrast p95/p5:  ambient {c_amb:.1f}x   embedding {c_emb:.1f}x")
    print(f"  log-density sd:           ambient {la.std():.3f}    embedding {lz.std():.3f}")
    print(f"\n  regression of embedded on ambient log-density:")
    print(f"    slope {b:+.3f} per ambient sd,  R^2 = {r2:.3f}")
    # A map that reproduced the contrast exactly would have slope == la.std(),
    # since then the embedded log-density moves one-for-one with the ambient one.
    print(f"    -> {100 * b / la.std():.0f}% of the licensed contrast reproduced "
          f"(slope {b:.3f} of an ideal {la.std():.3f})")
    print(f"    -> the ambient graph licenses {100 * r2:.0f}% of the map's density "
          f"structure; {100 * (1 - r2):.0f}% is unexplained")
    print(f"\n  Moran's I (spatial organisation, 0 = salt-and-pepper):")
    print(f"    ambient log-density over the ambient graph : {morans_i(la, nb_a):+.3f}")
    print(f"    embedded log-density over the embedding    : {morans_i(lz, nb_z):+.3f}")
    mi_res = morans_i(resid, nb_z)
    print(f"    UNEXPLAINED residual over the embedding    : {mi_res:+.3f}")
    print(f"  Clark-Evans R = {clark_evans(Z):.3f}")

    verdict = ("the unlicensed density is spatially organised: the map is "
               "manufacturing coherent clumps"
               if mi_res > 0.3 and r2 < 0.75 else
               "the unlicensed part is not strongly organised; the clumping "
               "largely tracks real density")
    print(f"\n  -> {verdict}")

    _plot(run, Z, la, lz, resid, rho_a, rho_z, b, r2, mi_res, dim, args)


def _plot(run, Z, la, lz, resid, rho_a, rho_z, b, r2, mi_res, dim, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))
    # Above 2-D the embedding panels show the first two coordinates; everything
    # measured stays in the full embedding dimension.
    Z_full = Z
    d_out = Z.shape[1]
    Z = Z[:, :2] if d_out > 2 else Z
    lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
    mid, half = 0.5 * (lo + hi), 0.55 * float((hi - lo).max())

    def emb(ax, c, title, cmap="viridis", sym=False):
        v = np.percentile(np.abs(c), 98) if sym else None
        s = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=4, linewidths=0,
                       vmin=-v if sym else np.percentile(c, 2),
                       vmax=v if sym else np.percentile(c, 98))
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        fig.colorbar(s, ax=ax, fraction=0.046)

    ax = axs[0, 0]
    ax.scatter(la, lz, s=4, c="0.4", alpha=0.35, linewidths=0)
    xs = np.linspace(la.min(), la.max(), 10)
    ax.plot(xs, lz.mean() + b * (xs - la.mean()) / la.std(), color="crimson", lw=2,
            label=f"fit (R²={r2:.2f})")
    ax.set_xlabel(f"log10 ambient density  (dim {dim:.1f})")
    ax.set_ylabel("log10 embedding density (dim 2)")
    ax.legend(fontsize=8)
    ax.set_title("does ambient density explain embedding density?")

    emb(axs[0, 1], la, "ambient density (what the graph licenses)")
    emb(axs[0, 2], lz, "embedding density (what the map shows)")
    emb(axs[1, 0], resid, f"UNEXPLAINED density\nMoran's I = {mi_res:+.2f}",
        cmap="coolwarm", sym=True)

    ax = axs[1, 1]
    for v, lab, col in ((la - la.mean(), "ambient", "navy"),
                        (lz - lz.mean(), "embedding", "crimson")):
        ax.hist(v, bins=60, histtype="step", lw=1.8, color=col, density=True,
                label=f"{lab} (sd {v.std():.2f})")
    ax.set_xlabel("centred log10 local density")
    ax.set_ylabel("density of points")
    ax.legend(fontsize=8)
    ax.set_title("contrast on both sides, same axis")

    # Poisson reference: in d dimensions the m-th neighbour radius grows as
    # m**(1/d), so a layout that clumps below the neighbourhood scale falls
    # under the line.
    ax = axs[1, 2]
    ks = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50])
    from sklearn.neighbors import NearestNeighbors

    dd = (
        NearestNeighbors(n_neighbors=int(ks.max()) + 1)
        .fit(Z_full)
        .kneighbors(Z_full)[0][:, 1:]
    )
    obs = np.median(dd[:, ks - 1] / dd[:, ks.max() - 1:ks.max()], axis=0)
    ax.loglog(ks, obs, "o-", color="crimson", lw=1.8, label="embedding")
    ax.loglog(ks, (ks / ks.max()) ** (1.0 / d_out), "--", color="k", lw=1.5,
              label=f"uniform in {d_out}-D")
    ax.set_xlabel("neighbour rank m")
    ax.set_ylabel(f"median r_m / r_{ks.max()}")
    ax.legend(fontsize=8)
    ax.set_title("packing curve: below the line = clumped\n"
                 "at scales finer than the neighbourhood")

    fig.suptitle(f"{run.name}: is the clumping licensed by the graph?", fontsize=13)
    fig.tight_layout()
    out = args.out or (run / "clumpiness.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

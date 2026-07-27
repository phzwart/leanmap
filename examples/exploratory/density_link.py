"""Ambient ↔ embed local-density correspondence plots.

Library use via :func:`save_density_link`, or CLI on saved arrays::

    python examples/exploratory/density_link.py \\
      --X examples/exploratory/data/s_curve_X.npy \\
      --Z examples/out/exploratory/s_curve/lambda_geo__0.5/Z.npy \\
      --out /tmp/density_link.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


def save_density_link(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    path: PathLike,
    k: int = 15,
    title: str = "local density (ambient vs embed)",
    npz_path: Optional[PathLike] = None,
) -> dict:
    """Four-panel density correspondence figure + optional ``.npz`` dump.

    Panels: (1) log dens ambient vs embed, (2) Z colored by ambient dens,
    (3) Z colored by embed dens, (4) residual (embed denser / sparser than
    predicted from ambient).
    """
    import matplotlib.pyplot as plt
    from leanmap.evaluate import density_correspondence

    dc = density_correspondence(X, Z, k=k)
    dens_a = np.asarray(dc["dens_ambient"], dtype=np.float64)
    dens_z = np.asarray(dc["dens_embed"], dtype=np.float64)
    resid = np.asarray(dc["log_resid_embed_vs_ambient"], dtype=np.float64)
    la = np.log10(np.maximum(dens_a, 1e-12))
    lz = np.log10(np.maximum(dens_z, 1e-12))
    slope = float(dc["fit_slope"])
    intercept = float(dc["fit_intercept"])
    rho = float(dc["spearman"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    fig.suptitle(f"{title} — k={k}", fontsize=12)

    ax = axes[0, 0]
    hb = ax.hexbin(la, lz, gridsize=45, cmap="viridis", mincnt=1, bins="log")
    xline = np.linspace(la.min(), la.max(), 50)
    ax.plot(xline, intercept + slope * xline, color="cyan", lw=1.5, label=f"slope={slope:.2f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("log10 density ambient")
    ax.set_ylabel("log10 density embed")
    ax.set_title(f"density correspondence  Spearman={rho:.3f}")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=la, s=6, cmap="viridis", linewidths=0)
    ax.set_title("projection · log10 dens ambient")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=lz, s=6, cmap="viridis", linewidths=0)
    ax.set_title("projection · log10 dens embed")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    lim = float(np.percentile(np.abs(resid), 98)) if resid.size else 1.0
    lim = max(lim, 1e-3)
    sc = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=resid,
        s=6,
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        linewidths=0,
    )
    ax.set_title("Δ log10 dens (embed − predicted from ambient)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(">0 denser in Z")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    if npz_path is not None:
        npz_path = Path(npz_path)
        np.savez_compressed(
            npz_path,
            dens_ambient=dens_a.astype(np.float32),
            dens_embed=dens_z.astype(np.float32),
            mean_knn_ambient=np.asarray(dc["mean_knn_ambient"]),
            mean_knn_embed=np.asarray(dc["mean_knn_embed"]),
            log_resid_embed_vs_ambient=resid.astype(np.float32),
            k=np.int32(k),
            spearman=np.float32(rho),
            pearson_log=np.float32(dc["pearson_log"]),
            fit_slope=np.float32(slope),
            fit_intercept=np.float32(intercept),
        )
    return {
        "spearman": rho,
        "pearson_log": float(dc["pearson_log"]),
        "fit_slope": slope,
        "fit_intercept": intercept,
        "k": int(k),
        "path": str(path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True, help="ambient features (N, D) .npy")
    ap.add_argument("--Z", required=True, help="embedding (N, d) .npy")
    ap.add_argument("--out", type=Path, required=True, help="output PNG path")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--title", default="local density (ambient vs embed)")
    args = ap.parse_args(argv)
    X = np.load(args.X).astype(np.float32)
    Z = np.load(args.Z).astype(np.float32)
    if len(X) != len(Z):
        raise SystemExit(f"N mismatch: X={len(X)} Z={len(Z)}")
    stats = save_density_link(
        X,
        Z,
        path=args.out,
        k=args.k,
        title=args.title,
        npz_path=args.out.with_suffix(".npz"),
    )
    print(
        f"wrote {stats['path']}  density Spearman={stats['spearman']:.3f} "
        f"pearson_log={stats['pearson_log']:.3f} slope={stats['fit_slope']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

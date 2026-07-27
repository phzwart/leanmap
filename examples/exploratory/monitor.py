#!/usr/bin/env python
"""Per-epoch layout-uniformity trace.

``spacing_cv`` and ``area_sd`` are already scored once per run by
``metrics_run.uniformity_metrics``, which tells you *that* a layout knotted up
but not *when*. Clumping is a runaway: below the ``b = 1`` boundary of the
attraction curve the pull on a pair grows as it tightens, so the failure shows
as spacing_cv climbing through training rather than as a bad initialisation.
That is only visible per epoch.

Read the trace as follows. ``spacing_cv`` is the alarm; the reference points on
a uniformly sampled s-curve are ~0.18 for the true flattening and a Poisson
sample, 0.30 for PCA-2D and 0.37 for UMAP. The density Spearman is logged for
context but is *not* an alarm: leanmap scores 0.709 there against UMAP's 0.248
while being far clumpier, because a rank correlation is blind to knots and
voids. Its informative counterpart is ``area_sd``, the spread of the local
magnification that the rank correlation throws away.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Uniformity is a property of the layout, not of the sample size, so a subsample
# is enough -- and the per-epoch kNN search has to stay cheap.
MAX_POINTS = 2000
POISSON_FLOOR = 0.18
UMAP_REF = 0.37


def _knn_radius(A: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
    d, _ = nn.kneighbors(A)
    return d[:, -1]


def uniformity_monitor(
    X_ref: np.ndarray,
    out_csv: Path,
    *,
    every: int = 5,
    k: int = 10,
    max_points: int = MAX_POINTS,
    seed: int = 0,
) -> Callable[[int, object, dict], None]:
    """Log spacing_cv, area_sd and density Spearman per epoch.

    Returns a ``cb(epoch, model, metrics)`` callback for ``fit``/``fit_embed``.
    The ambient radii are computed once here; only the embedded ones change
    from epoch to epoch.
    """
    import torch
    from scipy.stats import spearmanr

    X = np.asarray(X_ref, dtype=np.float32)
    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X = X[np.sort(idx)]
    X_t = torch.as_tensor(X)
    r_x = _knn_radius(X.astype(np.float64), k)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # ``geom`` is the attraction/repulsion term the min_dist curve feeds, so it
    # is the one training loss worth reading next to the uniformity numbers.
    fields = ["epoch", "spacing_cv", "area_sd", "density_spearman", "geom", "retention"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def cb(epoch: int, model, metrics: dict) -> None:
        if epoch % every and epoch != 1:
            return
        was_training = model.training
        model.eval()
        with torch.no_grad():
            Z, _ = model.embed(X_t.to(next(model.parameters()).device), return_score=False)
        if was_training:
            model.train()
        Z = Z.detach().cpu().numpy().astype(np.float64)
        r_z = _knn_radius(Z, k)
        # kNN density is a monotone decreasing function of the kNN radius in
        # either space, so the rank correlation of the radii IS the density
        # Spearman -- no density estimate needed.
        rho = float(spearmanr(r_x, r_z).statistic)
        row = {
            "epoch": int(epoch),
            "spacing_cv": float(r_z.std() / max(r_z.mean(), 1e-12)),
            "area_sd": float(
                np.log(np.clip(r_z, 1e-12, None) / np.clip(r_x, 1e-12, None)).std()
            ),
            "density_spearman": rho,
            "geom": float(metrics.get("geom", float("nan"))),
            "retention": float(metrics.get("retention", float("nan"))),
        }
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    return cb


def plot_trace(csv_path: Path, png_path: Path, *, title: str = "") -> Optional[Path]:
    """Plot spacing_cv vs epoch against the Poisson floor and the UMAP mark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path, png_path = Path(csv_path), Path(png_path)
    if not csv_path.is_file():
        return None
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        return None

    ep = [float(r["epoch"]) for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    ax.plot(ep, [float(r["spacing_cv"]) for r in rows], "o-", ms=3, color="C3")
    ax.axhline(POISSON_FLOOR, ls="--", lw=1, color="0.4", label=f"Poisson {POISSON_FLOOR}")
    ax.axhline(UMAP_REF, ls=":", lw=1, color="C0", label=f"UMAP {UMAP_REF}")
    ax.set_ylabel("spacing_cv")
    ax.legend(fontsize=8)
    ax.set_title(title or csv_path.parent.name)

    ax2.plot(ep, [float(r["area_sd"]) for r in rows], "o-", ms=3, color="C2", label="area_sd")
    ax2.plot(
        ep,
        [float(r["density_spearman"]) for r in rows],
        "o-",
        ms=3,
        color="0.5",
        label="density rho",
    )
    ax2.set_xlabel("epoch")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path

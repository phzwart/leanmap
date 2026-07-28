"""Shared helpers for curated leanmap toy demos."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Before importing torch / leanmap (MPS missing-op fallback).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = Path(__file__).resolve().parent / "out"


def default_config(n: int, epochs: int = 60):
    from leanmap import PLANEConfig

    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = int(epochs)
    cfg.batch_edges = min(2048, max(512, n))
    return cfg


def fit_embed(
    X: np.ndarray,
    *,
    epochs: int = 60,
    seed: int = 0,
    device: Optional[str] = None,
    pyramid_level_weights=None,
    pyramid_scales=None,
    pyramid_coarse_backbone=None,
    pyramid_min_reps=None,
    pca_skip: Optional[bool] = None,
    width: Optional[int] = None,
    depth: Optional[int] = None,
    n_landmarks: Optional[int] = None,
    learn_landmarks: Optional[bool] = None,
    lr: Optional[float] = None,
    batch_edges: Optional[int] = None,
    min_dist: Optional[float] = None,
    lr_after: Optional[float] = None,
    lr_switch_epochs: Optional[int] = None,
    n_negatives: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    local_connectivity: Optional[int] = None,
    lambda_lm: Optional[float] = None,
    tau_scale: Optional[float] = None,
    learn_tau: Optional[bool] = None,
    tau_init: Optional[float] = None,
    landmark_geodesic: Optional[bool] = None,
    landmark_poisson: Optional[bool] = None,
    lambda_frame: Optional[float] = None,
    frame_neighbors: Optional[int] = None,
    frame_tangent: Optional[bool] = None,
    frame_ramp=None,
    lambda_geo: Optional[float] = None,
    geo_ramp=None,
    metric="l2",
    callbacks=None,
    init_state_dict=None,
    **config_overrides,
):
    from leanmap import fit

    X = np.asarray(X, dtype=np.float32)
    cfg = default_config(len(X), epochs=epochs)
    cfg.seed = int(seed)
    if device is not None:
        cfg.device = device
    if pyramid_scales is not None:
        cfg.pyramid_scales = int(pyramid_scales)
        if int(pyramid_scales) == 0:
            cfg.pyramid_level_weights = None
            cfg.pyramid_coarse_backbone = 0.0
    if pyramid_level_weights is not None:
        cfg.pyramid_level_weights = tuple(float(w) for w in pyramid_level_weights)
    if pyramid_coarse_backbone is not None:
        cfg.pyramid_coarse_backbone = float(pyramid_coarse_backbone)
    if pyramid_min_reps is not None:
        cfg.pyramid_min_reps = int(pyramid_min_reps)
    if pca_skip is not None:
        cfg.pca_skip = bool(pca_skip)
    if width is not None:
        cfg.width = int(width)
    if depth is not None:
        cfg.depth = int(depth)
    if n_landmarks is not None:
        cfg.n_landmarks = int(n_landmarks)
    if learn_landmarks is not None:
        cfg.learn_landmarks = bool(learn_landmarks)
    if lr is not None:
        cfg.lr = float(lr)
    if batch_edges is not None:
        cfg.batch_edges = int(batch_edges)
    if min_dist is not None:
        cfg.min_dist = float(min_dist)
    if lr_after is not None:
        cfg.lr_after = float(lr_after)
    if lr_switch_epochs is not None:
        cfg.lr_switch_epochs = int(lr_switch_epochs)
    if n_negatives is not None:
        cfg.n_negatives = int(n_negatives)
    if n_neighbors is not None:
        cfg.n_neighbors = int(n_neighbors)
    if local_connectivity is not None:
        cfg.local_connectivity = int(local_connectivity)
    if lambda_lm is not None:
        cfg.lambda_lm = float(lambda_lm)
    if tau_scale is not None:
        cfg.tau_scale = float(tau_scale)
    if learn_tau is not None:
        cfg.learn_tau = bool(learn_tau)
    if tau_init is not None:
        cfg.tau_init = float(tau_init)
    if landmark_geodesic is not None:
        cfg.landmark_geodesic = bool(landmark_geodesic)
    if landmark_poisson is not None:
        cfg.landmark_poisson = bool(landmark_poisson)
    if lambda_frame is not None:
        cfg.lambda_frame = float(lambda_frame)
    if frame_neighbors is not None:
        cfg.frame_neighbors = int(frame_neighbors)
    if frame_tangent is not None:
        cfg.frame_tangent = bool(frame_tangent)
    if frame_ramp is not None:
        cfg.frame_ramp = (float(frame_ramp[0]), float(frame_ramp[1]))
    if lambda_geo is not None:
        cfg.lambda_geo = float(lambda_geo)
    if geo_ramp is not None:
        cfg.geo_ramp = (float(geo_ramp[0]), float(geo_ramp[1]))
    # Any remaining kwarg must name a real config field, so a typo in a sweep
    # overlay fails loudly instead of being silently ignored.
    for key, value in config_overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown PLANEConfig field {key!r} in fit_embed overlay")
        setattr(cfg, key, value)
    result = fit(
        X,
        dist_fn=metric,
        config=cfg,
        callbacks=callbacks,
        init_state_dict=init_state_dict,
    )
    import torch

    with torch.no_grad():
        Z, score = result.embed(X)
    return result, Z.detach().cpu().numpy(), score.detach().cpu().numpy()


def save_scatter(
    Z: np.ndarray,
    color,
    *,
    title: str,
    path: Path,
    cmap: str = "viridis",
    colorbar_label: str = "",
    overlay: Optional[np.ndarray] = None,
    overlay_label: str = "fresh (out-of-sample)",
) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, s=6, cmap=cmap, linewidths=0)
    if overlay is not None:
        ax.scatter(
            overlay[:, 0],
            overlay[:, 1],
            c="red",
            s=12,
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            label=overlay_label,
            zorder=3,
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_density(
    Z: np.ndarray,
    *,
    title: str,
    path: Path,
    gridsize: int = 45,
    cmap: str = "magma",
) -> Path:
    """2-D density (hexbin histogram) of embedded points."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    hb = ax.hexbin(Z[:, 0], Z[:, 1], gridsize=gridsize, cmap=cmap, mincnt=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("count per bin")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_shepard(
    d_orig: np.ndarray,
    d_embed: np.ndarray,
    *,
    title: str,
    path: Path,
    xlabel: str = "original distance",
    ylabel: str = "embedding distance",
    gridsize: int = 60,
    cmap: str = "viridis",
    max_points: int = 20000,
    seed: int = 0,
) -> Path:
    """Shepard diagram: original (or geodesic) distance vs embedding distance.

    Draws a density hexbin, the least-squares isometric line
    ``d_embed ≈ d_orig / alpha``, and annotates Spearman / stress.
    """
    import matplotlib.pyplot as plt
    from leanmap.evaluate import shepard_stats

    d_orig = np.asarray(d_orig, dtype=np.float64).ravel()
    d_embed = np.asarray(d_embed, dtype=np.float64).ravel()
    st = shepard_stats(d_orig, d_embed)
    alpha = st["alpha"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if d_orig.size > max_points:
        rng = np.random.default_rng(seed)
        take = rng.choice(d_orig.size, size=max_points, replace=False)
        xo, yo = d_orig[take], d_embed[take]
    else:
        xo, yo = d_orig, d_embed
    hb = ax.hexbin(xo, yo, gridsize=gridsize, cmap=cmap, mincnt=1, bins="log")
    if np.isfinite(alpha) and alpha > 0 and xo.size:
        xmax = float(np.percentile(xo, 99.5))
        xs = np.linspace(0.0, xmax, 100)
        ax.plot(xs, xs / alpha, color="crimson", lw=1.5, label=f"iso (α={alpha:.3g})")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log count")
    rho, stress, n = st["spearman"], st["stress"], st["n_pairs"]
    ax.text(
        0.98,
        0.02,
        f"Spearman={rho:.3f}\nstress={stress:.3f}\npairs={n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def split_train_holdout(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_hold = max(1, int(round(holdout_frac * n)))
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]
    X_train, X_hold = X[train_idx], X[hold_idx]
    if y is None:
        return X_train, X_hold, None, None
    return X_train, X_hold, y[train_idx], y[hold_idx]

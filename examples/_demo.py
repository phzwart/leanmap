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


def _pair_auc(lo: np.ndarray, hi: np.ndarray) -> float:
    """``P(hi > lo)`` with ties at one half, via the rank-sum identity.

    Mirrors ``leanmap.classaxis._auc`` so the matrix drawn here agrees cell-for-cell
    with the ``order_*`` means the library reports; a tie convention that disagreed
    would make the picture and the number quietly tell different stories.
    """
    n_lo, n_hi = lo.shape[0], hi.shape[0]
    if n_lo == 0 or n_hi == 0:
        return float("nan")
    both = np.concatenate([lo, hi])
    order = np.argsort(both, kind="stable")
    ranks = np.empty(both.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, both.shape[0] + 1, dtype=np.float64)
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(counts.shape[0], dtype=np.float64)
    np.add.at(tie_mean, inv, ranks)
    tie_mean /= counts
    ranks = tie_mean[inv]
    r_hi = ranks[n_lo:].sum()
    return float((r_hi - n_hi * (n_hi + 1) / 2.0) / (n_lo * n_hi))


def axis_projection(Z: np.ndarray, report: dict, ax_spec) -> Tuple[np.ndarray, np.ndarray]:
    """``(direction, positions)`` for one :class:`~leanmap.ClassAxis`.

    A pinned axis *is* a basis vector. A free-direction axis has its direction
    reported as ``dir_<name>_<j>`` by :func:`leanmap.class_axis_report`, which is
    read back here rather than recomputed, so the picture is guaranteed to show the
    same direction the score was taken along.
    """
    d = Z.shape[1]
    if ax_spec.is_pinned:
        u = np.zeros(d, dtype=np.float64)
        u[ax_spec.axis] = 1.0
    else:
        u = np.asarray(
            [report[f"dir_{ax_spec.name}_{j}"] for j in range(d)], dtype=np.float64
        )
    return u, np.dot(Z, u)


def save_class_axes(
    Z,
    labels,
    axes,
    *,
    path: Path,
    report: Optional[dict] = None,
    title: str = "class axes",
    class_names: Optional[dict] = None,
) -> Path:
    """Draw the constrained axes three ways: as a frame, a ruler, and a pair matrix.

    ``class_axis_report`` compresses each axis to two numbers, an overall and an
    adjacent-pair ordering accuracy, and a direction to four more. That is enough to
    know whether an ordering took but not *where* it failed, and a direction read as
    a list of components is a geometric object in the least legible available form.
    Each panel here recovers something the scalars drop:

    **The frame** (top) puts every axis in the embedding at once: an arrow per axis
    from the centroid, scaled by the spread of the data along it, so a short arrow
    means an ordering that was satisfied without using much room. Its purpose is to
    show the *angles between* orderings -- the failure mode where a free direction
    tilts until it lies inside a pinned coordinate and has stopped being a separate
    direction at all.

    **The ruler** (middle, one per axis) is the marginal along the axis, one violin
    per class, stacked in the order you requested. This is the panel to read first:
    the requested ordering is correct exactly when the violins ascend, and the pair
    that spoils it is visible instead of averaged away. Classes sharing a rank are
    drawn as one block, since the request does not distinguish them. The number at
    the right of each row is the ordering accuracy against the next rank up, so a
    weak step is legible as a number too.

    **The pair matrix** (bottom, one per axis) is every ordered class pair's
    accuracy, rows and columns in requested rank order. Blue is satisfied, red
    inverted, white indifferent; grey cells were never constrained, either because
    the two classes share a rank or because the pair is below the diagonal. A
    well-ordered axis is a uniformly blue upper triangle, and a block structure in
    the grey is the coarse orderings showing themselves.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Arc

    try:  # accept torch tensors without importing torch at module scope
        Z = Z.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        labels = labels.detach().cpu().numpy()
    except AttributeError:
        pass
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    axes = list(axes)

    if report is None:
        import torch

        from leanmap import class_axis_report

        report = class_axis_report(
            torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(labels), axes
        )

    present = np.unique(labels)
    name_of = class_names or {}
    palette = plt.get_cmap("tab10" if len(present) <= 10 else "tab20")
    colour = {int(c): palette(i % palette.N) for i, c in enumerate(present)}

    n_ax = len(axes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.2 * n_ax, 12.0))
    gs = fig.add_gridspec(
        3, n_ax, height_ratios=[1.15, 1.0, 1.0], hspace=0.42, wspace=0.28
    )

    # ---- panel 1: the frame -------------------------------------------------
    dirs = [axis_projection(Z, report, a)[0] for a in axes]
    if Z.shape[1] == 2:
        B = np.eye(2)
        frame_lbl = ("$z_0$", "$z_1$")
    else:
        # Two axes span the plane worth looking at; orthonormalise so the panel is
        # a rotation of the embedding and the drawn angles are the real ones.
        b0 = dirs[0] / max(float(np.linalg.norm(dirs[0])), 1e-12)
        rest = dirs[1] if n_ax > 1 else np.eye(Z.shape[1])[(np.argmax(np.abs(b0)) + 1) % Z.shape[1]]
        b1 = rest - np.dot(rest, b0) * b0
        nb = float(np.linalg.norm(b1))
        b1 = b1 / nb if nb > 1e-8 else np.roll(b0, 1)
        B = np.stack([b0, b1])
        frame_lbl = (f"{axes[0].name} direction", "orthogonal complement")
    A = np.dot(Z, B.T)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.scatter(
        A[:, 0], A[:, 1], c=[colour[int(c)] for c in labels], s=5, linewidths=0, alpha=0.45
    )
    centre = A.mean(axis=0)
    span = A.max(axis=0) - A.min(axis=0)
    reach = 0.38 * float(np.min(span))
    spreads = [float(np.std(np.dot(Z, w))) for w in dirs]
    widest = max(spreads + [1e-12])
    tips = []
    for a, u, spread in zip(axes, dirs, spreads):
        v = np.dot(B, u)  # the axis direction expressed in the drawn plane
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:  # axis is orthogonal to the drawn plane; nothing to point at
            tips.append(None)
            continue
        # Arrow length carries the spread of the data along the axis, relative to
        # the widest axis, so the panel says how much room each ordering used.
        tip = centre + reach * (v / nv) * spread / widest
        tips.append(tip)
        ax0.annotate(
            "",
            xy=tip,
            xytext=centre,
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color="black", shrinkA=0, shrinkB=0),
        )
        tag = a.name if a.is_pinned else f"{a.name} (tilt {report[f'tilt_{a.name}']:.1f}°)"
        ax0.text(
            *(tip + 0.05 * reach * (v / nv)),
            tag,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
        )
    # Limits from the data *and* the arrows, set before the aspect is locked:
    # 'equal' with the default datalim adjustment would otherwise inflate the wider
    # limit to satisfy the ratio and silently crop the arrows off the short side.
    corners = np.vstack([A] + [t.reshape(1, 2) for t in tips if t is not None])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pad = 0.12 * np.maximum(hi - lo, 1e-9)
    ax0.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax0.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax0.set_aspect("equal", adjustable="box")
    if n_ax >= 2:
        p0, p1 = np.dot(B, dirs[0]), np.dot(B, dirs[1])
        if np.linalg.norm(p0) > 1e-12 and np.linalg.norm(p1) > 1e-12:
            a0 = np.degrees(np.arctan2(p0[1], p0[0]))
            a1 = np.degrees(np.arctan2(p1[1], p1[0]))
            sweep = (a1 - a0 + 180.0) % 360.0 - 180.0
            r = 0.32 * reach
            ax0.add_patch(
                Arc(
                    centre,
                    2 * r,
                    2 * r,
                    theta1=min(a0, a0 + sweep),
                    theta2=max(a0, a0 + sweep),
                    color="crimson",
                    lw=1.8,
                )
            )
            mid = np.radians(a0 + sweep / 2.0)
            ax0.text(
                centre[0] + 1.18 * r * np.cos(mid),
                centre[1] + 1.18 * r * np.sin(mid),
                f"{abs(sweep):.1f}°",
                color="crimson",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )
    ax0.set_xlabel(frame_lbl[0])
    ax0.set_ylabel(frame_lbl[1])
    ax0.set_title(
        f"{title}: where the axes point\n"
        "arrow = axis direction, length = spread of the data along it"
    )

    # ---- panels 2 and 3: ruler and pair matrix, per axis --------------------
    for col, a in enumerate(axes):
        _, pos = axis_projection(Z, report, a)
        rnk = a.rank.detach().cpu().numpy().astype(np.float64)
        # Rank groups in requested order; classes tied at a rank form one block,
        # because the request does not distinguish them.
        by_rank: dict = {}
        for c in present:
            by_rank.setdefault(float(rnk[int(c)]), []).append(int(c))
        ordered_ranks = sorted(by_rank)
        rows = [(r, c) for r in ordered_ranks for c in by_rank[r]]

        axr = fig.add_subplot(gs[1, col])
        for i, (r, c) in enumerate(rows):
            v = pos[labels == c]
            if v.size < 2:
                continue
            parts = axr.violinplot(
                [v], positions=[i], vert=False, widths=0.85, showextrema=False
            )
            for body in parts["bodies"]:
                body.set_facecolor(colour[c])
                body.set_alpha(0.75)
                body.set_linewidth(0)
            axr.plot([v.mean()], [i], marker="|", color="black", ms=12, mew=1.6)
        # Separate rank blocks, and score each step against the next rank up.
        nxt = {r: ordered_ranks[k + 1] for k, r in enumerate(ordered_ranks[:-1])}
        right = float(np.max(pos))
        for k, r in enumerate(ordered_ranks[:-1]):
            edge = max(i for i, (rr, _) in enumerate(rows) if rr == r) + 0.5
            axr.axhline(edge, color="0.75", lw=0.8, ls=":")
            lo = np.concatenate([pos[labels == c] for c in by_rank[r]])
            hi = np.concatenate([pos[labels == c] for c in by_rank[nxt[r]]])
            step = _pair_auc(lo, hi)
            axr.text(
                right,
                edge,
                f" {step:.2f}",
                fontsize=8,
                va="center",
                ha="left",
                color="darkgreen" if step >= 0.9 else ("darkorange" if step >= 0.7 else "crimson"),
            )
        axr.set_yticks(range(len(rows)))
        axr.set_yticklabels([name_of.get(c, str(c)) for _, c in rows], fontsize=8)
        axr.set_xlabel(f"position along {a.name}")
        axr.set_ylabel("class, in requested order →")
        pin = f"pinned to $z_{a.axis}$" if a.is_pinned else "free direction"
        axr.set_title(
            f"{a.name}: the ruler ({pin})\n"
            f"order={report[f'order_{a.name}']:.3f} "
            f"adjacent={report[f'order_adjacent_{a.name}']:.3f}",
            fontsize=10,
        )
        axr.margins(y=0.02)

        axm = fig.add_subplot(gs[2, col])
        cls = [c for _, c in rows]
        K = len(cls)
        M = np.full((K, K), np.nan)
        for i, ci in enumerate(cls):
            for j, cj in enumerate(cls):
                if rnk[ci] >= rnk[cj]:
                    continue  # unconstrained: same rank, or the wrong way round
                M[i, j] = _pair_auc(pos[labels == ci], pos[labels == cj])
        cmap = colors.LinearSegmentedColormap.from_list(
            "auc", ["crimson", "white", "steelblue"]
        )
        cmap.set_bad("0.88")
        im = axm.imshow(
            np.ma.masked_invalid(M), cmap=cmap, vmin=0.0, vmax=1.0, origin="upper"
        )
        axm.set_xticks(range(K))
        axm.set_xticklabels([name_of.get(c, str(c)) for c in cls], fontsize=7, rotation=90)
        axm.set_yticks(range(K))
        axm.set_yticklabels([name_of.get(c, str(c)) for c in cls], fontsize=7)
        axm.set_xlabel("should be higher →")
        axm.set_ylabel("← should be lower")
        worst = np.nanmin(M) if np.isfinite(M).any() else float("nan")
        axm.set_title(
            f"{a.name}: every ordered pair\ngrey = unconstrained, worst pair = {worst:.2f}",
            fontsize=10,
        )
        fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04).set_label(
            "P(correct order)", fontsize=8
        )

    fig.savefig(path, dpi=150, bbox_inches="tight")
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

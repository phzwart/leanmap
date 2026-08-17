#!/usr/bin/env python
"""Both orderings, the class densities, and where held-out points land.

The layout is the one from ``digits_two_orderings.py``: ``z0`` is pinned to the
digit value, and parity rides on a direction the fit chose for itself -- including
its angle, which is not required to be square to ``z0``. The figure is drawn in the
orthonormalised frame ``(z0, u_perp)``, so it is a rotation of the embedding rather
than a projection and nothing is hidden by looking at it this way. Both requested
orderings are then the two axes: digit value left to right, parity bottom to top.

The one thing to keep in mind is that the vertical is ``u_perp``, not ``u``: the
part of the parity direction that digit value does not already explain. On digits
these differ by only a couple of degrees (printed as ``tilt``), so the distinction
is cosmetic here, but it would not be for two orderings that genuinely share a
direction.

Two things are drawn on top of each other because they answer different questions.

**Density contours** (thin lines, one per class) enclose 50% and 90% of each
class's mass, estimated from the training points. This is where the class *is*.

**Conformal acceptance regions** (solid lines) are the level sets where a class's
calibrated p-value crosses ``alpha`` -- where a class would be *admitted*. This is
where the class is *allowed to be*, and it is necessarily larger than the density,
since it has to cover ``1 - alpha`` of future points including the awkward ones.

They come out very much larger here, and the reason is worth knowing before
reading anything into their size: digits gives only 40 calibration points per
class, so the ``alpha=0.05`` threshold is the 95th percentile of 40 values --
pinned by its top two, and those are whichever two points the encoder placed
furthest from their own class. The regions are honest but coarse, which is why
they overlap enough that most held-out points come back as a set of two. More
calibration points per class tightens them; a larger ``alpha`` does too, at the
cost of the guarantee.

Where the fill fades to white, no class survives ``alpha`` and the honest answer
is "none of these" rather than a nearest label. A softmax has no such area.

Three disjoint splits, because "held out" has to mean it: ``train`` fits the
encoder and supplies the class clouds, ``calib`` supplies the per-class conformal
distributions, and ``test`` is touched only at the end.

Run::

    python examples/digits_two_orderings_regions.py --epochs 30
    python examples/digits_two_orderings_regions.py --replot   # figure only
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

from leanmap import (
    ClassAxisReadout,
    ClassRegionConformal,
    PLANEConfig,
    class_axis_report,
    fit,
    grouped_class_axis,
    ordinal_class_axis,
)
from leanmap.evaluate import density_correspondence

from _demo import save_class_axes

OUT_DIR = Path(__file__).resolve().parent / "out" / "digits_two_orderings"
CACHE = OUT_DIR / "regions_cache.npz"
N_CLASSES = 10
EVEN = [0, 2, 4, 6, 8]
ODD = [1, 3, 5, 7, 9]
PER_CLASS_CALIB = 40
PER_CLASS_TEST = 30
GRID = 300

# From examples/class_axis_sweep.py: a strong hinge that engages early costs less
# neighbourhood quality than a weak one arguing for the whole run.
LAMBDA = 16.0
MARGIN = 0.30
RAMP = (0.0, 0.1)
# Secondary factors want a fraction of the primary's force; past ~0.5 a
# free-direction term drags the layout while its direction is still forming.
PARITY_WEIGHT = 0.3

ALPHA = 0.05
# p at which a region is drawn fully saturated, so the colour gradient spends its
# range where it is informative rather than on the long tail above alpha.
FULL_P = 0.5
# Fractions of each class's mass enclosed by the density contours.
MASS_LEVELS = (0.5, 0.9)


def three_way_split(y, seed=0):
    """Stratified train / calib / test — every class must appear in all three."""
    rng = np.random.default_rng(seed)
    tr, cal, te = [], [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cal.append(idx[:PER_CLASS_CALIB])
        te.append(idx[PER_CLASS_CALIB : PER_CLASS_CALIB + PER_CLASS_TEST])
        tr.append(idx[PER_CLASS_CALIB + PER_CLASS_TEST :])
    return tuple(np.concatenate(a) for a in (tr, cal, te))


def frame(u: np.ndarray, pinned: int = 0) -> np.ndarray:
    """Orthonormal 2-D frame ``(e_pinned, u_perp)``, as rows.

    The frame has to be orthonormal for the figure to be a rotation of the
    embedding, and it has to be orthonormal for :func:`from_frame` to invert
    :func:`to_frame` at all -- the conformal grid is built by mapping plot
    coordinates *back* to embedding space, so an oblique basis would evaluate the
    p-values at the wrong points and quietly draw the wrong regions.

    ``u`` does not have to cooperate. The default free direction discovers its own
    angle and may lean into the pinned coordinate, so ``u`` is orthonormalised
    against ``e_pinned`` here rather than assumed square to it. The vertical axis is
    then the part of the parity direction that the digit axis does not already
    account for, which is the honest thing to plot: the component along ``z0`` is
    digit value, and drawing it twice would double-count it. ``tilt`` says how much
    was removed.
    """
    e = np.zeros_like(u)
    e[pinned] = 1.0
    v = u - np.dot(u, e) * e
    n = float(np.linalg.norm(v))
    if n < 1e-8:  # parity direction absorbed into the pinned axis; nothing left
        v = np.zeros_like(u)
        v[(pinned + 1) % len(u)] = 1.0
    else:
        v = v / n
    return np.stack([e, v])


def to_frame(Z: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(Z, B.T)


def from_frame(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A, B)


def p_field(cal: ClassRegionConformal, xs, ys, B):
    """Conformal p-value per class over the grid. Returns (K, ny, nx), classes."""
    gx, gy = np.meshgrid(xs, ys)
    A = np.stack([gx.ravel(), gy.ravel()], axis=1)
    Z = torch.as_tensor(from_frame(A, B), dtype=torch.float32)
    pv = cal.p_values(Z)
    classes = sorted(pv)
    P = np.stack([pv[c].numpy().reshape(gx.shape) for c in classes])
    return P, classes


def density_field(A_tr, y_tr, classes, xs, ys):
    """Per-class KDE on the grid, plus the level enclosing each mass fraction.

    The levels are found by sorting the class's density at its own *training
    points* and taking a quantile, which is the usual trick for turning a density
    into a "contains X% of the mass" contour without integrating the grid.
    """
    from scipy.stats import gaussian_kde

    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()])
    D, levels = [], []
    for c in classes:
        pts = A_tr[y_tr == c].T
        if pts.shape[1] < 5:
            D.append(np.zeros_like(gx))
            levels.append([np.nan] * len(MASS_LEVELS))
            continue
        kde = gaussian_kde(pts)
        D.append(kde(grid).reshape(gx.shape))
        at_pts = kde(pts)
        levels.append([np.quantile(at_pts, 1.0 - m) for m in MASS_LEVELS])
    return np.stack(D), np.asarray(levels)


def outcome(true_c: int, accepted: tuple) -> str:
    if not accepted:
        return "novel"
    if true_c not in accepted:
        return "wrong"
    return "exact" if len(accepted) == 1 else "ambiguous"


MARKERS = {
    "exact": ("o", "accepted, unambiguous"),
    "ambiguous": ("s", "accepted with alternatives"),
    "wrong": ("^", "true class rejected"),
    "novel": ("X", "no class accepted (novel)"),
}


def _soft_field(F, classes, colours, gamma=1.0, normalise=False):
    """Hue from the leading class, saturation from its value, white elsewhere.

    ``normalise`` divides each class's field by its own maximum, which densities
    need and p-values must not have. A density's units are arbitrary and its peak
    height is set by how *tight* the class is, so without normalising, argmax is
    won by the most compact class rather than the most plausible one -- digit 0
    forms a very small cluster and its raw peak would paint the map. Conformal
    p-values are already on a common scale, and rescaling them per class would
    turn "how admissible" into "admissible relative to this class's best case".
    """
    G = F / np.maximum(F.max(axis=(1, 2), keepdims=True), 1e-30) if normalise else F
    winner = G.argmax(axis=0)
    conf = np.clip(G.max(axis=0), 0.0, 1.0)[..., None] ** gamma
    return np.clip(colours[np.asarray(classes)[winner]] * conf + (1.0 - conf), 0.0, 1.0)


def make_figure(P, D, levels, classes, A_tr, y_tr, A_te, y_te, sets, xs, ys, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab10")
    colours = np.stack([cmap(i)[:3] for i in range(N_CLASSES)])
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    # gamma < 1 lifts the low end so the skirts of each class stay visible; the
    # conformal field is already a probability and needs no such help, only the
    # clip at FULL_P so its range is spent below the uninformative tail.
    dens_rgb = _soft_field(D, classes, colours, gamma=0.45, normalise=True)
    conf_rgb = _soft_field(np.clip(P / FULL_P, 0.0, 1.0), classes, colours)

    fig, axes = plt.subplots(1, 3, figsize=(21.5, 6.9))

    def mass_contours(ax, alpha=0.9, lw=0.9):
        for k, c in enumerate(classes):
            lv = sorted(v for v in levels[k] if np.isfinite(v) and v > 0)
            if lv:
                ax.contour(
                    xs, ys, D[k], levels=lv, colors=[colours[c]],
                    linewidths=lw, linestyles=":", alpha=alpha, zorder=2,
                )

    def accept_contours(ax, alpha=0.85, lw=1.1):
        for k, c in enumerate(classes):
            ax.contour(
                xs, ys, P[k], levels=[ALPHA], colors=[colours[c]],
                linewidths=lw, alpha=alpha, zorder=1,
            )

    axes[0].imshow(dens_rgb, extent=extent, origin="lower", aspect="auto", zorder=0)
    mass_contours(axes[0])
    axes[0].scatter(
        A_tr[:, 0], A_tr[:, 1], c=y_tr, cmap="tab10", vmin=-0.5, vmax=9.5,
        s=5, alpha=0.5, linewidths=0, zorder=3,
    )
    axes[0].set_title(
        "where each class is: training density\n"
        f"dotted = {int(MASS_LEVELS[0] * 100)}% and "
        f"{int(MASS_LEVELS[1] * 100)}% of the class's mass"
    )

    axes[1].imshow(conf_rgb, extent=extent, origin="lower", aspect="auto", zorder=0)
    accept_contours(axes[1])
    axes[1].set_title(
        f"where each class is admissible: conformal p > {ALPHA}\n"
        "wide because n_calib=40 per class puts the 95th percentile on 2 points"
    )

    axes[2].imshow(dens_rgb, extent=extent, origin="lower", aspect="auto", zorder=0)
    accept_contours(axes[2], alpha=0.5, lw=0.9)
    for name, (marker, _) in MARKERS.items():
        sel = np.asarray([outcome(int(t), s) == name for t, s in zip(y_te, sets)])
        if not sel.any():
            continue
        axes[2].scatter(
            A_te[sel, 0], A_te[sel, 1],
            c=y_te[sel], cmap="tab10", vmin=-0.5, vmax=9.5,
            marker=marker, s=58, alpha=0.95,
            edgecolors="black", linewidths=0.7, zorder=4,
        )
    axes[2].set_title(
        f"where held-out points land: {len(y_te)} never seen in fitting "
        f"or calibration\nfill = true digit, marker = conformal outcome at "
        f"alpha={ALPHA}"
    )
    axes[2].legend(
        handles=[
            Line2D([], [], marker=m, color="none", markerfacecolor="0.7",
                   markeredgecolor="black", markersize=9, label=lab)
            for m, lab in MARKERS.values()
        ],
        loc="lower left", fontsize=8.5, framealpha=0.92,
    )

    for ax in axes:
        ax.set_xlabel("$z_0$  —  pinned: digit value 0 → 9")
        ax.set_ylabel("chosen direction  —  even → odd")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.BoundaryNorm(np.arange(11) - 0.5, 10)
    )
    cb = fig.colorbar(
        sm, ax=axes.tolist(), ticks=np.arange(10), fraction=0.016, pad=0.012
    )
    cb.set_label("digit")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _object_array(seq):
    """Ragged tuples, kept ragged — numpy would otherwise square them off."""
    a = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        a[i] = tuple(v)
    return a


def replot() -> None:
    d = np.load(CACHE, allow_pickle=True)
    out = make_figure(
        d["P"], d["D"], d["levels"], list(d["classes"]), d["A_tr"], d["y_tr"],
        d["A_te"], d["y_te"], list(d["sets"]), d["xs"], d["ys"],
        OUT_DIR / "two_orderings_regions.png",
    )
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        replot()
        return

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    tr_i, cal_i, te_i = three_way_split(y, seed=args.seed)
    print(f"train={len(tr_i)} calib={len(cal_i)} test={len(te_i)} (disjoint)")

    digit = ordinal_class_axis(N_CLASSES, axis=0, name="digit")
    parity = grouped_class_axis(
        [EVEN, ODD], axis=None, name="parity", weight=PARITY_WEIGHT
    )
    cfg = replace(
        PLANEConfig.for_scale(len(tr_i)),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        dedup=False,
        lambda_class=LAMBDA,
        class_margin=MARGIN,
        class_ramp=RAMP,
    )
    res = fit(
        X[tr_i],
        dist_fn="l2",
        config=cfg,
        X_calib=X[cal_i],
        class_labels=y[tr_i],
        class_axes=[digit, parity],
    )

    X_tr = torch.as_tensor(X[tr_i])
    y_tr = y[tr_i]
    Z_tr, _ = res.model.embed(X_tr)
    Z_tr = Z_tr.detach()
    rep = class_axis_report(Z_tr, torch.as_tensor(y_tr), [digit, parity])
    u = np.asarray([rep[f"dir_parity_{j}"] for j in range(cfg.d_out)])
    B = frame(u, pinned=digit.axis)
    print(
        f"digit  order={rep['order_digit']:.3f} adjacent="
        f"{rep['order_adjacent_digit']:.3f}\n"
        f"parity order={rep['order_parity']:.3f} along {np.round(u, 3).tolist()}"
        f" tilt={rep['tilt_parity']:.1f} deg off square to z{digit.axis}"
    )
    print(
        "saved "
        + str(
            save_class_axes(
                Z_tr,
                torch.as_tensor(y_tr),
                [digit, parity],
                report=rep,
                path=OUT_DIR / "class_axes.png",
                title="digits: two requested orderings",
            )
        )
    )
    # Whether the densities being drawn mean anything: if embedded density does
    # not track ambient density, the contours describe the picture rather than
    # the data.
    dc = density_correspondence(np.asarray(X[tr_i]), Z_tr.numpy(), k=15)
    scalars = {k: float(v) for k, v in dc.items() if np.ndim(v) == 0}
    print(
        "ambient vs embedded local density (k=15): "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(scalars.items()))
    )

    readout = ClassAxisReadout.from_model(res.model, X_tr, torch.as_tensor(y_tr), digit)
    cal = ClassRegionConformal(readout).fit(
        res.model, torch.as_tensor(X[cal_i]), torch.as_tensor(y[cal_i])
    )

    # Held-out inference: one forward pass, no labels, no graph.
    Z_te, _ = res.model.embed(torch.as_tensor(X[te_i]))
    Z_te = Z_te.detach()
    y_te = y[te_i]
    sets = cal.prediction_set(Z_te, alpha=ALPHA)
    kinds = np.asarray([outcome(int(t), s) for t, s in zip(y_te, sets)])
    sizes = np.asarray([len(s) for s in sets])
    print(f"\nheld-out outcomes at alpha={ALPHA} (n={len(y_te)}):")
    for name, (_, label) in MARKERS.items():
        n = int((kinds == name).sum())
        print(f"  {label:32s} {n:4d}  {n / len(y_te):6.1%}")
    print(
        f"\ncoverage of the true digit "
        f"{float((kinds != 'wrong').mean() - (kinds == 'novel').mean()):.3f} "
        f"(target >= {1 - ALPHA:.2f}), mean set size {sizes.mean():.2f}"
    )
    # Parity is the coarser question, so it should be answered more often than
    # the digit: a set can be ambiguous about which digit and unanimous about
    # whether it is odd.
    par_of = {c: (0 if c in EVEN else 1) for c in range(N_CLASSES)}
    agree = np.asarray(
        [len({par_of[c] for c in s}) == 1 if s else False for s in sets]
    )
    par_ok = np.asarray(
        [
            bool(s) and len({par_of[c] for c in s}) == 1
            and par_of[int(t)] == par_of[next(iter(s))]
            for t, s in zip(y_te, sets)
        ]
    )
    print(
        f"parity decided by the prediction set: {agree.mean():.1%} of points, "
        f"correct on {par_ok.sum()}/{int(agree.sum())} of those"
    )

    A_tr, A_te = to_frame(Z_tr.numpy(), B), to_frame(Z_te.numpy(), B)
    pad = 0.08
    lo, hi = A_tr.min(axis=0), A_tr.max(axis=0)
    span = (hi - lo) * pad
    xs = np.linspace(lo[0] - span[0], hi[0] + span[0], GRID)
    ys = np.linspace(lo[1] - span[1], hi[1] + span[1], GRID)
    P, classes = p_field(cal, xs, ys, B)
    D, levels = density_field(A_tr, y_tr, classes, xs, ys)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        P=P, D=D, levels=levels, classes=np.asarray(classes), xs=xs, ys=ys,
        A_tr=A_tr, y_tr=y_tr, A_te=A_te, y_te=y_te, sets=_object_array(sets), u=u,
    )
    out = make_figure(
        P, D, levels, classes, A_tr, y_tr, A_te, y_te, sets, xs, ys,
        OUT_DIR / "two_orderings_regions.png",
    )
    print(f"\nsaved {out}\nsaved {CACHE} (re-render with --replot)")


if __name__ == "__main__":
    main()

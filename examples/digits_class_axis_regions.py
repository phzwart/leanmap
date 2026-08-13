#!/usr/bin/env python
"""Per-class conformal regions in the map, and where held-out points land.

The contours here are not a kernel density estimate of each class. They are the
*calibrated acceptance regions* of :class:`~leanmap.classaxis.ClassRegionConformal`
-- the level sets where that class's conformal p-value crosses alpha -- so the
picture is the decision geometry itself rather than a decoration drawn next to it.
Two consequences worth knowing before reading the figure:

* The regions fade to white between classes and at the edges. That is not a
  colour-map choice; it is where no class's p-value survives alpha, i.e. where
  the answer is "none of these" rather than a nearest label. A softmax has no
  such area.
* Regions overlap, and the overlaps are real. A point in one is genuinely
  admissible as either class at that alpha, which the prediction set reports as a
  set of size two rather than hiding behind a margin.

Three disjoint splits, because "held out" has to mean it: ``train`` fits the
encoder and supplies the class point clouds the regions are measured from,
``calib`` supplies the per-class conformal distributions, and ``test`` is touched
only at the end. The readout is built on train and calibrated on calib for the
same reason :class:`~leanmap.conformal.LandmarkSupport` is -- per-class region
distances are not rank-preserving, so fitting them on the calibration set would
void the guarantee.

Run::

    python examples/digits_class_axis_regions.py --epochs 40
"""

from __future__ import annotations

import argparse
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
    ordinal_class_axis,
)

OUT_DIR = Path(__file__).resolve().parent / "out" / "digits_class_axis"
N_CLASSES = 10
PER_CLASS_CALIB = 40
PER_CLASS_TEST = 30
GRID = 320
# p-value at which a class stops accepting a point. The outline is drawn here and
# the fill fades to white below it.
ALPHA = 0.05
# p at which a region is drawn fully saturated; above this the colour stops
# getting stronger, so the gradient spends its range where it is informative.
FULL_P = 0.5


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


def p_field(cal: ClassRegionConformal, xs, ys):
    """Conformal p-value per class over a grid. Returns (K, ny, nx) and classes."""
    gx, gy = np.meshgrid(xs, ys)
    Z = torch.as_tensor(
        np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32
    )
    pv = cal.p_values(Z)
    classes = sorted(pv)
    P = np.stack([pv[c].numpy().reshape(gx.shape) for c in classes])
    return P, classes


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


def make_figure(P, classes, Z_tr, y_tr, Z_te, y_te, sets, pos_te, xs, ys, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab10")
    colours = np.stack([cmap(i)[:3] for i in range(N_CLASSES)])

    # Soft region field: hue from the most plausible class, saturation from how
    # plausible it is, so the between-class gaps and the outside both go white.
    winner = P.argmax(axis=0)
    conf = np.clip(P.max(axis=0) / FULL_P, 0.0, 1.0)[..., None]
    rgb = np.clip(colours[np.asarray(classes)[winner]] * conf + (1.0 - conf), 0.0, 1.0)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    for ax in axes:
        ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=0)
        for k, c in enumerate(classes):
            ax.contour(
                xs,
                ys,
                P[k],
                levels=[ALPHA],
                colors=[colours[c]],
                linewidths=1.3,
                alpha=0.95,
                zorder=1,
            )
        ax.set_xlabel("z0  (ordered axis: digit 0 → 9)")
        ax.set_ylabel("z1  (free)")

    axes[0].scatter(
        Z_tr[:, 0], Z_tr[:, 1], c=y_tr, cmap="tab10", vmin=-0.5, vmax=9.5,
        s=5, alpha=0.55, linewidths=0, zorder=2,
    )
    axes[0].set_title(
        f"training points inside their conformal regions (outline at p={ALPHA})\n"
        "white = no class accepts a point here"
    )

    for name, (marker, _) in MARKERS.items():
        sel = np.asarray([outcome(int(t), s) == name for t, s in zip(y_te, sets)])
        if not sel.any():
            continue
        axes[1].scatter(
            Z_te[sel, 0], Z_te[sel, 1],
            c=y_te[sel], cmap="tab10", vmin=-0.5, vmax=9.5,
            marker=marker, s=64, alpha=0.95,
            edgecolors="black", linewidths=0.7, zorder=3,
        )
    axes[1].set_title(
        f"held-out inference: {len(y_te)} points never seen in fitting or "
        f"calibration\nfill colour = true digit, marker = conformal outcome at "
        f"alpha={ALPHA}"
    )
    axes[1].legend(
        handles=[
            Line2D([], [], marker=m, color="none", markerfacecolor="0.7",
                   markeredgecolor="black", markersize=9, label=lab)
            for m, lab in MARKERS.values()
        ],
        loc="best", fontsize=9, framealpha=0.92,
    )

    # Without this the reader cannot tell which region belongs to which digit.
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.BoundaryNorm(np.arange(11) - 0.5, 10)
    )
    cb = fig.colorbar(
        sm, ax=axes.tolist(), ticks=np.arange(10), fraction=0.022, pad=0.015
    )
    cb.set_label("digit")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


CACHE = OUT_DIR / "regions_cache.npz"


def _object_array(seq):
    """Ragged tuples, kept ragged — numpy would otherwise square them off."""
    a = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        a[i] = tuple(v)
    return a


def replot() -> None:
    """Re-render from the cached fields, so figure work costs no refit."""
    d = np.load(CACHE, allow_pickle=True)
    out = make_figure(
        d["P"], list(d["classes"]), d["Z_tr"], d["y_tr"], d["Z_te"], d["y_te"],
        list(d["sets"]), d["pos_te"], d["xs"], d["ys"],
        OUT_DIR / "class_regions.png",
    )
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lambda-class", type=float, default=1.0, dest="lam")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="re-render the figure from the cached fit instead of refitting",
    )
    args = ap.parse_args()
    if args.replot:
        replot()
        return

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    tr_i, cal_i, te_i = three_way_split(y, seed=args.seed)
    print(f"train={len(tr_i)} calib={len(cal_i)} test={len(te_i)} (disjoint)")

    cfg = PLANEConfig.for_scale(len(tr_i))
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.dedup = False
    cfg.lambda_class = args.lam
    ax = ordinal_class_axis(N_CLASSES, axis=0, name="digit")
    res = fit(
        X[tr_i],
        dist_fn="l2",
        config=cfg,
        X_calib=X[cal_i],
        class_labels=y[tr_i],
        class_axes=[ax],
    )

    X_tr = torch.as_tensor(X[tr_i])
    y_tr = y[tr_i]
    readout = ClassAxisReadout.from_model(res.model, X_tr, torch.as_tensor(y_tr), ax)
    cal = ClassRegionConformal(readout).fit(
        res.model, torch.as_tensor(X[cal_i]), torch.as_tensor(y[cal_i])
    )

    Z_tr, _ = res.model.embed(X_tr)
    Z_tr = Z_tr.detach()
    rep = class_axis_report(Z_tr, torch.as_tensor(y_tr), [ax])
    print(
        f"ordering accuracy {rep['order_digit']:.3f} "
        f"(adjacent {rep['order_adjacent_digit']:.3f}, chance 0.500)"
    )

    # Held-out inference: one forward pass, no labels, no graph.
    Z_te, _ = res.model.embed(torch.as_tensor(X[te_i]))
    Z_te = Z_te.detach()
    y_te = y[te_i]
    sets = cal.prediction_set(Z_te, alpha=ALPHA)
    pos_te = readout.position(Z_te).numpy()
    pv = cal.p_values(Z_te)

    kinds = np.asarray([outcome(int(t), s) for t, s in zip(y_te, sets)])
    sizes = np.asarray([len(s) for s in sets])
    print(f"\nheld-out outcomes at alpha={ALPHA} (n={len(y_te)}):")
    for name, (_, label) in MARKERS.items():
        n = int((kinds == name).sum())
        print(f"  {label:32s} {n:4d}  {n / len(y_te):6.1%}")
    print(
        f"\ncoverage of the true digit {float((kinds != 'wrong').mean() - (kinds == 'novel').mean()):.3f} "
        f"(target >= {1 - ALPHA:.2f}), mean set size {sizes.mean():.2f}"
    )
    print(
        f"position on the ordering: median |position - true digit| = "
        f"{float(np.median(np.abs(pos_te - y_te))):.2f}"
    )

    print("\nwhere the first ten held-out points landed:")
    print(f"{'true':>5s} {'z0':>7s} {'z1':>7s} {'position':>9s}  accepted (p)")
    for i in range(10):
        acc = ", ".join(
            f"{c}({float(pv[c][i]):.2f})" for c in sorted(sets[i])
        ) or "— none —"
        print(
            f"{int(y_te[i]):5d} {Z_te[i, 0]:7.2f} {Z_te[i, 1]:7.2f} "
            f"{pos_te[i]:9.2f}  {acc}"
        )

    pad = 0.08
    x0, x1 = float(Z_tr[:, 0].min()), float(Z_tr[:, 0].max())
    y0, y1 = float(Z_tr[:, 1].min()), float(Z_tr[:, 1].max())
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    xs = np.linspace(x0 - dx, x1 + dx, GRID)
    ys = np.linspace(y0 - dy, y1 + dy, GRID)
    P, classes = p_field(cal, xs, ys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        P=P, classes=np.asarray(classes), xs=xs, ys=ys,
        Z_tr=Z_tr.numpy(), y_tr=y_tr, Z_te=Z_te.numpy(), y_te=y_te,
        pos_te=pos_te, sets=_object_array(sets),
    )
    out = make_figure(
        P, classes, Z_tr.numpy(), y_tr, Z_te.numpy(), y_te, sets, pos_te, xs, ys,
        OUT_DIR / "class_regions.png",
    )
    print(f"\nsaved {out}\nsaved {CACHE} (re-render with --replot)")


if __name__ == "__main__":
    main()

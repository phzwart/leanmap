#!/usr/bin/env python
"""Ordering the digits 0-9 along a direction the fit discovers.

The graph losses set the shape of the cloud. One ordered label -- here the digit
value -- is one :class:`~leanmap.ClassAxis` with ``axis=None``, so the fit finds
a direction along which 0 sits below 1 sits below ... sits below 9, and leaves
the other coordinate to the neighbour graph. K classes do not ask for K
directions; they ask for one number line.

See :mod:`leanmap.classaxis`. Pinning the order to ``z0`` (``axis=0``) is the
stronger request, shown in ``digits_two_orderings.py``; this example is the
usual one: preserve the order along some projection.

Three fits, because the first one on its own proves nothing:

``off``
    ``lambda_class=0``. The same data and seed with the gauge term disabled.
    Its ordering accuracy is the baseline the term has to beat.

``on``
    The gauge fix applied. Reports ordering accuracy *and* 5-NN label accuracy,
    the second being the one that matters: if orientation were bought by
    distorting the layout, neighbourhood structure would degrade. Low friction
    is a measurement, not a design claim.

``null``
    The gauge fix applied to *shuffled* labels at the same strength. This is the
    control that says whether the ordered look is a property of the data or of
    the term. If ``null`` also orders, the term is bending the map to fit
    whatever labels it is handed and the ``on`` result means nothing. Shuffling
    keeps every marginal and destroys only the association, which is the same
    null the rest of this repo reports against.

Measured at 20 epochs, digit order 0-9 on a discovered direction
(``axis=None``; chance on a free direction is slightly above 0.5)::

    run                            order  adjacent    5-NN
    off (lambda_class=0)           0.638     0.556   0.942
    on (lambda_class=1)            0.848     0.676   0.913
    null (shuffled labels)         0.550     0.515   0.369

Three things to read off that, in order of importance.

The null stays at chance. The term *cannot* manufacture an ordering that the
features do not support, so the ``on`` improvement is about the data.

The null's 5-NN collapses to 0.377, and its ``z0`` range shrinks from about six
units to half a unit. That is the failure mode of an unsatisfiable ordering: with
random labels every point needs to be both left and right of every other, the
contradictory pulls average toward the centre, and the constrained axis collapses
while the free axis absorbs what is left of the layout. An ordering the features
cannot support does not degrade gracefully -- it destroys the coordinate it was
applied to, which is why the diagnostic warns instead of the term quietly
straining.

``on`` costs 0.034 of 5-NN accuracy. That is the price of the orientation, and it
is small but not zero. Digit identity is *nominal*, not ordinal -- there is no
sense in which a 4 lies between a 3 and a 5 -- so a 0-9 reading order is only
partly achievable here and adjacent-class accuracy stops at 0.669. That is the
honest answer for this dataset and the reason this example is a demonstration of
the diagnostic as much as of the mechanism. On genuinely ordered classes (a
stage, a severity, a cell-cycle phase) expect adjacent accuracy far higher and
the 5-NN cost lower, because the requested order is then already latent in the
features and the term only has to choose which way round it points.

Runs three fits, so it is roughly three times the cost of a plain digits fit.

Run::

    python examples/digits_class_axis.py --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

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
CALIB_PER_CLASS = 40


def _split(X, y, seed=0):
    """Stratified calibration split: a per-class conformal test needs every class."""
    rng = np.random.default_rng(seed)
    cal, tr = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cal.append(idx[:CALIB_PER_CLASS])
        tr.append(idx[CALIB_PER_CLASS:])
    return np.concatenate(tr), np.concatenate(cal)


def _config(args, n_train, lam):
    cfg = PLANEConfig.for_scale(n_train)
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.dedup = False
    cfg.lambda_class = lam
    return cfg


def _run(name, X_tr, y_tr, X_cal, args, lam, ax):
    cfg = _config(args, len(X_tr), lam)
    res = fit(
        X_tr,
        dist_fn="l2",
        config=cfg,
        X_calib=X_cal,
        class_labels=y_tr,
        class_axes=[ax],
    )
    Z, _ = res.model.embed(torch.as_tensor(X_tr))
    Z = Z.detach()
    report = class_axis_report(Z, torch.as_tensor(y_tr), [ax])
    knn = KNeighborsClassifier(n_neighbors=5).fit(Z.numpy(), y_tr)
    direction = None
    if not ax.is_pinned:
        direction = np.asarray(
            [report[f"dir_{ax.name}_{j}"] for j in range(Z.shape[1])],
            dtype=np.float64,
        )
    return {
        "name": name,
        "result": res,
        "Z": Z,
        "direction": direction,
        "order": report[f"order_{ax.name}"],
        "order_adjacent": report[f"order_adjacent_{ax.name}"],
        # Resubstitution 5-NN is fine as a *relative* structure measure across
        # runs that share data and seed; it is not a generalisation claim.
        "knn5": float(knn.score(Z.numpy(), y_tr)),
    }


def _scatter(runs, y, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6))
    for ax_p, run in zip(np.atleast_1d(axes), runs):
        z = run["Z"].numpy()
        u = run.get("direction")
        if u is not None and z.shape[1] >= 2:
            # Draw in the frame (discovered direction, its complement) so left
            # to right is the requested order, not whichever coordinate the
            # encoder happened to use.
            e = u / max(float(np.linalg.norm(u)), 1e-12)
            v = np.array([-e[1], e[0]])
            a = np.stack([z @ e, z @ v], axis=1)
            xlab, ylab = "projection on the discovered direction (0 → 9)", "orthogonal"
        else:
            a = z
            xlab, ylab = "z0", "z1"
        sc = ax_p.scatter(a[:, 0], a[:, 1], c=y, cmap="tab10", s=5, alpha=0.85)
        ax_p.set_title(
            f"{run['name']}\norder={run['order']:.3f} "
            f"adjacent={run['order_adjacent']:.3f} 5-NN={run['knn5']:.3f}"
        )
        ax_p.set_xlabel(xlab)
        ax_p.set_ylabel(ylab)
        ax_p.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=np.atleast_1d(axes).tolist(), label="digit", fraction=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lambda-class", type=float, default=1.0, dest="lam")
    args = ap.parse_args()

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    tr_i, cal_i = _split(X, y, seed=args.seed)
    X_tr, y_tr, X_cal, y_cal = X[tr_i], y[tr_i], X[cal_i], y[cal_i]
    ax = ordinal_class_axis(N_CLASSES, axis=None, name="digit")

    runs = [
        _run("off (lambda_class=0)", X_tr, y_tr, X_cal, args, 0.0, ax),
        _run(f"on (lambda_class={args.lam:g})", X_tr, y_tr, X_cal, args, args.lam, ax),
    ]
    y_shuf = np.random.default_rng(args.seed + 99).permutation(y_tr)
    runs.append(
        _run("null (shuffled labels)", X_tr, y_shuf, X_cal, args, args.lam, ax)
    )

    print(f"\nN_train={len(X_tr)} N_calib={len(X_cal)} epochs={args.epochs}\n")
    print(f"{'run':28s} {'order':>7s} {'adjacent':>9s} {'5-NN':>7s}")
    for r in runs:
        print(
            f"{r['name']:28s} {r['order']:7.3f} {r['order_adjacent']:9.3f} "
            f"{r['knn5']:7.3f}"
        )
    print("\nchance for order / adjacent is 0.500")
    print(
        "read: 'on' should beat 'off' on order while holding 5-NN, and 'null' "
        "should sit near chance on order.\nIf 'null' orders too, the term is "
        "bending the map to fit any labels and 'on' means nothing."
    )

    on = runs[1]
    readout = ClassAxisReadout.from_model(
        on["result"].model, torch.as_tensor(X_tr), torch.as_tensor(y_tr), ax
    )
    cal = ClassRegionConformal(readout).fit(
        on["result"].model, torch.as_tensor(X_cal), torch.as_tensor(y_cal)
    )
    Z_cal, _ = on["result"].model.embed(torch.as_tensor(X_cal))
    Z_cal = Z_cal.detach()
    sets = cal.prediction_set(Z_cal, alpha=0.05)
    pos = readout.position(Z_cal).numpy()
    sizes = np.array([len(s) for s in sets])
    covered = np.mean([int(y_cal[i]) in s for i, s in enumerate(sets)])
    print(
        f"\nper-class conformal at alpha=0.05: coverage of the true digit "
        f"{covered:.3f}, mean set size {sizes.mean():.2f}, "
        f"empty (novel) {float((sizes == 0).mean()):.3f}, "
        f"ambiguous (>1) {float((sizes > 1).mean()):.3f}"
    )
    err = np.abs(pos - y_cal)
    print(
        f"position on the ordering: median |position - true digit| = "
        f"{float(np.median(err)):.2f}"
    )

    out = _scatter(runs, y_tr, OUT_DIR / "class_axis.png")
    model_path = OUT_DIR / "digits_class_axis.pt"
    on["result"].save(str(model_path))
    print(f"\nsaved {out}\nsaved {model_path}")


if __name__ == "__main__":
    main()

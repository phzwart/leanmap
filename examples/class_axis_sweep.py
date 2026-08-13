#!/usr/bin/env python
"""What raising the ordering weight actually buys, and what it costs.

Three knobs affect how hard the class ordering is pressed, and they are not
interchangeable:

``lambda_class``
    The price of a *violated* pair. Because the term is a hinge, this scales the
    gradient on pairs that are still out of order and does nothing at all to
    pairs already ordered — so it buys ordering only where ordering is
    achievable, and where it is not it spends the layout instead.

``class_margin``
    What counts as satisfied: the gap an ordered pair must clear, as a fraction
    of the axis spread. This is the knob that makes an ordering *visible* rather
    than merely technically correct, and it is also where the term stops being a
    pure gauge fix — a large margin dictates spacing, which is geometry the
    neighbour graph was supposed to own.

``class_ramp``
    When it engages. Rotating a formed layout is much harder than growing one in
    the right orientation, so an earlier ramp is often the cheapest way to get
    more ordering per unit of ``lambda_class``.

Reports ordering accuracy against 5-NN label accuracy, because the second is the
price. Writes ``runs/class_axis_sweep.log``.

Run::

    python examples/class_axis_sweep.py --epochs 20
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

from leanmap import PLANEConfig, class_axis_report, fit, ordinal_class_axis

LOG = Path(__file__).resolve().parent.parent / "runs" / "class_axis_sweep.log"
N_CLASSES = 10
PER_CLASS_CALIB = 40

# (label, lambda_class, class_margin, class_ramp)
ARMS = [
    ("off", 0.0, 0.05, (0.2, 0.45)),
    ("lambda=1 (default)", 1.0, 0.05, (0.2, 0.45)),
    ("lambda=4", 4.0, 0.05, (0.2, 0.45)),
    ("lambda=16", 16.0, 0.05, (0.2, 0.45)),
    ("lambda=4, margin=0.30", 4.0, 0.30, (0.2, 0.45)),
    ("lambda=4, ramp early", 4.0, 0.05, (0.0, 0.1)),
    ("lambda=16, margin=0.30, early", 16.0, 0.30, (0.0, 0.1)),
]


def split(y, seed=0):
    rng = np.random.default_rng(seed)
    tr, cal = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cal.append(idx[:PER_CLASS_CALIB])
        tr.append(idx[PER_CLASS_CALIB:])
    return np.concatenate(tr), np.concatenate(cal)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    tr_i, cal_i = split(y, seed=args.seed)
    X_tr, y_tr, X_cal = X[tr_i], y[tr_i], X[cal_i]
    ax = ordinal_class_axis(N_CLASSES, axis=0, name="digit")
    # Same seed and data everywhere, so differences are the knobs only.
    base = PLANEConfig.for_scale(len(X_tr))
    base = replace(
        base, epochs=args.epochs, seed=args.seed, device=args.device, dedup=False
    )

    rows = []
    for name, lam, margin, ramp in ARMS:
        cfg = replace(
            base, lambda_class=lam, class_margin=margin, class_ramp=tuple(ramp)
        )
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
        rep = class_axis_report(Z, torch.as_tensor(y_tr), [ax])
        knn = float(
            KNeighborsClassifier(n_neighbors=5).fit(Z.numpy(), y_tr).score(Z.numpy(), y_tr)
        )
        # How stratified the axis actually looks: spread of the class medians
        # relative to the within-class spread along that axis.
        med = np.asarray([np.median(Z[y_tr == c, 0]) for c in range(N_CLASSES)])
        within = float(np.mean([Z[y_tr == c, 0].std() for c in range(N_CLASSES)]))
        rows.append(
            {
                "name": name,
                "order": rep["order_digit"],
                "adjacent": rep["order_adjacent_digit"],
                "knn5": knn,
                "separation": float(med.std() / max(within, 1e-9)),
            }
        )
        print(
            f"{name:32s} order={rows[-1]['order']:.3f} "
            f"adjacent={rows[-1]['adjacent']:.3f} 5-NN={knn:.3f} "
            f"sep={rows[-1]['separation']:.2f}",
            flush=True,
        )

    head = (
        f"{'arm':32s} {'order':>7s} {'adjacent':>9s} {'5-NN':>7s} {'sep':>6s}"
    )
    lines = [
        f"digits, N_train={len(X_tr)}, epochs={args.epochs}, seed={args.seed}",
        "order/adjacent chance = 0.500; sep = sd(class medians)/mean(within-class sd) on z0",
        "",
        head,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:32s} {r['order']:7.3f} {r['adjacent']:9.3f} "
            f"{r['knn5']:7.3f} {r['separation']:6.2f}"
        )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved {LOG}")


if __name__ == "__main__":
    main()

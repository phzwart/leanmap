#!/usr/bin/env python
"""Digit value pinned to one axis, parity on a direction the fit chooses.

Two orderings of the same labels, asked for with deliberately different force.

The digit value is the ordering a reader navigates by, so it gets a *pinned*
axis: ``z0`` must increase with the digit, which fixes one direction and its
sign. Parity is a coarse secondary factor, and there is no honest reason to
claim which way a map should lay out even versus odd -- so it gets a
*free-direction* axis, which asks only that the two groups come apart along some
direction and lets the fit choose which -- including its angle to ``z0``. The
default discovers that angle (read ``tilt_parity``); ``orthogonal=True`` forces
it square to the pinned axis and then the parity term cannot move ``z0``.

The arms are chosen to show what each choice costs:

``digit only``
    Parity is measured but never requested. This is the number the parity arms
    have to beat -- digits may already separate by parity incidentally.

``+ parity free dir``
    The request under discussion, at two weights.

``+ parity pinned to z1``
    Over-specifying: naming the second coordinate too. The ceiling refuses this
    outright at ``d_out=2``, and the arm is kept so the refusal is visible next
    to the request that is allowed — the same information, asked for in a way
    that leaves the layout a say.

``+ parity free dir, d_out=3``
    The same weak request with room to breathe: ``z0`` pinned still leaves a
    whole plane for parity to pick a direction in, so a free direction remains.

``arbitrary 5/5 split``
    Parity replaced by a meaningless balanced split of the digits, scored both on
    that split and on true parity. The point is not a null for the measurement --
    the ``digit only`` arm supplies that -- but the sterner warning: if an
    arbitrary grouping also separates cleanly, then a high parity score is
    evidence that the *term works*, not that parity means anything.

Writes ``runs/digits_two_orderings.log`` and ``runs/digits_two_orderings.png``.

Run::

    python examples/digits_two_orderings.py --epochs 20
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

from leanmap import (
    PLANEConfig,
    class_axis_report,
    fit,
    grouped_class_axis,
    ordinal_class_axis,
)

RUNS = Path(__file__).resolve().parent.parent / "runs"
N_CLASSES = 10
PER_CLASS_CALIB = 40
EVEN = [0, 2, 4, 6, 8]
ODD = [1, 3, 5, 7, 9]

# Best arm from examples/class_axis_sweep.py: a strong hinge that engages early
# costs less neighbourhood quality than a weak one that argues for the whole run.
LAMBDA = 16.0
MARGIN = 0.30
RAMP = (0.0, 0.1)


def split(y, seed=0):
    rng = np.random.default_rng(seed)
    tr, cal = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cal.append(idx[:PER_CLASS_CALIB])
        tr.append(idx[PER_CLASS_CALIB:])
    return np.concatenate(tr), np.concatenate(cal)


def sham_split(seed):
    rng = np.random.default_rng(seed + 99)
    shuffled = rng.permutation(N_CLASSES)
    return [sorted(shuffled[:5].tolist()), sorted(shuffled[5:].tolist())]


def arms(seed):
    """(name, d_out, secondary axis or None) -- the digit axis is common to all."""
    sham = sham_split(seed)
    return [
        ("digit only", 2, None),
        (
            "+ parity free dir, w=0.3",
            2,
            grouped_class_axis([EVEN, ODD], axis=None, name="parity", weight=0.3),
        ),
        (
            "+ parity free dir, w=1.0",
            2,
            grouped_class_axis([EVEN, ODD], axis=None, name="parity", weight=1.0),
        ),
        (
            "+ parity pinned to z1, w=0.3",
            2,
            grouped_class_axis([EVEN, ODD], axis=1, name="parity", weight=0.3),
        ),
        (
            "+ parity free dir, w=0.3, d=3",
            3,
            grouped_class_axis([EVEN, ODD], axis=None, name="parity", weight=0.3),
        ),
        (
            "arbitrary 5/5 split",
            2,
            grouped_class_axis(sham, axis=None, name="parity", weight=0.3),
        ),
    ]


def make_figure(panels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.6))
    axs = np.atleast_1d(axs)
    cmap = plt.get_cmap("tab10")
    for ax, p in zip(axs, panels):
        Z, y, u, name, rep = p["Z"], p["y"], p["u"], p["name"], p["rep"]
        # Plot z0 against the parity direction: the two things that were asked
        # for, which in d_out=3 are not two of the coordinates.
        # np.dot, not @: numpy 2.2's matmul emits spurious warnings on this path.
        h = np.dot(Z, u)
        for c in range(N_CLASSES):
            m = y == c
            ax.scatter(
                Z[m, 0],
                h[m],
                s=14,
                color=cmap(c % 10),
                marker="o" if c % 2 == 0 else "^",
                alpha=0.75,
                linewidths=0.0,
                label=f"{c}",
            )
        ax.set_xlabel("$z_0$  (digit value, pinned)")
        ax.set_ylabel("projection on the parity direction")
        ax.set_title(
            f"{name}\n"
            f"digit adj={rep['order_adjacent_digit']:.3f}  "
            f"parity={rep['order_parity']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.15)
    axs[0].legend(
        title="digit (o even, ^ odd)", fontsize=7, title_fontsize=7,
        loc="upper left", ncol=2, framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    y_t = torch.as_tensor(y_tr)

    digit = ordinal_class_axis(N_CLASSES, axis=0, name="digit")
    # Parity as it is *measured*, always the true one, so the null arm is scored
    # on the question we care about rather than on the grouping it was trained on.
    parity_true = grouped_class_axis([EVEN, ODD], axis=None, name="parity")
    base = replace(
        PLANEConfig.for_scale(len(X_tr)),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        dedup=False,
        lambda_class=LAMBDA,
        class_margin=MARGIN,
        class_ramp=RAMP,
    )

    rows, panels = [], []
    for name, d_out, parity in arms(args.seed):
        axes = [digit] + ([parity] if parity is not None else [])
        try:
            res = fit(
                X_tr,
                dist_fn="l2",
                config=replace(base, d_out=d_out),
                X_calib=X_cal,
                class_labels=y_tr,
                class_axes=axes,
            )
        except ValueError as exc:
            # The ceiling refusing an arm is a result, not a failure: it is the
            # reason axis=None exists.
            rows.append({"name": name, "d_out": d_out, "refused": str(exc)})
            print(f"{name:30s} d={d_out} REFUSED: {str(exc).split(';')[0]}", flush=True)
            continue
        Z, _ = res.model.embed(torch.as_tensor(X_tr))
        Z = Z.detach()
        rep = class_axis_report(Z, y_t, [digit, parity_true])
        u = np.asarray([rep[f"dir_parity_{j}"] for j in range(d_out)])
        # Also score the grouping that was actually requested, which is the same
        # thing except in the arbitrary-split arm — where the gap between the two
        # is the whole point.
        requested = (
            class_axis_report(Z, y_t, [replace(parity, name="req")])["order_req"]
            if parity is not None
            else float("nan")
        )
        knn = float(
            KNeighborsClassifier(n_neighbors=5)
            .fit(Z.numpy(), y_tr)
            .score(Z.numpy(), y_tr)
        )
        rows.append(
            {
                "name": name,
                "d_out": d_out,
                "refused": None,
                "digit": rep["order_digit"],
                "adjacent": rep["order_adjacent_digit"],
                "parity": rep["order_parity"],
                "requested": requested,
                "knn5": knn,
                "u": u,
            }
        )
        print(
            f"{name:30s} d={d_out} digit={rep['order_digit']:.3f} "
            f"adj={rep['order_adjacent_digit']:.3f} "
            f"requested={requested:.3f} parity={rep['order_parity']:.3f} "
            f"5-NN={knn:.3f} u={np.round(u, 2).tolist()}",
            flush=True,
        )
        if name in (
            "digit only",
            "+ parity free dir, w=0.3",
            "+ parity free dir, w=0.3, d=3",
        ):
            panels.append({"Z": Z.numpy(), "y": y_tr, "u": u, "name": name, "rep": rep})

    head = (
        f"{'arm':30s} {'d':>2s} {'digit':>7s} {'adjacent':>9s} {'asked':>7s} "
        f"{'parity':>7s} {'5-NN':>7s}  direction"
    )
    sham = sham_split(args.seed)
    shared = len(set(sham[0]) & set(EVEN))
    shared = max(shared, 5 - shared)
    lines = [
        f"digits, N_train={len(X_tr)}, epochs={args.epochs}, seed={args.seed}",
        f"lambda_class={LAMBDA} margin={MARGIN} ramp={RAMP}",
        "digit is pinned to z0 in every arm; the secondary axis varies.",
        "'asked' scores the grouping that arm requested, 'parity' always scores",
        "true even/odd; they differ only in the arbitrary-split arm. Both are read",
        "along the best-separating direction, which is what a free-direction axis",
        "asks for. 'digit only' is the baseline for 'parity': a layout never asked",
        "to encode parity, so it carries both the incidental signal and the upward",
        "bias of a fitted direction.",
        "",
        f"the arbitrary split is {sham[0]} vs {sham[1]}, which happens to share",
        f"{shared} of 5 members with a parity class -- so read its 'asked' column,",
        "not its 'parity' column. A random 5/5 split cannot be parity-neutral and",
        "this draw is strongly parity-aligned; what it is here to show is that an",
        "arbitrary grouping separates just as cleanly as a meaningful one.",
        "",
        head,
    ]
    for r in rows:
        if r.get("refused"):
            lines.append(
                f"{r['name']:30s} {r['d_out']:2d} "
                f"{'refused by the d_out - 1 ceiling':>49s}"
            )
            continue
        lines.append(
            f"{r['name']:30s} {r['d_out']:2d} {r['digit']:7.3f} "
            f"{r['adjacent']:9.3f} {r['requested']:7.3f} {r['parity']:7.3f} "
            f"{r['knn5']:7.3f}  {np.round(r['u'], 3).tolist()}"
        )
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "digits_two_orderings.log").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    if panels:
        make_figure(panels, RUNS / "digits_two_orderings.png")
        print(f"\nsaved {RUNS / 'digits_two_orderings.png'}")
    print(f"saved {RUNS / 'digits_two_orderings.log'}")


if __name__ == "__main__":
    main()

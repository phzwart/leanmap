#!/usr/bin/env python
"""What do the requested orderings mean in pixels? A biplot, drawn in feature space.

``class_axis_report`` says whether an ordering took; :func:`leanmap.axis_loadings`
says what it *means*, by taking the gradient of position-along-the-axis with respect
to the input. Because the input is an 8x8 image, so is the loading, and "what does my
digit axis mean" becomes a picture rather than a 64-vector.

The point of this example is the caveat, not the pictures. A biplot presumes a linear
map, and a neural embedding is not one, so the gradient is a property of the point it
is taken at. Averaging it over the data produces something that always *looks* like an
answer. Whether it is one is what ``stability`` measures -- the resultant length of
the per-point unit loadings, ``1.0`` if every point agrees.

Three rows, and the third is the only one that needs no apology:

1. **Default fit, mean Jacobian.** Stability lands near 0.4, and the 10th percentile
   of per-point agreement is negative: for a good fraction of points the mean loading
   points the *opposite* way along the axis from their own gradient. The image is an
   average over evidence that disagrees in sign.
2. **``pca_skip=True``, mean Jacobian.** Same quantity for a fit that has an
   explicitly linear component. Usually steadier, still a mean over a nonlinear map.
3. **``pca_skip=True``, the linear component alone.** With ``z = W x_n + residual``,
   the loading of ``W`` is exact and the same everywhere by construction, so this one
   is a biplot in the original sense rather than an average standing in for one. The
   price is that it describes only the part of the map that is linear.

Run::

    python examples/digits_axis_loadings.py --epochs 20
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

from leanmap import (
    PLANEConfig,
    axis_loadings,
    class_axis_report,
    fit,
    grouped_class_axis,
    ordinal_class_axis,
)

OUT_DIR = Path(__file__).resolve().parent / "out" / "digits_axis_loadings"
EVEN = [0, 2, 4, 6, 8]
ODD = [1, 3, 5, 7, 9]
LAMBDA = 16.0
MARGIN = 0.30
RAMP = (0.0, 0.1)
PARITY_WEIGHT = 0.3
SHAPE = (8, 8)


def run(X, y, axes, *, epochs, seed, pca_skip):
    cfg = replace(
        PLANEConfig.for_scale(len(X)),
        epochs=epochs,
        seed=seed,
        dedup=False,
        lambda_class=LAMBDA,
        class_margin=MARGIN,
        class_ramp=RAMP,
        pca_skip=pca_skip,
    )
    res = fit(X, dist_fn="l2", config=cfg, class_labels=y, class_axes=axes)
    Z, _ = res.model.embed(res.X_train)
    yt = res.class_labels_train
    rep = class_axis_report(Z.detach(), yt, axes)
    # Hand the report's own directions over, so the loading is taken along exactly
    # the direction the ordering was scored along.
    dirs = {
        a.name: (
            np.eye(cfg.d_out)[a.axis]
            if a.is_pinned
            else np.asarray([rep[f"dir_{a.name}_{j}"] for j in range(cfg.d_out)])
        )
        for a in axes
    }
    ld = axis_loadings(res.model, res.X_train, axes, directions=dirs, sample=512)
    return res, rep, ld


def linear_loadings(res, axes, ld):
    """Loading of the linear skip alone: exact, and the same at every point."""
    enc = res.model.encoder
    if getattr(enc, "pca", None) is None:
        return None
    W = enc.pca.weight.detach().cpu()  # (d_out, D), acts on standardised input
    out = {}
    for a in axes:
        u = torch.as_tensor(ld.direction[a.name], dtype=torch.float32)
        out[a.name] = (W.T @ u).numpy().astype(np.float64)
    return out


def panel(ax, vec, feature_std, *, title):
    import matplotlib.pyplot as plt

    img = np.asarray(vec, dtype=np.float64).reshape(SHAPE)
    # Features that never varied have no "one standard deviation", so their loading
    # is arbitrary rather than small. They have to be greyed rather than left to the
    # colormap: on a diverging scale an unmasked bad value renders white, which is
    # exactly the colour of "this pixel does not matter" -- the opposite of unknown.
    dead = (np.asarray(feature_std) <= 1e-5).reshape(SHAPE)
    img = np.ma.array(img, mask=dead)
    lim = float(np.max(np.abs(img))) if img.count() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.6")
    im = ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return im


def make_figure(rows, path: Path) -> Path:
    import matplotlib.pyplot as plt

    n_row = len(rows)
    n_col = max(len(r["vecs"]) for r in rows)
    fig, axs = plt.subplots(
        n_row, n_col, figsize=(3.1 * n_col, 3.5 * n_row), squeeze=False
    )
    for i, row in enumerate(rows):
        for j, (name, vec) in enumerate(row["vecs"].items()):
            im = panel(
                axs[i][j],
                vec,
                row["feature_std"],
                title=f"{name}\n{row['note'](name)}",
            )
            fig.colorbar(im, ax=axs[i][j], fraction=0.046, pad=0.04)
        for j in range(len(row["vecs"]), n_col):
            axs[i][j].axis("off")
        axs[i][0].set_ylabel(row["label"], fontsize=10, fontweight="bold")
    fig.suptitle(
        "what the requested orderings mean in pixels\n"
        "red = raises position along the axis, blue = lowers it, grey = pixel never varied",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    digit = ordinal_class_axis(10, axis=0, name="digit")
    parity = grouped_class_axis(
        [EVEN, ODD], axis=None, name="parity", weight=PARITY_WEIGHT
    )
    axes = [digit, parity]

    rows = []
    for pca_skip in (False, True):
        res, rep, ld = run(
            X, y, axes, epochs=args.epochs, seed=args.seed, pca_skip=pca_skip
        )
        tag = f"pca_skip={pca_skip}"
        print(
            f"\n{tag}: digit adj={rep['order_adjacent_digit']:.3f} "
            f"parity={rep['order_parity']:.3f} tilt={rep['tilt_parity']:.1f}"
        )
        for a in axes:
            print(f"  {a.name:6s} loading stability = {ld.stability[a.name]:.3f}")
        rows.append(
            {
                "label": f"{tag}\nmean Jacobian",
                "vecs": {a.name: ld.loading[a.name] for a in axes},
                "feature_std": ld.feature_std,
                "note": lambda n, ld=ld: f"stability {ld.stability[n]:.2f}"
                + ("  (trust it)" if ld.stability[n] > 0.8 else "  (an average of disagreement)"),
            }
        )
        lin = linear_loadings(res, axes, ld)
        if lin is not None:
            print("  linear component present: its loading is exact and global")
            rows.append(
                {
                    "label": f"{tag}\nlinear part only",
                    "vecs": lin,
                    "feature_std": ld.feature_std,
                    "note": lambda n: "exact, same at every point",
                }
            )
    print("\nsaved", make_figure(rows, OUT_DIR / "axis_loadings.png"))


if __name__ == "__main__":
    main()

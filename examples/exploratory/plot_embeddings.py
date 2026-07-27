#!/usr/bin/env python
"""Side-by-side embeddings with the structured probes overlaid.

The top row is the map itself, coloured by digit class. The bottom row replaces
the digits with a density field so the *empty* parts of each map are visible,
which is what actually determines whether an off-manifold point is detectable:
a probe is only far from the data if the map left somewhere for it to go.

Usage::

    python examples/exploratory/plot_embeddings.py \\
      --y examples/exploratory/data/digits_y.npy \\
      --probe-kind examples/exploratory/data/digits_probes_kind.npy \\
      --Z leanmap=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --Z umap=examples/out/exploratory/digits_holdout/reference/umap_default__none__seed0 \\
      --out examples/out/exploratory/digits_emd/embeddings.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_z(spec: str):
    name, path = spec.split("=", 1)
    return name, Path(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Z", action="append", required=True, help="name=run_dir")
    ap.add_argument("--y", default=None, help="digit labels .npy")
    ap.add_argument("--probe-kind", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.load(args.y) if args.y else None
    kinds = (
        np.asarray([str(v) for v in np.load(args.probe_kind, allow_pickle=True)])
        if args.probe_kind
        else None
    )

    runs = []
    for spec in args.Z:
        name, d = _parse_z(spec)
        Z = np.load(d / "Z.npy").astype(np.float64)
        Zp = np.load(d / "Z_probe.npy").astype(np.float64) if (d / "Z_probe.npy").is_file() else None
        runs.append((name, Z, Zp))

    ncol = len(runs)
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 9.6))
    axes = np.atleast_2d(axes)
    if ncol == 1:
        axes = axes.reshape(2, 1)

    for j, (name, Z, Zp) in enumerate(runs):
        ok = np.isfinite(Z).all(axis=1)
        ax = axes[0, j]
        if y is not None:
            sc = ax.scatter(
                Z[ok, 0], Z[ok, 1], c=np.asarray(y)[ok], cmap="tab10", s=5,
                linewidths=0, alpha=0.75,
            )
            if j == ncol - 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, ticks=range(10))
                cb.set_label("digit")
        else:
            ax.scatter(Z[ok, 0], Z[ok, 1], s=5, c="steelblue", linewidths=0, alpha=0.7)
        if Zp is not None:
            okp = np.isfinite(Zp).all(axis=1)
            ax.scatter(
                Zp[okp, 0], Zp[okp, 1], s=26, c="k", marker="x", linewidths=1.0,
                label="probes",
            )
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{name}: digits by class, probes overlaid")
        ax.set_xticks([])
        ax.set_yticks([])

        # Density view: where the map is empty is where an outlier can be seen.
        ax = axes[1, j]
        ax.hexbin(Z[ok, 0], Z[ok, 1], gridsize=48, cmap="Greys", mincnt=1, bins="log")
        if Zp is not None and kinds is not None:
            okp = np.isfinite(Zp).all(axis=1)
            fams = sorted(set(kinds[: len(Zp)][okp].tolist()))
            cmap = plt.get_cmap("tab20")
            for i, fam in enumerate(fams):
                m = (kinds[: len(Zp)] == fam) & okp
                ax.scatter(
                    Zp[m, 0], Zp[m, 1], s=20, color=cmap(i % 20), linewidths=0.3,
                    edgecolors="k", label=fam,
                )
            if j == 0:
                ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.85)
        elif Zp is not None:
            okp = np.isfinite(Zp).all(axis=1)
            ax.scatter(Zp[okp, 0], Zp[okp, 1], s=18, c="crimson", linewidths=0)
        ax.set_title(f"{name}: digit density (grey) vs probes")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Where structured probes land — never seen in training, ink-matched to real digits",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=145)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

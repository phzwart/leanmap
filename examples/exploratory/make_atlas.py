#!/usr/bin/env python
"""Rebuild summary.csv and a small-multiples atlas from saved exploratory runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def build_atlas(name_dir: Path, *, cols: int = 4, thumb: float = 2.4) -> Path:
    """Compose ``atlas.png`` from per-run ``scatter.png`` files."""
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    name_dir = Path(name_dir)
    # Prefer summary order if present; else filesystem order.
    run_dirs = []
    summary = name_dir / "summary.csv"
    if summary.is_file() and summary.stat().st_size > 0:
        import csv

        with summary.open() as f:
            for row in csv.DictReader(f):
                rid = row.get("run_id") or row.get("path")
                if not rid:
                    continue
                d = name_dir / rid
                if (d / "scatter.png").is_file():
                    run_dirs.append(d)
    if not run_dirs:
        run_dirs = sorted(
            p.parent for p in name_dir.glob("*/scatter.png")
        )

    if not run_dirs:
        raise FileNotFoundError(f"no scatter.png under {name_dir}")

    n = len(run_dirs)
    ncols = max(1, min(cols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(thumb * ncols, thumb * nrows),
        squeeze=False,
    )
    for i, d in enumerate(run_dirs):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        img = mpimg.imread(d / "scatter.png")
        ax.imshow(img)
        ax.set_title(d.name, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(name_dir.name, fontsize=11)
    fig.tight_layout()
    out = name_dir / "atlas.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "name_dir",
        type=Path,
        help="directory containing per-run subfolders (e.g. examples/out/exploratory/s_curve)",
    )
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(argv)

    name_dir = args.name_dir
    if not name_dir.is_dir():
        print(f"not a directory: {name_dir}", file=sys.stderr)
        return 2

    # Refresh summary if master helpers are available.
    try:
        from master import write_summary

        summary = write_summary(name_dir)
        print(f"wrote {summary}")
    except Exception as exc:  # noqa: BLE001
        print(f"summary refresh skipped: {exc}", file=sys.stderr)

    atlas = build_atlas(name_dir, cols=args.cols)
    print(f"wrote {atlas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

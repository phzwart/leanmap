#!/usr/bin/env python
"""Run Phase-1 embeddings on the three prepared feeds.

Idempotent (skips finished runs unless ``--force``). Embeddings land as
``examples/out/exploratory/{name}/{run_id}/Z.npy``.

Usage::

    python examples/exploratory/run_phase1.py
    python examples/exploratory/run_phase1.py --epochs 60 --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from master import main as master_main  # noqa: E402

DATA = _HERE / "data"

FEEDS = (
    {
        "name": "s_curve",
        "X": DATA / "s_curve_X.npy",
        "color": DATA / "s_curve_t.npy",
        "extra": ["--colorbar-label", "manifold parameter"],
    },
    {
        "name": "swiss_cone",
        "X": DATA / "swiss_cone_X.npy",
        "color": DATA / "swiss_cone_t.npy",
        "extra": ["--colorbar-label", "manifold parameter"],
    },
    {
        "name": "digits",
        "X": DATA / "digits_X.npy",
        "color": DATA / "digits_y.npy",
        "extra": ["--cmap", "tab10", "--colorbar-label", "digit"],
    },
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--only",
        default=None,
        help="optional feed name filter: s_curve | swiss_cone | digits",
    )
    ap.add_argument(
        "--sweep-only",
        default=None,
        dest="sweep_only",
        help="forwarded to master --only (axis / run_id filter)",
    )
    args = ap.parse_args(argv)

    feeds = FEEDS
    if args.only:
        feeds = tuple(f for f in FEEDS if f["name"] == args.only)
        if not feeds:
            print(f"unknown feed {args.only!r}", file=sys.stderr)
            return 2

    for feed in feeds:
        for key in ("X", "color"):
            path = feed[key]
            if not path.is_file():
                print(
                    f"missing {path}; run: python examples/exploratory/prepare_feeds.py",
                    file=sys.stderr,
                )
                return 2
        argv = [
            "--X",
            str(feed["X"]),
            "--color",
            str(feed["color"]),
            "--name",
            feed["name"],
            "--sweep",
            "phase1",
            "--epochs",
            str(args.epochs),
            "--atlas",
            *feed["extra"],
        ]
        if args.device:
            argv.extend(["--device", args.device])
        if args.force:
            argv.append("--force")
        if args.sweep_only:
            argv.extend(["--only", args.sweep_only])
        print(f"======== {feed['name']} ========", flush=True)
        rc = master_main(argv)
        if rc:
            return rc

    print(f"all feeds finished → {_EXAMPLES / 'out' / 'exploratory'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

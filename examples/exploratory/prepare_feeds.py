#!/usr/bin/env python
"""Dump paper-scope exploratory feeds to ``examples/exploratory/data/``.

Writes arrays for:

- S-curve (``s_curve_X.npy``, ``s_curve_t.npy``, ``s_curve_tbin.npy``)
- Classic swiss roll (``swiss_roll_X.npy``, ``swiss_roll_t.npy``,
  ``swiss_roll_tbin.npy``)
- 8×8 digits (``digits_X.npy``, ``digits_y.npy``; full set is 1797 points)
- Iris (``iris_X.npy``, ``iris_y.npy``; N=150)

Usage::

    python examples/exploratory/prepare_feeds.py
    python examples/exploratory/prepare_feeds.py --n 2000 --out examples/exploratory/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from quantile_bins import quantile_bins  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"  wrote {path}  shape={arr.shape} dtype={arr.dtype}")


def prepare_s_curve(out: Path, n: int, seed: int, noise: float, bins: int) -> None:
    from sklearn.datasets import make_s_curve

    X, t = make_s_curve(n_samples=n, noise=noise, random_state=seed)
    t = np.asarray(t, dtype=np.float64)
    _write(out / "s_curve_X.npy", X.astype(np.float32))
    _write(out / "s_curve_t.npy", t)
    _write(out / "s_curve_tbin.npy", quantile_bins(t, bins))


def prepare_swiss_roll(out: Path, n: int, seed: int, noise: float, bins: int) -> None:
    from sklearn.datasets import make_swiss_roll

    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    t = np.asarray(t, dtype=np.float64)
    _write(out / "swiss_roll_X.npy", X.astype(np.float32))
    _write(out / "swiss_roll_t.npy", t)
    _write(out / "swiss_roll_tbin.npy", quantile_bins(t, bins))


def prepare_digits(out: Path, n: int | None, seed: int) -> None:
    """Sklearn 8×8 digits (MNIST-like). Caps at available samples (1797)."""
    from sklearn.datasets import load_digits

    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    if n is not None and n < len(X):
        rng = np.random.default_rng(seed)
        keep = []
        per = max(1, n // 10)
        for c in range(10):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            keep.append(idx[:per])
        keep = np.concatenate(keep)[:n]
        X, y = X[keep], y[keep]
    _write(out / "digits_X.npy", X)
    _write(out / "digits_y.npy", y)


def prepare_iris(out: Path) -> None:
    """Sklearn iris (N=150, D=4, 3 classes) — small-N parametric showcase."""
    from sklearn.datasets import load_iris

    data = load_iris()
    _write(out / "iris_X.npy", data.data.astype(np.float32))
    _write(out / "iris_y.npy", data.target.astype(np.int64))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    ap.add_argument("--n", type=int, default=2000, help="samples for S-curve / swiss roll")
    ap.add_argument(
        "--digits-n",
        type=int,
        default=None,
        help="optional cap on digits samples (default: use full 1797)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.05, help="ambient noise for swiss roll")
    ap.add_argument(
        "--s-curve-noise",
        type=float,
        default=0.0,
        help="S-curve noise (default 0; sklearn gallery style)",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=8,
        help="quantile bins for continuous manifold parameters",
    )
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"preparing feeds → {out}")
    prepare_s_curve(out, args.n, args.seed, args.s_curve_noise, args.bins)
    prepare_swiss_roll(out, args.n, args.seed, args.noise, args.bins)
    prepare_digits(out, args.digits_n, args.seed)
    prepare_iris(out)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

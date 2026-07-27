#!/usr/bin/env python
"""Out-of-sample inference cost: leanmap's encoder against UMAP's transform().

This measures the thing leanmap was built for -- placing *new* points on an
already-fitted model -- rather than the one-off cost of fitting it. Two numbers
matter and they are different questions:

* **Throughput** on a large batch, for scoring a backlog.
* **Single-point latency**, for anything online, where per-call overhead is the
  whole cost and batching cannot hide it.

Model size on disk is reported alongside because it is the structural difference:
a parametric encoder is fixed-size, while ``UMAP.transform`` needs the training
data and its neighbour index kept around, so its artefact grows with N.

Every timing is on CPU for both methods, after a warm-up call that pays numba's
JIT compilation, and reported as a median over repeats.

Usage::

    python examples/exploratory/bench_inference.py \\
      --X examples/exploratory/data/digits_X.npy \\
      --leanmap examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --sklearn umap=examples/out/exploratory/digits_holdout/reference/umap_default__none__seed0 \\
      --sklearn pca2d=examples/out/exploratory/digits_holdout/reference/pca2d__none__seed0
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

BATCHES = (1, 8, 64, 512)

# Apple's Accelerate BLAS raises FP flags on large float32 matmuls that do not
# correspond to any bad value -- the results are finite and match float64 to the
# last digit. Checked, not assumed: --verify compares every transform against the
# embedding the run already committed to disk.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")


def _time(fn: Callable[[], object], repeats: int, warmup: int = 2) -> Tuple[float, float]:
    """Median and min wall time, in seconds, after warm-up."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def _leanmap_fn(run: Path, device: str | None):
    import torch

    from leanmap import load_plane

    model = load_plane(run / "model.pt", device=device)
    model.eval()

    def fn(P: np.ndarray):
        with torch.no_grad():
            z, cover = model.embed(torch.as_tensor(P))
        return z.detach().cpu().numpy(), cover.detach().cpu().numpy()

    return fn, (run / "model.pt").stat().st_size


def _sklearn_fn(run: Path):
    with open(run / "model.pkl", "rb") as fh:
        model = pickle.load(fh)

    def fn(P: np.ndarray):
        return np.asarray(model.transform(P))

    return fn, (run / "model.pkl").stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument("--leanmap", action="append", default=[], help="run dir with model.pt")
    ap.add_argument("--sklearn", action="append", default=[], help="name=run dir with model.pkl")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify", default=None, help="probe .npy to check against saved Z_probe")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    X = np.load(args.X).astype(np.float32)
    rng = np.random.default_rng(0)

    methods: List[Tuple[str, Callable, int]] = []
    paths: List[Tuple[str, str]] = []
    for spec in args.leanmap:
        name, path = (spec.split("=", 1) if "=" in spec else ("leanmap", spec))
        fn, size = _leanmap_fn(Path(path), args.device)
        methods.append((name, fn, size))
        paths.append((name, path))
    for spec in args.sklearn:
        name, path = (spec.split("=", 1) if "=" in spec else ("sklearn", spec))
        try:
            fn, size = _sklearn_fn(Path(path))
        except Exception as exc:  # noqa: BLE001
            print(f"skipping {name}: {type(exc).__name__}: {exc}")
            continue
        methods.append((name, fn, size))
        paths.append((name, path))
    if not methods:
        raise SystemExit("nothing to benchmark")

    import torch

    # A fast wrong answer is not a result. Re-place the probe set through each
    # loaded model and check it reproduces what that run already wrote out.
    if args.verify and Path(args.verify).is_file():
        P = np.load(args.verify).astype(np.float32)
        print("verifying each loaded model reproduces its own saved Z_probe.npy")
        for name, fn, _ in methods:
            run = Path(dict(paths)[name])
            got = fn(P)
            got = np.asarray(got[0] if isinstance(got, tuple) else got, dtype=np.float64)
            ref_f = run / "Z_probe.npy"
            if not ref_f.is_file():
                print(f"  {name:<10} no reference on disk; finite={np.isfinite(got).all()}")
                continue
            ref = np.load(ref_f).astype(np.float64)
            dev = float(np.abs(got - ref).max())
            scale = float(np.abs(ref).max()) or 1.0
            print(
                f"  {name:<10} finite={np.isfinite(got).all()}  "
                f"max |delta| vs saved = {dev:.2e} ({dev / scale:.1e} of range)"
            )

    print(f"\nCPU, torch threads = {torch.get_num_threads()}, "
          f"repeats = {args.repeats} (median reported), n_features = {X.shape[1]}")
    print(f"\n{'method':<12}{'model on disk':>15}" + "".join(f"{f'B={b}':>16}" for b in BATCHES))
    print("-" * (27 + 16 * len(BATCHES)))

    results: dict = {"batches": list(BATCHES), "methods": {}}
    per_point: dict = {}
    for name, fn, size in methods:
        cells, rec = [], {}
        for b in BATCHES:
            P = X[rng.choice(len(X), size=b, replace=False)].astype(np.float32)
            med, best = _time(lambda P=P: fn(P), args.repeats)
            rec[b] = {"median_s": med, "min_s": best, "per_point_us": med / b * 1e6}
            cells.append(f"{med * 1e3:>9.2f} ms" + f"{'':>3}")
        per_point[name] = rec[max(BATCHES)]["per_point_us"]
        results["methods"][name] = {"model_bytes": size, "timings": rec}
        print(f"{name:<12}{size / 1024:>12.0f} KB" + "".join(cells))

    print(f"\n{'method':<12}{'us / point at B=1':>20}{'us / point at B=512':>22}{'speedup vs slowest':>21}")
    slowest = max(per_point.values())
    for name, _, _ in methods:
        r = results["methods"][name]["timings"]
        print(
            f"{name:<12}{r[1]['per_point_us']:>20.1f}{r[max(BATCHES)]['per_point_us']:>22.2f}"
            f"{slowest / per_point[name]:>20.1f}x"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

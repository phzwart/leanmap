#!/usr/bin/env python
"""Illustrative: CatBoost depth needed in ambient X vs embedding Z.

Not part of the paper scorecard. Idea: a good embedding absorbs the nonlinear
bulk, so a shallow tree on Z should match a deeper tree on X.

  python examples/exploratory/catboost_complexity.py \\
    --X examples/exploratory/data/digits_X.npy \\
    --y examples/exploratory/data/digits_y.npy \\
    --run leanmap=examples/out/exploratory/paper_digits/recommended__default__seed0 \\
    --run umap=examples/out/exploratory/paper_digits/reference/umap_default__none__seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
_ROOT = _EXAMPLES.parent
for p in (_ROOT / "src", _EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


DEPTHS = (1, 2, 3, 4, 6)


def _load_split(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    split = np.load(run_dir / "split.npz")
    return split["train_idx"], split["hold_idx"]


def _load_Z(run_dir: Path) -> np.ndarray:
    z_path = run_dir / "Z.npy"
    if z_path.is_file():
        return np.load(z_path).astype(np.float32)
    # sklearn reference folders may store under different names
    for alt in ("embedding.npy", "Z_umap.npy", "Z_pca.npy"):
        if (run_dir / alt).is_file():
            return np.load(run_dir / alt).astype(np.float32)
    raise FileNotFoundError(f"no embedding under {run_dir}")


def catboost_acc(
    Feat: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    depth: int,
    seed: int,
    iterations: int,
) -> float:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        depth=int(depth),
        iterations=int(iterations),
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Feat[train_idx], y[train_idx])
    pred = model.predict(Feat[hold_idx]).reshape(-1)
    return float((pred == y[hold_idx]).mean())


def depth_curve(
    Feat: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    seed: int,
    iterations: int,
    depths: Tuple[int, ...] = DEPTHS,
) -> Dict[str, float]:
    return {
        f"d{d}": catboost_acc(
            Feat, y, train_idx, hold_idx, depth=d, seed=seed, iterations=iterations
        )
        for d in depths
    }


def min_depth_at(
    curve: Dict[str, float], *, target: float, depths: Tuple[int, ...] = DEPTHS
) -> Optional[int]:
    for d in depths:
        if curve[f"d{d}"] >= target:
            return int(d)
    return None


def evaluate_run(
    name: str,
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    iterations: int,
    ambient_curve: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    train_idx, hold_idx = _load_split(run_dir)
    Z = _load_Z(run_dir)
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"{name}: Z N={Z.shape[0]} != X N={X.shape[0]}")

    if ambient_curve is None:
        ambient_curve = depth_curve(
            X, y, train_idx, hold_idx, seed=seed, iterations=iterations
        )
    z_curve = depth_curve(Z, y, train_idx, hold_idx, seed=seed, iterations=iterations)

    # Target = ambient's best depth accuracy (what X can do at full budget).
    amb_best = max(ambient_curve.values())
    c_x = min_depth_at(ambient_curve, target=0.95 * amb_best)
    c_z = min_depth_at(z_curve, target=0.95 * amb_best)
    # Also: depth needed to match ambient's depth-1 accuracy (polish bar).
    amb_d1 = ambient_curve["d1"]
    c_z_vs_amb_d1 = min_depth_at(z_curve, target=amb_d1)

    return {
        "name": name,
        "run_dir": str(run_dir),
        "n_train": int(len(train_idx)),
        "n_hold": int(len(hold_idx)),
        "ambient": ambient_curve,
        "embedding": z_curve,
        "ambient_best": amb_best,
        "C_X_at_95pct_best": c_x,
        "C_Z_at_95pct_ambient_best": c_z,
        "delta_C": (None if c_x is None or c_z is None else int(c_x) - int(c_z)),
        "C_Z_to_match_ambient_d1": c_z_vs_amb_d1,
        "gap_at_d1": float(z_curve["d1"] - ambient_curve["d1"]),
        "gap_at_d6": float(z_curve["d6"] - ambient_curve["d6"]),
    }


def _parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        p = Path(spec)
        return p.name, p
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", type=Path, required=True)
    ap.add_argument("--y", type=Path, required=True)
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="name=path to a run dir with Z.npy + split.npz (repeatable)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="write JSON summary")
    args = ap.parse_args(argv)

    if not args.run:
        ap.error("pass at least one --run name=path")

    X = np.load(args.X).astype(np.float32)
    y = np.load(args.y).astype(np.int64).reshape(-1)
    if len(X) != len(y):
        raise SystemExit(f"X N={len(X)} != y N={len(y)}")

    runs = [_parse_run(s) for s in args.run]
    # Ambient curve once per distinct split (usually shared seed0).
    ambient_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []

    for name, run_dir in runs:
        train_idx, hold_idx = _load_split(run_dir)
        key = (tuple(train_idx.tolist()), tuple(hold_idx.tolist()))
        if key not in ambient_cache:
            print(f"ambient CatBoost ladder on split from {run_dir.name} …", flush=True)
            ambient_cache[key] = depth_curve(
                X,
                y,
                train_idx,
                hold_idx,
                seed=args.seed,
                iterations=args.iterations,
            )
        print(f"embedding CatBoost ladder: {name} …", flush=True)
        rows.append(
            evaluate_run(
                name,
                run_dir,
                X,
                y,
                seed=args.seed,
                iterations=args.iterations,
                ambient_curve=ambient_cache[key],
            )
        )

    # Compact table
    depths = DEPTHS
    hdr = (
        f"{'name':12s}  "
        + " ".join(f"X@d{d}" for d in depths)
        + " | "
        + " ".join(f"Z@d{d}" for d in depths)
        + " | Cx Cz ΔC  Z-X@d1"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        xa = r["ambient"]
        za = r["embedding"]
        xs = " ".join(f"{xa[f'd{d}']:5.3f}" for d in depths)
        zs = " ".join(f"{za[f'd{d}']:5.3f}" for d in depths)
        cx = r["C_X_at_95pct_best"]
        cz = r["C_Z_at_95pct_ambient_best"]
        dc = r["delta_C"]
        print(
            f"{r['name']:12s}  {xs} | {zs} | "
            f"{str(cx):>2} {str(cz):>2} {str(dc):>2}  {r['gap_at_d1']:+.3f}"
        )

    payload = {
        "X": str(args.X),
        "y": str(args.y),
        "iterations": args.iterations,
        "seed": args.seed,
        "depths": list(depths),
        "runs": rows,
    }
    out = args.out
    if out is None and len(runs) == 1:
        out = runs[0][1].parent / "catboost_complexity.json"
    elif out is None:
        out = runs[0][1].parent / "catboost_complexity.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

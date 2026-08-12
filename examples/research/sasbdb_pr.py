#!/usr/bin/env python
"""leanmap on SASBDB P(r) pair-distance distributions.

Each row is a pair-distance distribution resampled onto 100 bins spanning
``r`` in ``[0, dmax]``, so bin index is already a relative coordinate ``r/dmax``.
Default ``--normalize unit-sum`` makes each profile a distribution over the
relative-r bins (a scale-free shape descriptor).

Data is **not bundled**. Point ``--parquet`` at a SASBDB ``pr_profiles.parquet``
(or set ``LEANMAP_SASBDB_PARQUET``). Outputs land under ``examples/out/research/sasbdb_pr/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
_ROOT = _EXAMPLES.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _demo import fit_embed, save_density, save_scatter, save_shepard  # noqa: E402

META_COLS = ("sasbdb_code", "dmax", "rg_pr", "rg_guinier", "length_unit")
DEFAULT_OUT = _EXAMPLES / "out" / "research" / "sasbdb_pr"


def default_parquet() -> Path | None:
    env = os.environ.get("LEANMAP_SASBDB_PARQUET")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.home() / "Projects" / "SASDBD" / "data" / "catalog" / "pr_profiles.parquet",
        Path.home() / "Projects" / "sasdbd" / "data" / "catalog" / "pr_profiles.parquet",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_profiles(parquet: Path, column: str):
    import pyarrow.parquet as pq

    df = pq.read_table(parquet).to_pandas()
    P = np.stack(df[column].to_numpy()).astype(np.float64)
    return P, df


def normalize(P: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return P
    if mode == "unit-sum":
        return P / P.sum(axis=1, keepdims=True)
    if mode == "unit-max":
        return P / P.max(axis=1, keepdims=True)
    if mode == "unit-l2":
        return P / np.linalg.norm(P, axis=1, keepdims=True)
    raise ValueError(f"unknown normalize mode {mode!r}")


def quality_mask(P: np.ndarray, df) -> np.ndarray:
    """Drop profiles that cannot be a physical P(r)."""
    ok = np.isfinite(P).all(axis=1) & (P.sum(axis=1) > 0)
    ok &= P.min(axis=1) >= -1e-12
    ratio = df["rg_pr"].to_numpy(dtype=np.float64) / df["dmax"].to_numpy(dtype=np.float64)
    ok &= np.isfinite(ratio) & (ratio > 0.1) & (ratio < 0.6)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="path to pr_profiles.parquet (or set LEANMAP_SASBDB_PARQUET)",
    )
    ap.add_argument("--column", default="pr_norm", choices=("pr", "pr_norm"))
    ap.add_argument(
        "--normalize",
        default="unit-sum",
        choices=("unit-sum", "unit-max", "unit-l2", "raw"),
    )
    # pr_norm rows are discrete distributions on equal-width relative-r bins;
    # 1-D Wasserstein-1 (CDF L1) is the natural transport metric on that line.
    # L1 remains available as total-variation (up to a factor 2).
    ap.add_argument(
        "--metric",
        default="wasserstein1d",
        choices=(
            "wasserstein1d",
            "l1",
            "l2",
            "jensenshannon",
            "cosine",
            "correlation",
            "braycurtis",
        ),
    )
    ap.add_argument("--tau-scale", type=float, default=None)
    ap.add_argument("--learn-tau", dest="learn_tau", action="store_true", default=None)
    ap.add_argument("--no-learn-tau", dest="learn_tau", action="store_false")
    ap.add_argument(
        "--learn-landmarks", dest="learn_landmarks", action="store_true", default=None
    )
    ap.add_argument("--no-learn-landmarks", dest="learn_landmarks", action="store_false")
    ap.add_argument("--lambda-density", type=float, default=1.0)
    ap.add_argument("--n-landmarks", type=int, default=None)
    ap.add_argument("--min-dist", type=float, default=None, dest="min_dist")
    ap.add_argument("--lambda-geo", type=float, default=None, dest="lambda_geo")
    ap.add_argument("--n", type=int, default=0, help="random subsample size (0 = all)")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    parquet = args.parquet or default_parquet()
    if parquet is None or not Path(parquet).is_file():
        raise SystemExit(
            "SASBDB parquet not found. Pass --parquet PATH or set "
            "LEANMAP_SASBDB_PARQUET to pr_profiles.parquet "
            "(see examples/research/README.md)."
        )
    parquet = Path(parquet)

    try:
        P_all, df = load_profiles(parquet, args.column)
    except ImportError as exc:
        raise SystemExit(
            "Reading the parquet catalog requires pyarrow "
            f"(pip install pyarrow). Import failed: {exc}"
        ) from exc

    n_total = len(P_all)
    if args.no_filter:
        keep = np.isfinite(P_all).all(axis=1) & (P_all.sum(axis=1) > 0)
    else:
        keep = quality_mask(P_all, df)
    P_all, df = P_all[keep], df.loc[keep].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    n = len(P_all) if args.n <= 0 else min(int(args.n), len(P_all))
    idx = (
        rng.choice(len(P_all), size=n, replace=False)
        if n < len(P_all)
        else np.arange(n)
    )
    idx.sort()
    X = normalize(P_all[idx], args.normalize).astype(np.float32)
    meta = df.loc[idx, [c for c in META_COLS if c in df.columns]].reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "meta.csv", index=False)

    dmax = meta["dmax"].to_numpy(dtype=np.float64)
    rg = meta["rg_pr"].to_numpy(dtype=np.float64)
    bins = np.linspace(0.0, 1.0, X.shape[1])
    w = X / X.sum(axis=1, keepdims=True)
    mode_pos = bins[np.argmax(X, axis=1)]
    mean_pos = w @ bins
    skew = w @ (bins**3) - 3 * mean_pos * (w @ (bins**2)) + 2 * mean_pos**3
    panels = [
        ("dmax", np.log10(dmax), "log10 Dmax", "viridis"),
        ("rg_over_dmax", rg / dmax, "Rg / Dmax", "coolwarm"),
        ("mode_pos", mode_pos, "peak position r/Dmax", "plasma"),
        ("skew", skew, "P(r) skewness (relative r)", "cividis"),
    ]

    print(
        f"{parquet.name}: {n_total} rows -> {len(P_all)} pass filter "
        f"(dropped {n_total - len(P_all)}) -> fitting {n} x {X.shape[1]} "
        f"[{args.column}, {args.normalize}, metric={args.metric}]\n"
        f"writing to {out}",
        flush=True,
    )

    result, Z, score = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        metric=args.metric,
        tau_scale=args.tau_scale,
        learn_tau=args.learn_tau,
        learn_landmarks=args.learn_landmarks,
        lambda_density=args.lambda_density,
        min_dist=args.min_dist,
        lambda_geo=args.lambda_geo,
        **({} if args.n_landmarks is None else {"n_landmarks": args.n_landmarks}),
    )
    del result

    np.save(out / "Z.npy", Z)
    for name, color, label, cmap in panels:
        save_scatter(
            Z,
            color,
            title=f"leanmap — SASBDB P(r), N={n} ({label})",
            path=out / f"scatter_{name}.png",
            cmap=cmap,
            colorbar_label=label,
        )
    save_density(
        Z, title=f"leanmap — SASBDB P(r) density, N={n}", path=out / "density.png"
    )

    from leanmap.evaluate import shepard_pairs_ambient
    from leanmap.metrics import get_metric

    d_orig, d_embed = shepard_pairs_ambient(
        X, Z, n_pairs=32768, seed=args.seed, dist_fn=get_metric(args.metric).fn
    )
    save_shepard(
        d_orig,
        d_embed,
        title=f"Shepard (ambient {args.metric}) — SASBDB P(r), N={n}",
        path=out / "shepard_ambient.png",
    )

    metrics = {
        "n": n,
        "n_rows": n_total,
        "n_filtered": int(n_total - len(P_all)),
        "d_in": int(X.shape[1]),
        "column": args.column,
        "normalize": args.normalize,
        "metric": args.metric,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "score_mean": float(np.mean(score)),
        "score_p95": float(np.percentile(score, 95)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

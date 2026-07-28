#!/usr/bin/env python
"""UMAP baseline on SASBDB P(r) profiles, for comparison against leanmap.

Shares ``sasbdb_pr``'s loader, sanity filter, normalization and subsampling so
both methods see byte-identical input, and reports the same metric battery.
Neighbour count, min_dist and the ambient metric are matched to the leanmap
defaults so the comparison is about the embedding, not the hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
_ROOT = _EXAMPLES.parent
for _p in (_EXAMPLES, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics_run import compute_metrics, write_json  # noqa: E402
from sasbdb_pr import (  # noqa: E402
    DEFAULT_PARQUET,
    META_COLS,
    daemonize,
    load_profiles,
    normalize,
    quality_mask,
)

from _demo import save_density, save_scatter, save_shepard  # noqa: E402

DEFAULT_OUT = _ROOT / "runs" / "sasbdb_pr_umap"
# leanmap metric name -> UMAP metric name.
METRICS = {
    "l1": "manhattan",
    "l2": "euclidean",
    "cosine": "cosine",
    "correlation": "correlation",
    "braycurtis": "braycurtis",
    "jensenshannon": "hellinger",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--column", default="pr", choices=("pr", "pr_norm"))
    ap.add_argument(
        "--normalize",
        default="unit-sum",
        choices=("unit-sum", "unit-max", "unit-l2", "raw"),
    )
    ap.add_argument("--metric", default="l1", choices=tuple(METRICS))
    ap.add_argument("--n", type=int, default=0, help="random subsample size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    # Matched to PLANEConfig defaults so the two maps are comparable.
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.5)
    ap.add_argument("--umap-epochs", type=int, default=None, help="UMAP n_epochs")
    # Plain UMAP flattens local density ~6x here (fit slope 0.157, Spearman
    # 0.246 vs ambient), so its voids are largely optimizer texture. densMAP
    # adds the density-preserving term.
    ap.add_argument("--densmap", action="store_true")
    ap.add_argument("--dens-lambda", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()

    if args.detach:
        out = Path(args.out)
        print(f"detaching; log -> {out.with_suffix('.log')}")
        daemonize(out.with_suffix(".log"), out.with_suffix(".pid"))

    P_all, df = load_profiles(args.parquet, args.column)
    n_total = len(P_all)
    if args.no_filter:
        keep = np.isfinite(P_all).all(axis=1) & (P_all.sum(axis=1) > 0)
    else:
        keep = quality_mask(P_all, df)
    P_all, df = P_all[keep], df.loc[keep].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    n = len(P_all) if args.n <= 0 else min(int(args.n), len(P_all))
    idx = rng.choice(len(P_all), size=n, replace=False) if n < len(P_all) else np.arange(n)
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

    umap_metric = METRICS[args.metric]
    print(
        f"{args.parquet.name}: {n_total} rows -> {len(P_all)} pass filter "
        f"-> UMAP on {n} x {X.shape[1]} "
        f"[{args.column}, {args.normalize}, metric={umap_metric}, "
        f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist}]\n"
        f"writing to {out}",
        flush=True,
    )

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=umap_metric,
        n_epochs=args.umap_epochs,
        random_state=args.seed,
        densmap=args.densmap,
        dens_lambda=args.dens_lambda,
        verbose=True,
    )
    t0 = time.perf_counter()
    Z = reducer.fit_transform(X).astype(np.float32)
    fit_seconds = time.perf_counter() - t0
    print(f"UMAP fit took {fit_seconds:.1f}s", flush=True)

    np.save(out / "Z.npy", Z)
    np.save(out / "X.npy", X)
    for name, color, label, cmap in panels:
        save_scatter(
            Z,
            color,
            title=f"UMAP — SASBDB P(r), N={n} ({label})",
            path=out / f"scatter_{name}.png",
            cmap=cmap,
            colorbar_label=label,
        )
    save_density(Z, title=f"UMAP — SASBDB P(r) density, N={n}", path=out / "density.png")

    from leanmap.evaluate import shepard_pairs_ambient

    d_orig, d_embed = shepard_pairs_ambient(X, Z, n_pairs=32768, seed=args.seed)
    save_shepard(
        d_orig,
        d_embed,
        title=f"Shepard (ambient L2) — UMAP P(r), N={n}",
        path=out / "shepard_ambient.png",
    )

    from leanmap.evaluate import density_correspondence

    dens = density_correspondence(X, Z, k=15)
    metrics = compute_metrics(X, Z, seed=args.seed)
    metrics.update(
        method="densmap" if args.densmap else "umap",
        densmap=bool(args.densmap),
        dens_spearman=float(dens["spearman"]),
        dens_pearson_log=float(dens["pearson_log"]),
        dens_fit_slope=float(dens["fit_slope"]),
        n=n,
        n_rows=n_total,
        n_filtered=int(n_total - len(P_all)),
        d_in=int(X.shape[1]),
        column=args.column,
        normalize=args.normalize,
        metric=umap_metric,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        seed=int(args.seed),
        fit_seconds=fit_seconds,
        spearman_dim1_logdmax=float(
            np.corrcoef(
                np.argsort(np.argsort(Z[:, 0])), np.argsort(np.argsort(np.log10(dmax)))
            )[0, 1]
        ),
    )
    write_json(out / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

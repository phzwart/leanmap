#!/usr/bin/env python
"""Master driver: ingest arrays -> axis sweep -> visual / metric artifacts.

Scores every run on the label + geometry + artifact battery, optionally against a
matched null and out of sample. Three defaults matter:

- ``--null`` refits the identical configuration on structureless data. Metric
  chance levels depend on the config, not just the data, so a null is only valid
  for the config it was run with.
- ``--holdout`` scores points the model never trained on. leanmap is parametric,
  and at small N in-sample metrics flatter the embedding.
- ``--seeds`` repeats runs so a difference can be compared against run-to-run
  spread before being believed.

Researcher-style usage::

    python examples/exploratory/master.py \\
      --X examples/exploratory/data/digits_X.npy \\
      --y examples/exploratory/data/digits_y.npy \\
      --name digits --sweep umap_match --holdout 0.2 --seeds 0 1 2 --null shuffle
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# examples/ on sys.path so `_demo` and local exploratory modules resolve.
_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from axes import BASELINE, list_sweeps, merged_overlay, resolve_runs  # noqa: E402
from calibrate import calibrate_tau_scale, predict_pyramid_levels  # noqa: E402
from ingest import default_run_name, ingest  # noqa: E402
from metrics_run import full_battery, read_json, write_json  # noqa: E402
from nulls import describe as describe_null  # noqa: E402
from nulls import make_null  # noqa: E402
from splits import save_split, split_indices  # noqa: E402

from _demo import fit_embed, save_scatter, save_shepard  # noqa: E402

DEFAULT_OUT = _EXAMPLES / "out" / "exploratory"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="leanmap exploratory sweep (generic array ingest)",
    )
    ap.add_argument("--X", required=True, help="feature matrix (N, D): .npy/.npz/.csv")
    ap.add_argument(
        "--color",
        "--y",
        dest="color",
        default=None,
        help="length-N color / label vector (used as labels when integral)",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="explicit label vector for label metrics (defaults to --y if integral)",
    )
    ap.add_argument("--name", default=None, help="run-tree tag under --out")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sweep", default="phase1", choices=list_sweeps())
    ap.add_argument("--only", default=None, help="restrict to one axis name or run_id")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--colorbar-label", default="")
    ap.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="fraction held out of training and scored out of sample (0 = in-sample)",
    )
    ap.add_argument(
        "--null",
        default="none",
        choices=("none", "shuffle", "gauss"),
        help="also refit each config on a matched null and score it",
    )
    ap.add_argument(
        "--target-perp",
        type=float,
        default=None,
        help="derive tau_scale per run for this median affinity perplexity",
    )
    ap.add_argument(
        "--shepard",
        choices=("ambient", "geodesic", "both", "none"),
        default="both",
    )
    ap.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="log layout uniformity every N epochs (0 = off)",
    )
    ap.add_argument("--force", action="store_true", help="re-run even if metrics exist")
    ap.add_argument("--dry-run", action="store_true", help="print planned runs, exit")
    ap.add_argument("--atlas", action="store_true", help="rebuild summary.csv + atlas.png")
    ap.add_argument(
        "--bar",
        type=Path,
        default=None,
        help="bar.json from reference.py; prints the gap to the best reference method",
    )
    ap.add_argument(
        "--probes",
        default=None,
        help="structured probe array (M, D) .npy; embedded out of sample, never trained on",
    )
    ap.add_argument(
        "--emd",
        default=None,
        help="EMD reference matrix .npy; adds EMD-referenced metrics to the battery",
    )
    return ap.parse_args(argv)


def _tag(run_id: str, *, null: str, seed: int, seeds: list) -> str:
    tag = run_id
    if null != "none":
        tag += f"__null-{null}"
    if len(seeds) > 1:
        tag += f"__seed{seed}"
    return tag


def _fit_kwargs(overlay: dict, *, epochs: int, seed: int, device) -> dict:
    kw = dict(overlay)
    # An overlay may pin epochs so a sweep can vary training length as an axis;
    # otherwise --epochs applies.
    kw["epochs"] = int(overlay.get("epochs", epochs))
    kw["seed"] = int(seed)
    if device is not None:
        kw["device"] = device
    return kw


def _config_payload(overlay: dict, *, epochs: int, seed: int, device) -> dict:
    payload = dict(BASELINE)
    payload.update(overlay)
    payload["epochs"] = int(overlay.get("epochs", epochs))
    payload["seed"] = int(seed)
    payload["device"] = device
    for k, v in list(payload.items()):
        if isinstance(v, tuple):
            payload[k] = list(v)
    return payload


def run_one(
    X: np.ndarray,
    color,
    *,
    run,
    out_dir: Path,
    epochs: int,
    seed: int,
    device,
    cmap: str,
    colorbar_label: str,
    shepard_mode: str,
    force: bool,
    labels: Optional[np.ndarray] = None,
    null: str = "none",
    holdout: float = 0.0,
    target_perp: Optional[float] = None,
    monitor: int = 0,
    probes: Optional[np.ndarray] = None,
    emd_cache: Optional[str] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        print(f"skip {out_dir.name} (exists; pass --force to redo)", flush=True)
        return read_json(metrics_path)

    X_fit = make_null(X, null, seed=seed)
    overlay = merged_overlay(run)

    # Derive tau_scale from the anchor geometry rather than trusting a literal.
    if target_perp is not None:
        L = int(overlay.get("n_landmarks", BASELINE["n_landmarks"]))
        overlay["tau_scale"] = calibrate_tau_scale(
            X_fit, L, target_perp=target_perp, seed=seed
        )
        overlay["learn_tau"] = False

    n = len(X_fit)
    train_idx, hold_idx = split_indices(n, holdout, seed)
    save_split(out_dir, train_idx, hold_idx, holdout=holdout, seed=seed)

    write_json(
        out_dir / "config.json",
        {
            "run_id": run.run_id,
            "axis": run.axis,
            "level": run.level,
            "null": null,
            "seed": int(seed),
            "holdout": float(holdout),
            "target_perp": target_perp,
            "n_train": int(len(train_idx)),
            "n_eval": int(len(hold_idx)),
            "overlay": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in run.overlay.items()
            },
            "config": _config_payload(overlay, epochs=epochs, seed=seed, device=device),
        },
    )

    print(
        f"fit {out_dir.name}: axis={run.axis} level={run.level} "
        f"null={null} seed={seed} N_train={len(train_idx)} D={X_fit.shape[1]} "
        f"epochs={overlay.get('epochs', epochs)}",
        flush=True,
    )
    callbacks = None
    trace_csv = out_dir / "uniformity_trace.csv"
    if monitor > 0:
        from monitor import uniformity_monitor

        callbacks = [
            uniformity_monitor(X_fit[train_idx], trace_csv, every=int(monitor), seed=seed)
        ]

    t0 = time.perf_counter()
    result, _Z_train, _score = fit_embed(
        X_fit[train_idx],
        callbacks=callbacks,
        **_fit_kwargs(overlay, epochs=epochs, seed=seed, device=device),
    )
    wall = time.perf_counter() - t0

    if monitor > 0:
        from monitor import plot_trace

        plot_trace(trace_csv, out_dir / "uniformity_trace.png", title=out_dir.name)

    import torch

    with torch.no_grad():
        Z_all, cover_all = result.embed(X_fit)
    Z_all = Z_all.detach().cpu().numpy()
    np.save(out_dir / "cover.npy", cover_all.detach().cpu().numpy().astype(np.float32))
    # The model is what lets a later script place *new* points -- structured
    # probes, a fresh holdout -- without paying for a refit.
    try:
        result.save(out_dir / "model.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"  model save failed: {exc}", flush=True)

    if probes is not None and len(probes):
        with torch.no_grad():
            Z_pr, cover_pr = result.embed(
                torch.as_tensor(np.asarray(probes, dtype=np.float32))
            )
        np.save(out_dir / "Z_probe.npy", Z_pr.detach().cpu().numpy().astype(np.float32))
        np.save(
            out_dir / "probe_cover.npy",
            cover_pr.detach().cpu().numpy().astype(np.float32),
        )

    color_vec = color if color is not None else np.zeros(n, dtype=np.float64)
    save_scatter(
        Z_all,
        color_vec,
        title=f"{run.axis} = {run.level}" + ("" if null == "none" else f"  [null:{null}]"),
        path=out_dir / "scatter.png",
        cmap=cmap,
        colorbar_label=colorbar_label,
    )

    n_neighbors = int(overlay.get("n_neighbors", BASELINE["n_neighbors"]))
    if shepard_mode != "none":
        from metrics_run import shepard_arrays

        modes = ["ambient", "geodesic"] if shepard_mode == "both" else [shepard_mode]
        for mode in modes:
            try:
                d_o, d_e, xlabel = shepard_arrays(
                    X_fit, Z_all, mode=mode, n_neighbors=n_neighbors, seed=seed
                )
                if d_o.size:
                    save_shepard(
                        d_o,
                        d_e,
                        title=f"Shepard ({mode}) - {out_dir.name}",
                        path=out_dir / f"shepard_{mode}.png",
                        xlabel=xlabel,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  shepard_{mode} failed: {exc}", flush=True)

    # Score out of sample when a holdout was reserved.
    y_eval = None if labels is None else np.asarray(labels)[hold_idx]
    metrics = full_battery(
        X_fit[hold_idx],
        Z_all[hold_idx],
        y=y_eval,
        n_neighbors=n_neighbors,
        seed=seed,
        emd_cache=emd_cache,
        emd_rows=hold_idx,
    )
    metrics.update(
        {
            "run_id": run.run_id,
            "axis": run.axis,
            "level": run.level,
            "null": null,
            "seed": int(seed),
            "holdout": float(holdout),
            "wall_s": wall,
            "N": int(n),
            "n_train": int(len(train_idx)),
            "n_eval": int(len(hold_idx)),
            "D": int(X_fit.shape[1]),
            "epochs": int(overlay.get("epochs", epochs)),
            "tau_scale": float(overlay.get("tau_scale", float("nan"))),
            "pyramid_scales": int(result.config.pyramid_scales),
            "pyramid_level_weights": list(result.config.pyramid_level_weights or []),
            "pyramid_coarse_backbone": float(result.config.pyramid_coarse_backbone),
            "n_landmarks": int(result.config.n_landmarks),
            "lambda_geo": float(result.config.lambda_geo),
            "lambda_frame": float(result.config.lambda_frame),
        }
    )
    write_json(metrics_path, metrics)
    np.save(out_dir / "Z.npy", Z_all.astype(np.float32))
    print(
        f"  done {out_dir.name}: wall={wall:.0f}s "
        f"acc_Z={metrics.get('label_acc_Z', float('nan')):.3f} "
        f"ARI={metrics.get('label_ari', float('nan')):.3f} "
        f"trust15={metrics.get('trust_15', float('nan')):.3f} "
        f"ov15={metrics.get('knn_overlap_15', float('nan')):.3f}",
        flush=True,
    )
    return metrics


def write_summary(name_dir: Path) -> Path:
    """Collect metrics.json rows into summary.csv."""
    rows = []
    for metrics_path in sorted(name_dir.glob("*/metrics.json")):
        m = read_json(metrics_path)
        m["path"] = str(metrics_path.parent.relative_to(name_dir))
        rows.append(m)
    out = name_dir / "summary.csv"
    if not rows:
        out.write_text("")
        return out
    prefer = [
        "run_id",
        "axis",
        "level",
        "null",
        "seed",
        "label_acc_Z",
        "label_acc_X",
        "label_ari",
        "emd_spearman",
        "emd_spearman_global",
        "emd_knn_overlap_15",
        "label_sil_Z",
        "trust_15",
        "cont_15",
        "knn_overlap_15",
        "ambient_spearman",
        "geodesic_spearman",
        "density_spearman",
        "spacing_cv",
        "area_sd",
        "kmeans_sil_Z",
        "kmeans_sil_X",
        "kmeans_sil_floor_2d",
        "wall_s",
        "N",
        "D",
        "epochs",
        "path",
    ]
    keys = [k for k in prefer if any(k in r for r in rows)]
    for k in sorted({kk for r in rows for kk in r}):
        if k not in keys:
            keys.append(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return out


def report_gap(rows: list, bar_path: Path) -> None:
    """Print the best run against the best reference method."""
    if not bar_path.is_file():
        print(f"(no bar at {bar_path}; skipping gap report)")
        return
    bar = read_json(bar_path)
    ref = [r for r in bar.get("rows", []) if r.get("null") == "none"]
    real = [r for r in rows if r.get("null") == "none"]
    if not ref or not real or "label_acc_Z" not in ref[0]:
        return
    best_ref = max(ref, key=lambda r: r.get("label_acc_Z", 0.0))
    best_run = max(real, key=lambda r: r.get("label_acc_Z", 0.0))
    keys = [
        ("label_acc_Z", "label 5NN acc"),
        ("label_ari", "ARI vs truth"),
        ("label_sil_Z", "silhouette of labels"),
        ("trust_15", "trustworthiness@15"),
        ("knn_overlap_15", "kNN overlap@15"),
        ("geodesic_spearman", "geodesic rho"),
        ("density_spearman", "density rho"),
    ]
    print(f"\nGAP TO BAR ({best_ref['method']}) - best leanmap run: {best_run['run_id']}")
    print(f"{'metric':24s} {'leanmap':>9} {'bar':>9} {'delta':>9}")
    print("-" * 55)
    for k, lbl in keys:
        a, b = best_run.get(k), best_ref.get(k)
        if a is None or b is None:
            continue
        print(f"{lbl:24s} {a:>9.3f} {b:>9.3f} {a - b:>+9.3f}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    X, color = ingest(args.X, args.color)
    labels = None
    if args.labels:
        labels = np.load(args.labels)
    elif color is not None and np.issubdtype(np.asarray(color).dtype, np.integer):
        labels = np.asarray(color)
    name = default_run_name(args.X, args.name)
    runs = resolve_runs(args.sweep, only_axis=args.only)
    if not runs:
        print(f"no runs matched sweep={args.sweep!r} only={args.only!r}", file=sys.stderr)
        return 2

    probes = np.load(args.probes).astype(np.float32) if args.probes else None
    if probes is not None:
        print(f"probes: {len(probes)} structured outliers, embedded but never trained on")

    nulls = ["none"] if args.null == "none" else ["none", args.null]
    total = len(runs) * len(nulls) * len(args.seeds)
    n_lvl = predict_pyramid_levels(
        int(len(X) * (1.0 - args.holdout)),
        int(BASELINE.get("pyramid_scales", 3)),
    )
    print(
        f"name={name} N={len(X)} D={X.shape[1]} sweep={args.sweep} "
        f"runs={len(runs)} x nulls={len(nulls)} x seeds={len(args.seeds)} = {total} fits"
    )
    print(f"labels: {'yes' if labels is not None else 'no'}  holdout={args.holdout:.0%}")
    print(f"pyramid levels expected at this N: {n_lvl} (weight tuples should have {n_lvl})")
    for k in nulls:
        print(f"  null '{k}': {describe_null(k)}")
    if args.dry_run:
        for r in runs:
            delta = {k: v for k, v in r.overlay.items()}
            print(f"  {r.run_id:44s} axis={r.axis:24s} delta={delta or '{}'}")
        return 0

    name_dir = args.out / name
    name_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        name_dir / "ingest.json",
        {
            "X": str(Path(args.X).resolve()),
            "color": str(Path(args.color).resolve()) if args.color else None,
            "name": name,
            "N": int(len(X)),
            "D": int(X.shape[1]),
            "sweep": args.sweep,
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "holdout": args.holdout,
            "null": args.null,
            "target_perp": args.target_perp,
            "monitor": args.monitor,
            "probes": str(Path(args.probes).resolve()) if args.probes else None,
            "emd": str(Path(args.emd).resolve()) if args.emd else None,
        },
    )

    rows = []
    for run in runs:
        for null in nulls:
            for seed in args.seeds:
                tag = _tag(run.run_id, null=null, seed=seed, seeds=list(args.seeds))
                rows.append(
                    run_one(
                        X,
                        color,
                        run=run,
                        out_dir=name_dir / tag,
                        epochs=args.epochs,
                        seed=seed,
                        device=args.device,
                        cmap=args.cmap,
                        colorbar_label=args.colorbar_label,
                        shepard_mode=args.shepard,
                        force=args.force,
                        labels=labels,
                        null=null,
                        holdout=args.holdout,
                        target_perp=args.target_perp,
                        monitor=args.monitor,
                        probes=probes,
                        emd_cache=args.emd,
                    )
                )

    summary = write_summary(name_dir)
    print(f"wrote {summary}")
    if args.bar is not None:
        report_gap(rows, args.bar)
    elif (name_dir / "bar.json").is_file():
        report_gap(rows, name_dir / "bar.json")
    if args.atlas:
        from make_atlas import build_atlas

        print(f"wrote {build_atlas(name_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

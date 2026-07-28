#!/usr/bin/env python
"""Seeded regression baseline across the reference datasets.

Runs a fixed arm set at several seeds and aggregates each metric to mean +/- sd,
so a later change can be compared against run-to-run spread rather than against a
single number. Digits 5-NN alone carries a seed sd of 0.016-0.024, which is the
same size as several differences that have been read as real.

The same command reproduces the snapshot after a code change; point ``--tag`` at
a different directory and diff with ``compare``::

    python examples/exploratory/baseline_capture.py run --tag pre_review
    # ... change code ...
    python examples/exploratory/baseline_capture.py run --tag post_pr1
    python examples/exploratory/baseline_capture.py compare pre_review post_pr1

``--arms`` selects the arm set. ``recommended`` (default) is the shipping recipe
only, which is what a regression check needs; ``ladder`` adds the min_dist /
weights / geo arms used by the ablation stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_EXAMPLES, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from axes import RECOMMENDED, RunSpec, _rid  # noqa: E402
from ingest import ingest  # noqa: E402
from master import run_one, write_summary  # noqa: E402
from metrics_run import read_json, write_json  # noqa: E402

DEFAULT_ROOT = _EXAMPLES / "out" / "baseline"
DATA = _HERE / "data"

# Budget is deliberately below the 240-epoch paper recipe: a regression baseline
# only has to be comparable to its own re-run, and 240 epochs on three datasets
# at three seeds does not fit in a working session. Recorded in manifest.json.
DEFAULT_EPOCHS = 120
DEFAULT_SEEDS = (0, 1, 2)

# (name, X file, color file, is_label) -- label metrics only where labels exist.
DATASETS: Dict[str, Tuple[str, str, bool]] = {
    "digits": ("digits_X.npy", "digits_y.npy", True),
    "s_curve": ("s_curve_X.npy", "s_curve_tbin.npy", True),
    "swiss_roll": ("swiss_roll_X.npy", "swiss_roll_tbin.npy", True),
}

# Metrics worth tracking for regressions, in report order.
TRACKED = [
    "label_acc_Z",
    "label_ari",
    "trust_15",
    "cont_15",
    "knn_overlap_15",
    "ambient_spearman",
    "geodesic_spearman",
    "density_spearman",
    "spacing_cv",
    "area_sd",
    "wall_s",
]


def _arm(axis: str, level: str, epochs: int, **overlay: Any) -> RunSpec:
    ov = dict(RECOMMENDED)
    ov.update(overlay)
    ov["epochs"] = int(epochs)
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)


def build_arms(kind: str, epochs: int) -> List[RunSpec]:
    """Arm sets. Every ablation carries its own control so it is self-contained.

    Each set is compared *within itself* at matched seeds; do not read one set's
    arm against another set's, since they can differ in more than the axis named.
    """
    if kind == "recommended":
        return [_arm("recommended", "default", epochs)]
    if kind == "ladder":
        runs = [_arm("recommended", "default", epochs)]
        for md in (0.1, 0.2, 0.5, 0.8):
            runs.append(_arm("min_dist", str(md), epochs, min_dist=md))
        for g in (0.0, 0.5):
            runs.append(_arm("lambda_geo", str(g), epochs, lambda_geo=g))
        for label, weights in (("flat", (1.0, 1.0, 1.0)), ("ramp", (1.0, 2.0, 8.0))):
            runs.append(_arm("weights", label, epochs, pyramid_level_weights=weights))
        return runs

    # --- Stage 2c ablations -------------------------------------------------
    if kind == "conditioning":
        # a(x) is a deterministic function of x, so FiLM adds no information.
        # Concat is the baseline that tests whether the conditioning apparatus
        # earns its roles, temperatures, clamps, and perplexity calibration.
        return [
            _arm("cond", "film", epochs, conditioning="film"),
            _arm("cond", "concat", epochs, conditioning="concat"),
        ]
    if kind == "beta":
        return [
            _arm("beta", str(b), epochs, beta_multiplicity=b) for b in (0.0, 0.5, 1.0)
        ]
    if kind == "pyramid_geo":
        # Are a coarse-weighted pyramid and the geodesic anchor substitutes?
        return [
            _arm(
                "pygeo", "ramp_geo0", epochs,
                pyramid_level_weights=(1.0, 2.0, 8.0), lambda_geo=0.0,
            ),
            _arm(
                "pygeo", "flat_geo0.5", epochs,
                pyramid_level_weights=(1.0, 1.0, 1.0), lambda_geo=0.5,
            ),
            _arm(
                "pygeo", "ramp_geo0.5", epochs,
                pyramid_level_weights=(1.0, 2.0, 8.0), lambda_geo=0.5,
            ),
        ]
    if kind == "min_dist":
        return [_arm("mdist", str(md), epochs, min_dist=md) for md in (0.1, 0.2, 0.5, 0.8)]
    if kind == "speed":
        # Two ways to stop paying full price for the early epochs, which only
        # settle global layout: start from the landmark geodesic MDS instead of
        # from PCA, and climb the pyramid from the coarsest level instead of
        # mixing all scales from step 0. ``wall_s`` is the metric of interest
        # here, but it only counts if the quality columns hold, so the half-epoch
        # arm is the one that matters: same map, less time, or it did not work.
        warm, coarse = dict(warm_start_steps=300), dict(coarse_first_frac=0.3)
        return [
            _arm("speed", "flat", epochs),
            _arm("speed", "warm", epochs, **warm),
            _arm("speed", "coarse", epochs, **coarse),
            _arm("speed", "warm_coarse", epochs, **warm, **coarse),
            _arm("speed", "warm_coarse_half", max(1, epochs // 2), **warm, **coarse),
            _arm("speed", "flat_half", max(1, epochs // 2)),
        ]
    if kind == "density":
        # Strength of the term tying which neighbourhoods come out crowded to the
        # ambient graph. The s-curve is the control that must NOT move: uniform
        # sampling has no density ordering to reproduce, so any change there is
        # the mechanism inventing structure. Digits is where the shipping recipe
        # was tuned and is genuinely clustered, so it is where over-weighting this
        # term would show up as a regression in 5-NN accuracy.
        return [
            _arm("density", str(lam), epochs, lambda_density=lam)
            for lam in (0.0, 0.5, 1.0, 2.0)
        ]
    if kind == "anchor":
        # Frame on in both arms: the question is whether the Procrustes gauge
        # still earns its place once local rigidity is carrying metric fidelity.
        frame = dict(lambda_frame=0.5, frame_ramp=(0.5, 0.75), frame_tangent=True,
                     lambda_geo=0.5)
        return [
            _arm("anchor", "on", epochs, lambda_anchor=1.0, **frame),
            _arm("anchor", "off", epochs, lambda_anchor=0.0, **frame),
        ]
    if kind == "squash":
        # The squash changes the *magnitude* of coarse attraction, so the level
        # weights have to move with it or a re-tune reads as a regression.
        runs = [
            _arm("squash", "clamp_ramp", epochs,
                 pyramid_squash="quantile_clamp", pyramid_level_weights=(1.0, 2.0, 8.0)),
        ]
        # The rational squash raises the mean coarse membership by ~6.3x
        # (median -> 0.5 instead of q99 -> 1), so holding the *effective*
        # coarse attraction fixed means dividing the coarse level weights by
        # that factor, not raising them. Both directions are included because
        # the naive re-tune goes the wrong way.
        for label, weights in (
            ("rational_ramp", (1.0, 2.0, 8.0)),
            ("rational_steep", (1.0, 4.0, 16.0)),
            ("rational_matched", (1.0, 2.0 / 6.3, 8.0 / 6.3)),
            ("rational_flat", (1.0, 1.0, 1.0)),
            ("q99_ramp", (1.0, 2.0, 8.0)),
        ):
            runs.append(
                _arm("squash", label, epochs,
                     pyramid_squash="rational_q99" if label.startswith("q99") else "rational",
                     pyramid_level_weights=weights)
            )
        return runs
    if kind == "pca_lr":
        # The shipping recipe runs pca_skip=False at a high lr precisely because
        # a single flat rate couples the skip and the head. The multiplier is a
        # no-op without the skip, so the arms turn it back on and vary only the
        # differential; the control is the recipe as it ships.
        runs = [_arm("pcalr", "noskip_control", epochs, pca_skip=False, lr=2e-2)]
        for m in (1.0, 10.0, 20.0):
            runs.append(
                _arm("pcalr", f"skip_x{m:g}", epochs,
                     pca_skip=True, lr=1e-3, pca_lr_mult=m)
            )
        return runs
    raise KeyError(f"unknown arm set {kind!r} (choose one of {sorted(ARM_SETS)})")


ARM_SETS = (
    "recommended", "ladder", "conditioning", "beta", "pyramid_geo",
    "min_dist", "anchor", "squash", "pca_lr", "density", "speed",
)


def _mean_sd(values: Sequence[float]) -> Tuple[float, float, int]:
    vals = [float(v) for v in values if v is not None and not _isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), len(vals)


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def aggregate(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Collapse per-seed metrics.json into mean/sd keyed by dataset then arm."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        by_arm: Dict[str, List[Dict[str, Any]]] = {}
        for metrics_path in sorted(ds_dir.glob("*/metrics.json")):
            m = read_json(metrics_path)
            by_arm.setdefault(str(m.get("run_id", metrics_path.parent.name)), []).append(m)
        if not by_arm:
            continue
        ds_out: Dict[str, Dict[str, Any]] = {}
        for arm, rows in sorted(by_arm.items()):
            stats: Dict[str, Any] = {"n_seeds": len(rows)}
            for key in TRACKED:
                mean, sd, n = _mean_sd([r.get(key) for r in rows])
                if n:
                    stats[key] = {"mean": mean, "sd": sd, "n": n}
            ds_out[arm] = stats
        out[ds_dir.name] = ds_out
    return out


def write_aggregate(root: Path) -> Path:
    agg = aggregate(root)
    write_json(root / "aggregate.json", agg)
    csv_path = root / "aggregate.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "metric", "mean", "sd", "n_seeds"])
        for ds, arms in agg.items():
            for arm, stats in arms.items():
                for key in TRACKED:
                    if key in stats:
                        s = stats[key]
                        w.writerow([ds, arm, key, f"{s['mean']:.6g}", f"{s['sd']:.6g}", s["n"]])
    return csv_path


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root) / args.tag
    root.mkdir(parents=True, exist_ok=True)
    arms = build_arms(args.arms, args.epochs)
    datasets = args.datasets or list(DATASETS)

    total = len(arms) * len(datasets) * len(args.seeds)
    print(
        f"baseline tag={args.tag} arms={args.arms}({len(arms)}) "
        f"datasets={len(datasets)} seeds={len(args.seeds)} epochs={args.epochs} "
        f"=> {total} fits"
    )
    write_json(
        root / "manifest.json",
        {
            "tag": args.tag,
            "arms": args.arms,
            "arm_ids": [r.run_id for r in arms],
            "datasets": datasets,
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "holdout": args.holdout,
            "note": "pre/post comparison is only valid between tags sharing this manifest",
        },
    )

    for ds in datasets:
        x_file, c_file, has_labels = DATASETS[ds]
        X, color = ingest(DATA / x_file, DATA / c_file)
        labels = None
        if has_labels and color is not None:
            labels = np.asarray(color)
            if not np.issubdtype(labels.dtype, np.integer):
                labels = None
        ds_dir = root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        for run in arms:
            for seed in args.seeds:
                tag = f"{run.run_id}__seed{seed}"
                run_one(
                    X,
                    color,
                    run=run,
                    out_dir=ds_dir / tag,
                    epochs=args.epochs,
                    seed=seed,
                    device=args.device,
                    cmap="viridis",
                    colorbar_label="",
                    shepard_mode="none",
                    force=args.force,
                    labels=labels,
                    null="none",
                    holdout=args.holdout,
                )
        write_summary(ds_dir)

    csv_path = write_aggregate(root)
    print(f"wrote {csv_path}")
    _print_aggregate(aggregate(root))
    return 0


def _print_aggregate(agg: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    for ds, arms in agg.items():
        print(f"\n== {ds} ==")
        for arm, stats in arms.items():
            print(f"  {arm}  (n_seeds={stats['n_seeds']})")
            for key in TRACKED:
                if key in stats:
                    s = stats[key]
                    print(f"    {key:20s} {s['mean']:8.4f} +/- {s['sd']:.4f}")


def cmd_compare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    a = aggregate(root / args.before)
    b = aggregate(root / args.after)
    man_a = read_json(root / args.before / "manifest.json")
    man_b = read_json(root / args.after / "manifest.json")
    for key in ("arms", "datasets", "seeds", "epochs", "holdout"):
        if man_a.get(key) != man_b.get(key):
            print(
                f"WARNING: manifests differ on {key!r}: "
                f"{man_a.get(key)!r} vs {man_b.get(key)!r} -- comparison is not apples-to-apples"
            )

    print(f"\n{args.before} -> {args.after}")
    print("delta is flagged when |change| exceeds 2x the pooled seed sd.\n")
    header = f"{'dataset/arm':28s} {'metric':20s} {'before':>16s} {'after':>16s} {'delta':>9s}  flag"
    print(header)
    print("-" * len(header))
    n_flag = 0
    for ds in sorted(set(a) | set(b)):
        for arm in sorted(set(a.get(ds, {})) | set(b.get(ds, {}))):
            sa = a.get(ds, {}).get(arm, {})
            sb = b.get(ds, {}).get(arm, {})
            for key in TRACKED:
                if key == "wall_s" or key not in sa or key not in sb:
                    continue
                ma, sda = sa[key]["mean"], sa[key]["sd"]
                mb, sdb = sb[key]["mean"], sb[key]["sd"]
                delta = mb - ma
                pooled = math.sqrt(0.5 * (sda**2 + sdb**2))
                flag = ""
                if pooled > 0 and abs(delta) > 2 * pooled:
                    flag = "**"
                    n_flag += 1
                elif pooled == 0 and abs(delta) > 1e-9:
                    flag = "?"
                print(
                    f"{ds + '/' + arm:28s} {key:20s} "
                    f"{ma:8.4f}+/-{sda:<6.4f} {mb:8.4f}+/-{sdb:<6.4f} {delta:>+9.4f}  {flag}"
                )
    print(f"\n{n_flag} metric(s) moved by more than 2x pooled seed sd")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Compare arms *within* one tag against their own seed spread.

    An arm difference is only reported as resolved when it exceeds twice the
    pooled seed sd of the two arms. Most published single-seed differences on
    these datasets do not clear that bar, which is the whole reason this
    command exists.
    """
    root = Path(args.root) / args.tag
    agg = aggregate(root)
    metrics = args.metrics or ["label_acc_Z", "trust_15", "knn_overlap_15",
                               "geodesic_spearman", "spacing_cv", "area_sd"]
    for ds, arms in agg.items():
        names = sorted(arms)
        if not names:
            continue
        ref = args.baseline if args.baseline in names else names[0]
        print(f"\n== {ds} ==   (reference arm: {ref})")
        head = f"{'metric':20s}" + "".join(f"{n[:18]:>20s}" for n in names)
        print(head)
        print("-" * len(head))
        for key in metrics:
            if not any(key in arms[n] for n in names):
                continue
            row = f"{key:20s}"
            for n in names:
                s = arms[n].get(key)
                row += f"{s['mean']:>12.4f}+/-{s['sd']:<5.3f}" if s else f"{'—':>20s}"
            print(row)
            # Verdict line: which arms are resolved against the reference.
            base = arms[ref].get(key)
            if not base:
                continue
            verdicts = []
            for n in names:
                if n == ref:
                    continue
                s = arms[n].get(key)
                if not s:
                    continue
                delta = s["mean"] - base["mean"]
                pooled = math.sqrt(0.5 * (s["sd"] ** 2 + base["sd"] ** 2))
                if pooled > 0 and abs(delta) > 2 * pooled:
                    verdicts.append(f"{n}:{delta:+.3f}*")
                else:
                    verdicts.append(f"{n}:{delta:+.3f} ns")
            if verdicts:
                print(f"{'':20s}  vs {ref}: " + "  ".join(verdicts))
    print("\n* = |delta| > 2x pooled seed sd;  ns = within seed noise")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fit the arm set and aggregate")
    r.add_argument("--tag", required=True, help="snapshot name, e.g. pre_review")
    r.add_argument("--arms", default="recommended", choices=ARM_SETS)
    r.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=None)
    r.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    r.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    r.add_argument("--holdout", type=float, default=0.2)
    r.add_argument("--device", default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="compare arms within one tag")
    rep.add_argument("tag")
    rep.add_argument("--baseline", default="", help="reference arm id")
    rep.add_argument("--metrics", nargs="+", default=None)
    rep.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="diff two snapshots against seed spread")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

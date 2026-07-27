#!/usr/bin/env python
"""Set the reference bar: what UMAP/densMAP/PCA achieve on a dataset.

leanmap has to be judged against something. This scores the established methods
on the same battery leanmap will be scored on, plus a matched null for each, and
writes ``bar.json``.

Two things make this comparable to ``master.py`` in a way it was not before.

**Same rows.** Each method is fit on the training split only and the holdout is
placed through its own out-of-sample path -- ``transform()`` for UMAP/densMAP,
exact projection for PCA. Previously everything here was fit and scored on all
of X while leanmap was scored on a 20% holdout, so the bar was measured under
easier conditions than the thing it was the bar for. The all-N in-sample fit is
still run, as ``{method}_full``, because it is UMAP's upper bound and keeps the
older numbers interpretable.

**Saved embeddings.** ``Z.npy`` and the split are written per run, so a new
reference geometry can rescore these later without refitting anything.

``transform()`` is also checked for silent degeneracy -- collapse to a point,
wild inflation, swallowed exceptions -- because "UMAP is bad out of sample" is
only worth reporting if the call actually worked.

Usage::

    python examples/exploratory/reference.py \\
      --X examples/exploratory/data/digits_X.npy \\
      --y examples/exploratory/data/digits_y.npy \\
      --name digits --holdout 0.2 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_run import full_battery, write_json  # noqa: E402
from nulls import make_null  # noqa: E402
from splits import save_split, split_indices  # noqa: E402

DEFAULT_OUT = _EXAMPLES / "out" / "exploratory"


def _umap_model(seed: int, **kw):
    import umap

    return umap.UMAP(n_components=2, random_state=seed, **kw)


def _pca_model(seed: int):
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed)


def methods(n_neighbors: int):
    """Reference embedders as ``(name, factory(seed) -> estimator)``.

    Each estimator must support ``fit_transform`` and ``transform``; that is the
    only requirement for taking part in the holdout comparison.
    """
    return [
        ("umap_default", lambda seed: _umap_model(seed)),
        (f"umap_nn{n_neighbors}", lambda seed: _umap_model(seed, n_neighbors=n_neighbors)),
        ("densmap", lambda seed: _umap_model(seed, densmap=True)),
        ("pca2d", lambda seed: _pca_model(seed)),
    ]


def transform_sanity(Z_train: np.ndarray, Z_new: np.ndarray, tag: str) -> dict:
    """Cheap checks that an out-of-sample placement is not degenerate.

    A ``transform`` that quietly collapses every new point onto one spot would
    score terribly and look like a finding about the method. These fields make
    that distinguishable from genuine out-of-sample error.
    """
    out = {f"{tag}_n": int(len(Z_new))}
    if len(Z_new) == 0:
        return out
    finite = np.isfinite(Z_new).all(axis=1)
    out[f"{tag}_nonfinite_frac"] = float(1.0 - finite.mean())
    if not finite.any():
        out[f"{tag}_collapsed"] = True
        return out
    Zn = Z_new[finite]
    s_tr = float(np.mean(np.std(Z_train, axis=0))) or 1e-12
    s_new = float(np.mean(np.std(Zn, axis=0)))
    out[f"{tag}_spread_ratio"] = float(s_new / s_tr)
    out[f"{tag}_centroid_shift"] = float(
        np.linalg.norm(Zn.mean(axis=0) - Z_train.mean(axis=0)) / s_tr
    )
    out[f"{tag}_unique_frac"] = float(len(np.unique(np.round(Zn, 6), axis=0)) / len(Zn))
    out[f"{tag}_collapsed"] = bool(
        out[f"{tag}_spread_ratio"] < 0.05 or out[f"{tag}_unique_frac"] < 0.5
    )
    return out


def run_method(
    Xk: np.ndarray,
    *,
    name: str,
    factory,
    seed: int,
    holdout: float,
    probes: Optional[np.ndarray],
    in_sample: bool,
) -> dict:
    """Fit one reference method and place the holdout (and probes) with it."""
    n = len(Xk)
    if in_sample:
        train_idx = hold_idx = np.arange(n)
    else:
        train_idx, hold_idx = split_indices(n, holdout, seed)

    model = factory(seed)
    Z_all = np.full((n, 2), np.nan, dtype=np.float64)
    Z_train = np.asarray(model.fit_transform(Xk[train_idx]), dtype=np.float64)
    Z_all[train_idx] = Z_train

    # "cannot place new points at all" and "places them badly" are different
    # findings and must not be reported as the same thing. densMAP is the former.
    info: dict = {"transform_error": None, "holdout_supported": True}
    if in_sample:
        Z_all[hold_idx] = Z_train
    else:
        try:
            Z_all[hold_idx] = np.asarray(model.transform(Xk[hold_idx]), dtype=np.float64)
            info.update(transform_sanity(Z_train, Z_all[hold_idx], "holdout"))
        except Exception as exc:  # noqa: BLE001
            info["transform_error"] = f"{type(exc).__name__}: {exc}"
            info["holdout_supported"] = False

    Z_probe = None
    if probes is not None and len(probes) and info["holdout_supported"]:
        try:
            Z_probe = np.asarray(model.transform(probes), dtype=np.float64)
            info.update(transform_sanity(Z_train, Z_probe, "probe"))
        except Exception as exc:  # noqa: BLE001
            info["probe_transform_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "Z_all": Z_all,
        "Z_probe": Z_probe,
        "train_idx": train_idx,
        "hold_idx": hold_idx,
        "info": info,
        "model": model,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", "--color", dest="y", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-neighbors", type=int, default=10, help="match leanmap's graph k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=None, help="alias for a single --seeds")
    ap.add_argument("--holdout", type=float, default=0.2, help="match master.py")
    ap.add_argument("--probes", default=None, help="structured probe array (M, D) .npy")
    ap.add_argument(
        "--null",
        default="shuffle",
        help="null kind to also score each method on (none/shuffle/gauss)",
    )
    ap.add_argument(
        "--no-full",
        action="store_true",
        help="skip the all-N in-sample fits (kept by default as UMAP's ceiling)",
    )
    args = ap.parse_args(argv)

    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    X = np.load(args.X).astype(np.float32)
    y = np.load(args.y) if args.y else None
    probes = np.load(args.probes).astype(np.float32) if args.probes else None
    name = args.name or Path(args.X).stem
    name_dir = args.out / name
    ref_dir = name_dir / "reference"
    print(f"reference bar for {name}: N={len(X)} D={X.shape[1]}", flush=True)
    if y is not None:
        print(f"  labels: {len(np.unique(y))} classes", flush=True)
    if probes is not None:
        print(f"  probes: {len(probes)} structured outliers", flush=True)
    print(f"  holdout={args.holdout} seeds={seeds}", flush=True)

    rows = []
    kinds = ["none", args.null] if args.null != "none" else ["none"]
    regimes = [("holdout", False)] + ([] if args.no_full else [("full", True)])
    for kind in kinds:
        for seed in seeds:
            Xk = make_null(X, kind, seed=seed)
            for mname, factory in methods(args.n_neighbors):
                for regime, in_sample in regimes:
                    tag = mname if regime == "holdout" else f"{mname}_full"
                    t0 = time.perf_counter()
                    try:
                        res = run_method(
                            Xk,
                            name=mname,
                            factory=factory,
                            seed=seed,
                            holdout=args.holdout,
                            probes=probes,
                            in_sample=in_sample,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {tag} ({kind}) failed: {exc}", flush=True)
                        continue

                    hold_idx = res["hold_idx"]
                    Z_all = res["Z_all"]
                    ok = np.isfinite(Z_all[hold_idx]).all(axis=1)
                    if ok.sum() < 8:
                        m = {"scoring_error": "too few finite holdout points"}
                    else:
                        m = full_battery(
                            Xk[hold_idx][ok],
                            Z_all[hold_idx][ok],
                            y=None if y is None else np.asarray(y)[hold_idx][ok],
                            n_neighbors=args.n_neighbors,
                            seed=seed,
                        )
                    m.update(res["info"])
                    m.update(
                        {
                            "method": tag,
                            "base_method": mname,
                            "regime": regime,
                            "null": kind,
                            "seed": int(seed),
                            "holdout": float(0.0 if in_sample else args.holdout),
                            "wall_s": time.perf_counter() - t0,
                            "N": int(len(Xk)),
                            "D": int(Xk.shape[1]),
                            "n_train": int(len(res["train_idx"])),
                            "n_eval": int(len(hold_idx)),
                        }
                    )
                    rows.append(m)

                    run_dir = ref_dir / f"{tag}__{kind}__seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.save(run_dir / "Z.npy", Z_all.astype(np.float32))
                    save_split(
                        run_dir,
                        res["train_idx"],
                        res["hold_idx"],
                        holdout=0.0 if in_sample else args.holdout,
                        seed=seed,
                    )
                    if res["Z_probe"] is not None:
                        np.save(run_dir / "Z_probe.npy", res["Z_probe"].astype(np.float32))
                    # Persist the fitted estimator so new points can be placed
                    # later without refitting -- UMAP supports transform() on a
                    # trained model, it just has to be kept around to use it.
                    try:
                        import pickle

                        with open(run_dir / "model.pkl", "wb") as fh:
                            pickle.dump(res["model"], fh)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    model save failed for {tag}: {exc}", flush=True)
                    write_json(run_dir / "config.json", {k: v for k, v in m.items()})

                    warn = ""
                    if m.get("transform_error"):
                        warn = f"  !! no out-of-sample path: {m['transform_error']}"
                    elif m.get("holdout_collapsed"):
                        warn = "  !! holdout placement looks collapsed"
                    print(
                        f"  {tag:20s} null={kind:8s} seed={seed} "
                        f"acc_Z={m.get('label_acc_Z', float('nan')):.3f} "
                        f"trust15={m.get('trust_15', float('nan')):.3f} "
                        f"({m['wall_s']:.0f}s){warn}",
                        flush=True,
                    )

    name_dir.mkdir(parents=True, exist_ok=True)
    bar = {
        "name": name,
        "X": str(Path(args.X).resolve()),
        "y": str(Path(args.y).resolve()) if args.y else None,
        "probes": str(Path(args.probes).resolve()) if args.probes else None,
        "N": int(len(X)),
        "D": int(X.shape[1]),
        "n_neighbors": args.n_neighbors,
        "holdout": float(args.holdout),
        "seeds": [int(s) for s in seeds],
        "rows": rows,
    }
    path = write_json(name_dir / "bar.json", bar)
    print(f"\nwrote {path}")

    real = [r for r in rows if r["null"] == "none" and r["regime"] == "holdout"]
    if real and "label_acc_Z" in real[0]:
        best = max(real, key=lambda r: r.get("label_acc_Z", float("-inf")))
        print(
            f"BAR TO BEAT (held out): {best['method']} "
            f"label_acc_Z={best['label_acc_Z']:.3f} "
            f"ARI={best.get('label_ari', float('nan')):.3f} "
            f"trust_15={best.get('trust_15', float('nan')):.3f} "
            f"(ceiling in raw X = {best.get('label_acc_X', float('nan')):.3f})"
        )
    unsupported = sorted({r["method"] for r in rows if not r.get("holdout_supported", True)})
    if unsupported:
        print(f"NOTE: no out-of-sample path at all for {unsupported} (in-sample rows only)")
    degenerate = sorted({r["method"] for r in rows if r.get("holdout_collapsed")})
    if degenerate:
        print(f"WARNING: collapsed holdout placement for {degenerate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

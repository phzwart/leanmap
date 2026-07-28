#!/usr/bin/env python
"""Would you get the same map twice?

The rest of the battery scores whether one map is good. None of it asks whether
a refit reproduces it -- which is the property a saved ``.pt`` actually sells.

Two sources of variation are separated rather than conflated:

``seed``
    Identical training rows, different initialisation / sampling seed.
``subsample``
    Identical seed, a different 80% draw of the training rows.

Running both with one split seed held fixed is what makes the comparison
interpretable; varying the split *and* the model seed together (as a plain
multi-seed sweep does) cannot say which one moved the map.

Protocol. Every refit embeds the same probe set -- rows held out of *all* runs
in the group -- so no map has seen the points it is scored on. The Procrustes
similarity is fitted on half the probes and the residual is read on the other
half, because fitting and scoring on the same points lets the alignment absorb
part of the disagreement being measured.

Read the two numbers together. Coordinate disagreement alone overstates
instability: Procrustes cannot undo genuine topological ambiguity, such as which
arm of a branching structure ends up on which side. High rank agreement with
large coordinate disagreement is a gauge artefact. Rank agreement is primary.

Usage::

    python examples/exploratory/persistence_run.py \\
      --dataset digits --mode both --n-refits 3 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_EXAMPLES, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from axes import RECOMMENDED  # noqa: E402
from baseline_capture import DATA, DATASETS  # noqa: E402
from ingest import ingest  # noqa: E402
from metrics_run import write_json  # noqa: E402

from _demo import fit_embed  # noqa: E402

DEFAULT_OUT = _EXAMPLES / "out" / "persistence"


def _training_draws(
    n: int, mode: str, n_refits: int, holdout: float, split_seed: int
) -> List[np.ndarray]:
    """Training index sets, plus the probe set common to all of them."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_hold = int(round(holdout * n))
    base_train = perm[n_hold:]

    if mode == "seed":
        # Same rows every time; only the model seed varies.
        return [base_train.copy() for _ in range(n_refits)]

    # 80% draws of the base training pool, so the common probe set is still the
    # rows held out at the top level.
    draws = []
    for r in range(n_refits):
        sub = np.random.default_rng(split_seed + 1000 + r)
        keep = sub.permutation(len(base_train))[: int(round(0.8 * len(base_train)))]
        draws.append(np.sort(base_train[keep]))
    return draws


def run_group(
    X: np.ndarray,
    *,
    mode: str,
    n_refits: int,
    epochs: int,
    holdout: float,
    split_seed: int,
    overlay: Dict[str, Any],
    device: Optional[str],
) -> Dict[str, Any]:
    import torch

    from leanmap.evaluate import persistence_summary

    n = len(X)
    draws = _training_draws(n, mode, n_refits, holdout, split_seed)
    trained = set()
    for d in draws:
        trained |= set(d.tolist())
    probe_idx = np.array(sorted(set(range(n)) - trained), dtype=np.int64)
    if len(probe_idx) < 20:
        raise RuntimeError(
            f"only {len(probe_idx)} probe rows are held out of every refit; "
            "raise --holdout"
        )

    X_probe = X[probe_idx]
    embeddings = []
    for r, train_idx in enumerate(draws):
        seed = (split_seed + r) if mode == "seed" else split_seed
        kw = dict(overlay)
        kw["epochs"] = int(epochs)
        kw["seed"] = int(seed)
        if device is not None:
            kw["device"] = device
        print(
            f"  refit {r + 1}/{n_refits} mode={mode} seed={seed} "
            f"n_train={len(train_idx)}",
            flush=True,
        )
        result, _, _ = fit_embed(X[train_idx], **kw)
        with torch.no_grad():
            Z, _ = result.embed(X_probe)
        embeddings.append(Z.detach().cpu().numpy())

    out = persistence_summary(embeddings, k=15, anchor_frac=0.5, seed=split_seed)
    out.update(
        {
            "mode": mode,
            "n_probe": int(len(probe_idx)),
            "n_train": int(len(draws[0])),
            "epochs": int(epochs),
            "split_seed": int(split_seed),
        }
    )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=["digits"])
    ap.add_argument("--mode", choices=("seed", "subsample", "both"), default="both")
    ap.add_argument("--n-refits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="recommended")
    args = ap.parse_args(argv)

    modes = ["seed", "subsample"] if args.mode == "both" else [args.mode]
    overlay = dict(RECOMMENDED)
    overlay.pop("epochs", None)

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ds in args.dataset:
        x_file, c_file, _ = DATASETS[ds]
        X, _ = ingest(DATA / x_file, DATA / c_file)
        for mode in modes:
            print(f"{ds} / {mode}: {args.n_refits} refits", flush=True)
            row = run_group(
                np.asarray(X),
                mode=mode,
                n_refits=args.n_refits,
                epochs=args.epochs,
                holdout=args.holdout,
                split_seed=args.split_seed,
                overlay=overlay,
                device=args.device,
            )
            row["dataset"] = ds
            all_rows.append(row)
            print(
                f"  rank spearman {row['rank_spearman_mean']:.4f} "
                f"(worst {row['rank_spearman_worst']:.4f}), "
                f"jaccard@15 {row['rank_jaccard_15_mean']:.4f}, "
                f"coord disagreement {row['coord_disagreement_median']:.4f} "
                f"(worst {row['coord_disagreement_worst']:.4f})",
                flush=True,
            )

    path = args.out / f"persistence_{args.tag}.json"
    write_json(path, {"rows": all_rows})
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

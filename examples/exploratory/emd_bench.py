#!/usr/bin/env python
"""Score embeddings against the EMD reference: is leanmap better, or different?

Every geometry column in the usual battery calls pixel L2 the truth, and both
leanmap and UMAP are fit from a pixel-L2 kNN graph, so those columns partly
reward reproducing the input. This scores the same embeddings against EMD, which
no method here has seen, in three regimes:

``train``      rows the method was fit on
``holdout``    rows it never saw, placed through its own out-of-sample path
``gap``        train minus holdout, i.e. how much the map degrades on new data

plus a retrieval view (place a new image, are its map neighbours its true
perceptual neighbours) and a structured-probe view (smile, frown, ring: never in
training, and off-manifold by construction).

Differences between methods get a paired bootstrap over the *same* query points,
because the interesting differences here are comparable in size to the seed
spread and neither one alone settles anything.

Usage::

    python examples/exploratory/emd_bench.py \\
      --X examples/exploratory/data/digits_X.npy \\
      --emd examples/exploratory/data/digits_emd.npy \\
      --probes examples/exploratory/data/digits_probes_X.npy \\
      --probe-kind examples/exploratory/data/digits_probes_kind.npy \\
      --Z leanmap=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --Z umap=examples/out/exploratory/digits_holdout/reference/umap_default__none__seed0 \\
      --out examples/out/exploratory/digits_emd
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_run import write_json  # noqa: E402
from splits import load_split  # noqa: E402

DEFAULT_OUT = _EXAMPLES / "out" / "exploratory" / "emd_bench"


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC of ``pos`` scoring higher than ``neg`` (rank form, ties averaged)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if a.size < 8:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def paired_bootstrap_mean(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """CI on ``mean(a) - mean(b)`` for per-query values measured on the same queries.

    Per-query metrics (overlap, retrieval, hit rates) decompose exactly, so
    resampling queries is the honest paired test rather than an approximation.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 8:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(a.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(a.mean() - b.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(a.size),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def paired_bootstrap_spearman(
    ref: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 400,
    max_pairs: int = 20_000,
    seed: int = 0,
) -> Dict[str, float]:
    """CI on ``spearman(ref, a) - spearman(ref, b)`` by resampling pairs.

    This captures pair-sampling noise only. Run-to-run noise is a separate thing
    and is reported as the seed spread; neither substitutes for the other.
    """
    rng = np.random.default_rng(seed)
    n = ref.size
    if n < 32:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    take = min(n, max_pairs)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sel = rng.integers(0, n, size=take)
        diffs[i] = _spearman(ref[sel], a[sel]) - _spearman(ref[sel], b[sel])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(_spearman(ref, a) - _spearman(ref, b)),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# scoring one embedding
# --------------------------------------------------------------------------


def _pdist(Z: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(np.asarray(Z, dtype=np.float64)))


def _per_query_overlap(D_ref: np.ndarray, Z: np.ndarray, k: int) -> np.ndarray:
    """Per-point fraction of reference-kNN retained in the embedding."""
    n = len(D_ref)
    k = int(min(k, n - 1))
    nn_ref = np.argsort(D_ref, axis=1, kind="stable")[:, 1 : k + 1]
    nn_emb = np.argsort(_pdist(Z), axis=1, kind="stable")[:, 1 : k + 1]
    return np.array(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(nn_ref, nn_emb)]
    )


def _per_query_retrieval(
    D_qg: np.ndarray, Z_q: np.ndarray, Z_g: np.ndarray, k: int
) -> np.ndarray:
    k = int(min(k, D_qg.shape[1]))
    d_emb = np.linalg.norm(
        np.asarray(Z_q, np.float64)[:, None, :] - np.asarray(Z_g, np.float64)[None, :, :],
        axis=2,
    )
    nn_ref = np.argpartition(D_qg, k - 1, axis=1)[:, :k]
    nn_emb = np.argpartition(d_emb, k - 1, axis=1)[:, :k]
    return np.array(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(nn_ref, nn_emb)]
    )


def score_regime(
    D_ref: np.ndarray,
    Z: np.ndarray,
    *,
    prefix: str,
    k: int,
    seed: int,
    max_pairs: int = 200_000,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Fidelity of ``Z`` to reference distances ``D_ref`` on one set of rows."""
    from leanmap.emd import (
        geodesic_from_matrix,
        reference_knn_overlap,
        reference_shepard,
        reference_trust_continuity,
    )

    out: Dict[str, float] = {}
    out.update(reference_shepard(D_ref, Z, prefix=prefix, seed=seed, max_pairs=max_pairs))
    out.update(reference_knn_overlap(D_ref, Z, prefix=prefix, k=k))
    out.update(reference_trust_continuity(D_ref, Z, prefix=prefix, k_list=(k,)))

    # Geodesic under the *reference* metric: the strongest global statement,
    # since it asks about shortest paths through the data rather than raw pairs.
    try:
        G = geodesic_from_matrix(D_ref, n_neighbors=k)
        iu = np.triu_indices(len(D_ref), k=1)
        g, e = G[iu], _pdist(Z)[iu]
        m = np.isfinite(g) & np.isfinite(e)
        out[f"{prefix}_geodesic_spearman"] = _spearman(g[m], e[m])
    except Exception as exc:  # noqa: BLE001
        out[f"{prefix}_geodesic_error"] = str(exc)

    iu = np.triu_indices(len(D_ref), k=1)
    arrays = {
        "ref_pairs": D_ref[iu],
        "emb_pairs": _pdist(Z)[iu],
        "overlap": _per_query_overlap(D_ref, Z, k),
    }
    return out, arrays


def score_run(
    run: dict,
    *,
    D_emd: np.ndarray,
    D_l2: np.ndarray,
    n_images: int,
    k: int,
    seed: int,
) -> dict:
    """Score one embedding in the train and holdout regimes, plus retrieval."""
    Z = run["Z"]
    train_idx, hold_idx = run["train_idx"], run["hold_idx"]
    res: Dict[str, object] = {"name": run["name"], "path": str(run["path"])}
    arrays: Dict[str, np.ndarray] = {}

    # Neighbourhood metrics depend on how many points are in the scored set --
    # overlap@k among 359 points is mechanically easier than among 1438 -- so
    # the train regime is subsampled to the holdout's size. Without this the
    # train-holdout gap measures set size as much as generalization. The
    # subsample is seeded, so every method is scored on identical rows.
    n_eval = min(len(hold_idx), len(train_idx))
    train_eval = np.random.default_rng(seed).choice(train_idx, size=n_eval, replace=False)
    res["n_train_eval"] = int(n_eval)
    res["n_hold_eval"] = int(len(hold_idx))

    for regime, rows in (("train", train_eval), ("holdout", hold_idx)):
        Zr = Z[rows]
        ok = np.isfinite(Zr).all(axis=1)
        if ok.sum() < 16:
            res[f"{regime}_error"] = "too few finite embedded points"
            continue
        rows_ok = rows[ok]
        for ref_name, D in (("emd", D_emd), ("l2", D_l2)):
            sub = np.asarray(D[np.ix_(rows_ok, rows_ok)], dtype=np.float64)
            m, arr = score_regime(
                sub, Z[rows_ok], prefix=f"{ref_name}", k=k, seed=seed
            )
            for key, val in m.items():
                res[f"{regime}__{key}"] = val
            if ref_name == "emd":
                arrays[f"{regime}__ref_pairs"] = arr["ref_pairs"]
                arrays[f"{regime}__emb_pairs"] = arr["emb_pairs"]
                arrays[f"{regime}__overlap"] = arr["overlap"]

    # Retrieval: a held-out image queried against the training gallery.
    ok_h = np.isfinite(Z[hold_idx]).all(axis=1)
    ok_t = np.isfinite(Z[train_idx]).all(axis=1)
    if ok_h.sum() > 8 and ok_t.sum() > k:
        q, g = hold_idx[ok_h], train_idx[ok_t]
        for ref_name, D in (("emd", D_emd), ("l2", D_l2)):
            per_q = _per_query_retrieval(
                np.asarray(D[np.ix_(q, g)], dtype=np.float64), Z[q], Z[g], k
            )
            res[f"retrieval__{ref_name}_overlap_{k}"] = float(np.mean(per_q))
            if ref_name == "emd":
                arrays["retrieval__per_query"] = per_q

    # Generalization gap, the number that says how much is lost out of sample.
    for key in list(res.keys()):
        if not key.startswith("train__"):
            continue
        tail = key[len("train__") :]
        h = res.get(f"holdout__{tail}")
        if isinstance(res[key], float) and isinstance(h, float):
            res[f"gap__{tail}"] = float(res[key] - h)

    res["_arrays"] = arrays
    return res


# --------------------------------------------------------------------------
# structured probes
# --------------------------------------------------------------------------


def score_probes(
    run: dict,
    *,
    D_emd: np.ndarray,
    D_l2: np.ndarray,
    n_images: int,
    probe_kind: Optional[np.ndarray],
    k: int,
    seed: int,
) -> dict:
    """How each map treats structured images it has never seen.

    Nothing here claims a correct 2-D location for a smiley -- these points are
    off-manifold by construction. The claims are narrow: probes should look
    farther from the training data than held-out digits do, distinct patterns
    should stay distinct, and the placement should not collapse.
    """
    Z_probe = run.get("Z_probe")
    if Z_probe is None:
        return {"probe_available": False}
    Z, train_idx, hold_idx = run["Z"], run["train_idx"], run["hold_idx"]
    n_probe = len(Z_probe)
    probe_rows = np.arange(n_images, n_images + n_probe)

    ok_t = np.isfinite(Z[train_idx]).all(axis=1)
    ok_h = np.isfinite(Z[hold_idx]).all(axis=1)
    ok_p = np.isfinite(Z_probe).all(axis=1)
    if ok_t.sum() < k or ok_p.sum() < 4 or ok_h.sum() < 4:
        return {"probe_available": False, "probe_error": "non-finite placements"}
    tr, ho = train_idx[ok_t], hold_idx[ok_h]
    Zt, Zh, Zp = Z[tr], Z[ho], Z_probe[ok_p]
    prow = probe_rows[ok_p]

    def _nn_dist(A, B):
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min(axis=1)

    out: Dict[str, object] = {
        "probe_available": True,
        "n_probe": int(ok_p.sum()),
        "probe_spread_ratio": float(
            np.mean(np.std(Zp, axis=0)) / max(np.mean(np.std(Zt, axis=0)), 1e-12)
        ),
    }

    # Separability from the embedding alone: distance to the nearest training
    # point in the map. Computed identically for every method.
    s_probe = _nn_dist(Zp, Zt)
    s_hold = _nn_dist(Zh, Zt)
    out["ood_auroc_embed"] = _auroc(s_probe, s_hold)
    out["_per_probe_embed"] = s_probe
    out["_per_hold_embed"] = s_hold

    # The raw nearest-neighbour distances behind that AUROC. Raw values are in
    # each embedding's own units and are *not* comparable across methods -- the
    # maps differ in scale by an order of magnitude -- so the comparable number
    # is the ratio to what a genuine held-out digit does. A ratio of 1.0 means a
    # probe sits as close to the training data as a real new digit.
    med_h = float(np.median(s_hold))
    out["holdout_nn_median"] = med_h
    out["probe_nn_median"] = float(np.median(s_probe))
    out["probe_nn_ratio"] = float(np.median(s_probe) / max(med_h, 1e-12))
    out["embed_scale"] = float(np.mean(np.std(Zt, axis=0)))
    # Operating point: how many probes clear a threshold that admits 5% of real
    # held-out digits. More actionable than AUROC if you have to pick a cutoff.
    thr = float(np.percentile(s_hold, 95))
    out["probe_tpr_at_5fpr"] = float(np.mean(s_probe > thr))

    # Does a probe land *on* the data or in a gap? Local density answers this;
    # the convex hull below does not, because in a clustered embedding the hull
    # swallows all the empty space between clusters and a probe dropped in a
    # void still counts as "inside". The neighbourhood radius is the training
    # set's own 15-NN scale, so the counts are directly comparable to what a
    # real point sees.
    from scipy.spatial import cKDTree

    tree = cKDTree(Zt)
    d15, _ = tree.query(Zt, k=min(16, len(Zt)))
    r_scale = float(np.median(d15[:, -1]))
    n_probe = np.array([len(v) for v in tree.query_ball_point(Zp, r_scale)])
    n_hold = np.array([len(v) for v in tree.query_ball_point(Zh, r_scale)])
    out["neighborhood_radius"] = r_scale
    out["holdout_local_density"] = float(np.median(n_hold))
    out["probe_local_density"] = float(np.median(n_probe))
    out["probe_density_ratio"] = float(
        np.median(n_probe) / max(np.median(n_hold), 1e-12)
    )
    # The sharpest form: a probe with no training point in its neighbourhood at
    # all has been placed somewhere the map left empty.
    out["probe_isolated_frac"] = float(np.mean(n_probe == 0))
    out["holdout_isolated_frac"] = float(np.mean(n_hold == 0))

    try:
        from scipy.spatial import Delaunay

        hull = Delaunay(Zt)
        out["probe_in_hull"] = float((hull.find_simplex(Zp) >= 0).mean())
        out["holdout_in_hull"] = float((hull.find_simplex(Zh) >= 0).mean())
    except Exception as exc:  # noqa: BLE001
        out["hull_error"] = str(exc)

    # How far out radially, against the training embedding's own extent.
    centre = Zt.mean(axis=0)
    r95 = float(np.percentile(np.linalg.norm(Zt - centre, axis=1), 95)) or 1e-12
    out["probe_radius_ratio"] = float(
        np.median(np.linalg.norm(Zp - centre, axis=1)) / r95
    )
    out["holdout_radius_ratio"] = float(
        np.median(np.linalg.norm(Zh - centre, axis=1)) / r95
    )

    # Ceilings: what the reference metrics themselves achieve. Reading the
    # embedding number without these is meaningless -- it says nothing about how
    # detectable these probes were in the first place.
    for ref_name, D in (("emd", D_emd), ("l2", D_l2)):
        d_p = np.asarray(D[np.ix_(prow, tr)], dtype=np.float64).min(axis=1)
        d_h = np.asarray(D[np.ix_(ho, tr)], dtype=np.float64).min(axis=1)
        out[f"ood_auroc_{ref_name}"] = _auroc(d_p, d_h)

    # leanmap's shipped OOD score, when the run has one. UMAP has no equivalent,
    # so this is a capability column and not a head-to-head win.
    cover_p = run.get("probe_cover")
    cover_all = run.get("cover")
    if cover_p is not None and cover_all is not None:
        cp = np.asarray(cover_p)[ok_p]
        ch = np.asarray(cover_all)[ho]
        out["ood_auroc_cover"] = _auroc(cp, ch)
        # Same estimator as ConformalCalibrator.p_value: calibrate on real
        # held-out digits, then ask how extreme each probe's cover score is.
        calib = np.sort(ch)
        n_ge = len(calib) - np.searchsorted(calib, cp, side="left")
        pvals = (1.0 + n_ge.astype(np.float64)) / (len(calib) + 1.0)
        out["conformal_frac_flagged_05"] = float(np.mean(pvals <= 0.05))
        out["conformal_median_p"] = float(np.median(pvals))

    # Internal consistency: does the probe family keep its own EMD geometry?
    if ok_p.sum() > 8:
        D_pp = np.asarray(D_emd[np.ix_(prow, prow)], dtype=np.float64)
        iu = np.triu_indices(len(prow), k=1)
        out["probe_emd_spearman"] = _spearman(D_pp[iu], _pdist(Zp)[iu])

    # Do distinct patterns stay distinct, or pile into one "weird" corner?
    if probe_kind is not None:
        kinds = np.asarray([str(v) for v in probe_kind])[ok_p]
        d_emb = _pdist(Zp)
        same = kinds[:, None] == kinds[None, :]
        iu = np.triu_indices(len(kinds), k=1)
        w = d_emb[iu][same[iu]]
        b = d_emb[iu][~same[iu]]
        if w.size and b.size:
            out["probe_discriminability"] = float(np.mean(b) / max(np.mean(w), 1e-12))

        # Per family, because the pooled number hides the interesting spread:
        # a checkerboard and a smiley are not equally far from a digit, and a
        # detector can be carried entirely by the easy families.
        for fam in sorted(set(kinds.tolist())):
            m = kinds == fam
            if m.sum() < 3:
                continue
            out[f"famauroc_embed__{fam}"] = _auroc(s_probe[m], s_hold)
            out[f"famnn_ratio__{fam}"] = float(
                np.median(s_probe[m]) / max(med_h, 1e-12)
            )
            out[f"famtpr__{fam}"] = float(np.mean(s_probe[m] > thr))
            for ref_name, D in (("emd", D_emd), ("l2", D_l2)):
                d_p = np.asarray(D[np.ix_(prow[m], tr)], dtype=np.float64).min(axis=1)
                d_h = np.asarray(D[np.ix_(ho, tr)], dtype=np.float64).min(axis=1)
                out[f"famauroc_{ref_name}__{fam}"] = _auroc(d_p, d_h)
            if cover_p is not None and cover_all is not None:
                out[f"famauroc_cover__{fam}"] = _auroc(cp[m], ch)
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_run(name: str, path: Path, *, n: int, holdout: float, seed: int) -> dict:
    """Load an embedding plus whatever out-of-sample artifacts it saved."""
    path = Path(path)
    run_dir = path if path.is_dir() else path.parent
    z_path = path if path.is_file() else run_dir / "Z.npy"
    Z = np.load(z_path).astype(np.float64)
    train_idx, hold_idx = load_split(run_dir, n=len(Z), holdout=holdout, seed=seed)
    out = {
        "name": name,
        "path": run_dir,
        "Z": Z,
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "hold_idx": np.asarray(hold_idx, dtype=np.int64),
    }
    for key, fname in (
        ("Z_probe", "Z_probe.npy"),
        ("probe_cover", "probe_cover.npy"),
        ("cover", "cover.npy"),
    ):
        f = run_dir / fname
        if f.is_file():
            out[key] = np.load(f).astype(np.float64)
    return out


def _parse_z(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--Z expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

HEADLINE = [
    ("holdout__emd_spearman", "EMD Shepard (holdout)"),
    ("holdout__emd_spearman_global", "  far band"),
    ("holdout__emd_knn_overlap", "EMD kNN overlap (holdout)"),
    ("holdout__emd_trust", "EMD trust (holdout)"),
    ("holdout__emd_geodesic_spearman", "EMD geodesic (holdout)"),
    ("retrieval__emd_overlap", "EMD retrieval (new->train)"),
    ("gap__emd_knn_overlap", "  train-holdout gap"),
]


def _resolve(res: dict, stem: str, k: int) -> Optional[float]:
    for key in (stem, f"{stem}_{k}"):
        if key in res and isinstance(res[key], float):
            return res[key]
    return None


def aggregate(results: List[dict], k: int) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Mean and sd across seeds, grouped by method name."""
    by_name: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_name[r["name"]].append(r)
    agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, runs in by_name.items():
        keys = {
            key
            for r in runs
            for key, v in r.items()
            if isinstance(v, float) and not key.startswith("_")
        }
        agg[name] = {}
        for key in keys:
            vals = np.array(
                [r[key] for r in runs if isinstance(r.get(key), float)], dtype=np.float64
            )
            vals = vals[np.isfinite(vals)]
            if vals.size:
                agg[name][key] = (float(vals.mean()), float(vals.std()))
    return agg


def print_table(agg: Dict[str, Dict[str, Tuple[float, float]]], k: int) -> None:
    names = sorted(agg)
    width = max(28, max((len(n) for n in names), default=10) + 2)
    print("\n" + "=" * (30 + width * len(names)))
    print("EMD-referenced comparison (mean +/- sd over seeds)")
    print("=" * (30 + width * len(names)))
    head = f"{'metric':<30}" + "".join(f"{n:>{width}}" for n in names)
    print(head)
    for stem, label in HEADLINE:
        cells = []
        any_val = False
        for n in names:
            v = None
            for key in (stem, f"{stem}_{k}"):
                if key in agg[n]:
                    v = agg[n][key]
                    break
            if v is None:
                cells.append(f"{'-':>{width}}")
            else:
                any_val = True
                cells.append(f"{v[0]:>{width - 8}.3f} +/-{v[1]:.3f}")
        if any_val:
            print(f"{label:<30}" + "".join(cells))
    print("-" * (30 + width * len(names)))
    for stem, label in (
        ("embed_scale", "embedding scale (train sd)"),
        ("holdout_nn_median", "NN dist: holdout digit->train"),
        ("probe_nn_median", "NN dist: probe->train"),
        ("probe_nn_ratio", "  ratio probe/digit"),
        ("holdout_local_density", "nbrs in r: holdout digit"),
        ("probe_local_density", "nbrs in r: probe"),
        ("probe_density_ratio", "  density ratio probe/digit"),
        ("probe_isolated_frac", "probes with zero neighbours"),
        ("holdout_in_hull", "in train hull: holdout digit"),
        ("probe_in_hull", "in train hull: probe"),
        ("holdout_radius_ratio", "radius vs train p95: digit"),
        ("probe_radius_ratio", "radius vs train p95: probe"),
        ("probe_tpr_at_5fpr", "probes caught at 5% FPR"),
        ("ood_auroc_embed", "probe AUROC from map"),
        ("ood_auroc_emd", "  ceiling: EMD to train"),
        ("ood_auroc_l2", "  ceiling: L2 to train"),
        ("ood_auroc_cover", "  leanmap cover score"),
        ("conformal_frac_flagged_05", "  conformal flagged @0.05"),
        ("probe_discriminability", "probe pattern separation"),
        ("probe_emd_spearman", "probe internal EMD fidelity"),
    ):
        cells, any_val = [], False
        for n in names:
            v = agg[n].get(stem)
            if v is None:
                cells.append(f"{'-':>{width}}")
            else:
                any_val = True
                cells.append(f"{v[0]:>{width - 8}.3f} +/-{v[1]:.3f}")
        if any_val:
            print(f"{label:<30}" + "".join(cells))

    fams = sorted(
        {k.split("__", 1)[1] for n in names for k in agg[n] if k.startswith("famauroc_embed__")}
    )
    if fams:
        print("-" * (30 + width * len(names)))
        print("probe AUROC by family (EMD ceiling in brackets)")
        for fam in fams:
            cells = []
            for n in names:
                v = agg[n].get(f"famauroc_embed__{fam}")
                c = agg[n].get(f"famauroc_emd__{fam}")
                if v is None:
                    cells.append(f"{'-':>{width}}")
                else:
                    ceil = "" if c is None else f" [{c[0]:.2f}]"
                    cells.append(f"{v[0]:>{width - 7}.3f}{ceil}")
            print(f"  {fam:<28}" + "".join(cells))

        print("-" * (30 + width * len(names)))
        print("median NN distance to nearest training point, / that of a held-out digit")
        for fam in fams:
            cells = []
            for n in names:
                v = agg[n].get(f"famnn_ratio__{fam}")
                cells.append(
                    f"{'-':>{width}}" if v is None else f"{v[0]:>{width - 2}.2f}x"
                )
            print(f"  {fam:<28}" + "".join(cells))


def compare_pairs(
    results: List[dict], *, k: int, n_boot: int, seed: int
) -> List[dict]:
    """Paired bootstrap between every pair of methods, seed by seed."""
    by_name: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_name[r["name"]].append(r)
    names = sorted(by_name)
    comps = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for ra, rb in zip(by_name[a], by_name[b]):
                aa, ab = ra.get("_arrays", {}), rb.get("_arrays", {})
                row = {"a": a, "b": b, "seed_index": len(comps)}
                if "holdout__overlap" in aa and "holdout__overlap" in ab:
                    if len(aa["holdout__overlap"]) == len(ab["holdout__overlap"]):
                        row["overlap"] = paired_bootstrap_mean(
                            aa["holdout__overlap"],
                            ab["holdout__overlap"],
                            n_boot=n_boot,
                            seed=seed,
                        )
                if "retrieval__per_query" in aa and "retrieval__per_query" in ab:
                    if len(aa["retrieval__per_query"]) == len(ab["retrieval__per_query"]):
                        row["retrieval"] = paired_bootstrap_mean(
                            aa["retrieval__per_query"],
                            ab["retrieval__per_query"],
                            n_boot=n_boot,
                            seed=seed,
                        )
                if "holdout__ref_pairs" in aa and "holdout__ref_pairs" in ab:
                    if len(aa["holdout__emb_pairs"]) == len(ab["holdout__emb_pairs"]):
                        row["shepard"] = paired_bootstrap_spearman(
                            aa["holdout__ref_pairs"],
                            aa["holdout__emb_pairs"],
                            ab["holdout__emb_pairs"],
                            n_boot=max(50, n_boot // 5),
                            seed=seed,
                        )
                comps.append(row)
    return comps


def print_comparisons(comps: List[dict]) -> None:
    if not comps:
        return
    print("\npaired bootstrap on identical query points (95% CI on a - b)")
    seen = set()
    for c in comps:
        key = (c["a"], c["b"])
        for metric in ("overlap", "retrieval", "shepard"):
            d = c.get(metric)
            if not d or not np.isfinite(d.get("diff", np.nan)):
                continue
            tag = (key, metric)
            if tag in seen:
                continue
            seen.add(tag)
            verdict = "significant" if d.get("excludes_zero") else "includes zero"
            print(
                f"  {c['a']:>10} - {c['b']:<10} {metric:<10} "
                f"{d['diff']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  {verdict}"
            )


def make_figure(
    results: List[dict],
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    runs: List[dict],
    path: Path,
    k: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(agg)
    first = {}
    for r in results:
        first.setdefault(r["name"], r)
    run_by_name = {}
    for r in runs:
        run_by_name.setdefault(r["name"], r)

    ncol = max(len(names), 3)
    fig, axes = plt.subplots(3, ncol, figsize=(4.2 * ncol, 11.5))
    axes = np.atleast_2d(axes)
    fig.suptitle("Embeddings scored against the EMD reference (holdout)", fontsize=13)

    for j, name in enumerate(names):
        ax = axes[0, j]
        arr = first[name].get("_arrays", {})
        ref, emb = arr.get("holdout__ref_pairs"), arr.get("holdout__emb_pairs")
        if ref is not None and ref.size:
            take = min(ref.size, 40000)
            sel = np.random.default_rng(0).choice(ref.size, take, replace=False)
            ax.hexbin(ref[sel], emb[sel], gridsize=45, cmap="viridis", mincnt=1, bins="log")
            rho = agg[name].get("holdout__emd_spearman", (float("nan"), 0))[0]
            ax.set_title(f"{name}  Spearman={rho:.3f}")
        ax.set_xlabel("EMD distance")
        ax.set_ylabel("embedding distance")
    for j in range(len(names), ncol):
        axes[0, j].axis("off")

    ax = axes[1, 0]
    bands = ("local", "mid", "global")
    w = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        vals = [agg[name].get(f"holdout__emd_spearman_{b}", (np.nan, 0))[0] for b in bands]
        errs = [agg[name].get(f"holdout__emd_spearman_{b}", (np.nan, 0))[1] for b in bands]
        ax.bar(np.arange(3) + i * w, vals, w, yerr=errs, capsize=3, label=name)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Spearman vs EMD")
    ax.set_title("fidelity by EMD distance band")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for i, name in enumerate(names):
        tr = agg[name].get(f"train__emd_knn_overlap_{k}", (np.nan, 0))
        ho = agg[name].get(f"holdout__emd_knn_overlap_{k}", (np.nan, 0))
        ax.bar(i - 0.2, tr[0], 0.4, yerr=tr[1], capsize=3, color="tab:blue")
        ax.bar(i + 0.2, ho[0], 0.4, yerr=ho[1], capsize=3, color="tab:orange")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"EMD kNN overlap@{k}")
    ax.set_title("train (blue) vs holdout (orange)")

    ax = axes[1, 2] if ncol > 2 else axes[1, 1]
    ceiling = None
    for i, name in enumerate(names):
        v = agg[name].get("ood_auroc_embed")
        if v is not None:
            ax.bar(i, v[0], 0.6, yerr=v[1], capsize=3, color="tab:purple")
        c = agg[name].get("ood_auroc_emd")
        if c is not None:
            ceiling = c[0]
        cov = agg[name].get("ood_auroc_cover")
        if cov is not None:
            ax.plot([i], [cov[0]], "kv", markersize=8, label="cover score" if i == 0 else None)
    if ceiling is not None:
        ax.axhline(ceiling, color="tab:red", ls="--", lw=1.2, label=f"EMD ceiling {ceiling:.2f}")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AUROC probes vs holdout digits")
    ax.set_title("structured probe detection")
    ax.legend(fontsize=7)
    for j in range(3, ncol):
        axes[1, j].axis("off")

    for j, name in enumerate(names):
        ax = axes[2, j]
        run = run_by_name.get(name)
        if run is None:
            ax.axis("off")
            continue
        Z = run["Z"]
        ho = run["hold_idx"]
        ok = np.isfinite(Z[ho]).all(axis=1)
        ax.scatter(Z[ho][ok, 0], Z[ho][ok, 1], s=6, c="lightsteelblue", linewidths=0, label="holdout digits")
        Zp = run.get("Z_probe")
        if Zp is not None:
            okp = np.isfinite(Zp).all(axis=1)
            ax.scatter(Zp[okp, 0], Zp[okp, 1], s=14, c="crimson", marker="^", linewidths=0, label="probes")
        ax.set_title(f"{name}: where probes land")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="best")
    for j in range(len(names), ncol):
        axes[2, j].axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument("--emd", required=True, help="EMD reference matrix .npy")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--probe-kind", default=None)
    ap.add_argument("--Z", action="append", required=True, help="name=path (run dir or Z.npy)")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--holdout", type=float, default=0.2, help="only for legacy runs with no split.npz")
    ap.add_argument("--seed", type=int, default=0, help="only for legacy runs with no split.npz")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    from scipy.spatial.distance import pdist, squareform

    X = np.load(args.X).astype(np.float64)
    n_images = len(X)
    D_emd_all = np.load(args.emd).astype(np.float64)
    probes = np.load(args.probes).astype(np.float64) if args.probes else None
    probe_kind = np.load(args.probe_kind, allow_pickle=True) if args.probe_kind else None
    n_probe = 0 if probes is None else len(probes)
    if len(D_emd_all) != n_images + n_probe:
        raise SystemExit(
            f"EMD matrix is {len(D_emd_all)} rows but X+probes is {n_images + n_probe}"
        )

    A = np.concatenate([X, probes], axis=0) if probes is not None else X
    D_l2_all = squareform(pdist(A))
    print(f"reference: {n_images} images + {n_probe} probes, k={args.k}")

    runs = []
    for spec in args.Z:
        name, path = _parse_z(spec)
        try:
            runs.append(
                load_run(name, path, n=n_images, holdout=args.holdout, seed=args.seed)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name}: {exc}")
    if not runs:
        raise SystemExit("no embeddings could be loaded")

    results = []
    for run in runs:
        print(f"  scoring {run['name']} ({run['path'].name})", flush=True)
        res = score_run(
            run,
            D_emd=D_emd_all,
            D_l2=D_l2_all,
            n_images=n_images,
            k=args.k,
            seed=args.seed,
        )
        res.update(
            score_probes(
                run,
                D_emd=D_emd_all,
                D_l2=D_l2_all,
                n_images=n_images,
                probe_kind=probe_kind,
                k=args.k,
                seed=args.seed,
            )
        )
        results.append(res)

    agg = aggregate(results, args.k)
    print_table(agg, args.k)
    comps = compare_pairs(results, k=args.k, n_boot=args.bootstrap, seed=args.seed)
    print_comparisons(comps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": str(Path(args.X).resolve()),
        "emd": str(Path(args.emd).resolve()),
        "k": int(args.k),
        "n_images": int(n_images),
        "n_probes": int(n_probe),
        "runs": [
            {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in results
        ],
        "aggregate": {n: {kk: list(vv) for kk, vv in d.items()} for n, d in agg.items()},
        "comparisons": comps,
    }
    write_json(out_dir / "emd_bench.json", payload)
    try:
        make_figure(results, agg, runs, out_dir / "emd_bench.png", args.k)
        print(f"\nwrote {out_dir / 'emd_bench.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"figure failed: {exc}")
    print(f"wrote {out_dir / 'emd_bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

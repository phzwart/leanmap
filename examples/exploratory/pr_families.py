#!/usr/bin/env python
"""The fine-scale density structure: is it real, is it in the graph, why does UMAP lose it?

The landmark sweep (``pr_landmark_scale.py``) ruled out the mixture-of-experts
tessellation as the source of the short-range density structure leanmap draws --
its correlation length refused to follow a 16-fold change in cell size. That
leaves the possibility that the structure is simply real, at a scale of a few
tens of profiles, and that the clumpiness audit's single-bandwidth regressor was
blind to it. This script argues that in three steps, each falsifiable on its own.

**A -- the variation is real.** Groups of profiles that sit implausibly close
together in ambient space are found using ambient geometry alone, with no
reference to any embedding, so the selection cannot be circular. Their P(r)
curves are plotted so tightness can be judged by eye, against a control of
equally sized neighbour groups that were *not* selected for tightness. Then the
independent test: SASBDB sidecars carry the sample's protein name and UniProt
accession, which no part of the pipeline ever sees. If these tight groups are
real, they should be biologically coherent -- repeat depositions, concentration
series, mutants of one protein -- far more often than label-shuffled chance.

**B -- it is in the graph, and the audit could not see it.** Two things must
hold together. The groups have to be visible in the ambient kNN graph as genuine
density spikes, and the ``k=15`` ambient density used as the audit's regressor
has to be unable to represent them: a family of ``m < 15`` members has its 15th
neighbour *outside* the family, so that estimator smooths the spike away. The
decisive test is to hand the regression ambient density at several bandwidths at
once. If the leftover collapses, it was real ambient structure measured at the
wrong scale, not structure the map invented.

**C -- why UMAP does not show it.** UMAP normalises each point's neighbourhood
by its own local scale: it subtracts ``rho``, the nearest-neighbour distance, and
picks ``sigma`` so the fuzzy memberships sum to a fixed target. That is exactly
the quantity local density is carried in, so the graph UMAP optimises is density
blind by construction, and ``min_dist`` then imposes a floor on how close any two
points may sit. Both are measured here: how tightly each family survives into
leanmap, UMAP and densMAP, and the ``rho``/``sigma`` normalisation itself.

Usage::

    python examples/exploratory/pr_families.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pr_clumpiness import density, intrinsic_dim, knn_dist, morans_i  # noqa: E402
from pr_licensed import local_dim  # noqa: E402

DEFAULT_SASDBD = Path("/Users/phzwart/Projects/sasdbd")
K_GRID = (5, 15, 32, 100, 300, 900)
SCALES = (3, 5, 15, 50)


# --------------------------------------------------------------------------- #
# data


def knn(A: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Distances and indices of the ``k`` nearest neighbours, self excluded."""
    from sklearn.neighbors import NearestNeighbors

    d, i = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(A).kneighbors(A)
    return d[:, 1:], i[:, 1:]


def load_rich_meta(codes: List[str], root: Path, cache: Path):
    """Protein name / UniProt / organism per SASBDB code, from the entry sidecars.

    Cached because it is thousands of small JSON reads. These fields never enter
    the embedding, which is what makes them usable as an independent label.
    """
    import pandas as pd

    if cache.exists():
        df = pd.read_csv(cache)
        if len(df) == len(codes) and (df["sasbdb_code"].astype(str).values == np.asarray(codes)).all():
            return df

    rows = []
    for c in codes:
        f = root / "data" / "entries" / c / "sidecars" / f"{c}_saxs_analysis.json"
        name = uni = org = oligo = None
        conc = np.nan
        if f.exists():
            try:
                s = json.load(open(f)).get("sample") or {}
                name, uni = s.get("name"), s.get("uniprot")
                org, oligo = s.get("organism"), s.get("oligomeric_state")
                conc = s.get("concentration_mg_ml", np.nan)
            except Exception:
                pass
        rows.append((c, name, uni, org, oligo, conc))
    df = pd.DataFrame(
        rows, columns=["sasbdb_code", "name", "uniprot", "organism", "oligomer", "conc"]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- #
# A: tight families, found from ambient geometry only


def tight_families(
    X: np.ndarray, k: int, metric: str, pct: float, min_size: int
) -> Tuple[np.ndarray, float]:
    """Connected groups joined only by implausibly short ambient edges.

    The threshold is the ``pct``-th percentile of first-neighbour distances, so
    "implausibly short" is calibrated against this dataset rather than picked.
    Nothing here touches an embedding, so downstream comparisons stay honest.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    d, i = knn(X, k, metric)
    thr = float(np.percentile(d[:, 0], pct))
    src, dst = np.nonzero(d <= thr)
    lab = -np.ones(len(X), dtype=int)
    if not len(src):
        return lab, thr
    g = coo_matrix(
        (np.ones(len(src)), (src, i[src, dst])), shape=(len(X), len(X))
    )
    n, comp = connected_components(g, directed=False)
    keep = 0
    for c in range(n):
        m = np.nonzero(comp == c)[0]
        if len(m) >= min_size:
            lab[m] = keep
            keep += 1
    return lab, thr


def pseudo_families(
    X: np.ndarray, sizes: List[int], k: int, metric: str, seed: int
) -> List[np.ndarray]:
    """Control groups: a random seed point plus its nearest neighbours.

    Matched in size to the real families and drawn from the same graph, but
    selected without regard to tightness -- so any difference in spread is due to
    the tightness criterion and not to group size or to being neighbours.
    """
    _, i = knn(X, max(max(sizes) if sizes else 1, k), metric)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(X), size=len(sizes), replace=False)
    return [np.concatenate([[p], i[p, : s - 1]]) for p, s in zip(picks, sizes)]


def spread(X: np.ndarray, idx: np.ndarray) -> float:
    """Worst deviation from the group's median curve, as a fraction of its peak."""
    med = np.median(X[idx], axis=0)
    return float(np.abs(X[idx] - med).max() / max(med.max(), 1e-12))


def coherence(labels: np.ndarray, tag: np.ndarray, seed: int, n_perm: int = 2000):
    """Fraction of families sharing one protein label, against a shuffled null.

    The null permutes labels among exactly the points that belong to families,
    holding family sizes fixed, so it answers: given these group sizes and this
    label pool, how often would coherence arise by chance?
    """
    fams = [np.nonzero(labels == c)[0] for c in range(labels.max() + 1)]
    fams = [f for f in fams if (tag[f] != "").all()]
    if not fams:
        return float("nan"), float("nan"), 0

    def pure(t: np.ndarray) -> float:
        return float(np.mean([len(set(t[f])) == 1 for f in fams]))

    obs = pure(tag)
    inside = np.unique(np.concatenate(fams))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    t = tag.copy()
    for b in range(n_perm):
        t[inside] = rng.permutation(tag[inside])
        null[b] = pure(t)
    return obs, float(null.mean()), len(fams)


# --------------------------------------------------------------------------- #
# B: regression that survives the heavy tails of a fine-bandwidth density


def cv_fit(x: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5):
    """Out-of-fold spline fit, conditioned to survive near-duplicate points.

    ``pr_licensed`` fits raw splines by ordinary least squares, which is fine at
    ``k=15`` but explodes once a fine bandwidth is included: a near-duplicate pair
    has a third-neighbour radius near zero, so its dimension-corrected density is
    astronomically large and the design matrix becomes unusable. Ranking the
    features first bounds the tails, and ridge keeps the many spline columns from
    fighting each other. Without this the leftover would look like it collapsed
    when in truth the prediction had merely blown up.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer

    x = x.reshape(len(y), -1)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    pred = np.empty_like(y)
    for f in range(folds):
        te = fold == f
        m = make_pipeline(
            QuantileTransformer(
                output_distribution="normal", n_quantiles=min(1000, int((~te).sum()))
            ),
            SplineTransformer(n_knots=6, degree=3, extrapolation="linear"),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        pred[te] = m.fit(x[~te], y[~te]).predict(x[te])
    resid = y - pred
    return float(1.0 - resid.var() / y.var()), resid


# --------------------------------------------------------------------------- #
# C: what UMAP does to the graph


def umap_normalisation(X: np.ndarray, k: int, metric: str):
    """UMAP's per-point ``rho`` and ``sigma``, on this dataset's own kNN distances.

    Returns ``None`` when umap is unavailable; the panel is then skipped rather
    than faked.
    """
    try:
        from umap.umap_ import smooth_knn_dist
    except Exception:
        return None
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=k, metric=metric).fit(X).kneighbors(X)
    d = np.ascontiguousarray(d, dtype=np.float32)
    sigmas, rhos = smooth_knn_dist(d, float(k))
    return np.asarray(sigmas, float), np.asarray(rhos, float), d


def compaction(Z: np.ndarray, groups: List[np.ndarray]) -> np.ndarray:
    """Median within-group spacing in units of the map's typical spacing.

    Scale free, so maps with different overall extent compare directly. A
    density-faithful map keeps a near-duplicate family far below 1; a map with a
    separation floor pushes it towards 1.
    """
    from scipy.spatial.distance import pdist

    unit = np.median(knn(Z, 1, "euclidean")[0][:, 0])
    return np.array([np.median(pdist(Z[g])) / max(unit, 1e-12) for g in groups])


# --------------------------------------------------------------------------- #
# figures


def fig_families(X, fams, pseudo, meta, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(6, len(fams))
    fig, ax = plt.subplots(2, n, figsize=(3.1 * n, 6.6), sharex=True)
    ax = np.atleast_2d(ax)
    pop = np.median(X, axis=0)
    r = np.arange(X.shape[1]) / X.shape[1]

    for j in range(n):
        for row, (grp, title) in enumerate(
            ((fams[j], "tight family"), (pseudo[j], "control"))
        ):
            a = ax[row, j]
            a.plot(r, pop, color="0.75", lw=2.5, zorder=0)
            for m in grp:
                a.plot(r, X[m], lw=1.0, alpha=0.85)
            names = [str(x) for x in meta["name"].values[grp] if isinstance(x, str)]
            who = names[0][:26] if names else "?"
            uniq = len(set(names))
            a.set_title(
                f"{title}  n={len(grp)}\n{who}\n{uniq} distinct name(s), "
                f"spread {100 * spread(X, grp):.1f}%",
                fontsize=7.5,
            )
            a.tick_params(labelsize=7)
            if j == 0:
                a.set_ylabel("P(r), unit sum")
            if row == 1:
                a.set_xlabel("r / dmax")
    fig.suptitle(
        "A: profiles in the tightest ambient groups, against size-matched neighbour "
        "groups\n(grey = population median; selection used ambient distance only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_graph(X, d1, thr, fams, fam_sizes, r_in, r15, mi_tab, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    inside = np.concatenate(fams) if fams else np.array([], int)
    # Exact duplicates sit at r1 = 0 and would push the log axis to -inf, so they
    # are counted in the label instead of plotted.
    pos = d1 > 0
    n_dup = int((~pos).sum())
    bins = np.linspace(np.log10(d1[pos].min()), np.log10(d1[pos].max()), 70)
    ax[0].hist(np.log10(d1[pos]), bins=bins, color="0.8", label="all points")
    if len(inside):
        sel = inside[d1[inside] > 0]
        ax[0].hist(np.log10(d1[sel]), bins=bins, color="crimson", label="in a tight family")
    ax[0].axvline(np.log10(thr), color="k", ls="--", lw=1, label="tightness threshold")
    ax[0].set_xlabel("log10 first-neighbour distance")
    ax[0].set_ylabel("points")
    ax[0].set_title(
        f"the tight groups are real features of the graph\n({n_dup} exact duplicates "
        "off-scale to the left)",
        fontsize=10,
    )
    ax[0].legend(fontsize=8)

    keep = r_in > 0
    ax[1].scatter(fam_sizes[keep], r15[keep] / r_in[keep], s=26, color="crimson")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].axvline(15, color="0.5", ls="--", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("family size (log)")
    ax[1].set_ylabel("15th-neighbour radius / within-family radius")
    ax[1].set_title(
        "a k=15 estimator cannot resolve them:\nsmall families spill past their 15th "
        f"neighbour\n({int((~keep).sum())} all-duplicate families omitted)",
        fontsize=10,
    )

    ks = [k for k, _, _ in mi_tab]
    ax[2].plot(ks, [b for _, b, _ in mi_tab], "o-", label="regressor: ambient k=15")
    ax[2].plot(ks, [m for _, _, m in mi_tab], "s-", label="regressor: multi-scale")
    ax[2].set_xscale("log")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set_xlabel("neighbourhood size k")
    ax[2].set_ylabel("Moran's I of leftover")
    ax[2].set_title(
        "but resolving them changes nothing:\nextra bandwidths leave the leftover intact",
        fontsize=10,
    )
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(
        "B: the families are genuine graph features, yet they are not what the map's "
        "leftover density is made of",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_umap(
    comp: Dict[str, np.ndarray],
    comp_ctl: Dict[str, np.ndarray],
    norm,
    r1: np.ndarray,
    r15: np.ndarray,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    names = list(comp)
    pos = np.arange(len(names))
    ax[0].boxplot([comp[n] for n in names], positions=pos - 0.16, widths=0.28)
    ax[0].boxplot([comp_ctl[n] for n in names], positions=pos + 0.16, widths=0.28)
    for i, n in enumerate(names):
        ax[0].scatter(
            np.full(len(comp[n]), i - 0.16), comp[n], s=10, color="crimson", zorder=3
        )
        ax[0].scatter(
            np.full(len(comp_ctl[n]), i + 0.16), comp_ctl[n], s=10, color="0.6", zorder=3
        )
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("within-family spacing / typical spacing")
    ax[0].set_title(
        "how tightly each map keeps the families\n(red = families, grey = controls; "
        "1 = ordinary spacing)",
        fontsize=10,
    )

    if norm is not None:
        sig, rho, dfull = norm
        ok = (r15 > 0) & (sig > 0)
        ax[1].scatter(np.log10(r15[ok]), np.log10(sig[ok]), s=5, alpha=0.3)
        lo, hi = np.log10(r15[ok]).min(), np.log10(r15[ok]).max()
        ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="slope 1")
        b = np.polyfit(np.log10(r15[ok]), np.log10(sig[ok]), 1)[0]
        ax[1].set_xlabel("log10 raw 15th-neighbour radius")
        ax[1].set_ylabel("log10 UMAP sigma")
        ax[1].set_title(
            f"UMAP's bandwidth tracks local scale\n(slope {b:.2f}: density divides out)",
            fontsize=10,
        )
        ax[1].legend(fontsize=8)

        # However tight a pair really is, its normalised separation is identically
        # zero, because rho is that very distance and is subtracted. The 15th
        # neighbour is drawn alongside to show the graph does keep *some* variation
        # further out -- just none at the scale that defines a family.
        p = r1 > 0
        ax[2].scatter(
            np.log10(r1[p]),
            (r1[p] - rho[p]) / np.maximum(sig[p], 1e-12),
            s=6,
            alpha=0.5,
            color="crimson",
            label="nearest neighbour",
        )
        ax[2].scatter(
            np.log10(r1[p]),
            (r15[p] - rho[p]) / np.maximum(sig[p], 1e-12),
            s=6,
            alpha=0.35,
            color="steelblue",
            label="15th neighbour",
        )
        ax[2].set_xlabel("log10 raw first-neighbour distance (true tightness)")
        ax[2].set_ylabel("distance UMAP's graph actually uses")
        ax[2].set_title(
            "tightness is erased exactly, not approximately\n(nearest neighbour is at 0 "
            "for every point)",
            fontsize=10,
        )
        ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("C: UMAP normalises away the very contrast that makes families tight", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/sasbdb_pr_density")
    ap.add_argument(
        "--compare",
        nargs="+",
        default=["umap=runs/sasbdb_pr_umap", "densmap=runs/sasbdb_pr_densmap"],
    )
    ap.add_argument("--sasdbd", type=Path, default=DEFAULT_SASDBD)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--tight-pct", type=float, default=5.0)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/families")
    args = ap.parse_args()

    import pandas as pd

    run = _ROOT / args.run
    out = _ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(run / "X.npy").astype(np.float64)
    Z = np.load(run / "Z.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")
    rich = load_rich_meta(
        [str(c) for c in meta["sasbdb_code"]], args.sasdbd, out / "rich_meta.csv"
    )

    # ---- A ---------------------------------------------------------------- #
    lab, thr = tight_families(X, args.k, args.metric, args.tight_pct, args.min_size)
    fams = [np.nonzero(lab == c)[0] for c in range(lab.max() + 1)]
    fams.sort(key=lambda f: (-len(f), spread(X, f)))
    sizes = [len(f) for f in fams]
    pseudo = pseudo_families(X, sizes, args.k, args.metric, args.seed)

    print(f"tight-edge threshold (p{args.tight_pct:g} of 1-NN): {thr:.4g}")
    print(f"families (>= {args.min_size}): {len(fams)}  covering {sum(sizes)} points")
    s_fam = np.array([spread(X, f) for f in fams])
    s_ctl = np.array([spread(X, g) for g in pseudo])
    print(
        f"within-group spread, % of peak: families median {100 * np.median(s_fam):.2f}"
        f"   controls median {100 * np.median(s_ctl):.2f}"
    )

    for key, col in (("uniprot", "uniprot"), ("name", "name")):
        tag = rich[col].fillna("").astype(str).values
        obs, null, nf = coherence(lab, tag, args.seed)
        print(
            f"biological coherence by {key:<8} observed {obs:.3f}  shuffled {null:.3f}"
            f"  ({nf} families with labels)"
        )

    fig_families(X, fams, pseudo, rich, out / "A_families.png")

    # ---- B ---------------------------------------------------------------- #
    from scipy.spatial.distance import pdist

    d_all, _ = knn(X, max(SCALES), args.metric)
    d1 = d_all[:, 0]
    r_in = np.array([np.median(pdist(X[f], metric="cityblock")) for f in fams])
    r15 = np.array([np.median(d_all[f, args.k - 1]) for f in fams])

    dim = intrinsic_dim(knn_dist(X, 10, args.metric))
    la = {k: np.log10(density(d_all[:, :k], dim)) for k in SCALES}
    lz = np.log10(density(knn(Z, args.k, "euclidean")[0], float(Z.shape[1])))
    ldim = np.log10(np.clip(local_dim(d_all[:, : args.k]), 1e-6, None))
    base = np.column_stack([la[args.k], ldim])
    multi = np.column_stack([la[k] for k in SCALES] + [ldim])

    r2b, res_b = cv_fit(base, lz, seed=args.seed)
    r2m, res_m = cv_fit(multi, lz, seed=args.seed)
    print(f"\nR^2 on embedded log-density: k=15 only {r2b:.3f}   multi-scale {r2m:.3f}")
    print(f"leftover sd: k=15 only {res_b.std():.3f}   multi-scale {res_m.std():.3f}")
    if r2m < r2b:
        print("WARNING multi-scale fit is worse out of fold; its leftover is not comparable")
    mi_tab = []
    for kk in K_GRID:
        if kk >= len(Z) - 1:
            continue
        nb = knn(Z, kk, "euclidean")[1]
        mi_tab.append((kk, morans_i(res_b, nb), morans_i(res_m, nb)))
    print(f"{'k':>6}{'leftover I (k=15)':>20}{'leftover I (multi)':>20}")
    for kk, b, m in mi_tab:
        print(f"{kk:>6}{b:>20.3f}{m:>20.3f}")

    # Do these real families actually account for the map's unexplained density?
    # They are only a couple of percent of the points, so even a large per-point
    # effect could not carry a map-wide leftover. Stated explicitly so the real
    # structure of claim A is not silently credited with explaining claim B.
    inside = np.concatenate(fams) if fams else np.array([], int)
    mask = np.zeros(len(X), bool)
    mask[inside] = True
    share_pts = 100 * mask.mean()
    share_ss = 100 * (res_b[mask] ** 2).sum() / (res_b**2).sum()
    print(
        f"\nfamily members are {share_pts:.1f}% of points and carry {share_ss:.1f}% of "
        f"the leftover's sum of squares"
    )
    print(
        f"|leftover| median: inside families {np.median(np.abs(res_b[mask])):.3f}"
        f"   elsewhere {np.median(np.abs(res_b[~mask])):.3f}"
    )

    fig_graph(X, d1, thr, fams, np.array(sizes), r_in, r15, mi_tab, out / "B_graph.png")

    # ---- C ---------------------------------------------------------------- #
    comp = {"leanmap": compaction(Z, fams)}
    comp_ctl = {"leanmap": compaction(Z, pseudo)}
    for spec in args.compare:
        name, _, path = spec.partition("=")
        Zc = np.load(_ROOT / path / "Z.npy").astype(np.float64)
        if len(Zc) != len(Z):
            print(f"skip {name}: {len(Zc)} rows != {len(Z)}")
            continue
        comp[name] = compaction(Zc, fams)
        comp_ctl[name] = compaction(Zc, pseudo)

    print(f"\n{'map':<12}{'families':>12}{'controls':>12}{'ratio':>9}")
    for name in comp:
        f_med, c_med = np.median(comp[name]), np.median(comp_ctl[name])
        print(f"{name:<12}{f_med:>12.3f}{c_med:>12.3f}{f_med / max(c_med, 1e-12):>9.2f}")

    norm = umap_normalisation(X, args.k, args.metric)
    if norm is not None:
        sig, rho, dfull = norm
        raw = dfull[:, -1]
        nrm = (raw - rho) / np.maximum(sig, 1e-12)
        rng_raw = np.percentile(raw, 99) / max(np.percentile(raw, 1), 1e-12)
        rng_nrm = np.percentile(nrm, 99) / max(np.percentile(nrm, 1), 1e-12)
        print(
            f"\n15th-neighbour radius, p99/p1: raw {rng_raw:.1f}x"
            f"   after UMAP rho/sigma {rng_nrm:.1f}x"
        )
        # The strongest form of the argument needs no statistics. At the default
        # local_connectivity of 1, rho is the distance to the nearest *distinct*
        # neighbour and is subtracted off, so that neighbour lands at normalised
        # distance zero with full membership -- whether it is a near-duplicate
        # profile or the far side of a sparse void. Exact duplicates are excluded
        # from the check because they have r1 = 0 and rho then takes the first
        # strictly positive distance instead.
        r1 = d_all[:, 0]
        ok = r1 > 0
        print(
            f"rho equals the nearest distinct neighbour: max |rho - r1| = "
            f"{np.abs(rho[ok] - r1[ok]).max():.2e} over {ok.sum()} points "
            f"with r1 > 0, while r1 itself spans {r1[ok].min():.2e} to {r1.max():.2e}"
        )
        print(
            f"  ({(~ok).sum()} points are exact duplicates of another profile)"
        )
        print(
            "  => that neighbour normalises to distance 0 for every point, so a "
            "near-duplicate pair\n     and an isolated pair enter the layout "
            "indistinguishably."
        )
    else:
        print("\numap not importable: skipping the rho/sigma panels")

    fig_umap(comp, comp_ctl, norm, d_all[:, 0], d_all[:, args.k - 1], out / "C_umap.png")


if __name__ == "__main__":
    main()

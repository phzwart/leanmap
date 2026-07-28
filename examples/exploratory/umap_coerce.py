#!/usr/bin/env python
"""Sweep UMAP settings to reproduce the structure the leanmap map shows.

Both methods are given byte-identical input (the reference run's ``X.npy``), so
anything that differs is the embedding, not the preprocessing. Each candidate is
scored on the things this dataset actually is, established earlier:

``knn``       fraction of ambient 15-NN preserved -- plain neighbourhood fidelity
``trust``     sklearn trustworthiness at k=15
``rho_amb``   Spearman of ambient against embedded distances on sampled pairs
``grad``      how locally coherent the P(r) shape gradient is in the map; the
              data is a continuum along peak position, so a good layout keeps
              that varying smoothly rather than interleaving it
``dens``      Spearman of ambient against embedded local density
``clark``     Clark-Evans R on the embedding: 1 is Poisson-uniform, below 1 is
              clumped, so this is the direct measure of the clumping question
``islands``   DBSCAN fragments and the share of points left as noise
``tear``      ambient neighbours placed more than 5x the median apart

The reference row is the leanmap run itself, so every number is read against it.

Usage::

    python examples/exploratory/umap_coerce.py
    python examples/exploratory/umap_coerce.py --quick --k 15
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pr_islands import find_islands

_ROOT = Path(__file__).resolve().parents[2]


def stage2() -> list[tuple[str, dict]]:
    """Push the two knobs that actually moved: the density term and min_dist."""
    out = []
    for lam in (2.0, 5.0, 10.0):
        for md in (0.0, 0.5):
            out.append((f"dens{lam:g}_md{md:g}",
                        dict(n_neighbors=150, min_dist=md, densmap=True, dens_lambda=lam)))
    return out


def configs(quick: bool) -> list[tuple[str, dict]]:
    """UMAP settings, grouped by the knob each one is meant to test."""
    grid = [
        # Neighbourhood scale: the main lever on local-vs-global structure.
        ("nn15_md0.5", dict(n_neighbors=15, min_dist=0.5)),
        ("nn50_md0.5", dict(n_neighbors=50, min_dist=0.5)),
        ("nn150_md0.5", dict(n_neighbors=150, min_dist=0.5)),
        ("nn400_md0.5", dict(n_neighbors=400, min_dist=0.5)),
        # min_dist only sets how tightly points may pack once placed.
        ("nn50_md0.0", dict(n_neighbors=50, min_dist=0.0)),
        ("nn50_md0.9", dict(n_neighbors=50, min_dist=0.9)),
        # Density-preserving objective.
        ("nn50_densmap", dict(n_neighbors=50, densmap=True, dens_lambda=2.0, min_dist=0.5)),
        ("nn150_densmap", dict(n_neighbors=150, densmap=True, dens_lambda=2.0, min_dist=0.5)),
    ]
    if quick:
        return grid
    return grid + [
        # Repulsion is what carves voids into a continuum; ease it off.
        ("nn50_repel0.2", dict(n_neighbors=50, min_dist=0.5, repulsion_strength=0.2)),
        ("nn50_neg2", dict(n_neighbors=50, min_dist=0.5, negative_sample_rate=2)),
        ("nn150_repel0.2", dict(n_neighbors=150, min_dist=0.5, repulsion_strength=0.2)),
        # Force every point to have fully-connected close neighbours, which
        # stops the graph from shattering into fragments.
        ("nn50_lc3", dict(n_neighbors=50, min_dist=0.5, local_connectivity=3)),
        # Union-biased fuzzy set operation keeps weak links alive.
        ("nn50_setop1_lc3", dict(n_neighbors=50, min_dist=0.5, local_connectivity=3,
                                 set_op_mix_ratio=1.0)),
        ("nn150_long", dict(n_neighbors=150, min_dist=0.5, n_epochs=1000)),
        # Start from the leanmap layout: tests whether UMAP's objective keeps it.
        ("nn150_init_leanmap", dict(n_neighbors=150, min_dist=0.5, init="__REF__")),
        ("nn150_combo", dict(n_neighbors=150, min_dist=0.5, repulsion_strength=0.2,
                             local_connectivity=3, n_epochs=1000)),
    ]


def clark_evans(Z: np.ndarray) -> float:
    """Nearest-neighbour spacing against the Poisson expectation; <1 is clumped."""
    from scipy.spatial import ConvexHull
    from sklearn.neighbors import NearestNeighbors

    d = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
    area = float(ConvexHull(Z).volume)
    return float(d.mean() / (0.5 * np.sqrt(area / len(Z))))


def gradient_coherence(Z: np.ndarray, g: np.ndarray, k: int) -> float:
    """1 - (local variance / global variance) of a scalar over embedding kNN."""
    from sklearn.neighbors import NearestNeighbors

    nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    return float(1.0 - np.mean(g[nb].var(axis=1)) / g.var())


def score(X, Z, amb_nb, amb_edges, grad, k, seed):
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    from leanmap.evaluate import density_correspondence

    emb_nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    knn = float(np.mean([len(set(a) & set(b)) for a, b in zip(amb_nb, emb_nb)]) / k)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), 32768)
    j = rng.integers(0, len(X), 32768)
    ok = i != j
    i, j = i[ok], j[ok]
    d_amb = np.abs(X[i] - X[j]).sum(axis=1)
    d_emb = np.linalg.norm(Z[i] - Z[j], axis=1)

    e0, e1 = amb_edges[:, 0], amb_edges[:, 1]
    de = np.linalg.norm(Z[e0] - Z[e1], axis=1)
    tear = float((de > 5 * np.median(de)).mean())

    lab, _ = find_islands(Z, 8.0, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dens = density_correspondence(X, Z, k=k)
    return {
        "knn": knn,
        "trust": float(trustworthiness(X, Z, n_neighbors=k, metric="manhattan")),
        "rho_amb": float(spearmanr(d_amb, d_emb).statistic),
        "grad": gradient_coherence(Z, grad, k),
        "dens": float(dens["spearman"]),
        "clark": clark_evans(Z),
        "islands": int(sum(1 for c in np.unique(lab) if c >= 0)),
        "noise": float((lab < 0).mean()),
        "tear": tear,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen",
                    help="leanmap run supplying X.npy and the reference Z.npy")
    ap.add_argument("--out", type=Path, default=_ROOT / "runs" / "umap_coerce")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="parameter grid only")
    ap.add_argument("--stage2", action="store_true",
                    help="only the density-term follow-up sweep")
    args = ap.parse_args()

    ref = args.ref if args.ref.is_absolute() else Path.cwd() / args.ref
    X = np.load(ref / "X.npy").astype(np.float32)
    Z_ref = np.load(ref / "Z.npy").astype(np.float64)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=args.k + 1, metric="manhattan").fit(X)
    amb_nb = nn.kneighbors(X)[1][:, 1:]
    amb_edges = np.column_stack([np.repeat(np.arange(len(X)), args.k), amb_nb.ravel()])

    bins = np.linspace(0.0, 1.0, X.shape[1])
    grad = (X / X.sum(axis=1, keepdims=True)) @ bins
    peak = bins[X.argmax(axis=1)]

    # UMAP wants an init on roughly its own scale.
    init_ref = 10.0 * (Z_ref - Z_ref.mean(0)) / np.abs(Z_ref - Z_ref.mean(0)).max()

    rows = [dict(name="leanmap (reference)", seconds=np.nan,
                 **score(X, Z_ref, amb_nb, amb_edges, grad, args.k, args.seed))]
    np.save(out / "Z_leanmap.npy", Z_ref.astype(np.float32))
    print(f"reference: {rows[0]}")

    import umap

    embeddings = {"leanmap (reference)": Z_ref}
    for name, kw in (stage2() if args.stage2 else configs(args.quick)):
        kw = dict(kw)
        if kw.get("init") == "__REF__":
            kw["init"] = init_ref
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = umap.UMAP(n_components=2, metric="manhattan",
                          random_state=args.seed, **kw).fit_transform(X)
        Z = np.asarray(Z, dtype=np.float64)
        dt = time.perf_counter() - t0
        s = score(X, Z, amb_nb, amb_edges, grad, args.k, args.seed)
        rows.append(dict(name=name, seconds=dt, **s))
        embeddings[name] = Z
        np.save(out / f"Z_{name}.npy", Z.astype(np.float32))
        print(f"  {name:22s} {dt:5.1f}s  knn={s['knn']:.3f} trust={s['trust']:.3f} "
              f"rho={s['rho_amb']:.3f} grad={s['grad']:.3f} dens={s['dens']:+.3f} "
              f"clark={s['clark']:.2f} islands={s['islands']:3d} tear={s['tear']:.3f}",
              flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False)
    print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    _plot(embeddings, tab, peak, out)


def _plot(embeddings, tab, peak, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(embeddings)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, (name, Z) in zip(axs, embeddings.items()):
        r = tab.loc[tab["name"] == name].iloc[0]
        ax.scatter(Z[:, 0], Z[:, 1], c=peak, cmap="plasma", s=2.5, linewidths=0)
        lo, hi = np.percentile(Z, [0.3, 99.7], axis=0)
        mid, half = 0.5 * (lo + hi), 0.55 * float((hi - lo).max())
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nknn {r['knn']:.2f}  grad {r['grad']:.2f}  "
                     f"clark {r['clark']:.2f}  isl {int(r['islands'])}", fontsize=8)
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle("coercing UMAP toward the leanmap structure — colour = P(r) peak position",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "grid.png", dpi=120)
    plt.close(fig)
    print(f"\nsaved {out / 'grid.png'} and {out / 'summary.csv'}")


if __name__ == "__main__":
    main()

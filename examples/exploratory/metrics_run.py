"""Evaluation helpers for exploratory runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def compute_metrics(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_neighbors: int = 10,
    seed: int = 0,
    include_geodesic: bool = True,
) -> Dict[str, Any]:
    """Trustworthiness / continuity + ambient (and optional geodesic) Shepard."""
    from leanmap.distance import EuclideanDistance
    from leanmap.evaluate import (
        shepard_pairs_ambient,
        shepard_stats,
        trustworthiness_continuity,
    )

    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    out: Dict[str, Any] = {}

    k_list = tuple(
        sorted({k for k in (5, 15, min(50, max(5, len(X) // 20))) if k < len(X)})
    )
    try:
        tc = trustworthiness_continuity(
            X, Z, EuclideanDistance(), k_list=k_list or (5,), n_sample=min(5000, len(X)), seed=seed
        )
        out.update(tc)
    except Exception as exc:  # noqa: BLE001
        out["trust_error"] = str(exc)

    d_orig, d_embed = shepard_pairs_ambient(X, Z, n_pairs=32768, seed=seed)
    st = shepard_stats(d_orig, d_embed)
    out["ambient_spearman"] = st["spearman"]
    out["ambient_stress"] = st["stress"]
    out["ambient_alpha"] = st["alpha"]
    out["ambient_pairs"] = st["n_pairs"]

    if include_geodesic:
        try:
            geo = _geodesic_on_knn(X, Z, n_neighbors=n_neighbors, seed=seed)
            out.update(geo)
        except Exception as exc:  # noqa: BLE001
            out["geodesic_error"] = str(exc)
            out["geodesic_spearman"] = float("nan")
            out["geodesic_stress"] = float("nan")
            out["geodesic_pairs"] = 0

    return out


def _geodesic_on_knn(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_neighbors: int,
    seed: int,
) -> Dict[str, float]:
    """Rebuild a light kNN fuzzy graph and score geodesic fidelity."""
    import torch
    from leanmap.evaluate import geodesic_fidelity, shepard_pairs_geodesic, shepard_stats
    from leanmap.graph import build_graph
    from leanmap.metrics import get_metric

    Xt = torch.as_tensor(X, dtype=torch.float32)
    metric = get_metric("l2")
    # Small landmark budget — evaluation graph only, not the trained one.
    n_lm = int(min(64, max(16, len(X) // 20)))
    graph, *_ = build_graph(
        Xt,
        metric,
        n_neighbors=int(n_neighbors),
        n_landmarks=n_lm,
        dedup=False,
        seed=seed,
    )
    # Map rep-graph distances onto embedding of those reps.
    rep_idx = graph.reps.rep_idx.cpu().numpy().astype(np.int64)
    Z_reps = Z[rep_idx]
    gd, ed = shepard_pairs_geodesic(
        graph.edges, graph.weights, Z_reps, n_sources=64, max_targets=512, seed=seed
    )
    if gd.size == 0:
        return {
            "geodesic_spearman": float("nan"),
            "geodesic_stress": float("nan"),
            "geodesic_pairs": 0,
        }
    st = shepard_stats(gd, ed)
    # Also expose via geodesic_fidelity keys for consistency.
    fid = geodesic_fidelity(graph.edges, graph.weights, Z_reps, seed=seed)
    return {
        "geodesic_spearman": fid["geodesic_spearman"],
        "geodesic_stress": fid["geodesic_stress"],
        "geodesic_pairs": fid["geodesic_pairs"],
        "geodesic_alpha": st["alpha"],
    }


def label_metrics(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
    with_ambient: bool = True,
) -> Dict[str, Any]:
    """How much of a *known* grouping survives the embedding.

    This is the only part of the battery that can distinguish "the map preserved
    real structure" from "the map invented structure". ``label_acc_X`` is the
    ceiling: what the raw features already support. A large ``label_acc_X`` with
    a small ``label_acc_Z`` means the embedding destroyed real structure.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    y = np.asarray(y)
    out: Dict[str, Any] = {}
    n_cls = int(len(np.unique(y)))
    out["n_classes"] = n_cls
    if n_cls < 2:
        return out

    out["label_acc_Z"] = float(
        cross_val_score(KNeighborsClassifier(k), Z, y, cv=5).mean()
    )
    if with_ambient:
        out["label_acc_X"] = float(
            cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean()
        )
        out["label_acc_retained"] = out["label_acc_Z"] / max(out["label_acc_X"], 1e-9)
    lab = KMeans(n_clusters=n_cls, n_init=10, random_state=seed).fit_predict(Z)
    out["label_ari"] = float(adjusted_rand_score(y, lab))
    out["label_sil_Z"] = float(silhouette_score(Z, y))
    if with_ambient:
        out["label_sil_X"] = float(silhouette_score(X, y))
    return out


def kmeans_silhouette_floor(
    n: int,
    k: int,
    *,
    seed: int = 0,
    reps: int = 3,
) -> float:
    """Chance level for k-means silhouette on a structureless 2-D cloud.

    k-means partitions of any 2-D cloud are compact by construction, so
    silhouette in Z has a high floor (~0.35 at k=13) and a "crisp cluster" read
    off a scatter plot is usually just this floor. Without it, silhouette in Z is
    uninterpretable.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    vals = []
    for r in range(reps):
        Zr = rng.normal(size=(n, 2))
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed + r).fit_predict(Zr)
        vals.append(silhouette_score(Zr, lab))
    return float(np.mean(vals))


def artifact_metrics(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    k: int = 10,
    seed: int = 0,
    include_floor: bool = True,
) -> Dict[str, Any]:
    """Detect decoder-manufactured islands.

    Clusters the embedding and scores that same partition in both spaces. High
    ``sil_Z`` with ``sil_X`` near zero means the groups exist only in the
    picture. ``sil_Z_excess_over_floor`` corrects for the 2-D k-means floor.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    sil_z = float(silhouette_score(Z, lab))
    sil_x = float(silhouette_score(X, lab))
    out = {
        "kmeans_k": int(k),
        "kmeans_sil_Z": sil_z,
        "kmeans_sil_X": sil_x,
        "kmeans_sil_gap": sil_z - sil_x,
    }
    if include_floor:
        floor = kmeans_silhouette_floor(len(Z), k, seed=seed)
        out["kmeans_sil_floor_2d"] = floor
        out["kmeans_sil_Z_excess_over_floor"] = sil_z - floor
    return out


def density_metrics(X: np.ndarray, Z: np.ndarray, *, k: int = 15) -> Dict[str, Any]:
    """Ambient vs embedded local-density correspondence (no plotting)."""
    from leanmap.evaluate import density_correspondence

    try:
        dc = density_correspondence(X, Z, k=k)
        return {
            "density_spearman": float(dc["spearman"]),
            "density_pearson_log": float(dc["pearson_log"]),
            "density_fit_slope": float(dc["fit_slope"]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"density_error": str(exc), "density_spearman": float("nan")}


def uniformity_metrics(X: np.ndarray, Z: np.ndarray, *, k: int = 10) -> Dict[str, Any]:
    """How evenly the layout spreads points, and how faithfully.

    Distinct from ``density_metrics``, which only asks whether dense ambient
    regions stay relatively dense: a layout can rank densities correctly while
    still being a field of knots and voids. ``spacing_cv`` is the coefficient of
    variation of the kNN radius in Z, and ``area_sd`` the spread of local
    magnification ``log(r_Z / r_X)``, which is 0 for an area-preserving map.

    Reference points on a uniformly sampled s-curve: the true flattening and a
    Poisson sample both give spacing_cv ~0.18, PCA-2D 0.30, UMAP 0.37.
    """
    from sklearn.neighbors import NearestNeighbors

    out: Dict[str, Any] = {}
    try:
        for name, A in (("x", np.asarray(X, np.float64)), ("z", np.asarray(Z, np.float64))):
            nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
            d, _ = nn.kneighbors(A)
            out[f"_r_{name}"] = d[:, -1]
        rz, rx = out.pop("_r_z"), out.pop("_r_x")
        out["spacing_cv"] = float(rz.std() / max(rz.mean(), 1e-12))
        ratio = np.log(np.clip(rz, 1e-12, None) / np.clip(rx, 1e-12, None))
        out["area_sd"] = float(ratio.std())
    except Exception as exc:  # noqa: BLE001
        out = {"uniformity_error": str(exc), "spacing_cv": float("nan")}
    return out


def banded_shepard(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_pairs: int = 32768,
    seed: int = 0,
) -> Dict[str, Any]:
    """Ambient Shepard correlation split by distance tercile.

    A single overall Spearman hides which scale is preserved; local structure
    can look fine while global ordering is destroyed.
    """
    from scipy.stats import spearmanr
    from leanmap.evaluate import shepard_pairs_ambient

    d_o, d_e = shepard_pairs_ambient(X, Z, n_pairs=n_pairs, seed=seed)
    if d_o.size == 0:
        return {}
    out: Dict[str, Any] = {}
    edges = np.quantile(d_o, [0.0, 1 / 3, 2 / 3, 1.0])
    for b, band in enumerate(("local", "mid", "global")):
        m = (d_o >= edges[b]) & (d_o <= edges[b + 1])
        out[f"ambient_spearman_{band}"] = (
            float(spearmanr(d_o[m], d_e[m]).correlation) if m.sum() > 32 else float("nan")
        )
    return out


def emd_metrics(
    Z: np.ndarray,
    *,
    emd_cache: PathLike,
    rows: np.ndarray,
    k: int = 15,
    seed: int = 0,
) -> Dict[str, Any]:
    """Score an embedding against a precomputed EMD reference.

    Every other metric here treats pixel L2 as the truth, and every embedder in
    this harness is fit from an L2 kNN graph, so those metrics partly reward
    reproducing the input. EMD is not shown to any method, which is what makes
    it able to arbitrate.

    ``rows`` selects which rows of the cached matrix (and of ``Z``) to score, so
    the caller decides whether this is the train or the holdout regime.
    """
    from leanmap.emd import reference_knn_overlap, reference_shepard

    rows = np.asarray(rows, dtype=np.int64)
    D = np.load(str(emd_cache), mmap_mode="r")
    D_sub = np.asarray(D[np.ix_(rows, rows)], dtype=np.float64)
    Zs = np.asarray(Z, dtype=np.float64)
    if len(Zs) != len(rows):
        raise ValueError(f"Z has {len(Zs)} rows but {len(rows)} were selected")
    out: Dict[str, Any] = {}
    out.update(reference_shepard(D_sub, Zs, prefix="emd", seed=seed))
    out.update(reference_knn_overlap(D_sub, Zs, prefix="emd", k=k))
    return out


def full_battery(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    n_neighbors: int = 10,
    seed: int = 0,
    include_geodesic: bool = True,
    artifact_k: Optional[int] = None,
    with_ambient_labels: bool = True,
    emd_cache: Optional[PathLike] = None,
    emd_rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Geometry + label + artifact metrics for one embedding.

    Note every number here is uncalibrated: interpret it only against a null
    refit with the same configuration (see :mod:`nulls`).
    """
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    out = compute_metrics(
        X, Z, n_neighbors=n_neighbors, seed=seed, include_geodesic=include_geodesic
    )
    out.update(banded_shepard(X, Z, seed=seed))
    out.update(density_metrics(X, Z, k=min(15, max(5, len(X) // 20))))
    out.update(uniformity_metrics(X, Z))
    if y is not None:
        out.update(
            label_metrics(X, Z, y, seed=seed, with_ambient=with_ambient_labels)
        )
    k_art = artifact_k
    if k_art is None:
        k_art = int(len(np.unique(y))) if y is not None else 10
    out.update(artifact_metrics(X, Z, k=k_art, seed=seed))
    if emd_cache is not None and emd_rows is not None:
        try:
            out.update(
                emd_metrics(
                    Z, emd_cache=emd_cache, rows=emd_rows, k=n_neighbors, seed=seed
                )
            )
        except Exception as exc:  # noqa: BLE001
            out["emd_error"] = str(exc)
    return out


def shepard_arrays(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    mode: str = "ambient",
    n_neighbors: int = 10,
    seed: int = 0,
):
    """Return ``(d_orig, d_embed, xlabel)`` for Shepard plotting."""
    from leanmap.evaluate import shepard_pairs_ambient

    if mode == "geodesic":
        import torch
        from leanmap.evaluate import shepard_pairs_geodesic
        from leanmap.graph import build_graph
        from leanmap.metrics import get_metric

        Xt = torch.as_tensor(X, dtype=torch.float32)
        graph, *_ = build_graph(
            Xt,
            get_metric("l2"),
            n_neighbors=int(n_neighbors),
            n_landmarks=int(min(64, max(16, len(X) // 20))),
            dedup=False,
            seed=seed,
        )
        rep_idx = graph.reps.rep_idx.cpu().numpy().astype(np.int64)
        d_o, d_e = shepard_pairs_geodesic(
            graph.edges, graph.weights, Z[rep_idx], seed=seed
        )
        return d_o, d_e, "graph geodesic distance"
    d_o, d_e = shepard_pairs_ambient(X, Z, seed=seed)
    return d_o, d_e, "ambient Euclidean distance"


def write_json(path: PathLike, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n")
    return path


def read_json(path: PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

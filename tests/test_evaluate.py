"""Tests for Shepard / geodesic-fidelity evaluation helpers."""

from __future__ import annotations

import numpy as np

from leanmap.evaluate import (
    geodesic_fidelity,
    neighborhood_rank_agreement,
    persistence_summary,
    procrustes_disagreement,
    shepard_pairs_ambient,
    shepard_pairs_geodesic,
    shepard_stats,
)


def _similarity(Z, angle=0.7, scale=2.5, shift=(3.0, -1.0)):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return scale * (Z @ R) + np.asarray(shift)


def test_procrustes_disagreement_zero_under_similarity():
    """A rotated, rescaled, translated copy is the same map in a different gauge."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 2))
    idx = rng.permutation(200)
    out = procrustes_disagreement(Z, _similarity(Z), idx[:100], idx[100:])
    assert out["median"] < 1e-4
    assert abs(out["scale"] - 2.5) < 1e-3


def test_procrustes_disagreement_scored_out_of_sample():
    """Alignment is fitted on anchors and scored on disjoint points."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 2))
    Z2 = _similarity(Z).copy()
    idx = rng.permutation(200)
    anchor, evaluate = idx[:100], idx[100:]
    # Perturb only the evaluation half: a fit on anchors cannot absorb it.
    Z2[evaluate] += rng.normal(scale=0.5, size=(len(evaluate), 2))
    out = procrustes_disagreement(Z, Z2, anchor, evaluate)
    assert out["median"] > 0.05


def test_rank_agreement_is_gauge_invariant():
    """Neighbour ranks survive rotation/scale but not a reshuffle."""
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(150, 2))
    same = neighborhood_rank_agreement(Z, _similarity(Z), k=10, seed=0)
    assert same["spearman"] > 0.999
    assert same["jaccard"] > 0.99

    shuffled = Z[rng.permutation(150)]
    diff = neighborhood_rank_agreement(Z, shuffled, k=10, seed=0)
    assert diff["spearman"] < 0.5
    assert diff["jaccard"] < same["jaccard"]


def test_persistence_summary_separates_gauge_from_instability():
    """High rank agreement with large coordinate disagreement is a gauge artefact."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(150, 2))
    gauge = persistence_summary([Z, _similarity(Z), _similarity(Z, angle=-1.2)], k=10)
    assert gauge["coord_disagreement_median"] < 1e-3
    assert gauge["rank_spearman_mean"] > 0.999

    noisy = [Z + rng.normal(scale=0.8, size=Z.shape) for _ in range(3)]
    unstable = persistence_summary(noisy, k=10)
    assert unstable["rank_spearman_mean"] < gauge["rank_spearman_mean"]
    assert unstable["coord_disagreement_worst"] > gauge["coord_disagreement_worst"]
    assert unstable["n_pairs"] == 3


def test_shepard_stats_perfect_scale():
    rng = np.random.default_rng(0)
    g = rng.uniform(0.1, 5.0, size=2000)
    e = g / 2.5  # exact isotropic scale
    st = shepard_stats(g, e)
    assert st["n_pairs"] == 2000
    assert st["spearman"] > 0.999
    assert abs(st["alpha"] - 2.5) < 1e-6
    assert st["stress"] < 1e-6


def test_shepard_pairs_ambient_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    Z = X[:, :2]  # not isometric, but pairs exist
    d_x, d_z = shepard_pairs_ambient(X, Z, n_pairs=500, seed=0)
    assert d_x.shape == d_z.shape
    assert d_x.size == 500
    assert np.all(d_x > 0) and np.all(d_z >= 0)


def test_shepard_pairs_geodesic_chain():
    # Path graph 0-1-2-...-9 with unit fuzzy weights; embedding = index on line.
    n = 10
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    weights = np.ones(n - 1, dtype=np.float64)
    Z = np.arange(n, dtype=np.float64).reshape(-1, 1)
    gd, ed = shepard_pairs_geodesic(
        edges, weights, Z, n_sources=n, max_targets=n, seed=0
    )
    assert gd.size > 0
    st = shepard_stats(gd, ed)
    assert st["spearman"] > 0.99
    gf = geodesic_fidelity(edges, weights, Z, n_sources=n, max_targets=n, seed=0)
    assert gf["geodesic_pairs"] == st["n_pairs"]
    assert abs(gf["geodesic_spearman"] - st["spearman"]) < 1e-12

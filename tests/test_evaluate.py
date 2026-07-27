"""Tests for Shepard / geodesic-fidelity evaluation helpers."""

from __future__ import annotations

import numpy as np

from leanmap.evaluate import (
    geodesic_fidelity,
    shepard_pairs_ambient,
    shepard_pairs_geodesic,
    shepard_stats,
)


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

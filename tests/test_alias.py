"""Alias table correctness and memmap-scale sampling tests."""

from __future__ import annotations

import resource
import sys

import numpy as np
import torch

from leanmap.build.pipeline import Graph, GraphStats, Representatives
from leanmap.sampling.alias import (
    TwoLevelAlias,
    _alias_draw,
    build_edge_alias,
    build_two_level_alias,
    freeze_alias_tables,
)
from leanmap.sampling.edges import EdgeSampler


def test_empirical_frequencies_match_weights_chi2():
    w = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)
    prob, alias = build_edge_alias(w)
    rng = np.random.default_rng(0)
    n_draw = 50_000
    draws = _alias_draw(n_draw, prob, alias, rng)
    counts = np.bincount(draws, minlength=len(w)).astype(np.float64)
    expected = n_draw * (w / w.sum())
    # Pearson chi-square; critical value χ²_{0.999}(4) ≈ 18.47 — use slack.
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < 30.0
    # Absolute frequency error within ~ few σ of Multinomial.
    abs_err = np.abs(counts / n_draw - w / w.sum())
    assert np.all(abs_err < 0.01)


def test_single_level_vs_two_level_same_normalized_mass():
    rng = np.random.default_rng(1)
    w = rng.random(257).astype(np.float64) + 0.05
    p1 = w / w.sum()
    tla = build_two_level_alias(w, shard_size=16)
    p2 = tla.normalized_weights
    assert p1.shape == p2.shape
    np.testing.assert_allclose(p1, p2, rtol=0.0, atol=1e-12)

    # Empirical: both generators hit the same mass within CLT noise.
    n_draw = 80_000
    single = _alias_draw(n_draw, *build_edge_alias(w), rng=np.random.default_rng(2))
    two = tla.draw_flat(n_draw, np.random.default_rng(3))
    f1 = np.bincount(single, minlength=len(w)) / n_draw
    f2 = np.bincount(two, minlength=len(w)) / n_draw
    # Max |f - p| should be small; compare generators to each other loosely.
    assert np.max(np.abs(f1 - p1)) < 0.01
    assert np.max(np.abs(f2 - p1)) < 0.01


def test_two_level_draw_shape_and_range():
    w = np.arange(1, 21, dtype=np.float64)
    tla = build_two_level_alias(w, shard_size=5)
    shards, local = tla.draw(1000, np.random.default_rng(0))
    assert shards.shape == (1000,)
    assert local.shape == (1000,)
    assert shards.min() >= 0 and shards.max() < tla.n_shards
    for s, loc in zip(shards, local):
        assert 0 <= int(loc) < int(tla._shard_sizes[int(s)])


def test_freeze_alias_tables_keys():
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64)
    weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    R = 3
    reps = Representatives(
        rep_idx=torch.arange(R, dtype=torch.int64),
        member_of=torch.arange(R, dtype=torch.int64),
        weight=torch.ones(R, dtype=torch.float32),
        offsets=torch.arange(R + 1, dtype=torch.int64),
        values=torch.arange(R, dtype=torch.int64),
    )
    graph = Graph(
        edges=edges,
        weights=weights,
        reps=reps,
        knn_idx=torch.zeros(R, 1, dtype=torch.int64),
        stats=GraphStats(n_reps=R),
    )
    tables = freeze_alias_tables(graph)
    assert set(tables) >= {"prob", "alias"}
    assert tables["prob"].shape == (3,)
    assert tables["alias"].shape == (3,)
    tables2 = freeze_alias_tables(graph, shard_size=2)
    assert isinstance(tables2["two_level"], TwoLevelAlias)
    assert tables2["shard_size"] == 2


def test_edge_sampler_store_memmap_path(tmp_path):
    """Store kwargs path: alias + member CSR (+ edges) without graph.edges copy."""
    X = torch.randn(12, 3)
    # Two cells, multiple members each; one edge between them.
    offsets = np.array([0, 6, 12], dtype=np.int64)
    values = np.arange(12, dtype=np.int64)
    edges = np.array([[0, 1]], dtype=np.int64)
    weights = np.array([1.0], dtype=np.float64)
    prob, alias = build_edge_alias(weights)

    # Write memmaps
    def _mm(name, arr):
        path = tmp_path / name
        fp = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        fp[:] = arr
        fp.flush()
        return np.memmap(path, dtype=arr.dtype, mode="r", shape=arr.shape)

    mm_prob = _mm("prob.dat", prob)
    mm_alias = _mm("alias.dat", alias.astype(np.int64))
    mm_off = _mm("offsets.dat", offsets)
    mm_val = _mm("values.dat", values)
    mm_edges = _mm("edges.dat", edges)
    mm_w = _mm("weights.dat", weights)

    # Minimal graph stub — edges tensor unused when edges= memmap provided.
    R = 2
    reps = Representatives(
        rep_idx=torch.tensor([0, 6], dtype=torch.int64),
        member_of=torch.arange(12, dtype=torch.int64) // 6,
        weight=torch.tensor([6.0, 6.0]),
        offsets=torch.as_tensor(offsets),
        values=torch.as_tensor(values),
    )
    graph = Graph(
        edges=torch.zeros(0, 2, dtype=torch.int64),  # empty / not used
        weights=torch.as_tensor(weights, dtype=torch.float32),
        reps=reps,
        knn_idx=torch.zeros(R, 1, dtype=torch.int64),
        stats=GraphStats(n_reps=R),
    )
    samp = EdgeSampler(
        X,
        graph,
        seed=0,
        alias_prob=mm_prob,
        alias_alias=mm_alias,
        member_offsets=mm_off,
        member_values=mm_val,
        edges=mm_edges,
        weights=mm_w,
    )
    xi, xj, w, eidx = samp.sample(8)
    assert xi.shape == (8, 3)
    assert xj.shape == (8, 3)
    assert eidx.shape == (8,)
    assert torch.all(eidx == 0)
    assert torch.allclose(w, torch.ones(8))


def test_large_shard_alias_rss_soft_ceiling():
    """TwoLevelAlias with 1e5 shards: draw loop must not explode RSS."""
    S = 100_000
    # One weight per shard — no giant edge endpoint tensor.
    weights = (np.arange(S, dtype=np.float64) % 97) + 1.0
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tla = build_two_level_alias(weights, shard_size=1)
    assert tla.n_shards == S
    rng = np.random.default_rng(0)
    for _ in range(10):
        tla.draw(1000, rng)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS — normalize to bytes.
    scale = 1 if sys.platform == "darwin" else 1024
    delta = (rss_after - rss_before) * scale
    # Soft ceiling: building 1e5 size-1 shards should stay well under ~1.5 GiB
    # of *additional* peak RSS (tables are O(S), not O(S²)).
    assert delta < int(1.5 * 1024**3)

"""Tests for streaming cover graph construction."""

from __future__ import annotations

import pytest
import torch

from leanmap.build.streaming import (
    StreamingBuildReport,
    build_graph_pyramid_streaming,
    build_graph_streaming,
    knn_overlap_jaccard,
)
from leanmap.metrics import wrap_metric


def test_knn_overlap_identical():
    knn = torch.stack([(torch.arange(8) + i + 1) % 10 for i in range(10)])
    assert knn_overlap_jaccard(knn, knn.clone(), n_sample=10, seed=0) == pytest.approx(
        1.0
    )


def test_streaming_covers_all_rows():
    torch.manual_seed(0)
    X = torch.randn(400, 8)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graph, M, top1, topc, report = build_graph_streaming(
        X,
        metric,
        ingest_batch=80,
        seed_size=100,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=None,
        delta="eps",
        seed=0,
        knn_mode="brute",
        compute_knn_overlap=True,
    )
    assert isinstance(report, StreamingBuildReport)
    assert int(graph.reps.member_of.shape[0]) == 400
    assert int((graph.reps.member_of >= 0).sum()) == 400
    assert int(graph.reps.member_of.max()) < int(graph.reps.rep_idx.shape[0])
    assert graph.edges.shape[0] > 0
    assert report.n_reps == int(graph.reps.rep_idx.shape[0])
    assert report.compression_ratio == pytest.approx(
        400.0 / max(report.n_reps, 1), rel=1e-5
    )
    assert "streaming" in graph.stats.extra
    st = graph.stats.extra["streaming"]
    assert "n_absorbed" in st and "n_spawned" in st
    assert "rounds" in st and len(st["rounds"]) == report.n_rounds
    assert report.n_rounds >= 1
    assert report.knn_overlap is not None
    assert 0.0 <= float(report.knn_overlap) <= 1.0


def test_streaming_duplicates_absorb():
    torch.manual_seed(1)
    base = torch.randn(50, 6)
    X = base.repeat(8, 1)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=1)
    _graph, _M, _a1, _ac, report = build_graph_streaming(
        X,
        metric,
        ingest_batch=60,
        seed_size=80,
        n_neighbors=10,
        n_landmarks=12,
        epsilon=1e-5,
        delta="eps",
        seed=1,
        knn_mode="brute",
        compute_knn_overlap=False,
    )
    assert report.n_absorbed > 0
    assert report.compression_ratio > 2.0


def test_streaming_pyramid_and_freeze_shape():
    torch.manual_seed(2)
    X = torch.randn(300, 8)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=2)
    graphs, M, a1, ac, report = build_graph_pyramid_streaming(
        X,
        metric,
        pyramid_scales=2,
        pyramid_min_reps=32,
        ingest_batch=70,
        seed_size=90,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=None,
        delta="eps",
        seed=2,
        knn_mode="brute",
    )
    assert len(graphs) >= 1
    assert M.shape[0] <= 16
    assert a1.shape[0] == 300
    assert ac.shape[0] == 300
    assert report.n_reps == int(graphs[0].reps.rep_idx.shape[0])

"""Tests for geodesic gauge level selection (PR-8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from leanmap.graph import build_graph_pyramid
from leanmap.landmarks import classical_mds
from leanmap.losses.geo import (
    DEFAULT_GAUGE_R_THRESHOLD,
    gauge_nu_diagnostic,
    landmark_geodesics_on_level,
    metric_edge_lengths,
    select_gauge_level,
)
from leanmap.metrics import wrap_metric
from leanmap.store.dirstore import DirStore


def test_select_gauge_level_thresholds():
    assert select_gauge_level(0) == 0
    assert select_gauge_level(1) == 0
    assert select_gauge_level(int(DEFAULT_GAUGE_R_THRESHOLD) - 1) == 0
    assert select_gauge_level(int(DEFAULT_GAUGE_R_THRESHOLD)) == 1
    assert select_gauge_level(int(DEFAULT_GAUGE_R_THRESHOLD) + 1) == 1
    assert select_gauge_level(10, threshold=10) == 1
    assert select_gauge_level(9, threshold=10) == 0


def test_metric_edge_lengths_not_squashed_weights():
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 5], [0, 7]], dtype=torch.int64)
    # Squashed affinities in (0, 1] — must not be used as lengths.
    weights = torch.tensor([0.9, 0.4, 0.15, 0.05], dtype=torch.float32)
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    lengths = metric_edge_lengths(X, edges, metric)
    assert lengths.shape == weights.shape
    assert not torch.allclose(lengths, weights)
    # Lengths are ambient metric distances of endpoints (incl. natural_scale).
    expected = torch.diagonal(metric(X[edges[:, 0]], X[edges[:, 1]]))
    assert torch.allclose(lengths, expected, rtol=1e-5, atol=1e-5)
    # And they are not a monotone rescaling of memberships either.
    assert float(torch.corrcoef(torch.stack([lengths, weights]))[0, 1].abs()) < 0.99


def test_gauge_nu_diagnostic_finite():
    torch.manual_seed(1)
    # Simple Euclidean distance matrix embeddable in 2-D.
    pts = torch.randn(12, 2)
    D = torch.cdist(pts, pts)
    Y, _ = classical_mds(D, d=2, return_diagnostics=True)
    nu = gauge_nu_diagnostic(Y, D)
    assert np.isfinite(nu)
    assert nu >= 0.0


def _level_geodesics(graphs, X, metric, M, level: int):
    g = graphs[level]
    X_rep = X[g.reps.rep_idx]
    lengths = metric_edge_lengths(X_rep, g.edges, metric)
    from leanmap.distance import chunked_cdist

    _, nn_idx = chunked_cdist(metric, M, X, topk=1, out_device=X.device)
    lm_raw = nn_idx[:, 0].cpu().to(torch.int64)
    lm_level = g.reps.member_of[lm_raw].to(torch.int64)
    G = landmark_geodesics_on_level(g, lengths, lm_level)
    return G


def test_level0_vs_level1_geodesics_agree():
    """On a multi-level pyramid, level-0 and level-1 landmark geodesics correlate."""
    torch.manual_seed(2)
    # Enough points + low min_reps so coarsening actually produces level 1.
    n = 400
    t = torch.linspace(0, 4 * np.pi, n)
    X = torch.stack([t * torch.cos(t), t * torch.sin(t), t], dim=1).float()
    X = X + 0.02 * torch.randn_like(X)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=2)
    graphs, M, _, _ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=16,
        n_neighbors=10,
        n_landmarks=24,
        epsilon=0.0,
        seed=2,
        knn_mode="brute",
        dedup=False,
    )
    assert len(graphs) >= 2, f"expected ≥2 pyramid levels, got {len(graphs)}"

    G0 = _level_geodesics(graphs, X, metric, M, 0)
    G1 = _level_geodesics(graphs, X, metric, M, 1)
    ii, jj = torch.triu_indices(G0.shape[0], G0.shape[0], offset=1)
    a = G0[ii, jj].numpy()
    b = G1[ii, jj].numpy()
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    assert mask.sum() >= 20, "need enough finite landmark pairs"
    rho = float(spearmanr(a[mask], b[mask]).correlation)
    rel = float(np.median(np.abs(a[mask] - b[mask]) / np.maximum(a[mask], 1e-12)))
    assert rho > 0.5 or rel < 0.75, (
        f"level-0 vs level-1 geodesics disagree: spearman={rho:.3f} rel_med={rel:.3f}"
    )


def test_dirstore_writes_gauge_json(tmp_path: Path):
    torch.manual_seed(3)
    X = torch.randn(60, 4)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=3)
    graphs, M, top1, topc = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=1,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        n_neighbors=8,
        n_landmarks=8,
        epsilon=0.0,
        seed=3,
        knn_mode="brute",
    )
    graphs[0].stats.extra["gauge_level"] = 0
    graphs[0].stats.extra["nu"] = 0.0123
    n = int(X.shape[0])
    store = DirStore(tmp_path / "store")
    store.save(
        graphs=graphs,
        M=M,
        assign_top1=top1,
        assign_topc=topc,
        train_idx=torch.arange(n, dtype=torch.int64),
        calib_idx=torch.arange(0, dtype=torch.int64),
        metric_name="l2",
        n_all=n,
        n_neighbors=8,
        epsilon=0.0,
        seed=3,
        dedup=True,
        X=X,
    )
    gauge_path = tmp_path / "store" / "gauge" / "gauge.json"
    assert gauge_path.exists()
    payload = json.loads(gauge_path.read_text())
    assert payload["gauge_level"] == 0
    assert payload["nu"] == pytest.approx(0.0123)

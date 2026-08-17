"""Tests for the PR-3 resolution contract (ε + δ)."""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from leanmap.build.resolution import solve_delta
from leanmap.graph import build_graph
from leanmap.metrics import wrap_metric


REQUIRED_REPORT_KEYS = {
    "delta",
    "eps_ref",
    "r_est",
    "r_band",
    "alpha_guard",
    "guard_ok",
    "mode",
}


def test_delta_none_matches_eps_n_reps():
    """delta=None / \"eps\" must keep today's ε-net (same R)."""
    torch.manual_seed(0)
    X = torch.randn(120, 6)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=0)
    kw = dict(n_neighbors=8, n_landmarks=16, epsilon=None, seed=0, knn_mode="brute")
    g0, *_ = build_graph(X, metric, **kw, delta=None)
    g1, *_ = build_graph(X, metric, **kw, delta="eps")
    g2, *_ = build_graph(X, metric, **kw, delta=float(g0.stats.epsilon))
    assert g0.reps.rep_idx.shape[0] == g1.reps.rep_idx.shape[0] == g2.reps.rep_idx.shape[0]
    assert g0.stats.delta == pytest.approx(g0.stats.epsilon)
    assert g1.stats.extra.get("delta_mode") == "eps"


def test_solve_delta_report_keys():
    rng = np.random.default_rng(0)
    # Loose cluster of 1-NN distances around a small scale.
    nn1 = np.abs(rng.normal(0.05, 0.01, size=500))
    delta, report = solve_delta(nn1, r_band=(100.0, 400.0), alpha_guard=0.5, n_rows=500)
    assert set(REQUIRED_REPORT_KEYS).issubset(report.keys())
    assert report["mode"] in {"eps", "calibrated", "auto_fallback"}
    assert isinstance(delta, float)
    assert delta >= float(report["eps_ref"])


def test_solve_delta_collapse_warning(caplog):
    """Duplicate-heavy probe: calibrating into a tiny R band warns and falls back."""
    # Many exact zeros + a few large gaps → large δ collapses the cover.
    nn1 = np.zeros(200, dtype=np.float64)
    nn1[::20] = 1.0
    with caplog.at_level(logging.WARNING):
        delta, report = solve_delta(
            nn1,
            r_band=(1.0, 2.0),
            alpha_guard=0.99,
            n_rows=200,
        )
    assert report["mode"] in {"eps", "auto_fallback"}
    assert delta == pytest.approx(report["eps_ref"])
    assert any("collapse" in r.message.lower() or "guard" in r.message.lower() for r in caplog.records) or (
        report.get("collapse_warned") or report.get("guard_ok") is False or report["mode"] == "eps"
    )


def test_solve_delta_auto_lands_in_loose_band():
    """Synthetic well-separated clusters: auto δ puts R in a loose band."""
    rng = np.random.default_rng(1)
    n_clusters = 40
    per = 50
    centers = rng.normal(scale=5.0, size=(n_clusters, 4))
    X = np.vstack([centers[i] + 0.02 * rng.normal(size=(per, 4)) for i in range(n_clusters)])
    # Pairwise on a probe subsample.
    take = min(400, X.shape[0])
    idx = rng.choice(X.shape[0], size=take, replace=False)
    P = X[idx]
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    D = np.sqrt(np.maximum(d2, 0.0))
    # Expect ~n_clusters survivors after a moderate radius.
    band = (25.0, 80.0)
    delta, report = solve_delta(D, r_band=band, alpha_guard=0.0, n_rows=X.shape[0])
    assert set(REQUIRED_REPORT_KEYS).issubset(report.keys())
    assert band[0] <= report["r_est"] <= band[1]
    assert report["guard_ok"] is True
    assert report["mode"] in {"calibrated", "eps"}
    assert delta >= float(report["eps_ref"])


def test_build_graph_delta_auto_records_stats():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    n_clusters = 30
    per = 20
    centers = rng.normal(scale=4.0, size=(n_clusters, 3))
    X = torch.as_tensor(
        np.vstack([centers[i] + 0.03 * rng.normal(size=(per, 3)) for i in range(n_clusters)]),
        dtype=torch.float32,
    )
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=8,
        n_landmarks=16,
        epsilon=None,
        delta="auto",
        seed=0,
        knn_mode="brute",
        r_band=(20.0, 120.0),
        alpha_guard=0.0,
    )
    assert graph.stats.epsilon > 0.0
    assert graph.stats.delta >= graph.stats.epsilon
    assert "delta_mode" in graph.stats.extra
    assert "delta_guard_ok" in graph.stats.extra
    assert graph.stats.extra.get("epsilon") == pytest.approx(graph.stats.epsilon)

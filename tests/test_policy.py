"""ExemplarPolicy p_t (PR-9) unit + integration smoke."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.graph import build_graph
from leanmap.metrics import wrap_metric
from leanmap.sampling.edges import EdgeSampler
from leanmap.sampling.policy import RATIO_CAP_DEFAULT, ExemplarPolicy, _cap_ratio
from leanmap.train.probes import maybe_refresh_policy, sufficiency_gates


def _tiny_graph(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(rng.normal(size=(n, 5)).astype(np.float32))
    metric = wrap_metric("l2", X=X, n_neighbors=6, seed=seed)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=6,
        n_landmarks=6,
        epsilon=1e-5,
        seed=seed,
        knn_mode="brute",
    )
    return X, graph


def test_uniform_from_graph_draws():
    X, graph = _tiny_graph()
    pol = ExemplarPolicy.from_graph(graph, mode="uniform", seed=0)
    idx, ends, iw = pol.sample_edges(16)
    assert idx.shape == (16,)
    assert ends.shape == (16, 2)
    assert iw.shape == (16,)
    assert np.isfinite(iw).all()
    # Uniform + edge-mass base => importance ratios ~1 before/after cap.
    assert np.all(iw <= RATIO_CAP_DEFAULT + 1e-9)
    assert np.all(iw >= 1.0 / RATIO_CAP_DEFAULT - 1e-9)


def test_reweight_caps_ratios():
    # Extreme mismatch between base w and p_t.
    w = np.array([1.0, 1.0, 100.0], dtype=np.float64)
    # Force a peaked p via sufficient_v1 + visits, then check cap helper + API.
    assert float(_cap_ratio(np.array([1000.0]), 10.0)[0]) == 10.0
    assert float(_cap_ratio(np.array([0.001]), 10.0)[0]) == 0.1

    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    pol = ExemplarPolicy(
        mode="sufficient_v1",
        edge_mass=w,
        edges=edges,
        n_cells=3,
        reweight=True,
        ratio_cap=10.0,
        seed=0,
    )
    # Make p_t very different from w via visits.
    pol.refresh({"edge_visits": np.array([1.0, 1.0, 1e6])})
    idx = np.arange(3)
    iw = pol.importance_weights(idx, family="edges")
    assert iw.shape == (3,)
    assert np.all(iw <= 10.0 + 1e-9)
    assert np.all(iw >= 0.1 - 1e-9)

    pol_u = ExemplarPolicy(
        mode="uniform",
        edge_mass=w,
        edges=edges,
        n_cells=3,
        reweight=False,
        seed=0,
    )
    ones = pol_u.importance_weights(idx, family="edges")
    assert np.allclose(ones, 1.0)


def test_sufficient_v1_builds_on_tiny_graph():
    _, graph = _tiny_graph(n=30, seed=1)
    pol = ExemplarPolicy.from_graph(graph, mode="sufficient_v1", seed=1)
    assert pol.mode == "sufficient_v1"
    assert pol.n_edges > 0
    idx, iw = pol.sample_indices(8, family="edges")
    assert idx.shape == (8,)
    assert iw.shape == (8,)
    # Path / class families also draw without error.
    _, _ = pol.sample_indices(4, family="class_ordinal")
    _, _ = pol.sample_indices(4, family="paths")
    pol.set_violation_hooks(path_violation=np.zeros(pol.n_edges), rebuild=True)
    pol.refresh({"cell_visits": np.ones(pol.n_cells) * 2.0})


def test_uniform_policy_leaves_edge_sampler_path():
    X, graph = _tiny_graph(n=50, seed=2)
    pol = ExemplarPolicy.from_graph(graph, mode="uniform", seed=2)
    # Prior training path: EdgeSampler on graph weights (bit-compat mass).
    samp = EdgeSampler(X, graph, seed=2)
    xi, xj, w, eidx = samp.sample(12)
    assert xi.shape[0] == 12
    assert xj.shape[0] == 12
    assert w.shape[0] == 12
    # Policy mass matches graph edge mass up to normalization.
    mass = pol.sampling_mass("edges")
    gw = graph.weights.detach().cpu().numpy().astype(np.float64)
    gw = gw / gw.sum()
    assert np.allclose(mass, gw, rtol=1e-5, atol=1e-8)
    # Importance on EdgeSampler draws stays finite under uniform.
    iw = pol.importance_weights(eidx.cpu().numpy())
    assert np.isfinite(iw).all()


def test_sufficiency_gates_and_refresh():
    _, graph = _tiny_graph(n=24, seed=3)
    pol = ExemplarPolicy.from_graph(graph, mode="sufficient_v1", seed=3)
    assert sufficiency_gates({"cell_coverage": 0.99, "probe_gap": 0.01})
    assert not sufficiency_gates({"cell_coverage": 0.1})
    assert maybe_refresh_policy(pol, True) is False
    assert maybe_refresh_policy(pol, False, stats={"edge_visits": np.ones(pol.n_edges)})
    assert maybe_refresh_policy(pol, {"cell_coverage": 0.5}) is True

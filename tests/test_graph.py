"""Tests for graph construction."""

from __future__ import annotations

import numpy as np
import torch

from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from leanmap.distance import EuclideanDistance
from leanmap.graph import (
    build_graph,
    build_graph_pyramid,
    build_representatives,
    estimate_epsilon,
    knn_representatives,
    smooth_knn,
)
from leanmap.landmarks import assign_buckets, fps_init
from leanmap.metrics import wrap_metric


def _n_components(graph) -> int:
    R = int(graph.reps.rep_idx.shape[0])
    e = graph.edges.cpu().numpy()
    if e.shape[0] == 0:
        return R
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(R, R))
    return int(connected_components(A, directed=False)[0])


def _hop_diameter(graph, n_src: int = 16, seed: int = 0) -> float:
    """Approximate unweighted graph diameter via BFS from sampled sources."""
    R = int(graph.reps.rep_idx.shape[0])
    e = graph.edges.cpu().numpy()
    if e.shape[0] == 0:
        return 0.0
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(R, R)).tocsr()
    rng = np.random.default_rng(seed)
    src = rng.choice(R, size=min(n_src, R), replace=False)
    D = dijkstra(A, directed=False, unweighted=True, indices=src)
    D[~np.isfinite(D)] = 0.0
    return float(D.max())


def test_duplicates_compress():
    torch.manual_seed(0)
    base = torch.randn(100, 8)
    X = base.repeat(5, 1)  # 5x duplicates → N=500, expect R ~ 100
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graph, M, top1, topc = build_graph(
        X, metric, n_neighbors=10, n_landmarks=16, epsilon=1e-6, seed=0, knn_mode="brute"
    )
    # compression ≈ 5
    assert graph.stats.compression_ratio > 3.0
    assert graph.stats.dedup is True


def test_dedup_default_estimates_and_compresses():
    """With epsilon=None, exact dups used to force eps=0 and skip; now compress."""
    torch.manual_seed(0)
    base = torch.randn(80, 6)
    X = base.repeat(4, 1)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graph, *_ = build_graph(
        X, metric, n_neighbors=10, n_landmarks=16, epsilon=None, seed=0, knn_mode="brute"
    )
    assert graph.stats.dedup is True
    assert graph.stats.epsilon > 0.0
    assert graph.stats.compression_ratio > 2.0
    assert graph.reps.rep_idx.shape[0] < X.shape[0]


def test_dedup_false_keeps_all():
    torch.manual_seed(1)
    base = torch.randn(50, 5)
    X = base.repeat(3, 1)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=8,
        n_landmarks=12,
        dedup=False,
        seed=0,
        knn_mode="brute",
    )
    assert graph.stats.dedup is False
    assert graph.stats.epsilon == 0.0
    assert graph.reps.rep_idx.shape[0] == X.shape[0]
    assert graph.stats.compression_ratio == 1.0


def test_estimate_epsilon_positive_median_fallback():
    torch.manual_seed(5)
    A = torch.randn(30, 4)
    # many exact dups → low 1-NN quantile is 0; leftover uniques give positive median
    X = torch.cat([A, A, torch.randn(20, 4)], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    eps, diag = estimate_epsilon(X, metric, n_sample=80, quantile=0.01, seed=0)
    assert eps > 0.0
    assert diag["used_positive_median_fallback"] is True
    assert diag["frac_exact_zero"] > 0.0


def test_epsilon_zero_no_dedup():
    torch.manual_seed(1)
    X = torch.randn(200, 6)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    M = fps_init(X, metric, 8, seed=0)
    top1, _ = assign_buckets(X, M, metric, c=4)
    reps = build_representatives(X, top1, metric, epsilon=0.0, L=8, seed=0)
    assert reps.rep_idx.shape[0] == X.shape[0]


def test_symmetrised_max_and_no_orphans():
    torch.manual_seed(2)
    X = torch.randn(300, 8)
    metric = wrap_metric("l2", X=X, n_neighbors=15, seed=0)
    graph, *_ = build_graph(
        X, metric, n_neighbors=15, n_landmarks=32, epsilon=0.0, seed=0, knn_mode="brute"
    )
    assert float(graph.weights.max()) <= 1.0 + 1e-5
    assert float(graph.weights.min()) > 0.0
    # every rep appears in at least one edge (or isolated only if R=1)
    R = graph.reps.rep_idx.shape[0]
    seen = torch.zeros(R, dtype=torch.bool)
    seen[graph.edges[:, 0]] = True
    seen[graph.edges[:, 1]] = True
    # backbone + knn should cover almost all; allow tiny orphan fraction after backbone
    assert seen.float().mean() > 0.9


def test_two_blobs_components():
    torch.manual_seed(3)
    a = torch.randn(200, 4) + 10
    b = torch.randn(200, 4) - 10
    X = torch.cat([a, b], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
        lambda_backbone=0.01,
    )
    assert graph.stats.n_components_before_backbone == 2


def test_smooth_knn_degenerate():
    # all neighbours identical distance ~0 relative to rho
    knn = torch.zeros(50, 10)
    rho, sigma, diag = smooth_knn(knn, local_connectivity=1)
    assert diag["n_degenerate"] > 0


def test_ivf_recall():
    torch.manual_seed(4)
    X = torch.randn(2000, 16)
    metric = wrap_metric("l1", X=X, n_neighbors=15, seed=0)
    M = fps_init(X, metric, L=64, seed=0)
    _, topc = assign_buckets(X, M, metric, c=8)
    # Use X as "representatives"
    knn_d, knn_i, info = knn_representatives(
        X,
        metric,
        k=15,
        mode="ivf",
        landmarks=M,
        assign_topc=topc,
        c_search=8,
        metric=metric,
    )
    assert info["recall"] > 0.9
    assert knn_i.shape == (X.shape[0], 15)


def test_union_assign_topc_and_joint_ivf():
    from leanmap.graph import union_assign_topc

    torch.manual_seed(5)
    a = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.int64)
    b = torch.tensor([[4, 5], [2, 4], [1, 5]], dtype=torch.int64)
    u = union_assign_topc([a, b], c=4)
    assert u.shape == (3, 4)
    assert set(u[0].tolist()) - {-1} == {0, 1, 4, 5}

    X = torch.randn(800, 8)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    M = fps_init(X, metric, L=32, seed=0)
    _, topc = assign_buckets(X, M, metric, c=4)
    # Second partitioning: first 4 dims as a cheap extra view
    M2 = fps_init(X[:, :4], EuclideanDistance(), L=16, seed=1)
    _, topc2 = assign_buckets(X[:, :4], M2, EuclideanDistance(), c=4)
    knn_d, knn_i, info = knn_representatives(
        X,
        metric,
        k=10,
        mode="ivf",
        landmarks=M,
        assign_topc=topc,
        c_search=4,
        metric=metric,
        extra_assign_topc=[topc2],
    )
    assert info["recall"] > 0.85
    assert knn_i.shape == (X.shape[0], 10)


def test_graph_pyramid_levels():
    """Coarsening yields decreasing reps, preserves connectivity, and strengthens
    long-range ties (higher mean weighted degree at coarser scales)."""
    torch.manual_seed(0)
    # Elongated single-manifold: a noisy 1-D curve embedded in 8-D.
    t = torch.linspace(0.0, 1.0, 600).unsqueeze(1)
    X = torch.cat(
        [t * 10.0, torch.sin(t * 6.0), 0.05 * torch.randn(600, 6)], dim=1
    )
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, M, top1, topc = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=3,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
    )
    # multiple levels built, strictly decreasing representative counts
    assert len(graphs) >= 3
    reps = [int(g.reps.rep_idx.shape[0]) for g in graphs]
    assert all(reps[i] > reps[i + 1] for i in range(len(reps) - 1))

    # coarsening preserves connectivity (single component at every scale)
    comps = [_n_components(g) for g in graphs]
    assert comps[0] == 1
    assert all(c == comps[0] for c in comps)

    # coarser levels give long-range ties: smaller graph diameter (fewer hops
    # between far nodes) and larger cells (each rep aggregates more raw points)
    assert _hop_diameter(graphs[-1]) < _hop_diameter(graphs[0])
    cell_sizes = [float(g.reps.weight.float().mean().item()) for g in graphs]
    assert all(cell_sizes[i] < cell_sizes[i + 1] for i in range(len(cell_sizes) - 1))

    # coarse cells partition all raw points (samplers can expand any cell)
    for g in graphs:
        assert int(g.reps.weight.sum().item()) == X.shape[0]
        assert (g.reps.weight > 0).all()


def test_pyramid_preserves_disconnected_components():
    """Galerkin coarsening must NOT glue genuinely separate components
    (no Isomap short-circuit)."""
    torch.manual_seed(3)
    a = torch.randn(200, 4) + 20
    b = torch.randn(200, 4) - 20
    X = torch.cat([a, b], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        pyramid_coarse_backbone=0.0,  # isolate coarsening from cohesive MST
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
        lambda_backbone=0.0,  # keep the two blobs genuinely disconnected
    )
    comps = [_n_components(g) for g in graphs]
    assert comps[0] == 2
    assert all(c == 2 for c in comps)


def test_pyramid_coarse_backbone_connects_islands():
    """A strong MST skeleton on the coarsest level ties otherwise-separate
    regions into one component (global cohesion lever)."""
    torch.manual_seed(3)
    a = torch.randn(200, 4) + 20
    b = torch.randn(200, 4) - 20
    X = torch.cat([a, b], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        pyramid_coarse_backbone=1.0,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
        lambda_backbone=0.0,  # blobs disconnected without the coarse backbone
    )
    # coarsest level is now a single component thanks to the MST skeleton
    assert _n_components(graphs[-1]) == 1
    # and it carries strong (weight≈1) spanning edges
    assert float(graphs[-1].weights.max()) >= 0.99


def test_pyramid_scales_zero_is_single_level():
    torch.manual_seed(1)
    X = torch.randn(300, 8)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=0,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
    )
    assert len(graphs) == 1


def test_cohesive_defaults_match_config():
    """PLANEConfig cohesive defaults: 4 levels worth of weights + backbone on."""
    from leanmap.config import PLANEConfig

    cfg = PLANEConfig()
    assert cfg.pyramid_scales == 3
    assert list(cfg.pyramid_level_weights) == [1.0, 1.0, 2.0, 4.0]
    assert cfg.pyramid_coarse_backbone == 1.0
    # for_scale inherits cohesive pyramid settings
    scaled = PLANEConfig.for_scale(10_000)
    assert list(scaled.pyramid_level_weights) == [1.0, 1.0, 2.0, 4.0]
    assert scaled.pyramid_coarse_backbone == 1.0


def test_cohesive_pyramid_applies_backbone_by_default():
    """build_graph_pyramid default backbone connects islands on the coarsest level."""
    torch.manual_seed(0)
    a = torch.randn(80, 4) + torch.tensor([-8.0, 0.0, 0.0, 0.0])
    b = torch.randn(80, 4) + torch.tensor([8.0, 0.0, 0.0, 0.0])
    X = torch.cat([a, b], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        # omit pyramid_coarse_backbone → cohesive default 1.0
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
    )
    assert len(graphs) >= 2
    assert _n_components(graphs[-1]) == 1
    assert float(graphs[-1].weights.max()) >= 0.99
    # level-weight vector length for pyramid_scales=3 is 4; here scales=2 → ≤3
    from leanmap.config import PLANEConfig

    n_levels = len(graphs)
    weights = list(PLANEConfig().pyramid_level_weights)[:n_levels]
    assert len(weights) == n_levels


def test_coarse_backbone_is_noop_when_already_connected():
    """Regression: the backbone must not touch an already-connected level.

    It used to lay a full R-1 MST over the reps and max-merge it, overwriting
    hundreds of existing edges to the maximum weight and wrecking global
    geodesic/density fidelity even with nothing disconnected.
    """
    from leanmap.graph import _add_coarse_backbone

    torch.manual_seed(0)
    X = torch.randn(240, 5)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        pyramid_coarse_backbone=0.0,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
    )
    coarse = graphs[-1]
    assert _n_components(coarse) == 1, "precondition: coarsest level connected"
    before_e = coarse.edges.clone()
    before_w = coarse.weights.clone()

    out = _add_coarse_backbone(coarse, X, metric, 1.0)

    assert out.edges.shape == before_e.shape
    torch.testing.assert_close(out.weights, before_w)
    assert out.stats.extra["coarse_backbone_skipped"] is True
    assert out.stats.extra["coarse_backbone_bridges"] == 0


def test_coarse_backbone_adds_only_minimal_bridges():
    """When regions ARE disconnected, add exactly n_comp-1 strong bridges and
    leave every pre-existing weight untouched."""
    from leanmap.graph import _add_coarse_backbone

    torch.manual_seed(3)
    a = torch.randn(200, 4) + 20
    b = torch.randn(200, 4) - 20
    X = torch.cat([a, b], 0)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graphs, *_ = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        pyramid_coarse_backbone=0.0,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
        lambda_backbone=0.0,
    )
    coarse = graphs[-1]
    n_before = _n_components(coarse)
    assert n_before == 2
    n_edges_before = int(coarse.edges.shape[0])
    w_before = np.sort(coarse.weights.cpu().numpy())

    out = _add_coarse_backbone(coarse, X, metric, 1.0)

    assert _n_components(out) == 1
    assert out.stats.extra["coarse_backbone_bridges"] == n_before - 1
    assert int(out.edges.shape[0]) == n_edges_before + (n_before - 1)
    assert float(out.weights.max()) >= 0.99
    # every original weight survives unchanged; only bridges are new
    w_after = np.sort(out.weights.cpu().numpy())
    assert np.allclose(w_after[: len(w_before)], w_before, atol=1e-6)


def test_level_weights_keep_coarsest_when_truncated(caplog):
    """Regression: a 4-tuple on a 3-level pyramid must keep the COARSEST weight.

    Plain truncation dropped exactly the long-range term that anchors global
    structure, so the shipped (1, 1, 2, 4) default silently trained as (1, 1, 2).
    """
    import logging

    from leanmap import fit
    from leanmap.config import PLANEConfig

    torch.manual_seed(0)
    X = torch.randn(400, 6).numpy()
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 1
    cfg.pyramid_scales = 3
    cfg.pyramid_min_reps = 64  # force fewer levels than weights
    cfg.pyramid_level_weights = (1.0, 1.0, 2.0, 4.0)
    with caplog.at_level(logging.WARNING, logger="leanmap"):
        fit(X, dist_fn="l2", config=cfg)
    msgs = [r.getMessage() for r in caplog.records]
    warn = [m for m in msgs if "pyramid_level_weights has" in m]
    assert warn, f"expected a truncation warning, got {msgs}"
    assert "kept the coarsest weight 4" in warn[0]


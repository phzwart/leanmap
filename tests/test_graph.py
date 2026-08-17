"""Tests for graph construction."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_estimate_epsilon_uses_all_points():
    """The quantile is read off every row, not a fixed-size subsample.

    A subsample estimate drifts upward with N like (N/n_sub)^(1/m); reporting
    the scope lets a caller see which path was taken.
    """
    torch.manual_seed(3)
    X = torch.randn(500, 4)
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    eps, diag = estimate_epsilon(X, metric, n_sample=50, quantile=0.05, seed=0, metric=metric)
    assert diag["scope"] == "full"
    assert diag["subsample_correction"] == 1.0
    assert eps > 0.0


def test_estimate_epsilon_subsample_is_size_corrected():
    """Without a full pass, the subsample quantile is scaled by (n_sub/N)^(1/m)."""
    from leanmap.build import pipeline as pipeline_mod

    torch.manual_seed(4)
    X = torch.randn(800, 3)
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    eps_full, _ = estimate_epsilon(X, metric, quantile=0.1, seed=0, metric=metric)

    orig = pipeline_mod._one_nn_all
    pipeline_mod._one_nn_all = lambda *a, **k: None
    try:
        eps_sub, diag = estimate_epsilon(X, metric, n_sample=200, quantile=0.1, seed=0)
    finally:
        pipeline_mod._one_nn_all = orig

    assert diag["scope"] == "subsample"
    assert 0.0 < diag["subsample_correction"] < 1.0
    # The correction must move the subsample estimate toward the full-N value,
    # which is what stops epsilon from drifting with dataset size.
    raw = eps_sub / diag["subsample_correction"]
    assert abs(eps_sub - eps_full) < abs(raw - eps_full)


def test_halo_merges_across_landmark_buckets():
    """Near-duplicates split by a Voronoi boundary collapse into one cell."""
    from leanmap.graph import _halo_merge, build_representatives

    # Two tight pairs placed on opposite sides of the midplane between the
    # two landmark seeds, so each pair straddles the bucket boundary.
    eps = 1e-3
    X = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [-0.1 * eps, 0.0],
            [0.1 * eps, 0.0],
        ],
        dtype=torch.float32,
    )
    metric = wrap_metric("l2", X=X, n_neighbors=2, seed=0)
    M = X[torch.tensor([0, 1])]
    top1, topc = assign_buckets(X, M, metric, c=2)
    assert int(top1[2]) != int(top1[3])

    reps = build_representatives(X, top1, metric, epsilon=float(eps / metric.natural_scale), L=2, seed=0)
    r_before = int(reps.rep_idx.shape[0])
    merged, info = _halo_merge(X, reps, topc, metric, float(eps / metric.natural_scale))

    assert info["halo_merged"] == r_before - int(merged.rep_idx.shape[0])
    assert info["halo_merged"] >= 1
    # Membership stays a total, contiguous partition of all N points.
    assert int(merged.weight.sum()) == X.shape[0]
    assert int(merged.offsets[-1]) == X.shape[0]
    assert torch.equal(torch.sort(merged.values).values, torch.arange(X.shape[0]))
    assert int(merged.member_of.max()) == int(merged.rep_idx.shape[0]) - 1


def test_halo_merge_is_order_independent():
    """Union-find over pre-merge pairs: a chain collapses to one component."""
    from leanmap.graph import _halo_merge, build_representatives

    torch.manual_seed(11)
    X = torch.randn(120, 3)
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    M = fps_init(X, metric, 6, seed=0)
    top1, topc = assign_buckets(X, M, metric, c=4)
    eps = 0.35
    reps = build_representatives(X, top1, metric, epsilon=eps, L=6, seed=0)

    a, _ = _halo_merge(X, reps, topc, metric, eps)
    b, _ = _halo_merge(X, reps, topc, metric, eps)
    assert torch.equal(a.rep_idx, b.rep_idx)
    assert torch.equal(a.member_of, b.member_of)


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
    w = graphs[-1].weights
    # Valid memberships. The default "rational" squash never saturates, so the
    # top coarse edges keep their ranking instead of flattening to a common 1.
    assert float(w.min()) > 0.0 and float(w.max()) < 1.0
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



def test_rational_squash_is_monotone_and_unsaturating():
    """The top coarse edges must keep their ranking, not flatten to a tie."""
    from leanmap.graph import _squash_coarse_weights

    # Heavy tail: the top 1% spans two orders of magnitude.
    w = torch.cat([torch.linspace(0.1, 10.0, 990), torch.linspace(50.0, 5000.0, 10)])
    w = w.to(torch.float64)

    old = _squash_coarse_weights(w, mode="quantile_clamp")
    new = _squash_coarse_weights(w, mode="rational")

    top = w >= 50.0
    # Old scaling ties the entire top tail at exactly 1.0, discarding the order
    # among the very edges a (1, 2, 8) pyramid exists to exploit.
    assert int(old[top].unique().numel()) == 1
    assert float(old[top].max()) == 1.0
    # New scaling keeps every one of them distinct and strictly increasing.
    assert int(new[top].unique().numel()) == int(top.sum())
    assert bool((new[1:] >= new[:-1]).all())
    assert float(new.max()) < 1.0
    # Median maps to 0.5 by construction.
    q50 = float(torch.quantile(w, 0.5))
    assert abs(float(_squash_coarse_weights(torch.tensor([q50]).double(), "rational")) - 0.5) < 1e-6


def test_squash_modes_differ_in_magnitude():
    """Level weights tuned under one scaling do not transfer to the other."""
    from leanmap.graph import _squash_coarse_weights

    w = torch.distributions.LogNormal(0.0, 1.5).sample((2000,)).double()
    old = _squash_coarse_weights(w, mode="quantile_clamp")
    new = _squash_coarse_weights(w, mode="rational")
    assert float(new.mean()) > 1.5 * float(old.mean())


def test_squash_handles_degenerate_inputs():
    from leanmap.graph import _squash_coarse_weights

    assert _squash_coarse_weights(torch.zeros(0).double()).numel() == 0
    # A zero median must not divide by zero.
    w = torch.cat([torch.zeros(90), torch.ones(10)]).double()
    out = _squash_coarse_weights(w, mode="rational")
    assert bool(torch.isfinite(out).all())
    assert float(out.max()) < 1.0
    single = _squash_coarse_weights(torch.tensor([3.0]).double(), "rational")
    assert bool(torch.isfinite(single).all())


def test_pyramid_squash_is_selectable():
    torch.manual_seed(0)
    X = torch.randn(300, 5)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    kw = dict(
        pyramid_scales=2,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        n_neighbors=10,
        n_landmarks=16,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
    )
    g_new, *_ = build_graph_pyramid(X, metric, pyramid_squash="rational", **kw)
    g_old, *_ = build_graph_pyramid(X, metric, pyramid_squash="quantile_clamp", **kw)
    assert float(g_new[-1].weights.max()) < 1.0
    assert float(g_old[-1].weights.max()) == 1.0
    with pytest.raises(ValueError):
        build_graph_pyramid(X, metric, pyramid_squash="nope", **kw)


def test_q99_anchor_is_monotone_and_selective():
    """The default must fix the clamp's ties without diffusing coarse pull.

    Anchoring the rational map at the median is monotone but lifts the mean
    membership ~6x, making coarse attraction diffuse instead of selective;
    that measurably degraded density correspondence on clustered data.
    Anchoring at q99 keeps clamp's magnitude while removing the ties.
    """
    from leanmap.graph import _squash_coarse_weights

    torch.manual_seed(0)
    w = torch.distributions.LogNormal(0.0, 1.5).sample((5000,)).double()
    clamp = _squash_coarse_weights(w, "quantile_clamp")
    q50 = _squash_coarse_weights(w, "rational")
    q99 = _squash_coarse_weights(w, "rational_q99")

    # Monotone and unsaturating: no ties at the top, unlike the clamp.
    assert int((clamp == clamp.max()).sum()) > 10
    assert int((q99 == q99.max()).sum()) == 1
    assert float(q99.max()) < 1.0
    assert bool((q99[torch.argsort(w)][1:] >= q99[torch.argsort(w)][:-1]).all())

    # Selective: magnitude comparable to the clamp, unlike the median anchor.
    assert abs(float(q99.mean()) - float(clamp.mean())) < 0.05
    assert float(q50.mean()) > 4.0 * float(q99.mean())


def test_default_squash_is_q99():
    from leanmap.config import PLANEConfig

    assert PLANEConfig().pyramid_squash == "rational_q99"


def _brute_l2_knn(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact L2 kNN excluding self, matching leanmap's DistanceFn convention."""
    D = EuclideanDistance()(X, X)
    D = D.clone()
    D.fill_diagonal_(float("inf"))
    dist, idx = torch.topk(D, k=k, dim=1, largest=False)
    return idx.to(torch.int64), dist.to(torch.float32)


def test_precomputed_knn_injected_into_graph():
    torch.manual_seed(0)
    X = torch.randn(40, 4)
    k = 5
    knn_idx, knn_dist = _brute_l2_knn(X, k)
    metric = wrap_metric("l2", X=X, n_neighbors=k, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=k,
        n_landmarks=8,
        dedup=False,
        seed=0,
        knn_mode="brute",
        precomputed_knn=(knn_idx, knn_dist),
    )
    assert graph.stats.knn_mode == "precomputed"
    assert graph.stats.knn_recall is None
    assert torch.equal(graph.knn_idx, knn_idx.cpu())
    # Fuzzy edges exist and cover the supplied neighborhood width.
    assert graph.edges.shape[0] > 0
    assert graph.knn_idx.shape == (X.shape[0], k)


def test_precomputed_knn_with_dedup_uses_rep_space():
    """precomputed_knn is over representatives (R), allowed with dedup=True."""
    torch.manual_seed(0)
    X = torch.randn(40, 4)
    # Force no compression so R == N for a simple check with ambient knn.
    metric = wrap_metric("l2", X=X, n_neighbors=5, seed=0)
    knn_idx, knn_dist = _brute_l2_knn(X, 5)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=5,
        n_landmarks=8,
        dedup=True,
        epsilon=0.0,
        seed=0,
        knn_mode="brute",
        precomputed_knn=(knn_idx, knn_dist),
    )
    assert graph.stats.knn_mode == "precomputed"
    assert graph.knn_idx.shape[0] == X.shape[0]


def test_precomputed_knn_rejects_wrong_R_when_compressed():
    torch.manual_seed(0)
    X = torch.cat([torch.randn(1, 3).repeat(20, 1), torch.randn(20, 3)], dim=0)
    knn_idx, knn_dist = _brute_l2_knn(X, 3)
    metric = wrap_metric("l2", X=X, n_neighbors=3, seed=0)
    with pytest.raises(ValueError, match="expected R="):
        build_graph(
            X,
            metric,
            n_neighbors=3,
            n_landmarks=4,
            dedup=True,
            epsilon=0.5,
            seed=0,
            precomputed_knn=(knn_idx, knn_dist),
        )


def test_precomputed_knn_rejects_bad_shapes_self_and_oob():
    from leanmap.graph import validate_precomputed_knn

    n, k = 10, 3
    idx = torch.randint(0, n, (n, k))
    dist = torch.rand(n, k)
    # force a self-neighbor
    idx = idx.clone()
    idx[0, 0] = 0
    with pytest.raises(ValueError, match="self-neighbors"):
        validate_precomputed_knn(idx, dist, n)

    idx2 = torch.randint(0, n, (n, k))
    idx2[1, 0] = n  # OOB
    with pytest.raises(ValueError, match=r"\[0,"):
        validate_precomputed_knn(idx2, torch.rand(n, k), n)

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_precomputed_knn(torch.zeros(n, k, dtype=torch.long), torch.rand(n, k + 1), n)

    with pytest.raises(ValueError, match="rows"):
        validate_precomputed_knn(torch.zeros(n - 1, k, dtype=torch.long), torch.rand(n - 1, k), n)


def test_fit_precomputed_knn_requires_x_calib():
    from leanmap import PLANEConfig, fit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    Xt = torch.as_tensor(X)
    knn_idx, knn_dist = _brute_l2_knn(Xt, 5)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 1
    cfg.dedup = False
    cfg.n_landmarks = 8
    cfg.width, cfg.depth = 32, 2
    cfg.batch_edges = 128
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="X_calib"):
        fit(X, "l2", config=cfg, precomputed_knn=(knn_idx, knn_dist))


def test_fit_precomputed_knn_end_to_end():
    from leanmap import PLANEConfig, fit

    rng = np.random.default_rng(1)
    X_all = rng.normal(size=(80, 4)).astype(np.float32)
    X_train, X_cal = X_all[:60], X_all[60:]
    Xt = torch.as_tensor(X_train)
    knn_idx, knn_dist = _brute_l2_knn(Xt, 5)
    cfg = PLANEConfig.for_scale(len(X_train))
    cfg.epochs = 2
    cfg.dedup = False
    cfg.n_landmarks = 8
    cfg.width, cfg.depth = 32, 2
    cfg.batch_edges = 128
    cfg.n_neighbors = 5
    cfg.device = "cpu"
    result = fit(
        X_train,
        "l2",
        config=cfg,
        X_calib=X_cal,
        precomputed_knn=(knn_idx, knn_dist),
    )
    Z, _ = result.model.embed(Xt, return_score=False)
    assert Z.shape == (60, 2)

"""Edge sampler cell-expansion test."""

from __future__ import annotations

import torch

from leanmap.distance import EuclideanDistance
from leanmap.graph import build_graph
from leanmap.landmarks import LandmarkAffinity
from leanmap.metrics import wrap_metric
from leanmap.sampler import (
    EdgeSampler,
    OrdinalTripletSampler,
    StarSampler,
    estimate_retention_null,
)


def test_edge_sampler_cell_expansion_varies():
    torch.manual_seed(0)
    # Build data with duplicates so cells have multiple members
    base = torch.randn(50, 6)
    X = base.repeat_interleave(4, dim=0)  # 200 rows, ~50 cells
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=10,
        n_landmarks=8,
        epsilon=1e-5,
        seed=0,
        knn_mode="brute",
    )
    # Find a cell with >1 member
    multi = (graph.reps.weight > 1).nonzero(as_tuple=False).view(-1)
    assert multi.numel() > 0
    samp = EdgeSampler(X, graph, seed=0)
    # Force same edge repeatedly and check raw vectors can differ
    # Find an edge whose both cells have weight > 1 if possible
    found_diff = False
    for _ in range(50):
        xi1, xj1, _, e1 = samp.sample(1)
        # resample same edge index manually
        e = int(e1[0])
        ci, cj = int(graph.edges[e, 0]), int(graph.edges[e, 1])
        if graph.reps.weight[ci] <= 1 and graph.reps.weight[cj] <= 1:
            continue
        vecs_i = []
        for _ in range(20):
            # draw members of ci
            from leanmap.sampler import _cell_member

            idx = _cell_member(graph.reps, ci, samp.rng)
            vecs_i.append(idx)
        if len(set(vecs_i)) > 1:
            found_diff = True
            break
    assert found_diff


def test_star_sampler_shapes_and_neighbours():
    torch.manual_seed(0)
    X = torch.randn(120, 5)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=0)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=8,
        n_landmarks=16,
        epsilon=1e-5,
        seed=0,
        knn_mode="brute",
    )
    m = 6
    samp = StarSampler(X, graph, m=m, seed=0)
    B = 32
    x_c, x_nbr, mask = samp.sample(B)
    assert x_c.shape == (B, X.shape[1])
    assert x_nbr.shape == (B, m, X.shape[1])
    assert mask.shape == (B, m)
    # Mask is binary and every sampled centre has at least one valid neighbour.
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert (mask.sum(dim=1) > 0).all()
    # Padded neighbour slots (mask==0) are contiguous at the tail per row.
    for r in range(B):
        k = int(mask[r].sum().item())
        assert torch.all(mask[r, :k] == 1.0)
        assert torch.all(mask[r, k:] == 0.0)


# ---------------------------------------------------------------------------
# Empirical retention null
# ---------------------------------------------------------------------------


def _ordinal_setup(n=400, D=5, L=16, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, D)
    metric = wrap_metric("l2", X=X, n_neighbors=10, seed=seed)
    graph, M, top1, _ = build_graph(
        X, metric, n_neighbors=10, n_landmarks=L, epsilon=0.0,
        seed=seed, knn_mode="brute",
    )
    aff = LandmarkAffinity(M, EuclideanDistance(), probe_differentiable=False)
    ord_samp = OrdinalTripletSampler(X, top1, EuclideanDistance(), seed=seed)
    edge_samp = EdgeSampler(X, graph, seed=seed)
    return ord_samp, edge_samp, aff


def test_shuffling_ranks_destroys_the_ordinal_signal():
    """The null must score below the real sampler, or it is not a null."""
    ord_samp, edge_samp, aff = _ordinal_setup()
    x_i, x_j, _, _ = edge_samp.sample(512)
    _, _, _, real = ord_samp.sample(x_i, x_j, aff, shuffle_ranks=False)
    _, _, _, null = ord_samp.sample(x_i, x_j, aff, shuffle_ranks=True)
    assert null < real


def test_retention_null_is_measured_and_reproducible():
    ord_samp, edge_samp, aff = _ordinal_setup()
    a = estimate_retention_null(ord_samp, edge_samp, aff, n_batches=4, batch_size=256)
    assert 0.0 <= a <= 1.0
    # Same seeds ⇒ same estimate.
    ord2, edge2, aff2 = _ordinal_setup()
    b = estimate_retention_null(ord2, edge2, aff2, n_batches=4, batch_size=256)
    assert a == b


def test_shuffled_sampling_does_not_poison_last_retention():
    """A diagnostic pass must not overwrite the trainer's live retention."""
    ord_samp, edge_samp, aff = _ordinal_setup()
    x_i, x_j, _, _ = edge_samp.sample(256)
    ord_samp.sample(x_i, x_j, aff, shuffle_ranks=False)
    real = ord_samp.last_retention
    ord_samp.sample(x_i, x_j, aff, shuffle_ranks=True)
    assert ord_samp.last_retention == real

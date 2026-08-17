"""PR-10 distributed build bunches (ws=1 + unit helpers)."""
from __future__ import annotations

import importlib
import sys

import pytest
import torch

from leanmap.build.bunches import (
    build_graph_bunches,
    distributed_union_find,
    fill_knn_rows,
    halo_fraction,
    knn_completeness_audit,
    margin_halo,
    owned_net,
    partition_bunches,
    probe_shards,
    reconcile_landmarks,
    stitch_graph,
    uf_link,
    cut_mass,
)
from leanmap.build.pipeline import build_graph
from leanmap.metrics import wrap_metric


def test_union_find_determinism_parallel_merges():
    """Same undirected edges → identical roots regardless of merge order."""
    n = 8
    edges = [(0, 1), (1, 2), (3, 4), (2, 4), (5, 6), (6, 7), (0, 7)]

    def roots_from(order):
        parent = list(range(n))
        for a, b in order:
            uf_link(parent, a, b)
        return distributed_union_find(parent)

    orders = [
        edges,
        list(reversed(edges)),
        edges[::2] + edges[1::2],
        sorted(edges, key=lambda e: (e[1], e[0])),
    ]
    ref = roots_from(orders[0]).tolist()
    for order in orders[1:]:
        assert roots_from(order).tolist() == ref

    # Multi-array merge (mock per-rank parent pointers) matches single-pass.
    parent_a = list(range(n))
    for a, b in edges[:4]:
        uf_link(parent_a, a, b)
    parent_b = list(range(n))
    for a, b in edges[4:]:
        uf_link(parent_b, a, b)
    merged = distributed_union_find([parent_a, parent_b])
    assert merged.tolist() == ref


def test_knn_completeness_audit_full_and_thin():
    full = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    miss = knn_completeness_audit(4, full)
    assert miss == pytest.approx(0.0)
    assert knn_completeness_audit(4, full, strict=True) == pytest.approx(0.0)

    thin = full.clone()
    thin[:, -2:] = -1  # deliberately incomplete neighbour slots
    miss_thin = knn_completeness_audit(4, thin)
    assert miss_thin == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="completeness audit failed"):
        knn_completeness_audit(4, thin, strict=True)


def test_ws1_build_graph_bunches_matches_build_graph():
    torch.manual_seed(0)
    X = torch.randn(48, 6)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=0)
    kw = dict(
        n_neighbors=8,
        n_landmarks=12,
        seed=0,
        knn_mode="brute",
        epsilon=None,
    )
    g_std, M_std, top1_std, topc_std = build_graph(X, metric, **kw)
    g_b, M_b, top1_b, topc_b = build_graph_bunches(X, metric, **kw)

    assert int(g_std.reps.rep_idx.shape[0]) == int(g_b.reps.rep_idx.shape[0])
    assert torch.equal(g_std.reps.rep_idx, g_b.reps.rep_idx)
    assert torch.equal(g_std.reps.member_of, g_b.reps.member_of)
    assert torch.equal(g_std.edges, g_b.edges)
    assert torch.allclose(g_std.weights, g_b.weights)
    assert torch.equal(M_std, M_b)
    assert torch.equal(top1_std, top1_b)
    assert torch.equal(topc_std, topc_b)


def test_bunches_import_does_not_require_mpi4py(monkeypatch):
    """Top-level import of bunches must work without mpi4py installed."""
    # Simulate missing mpi4py / mpi4py.MPI.
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)
    sys.modules.pop("leanmap.build.bunches", None)
    mod = importlib.import_module("leanmap.build.bunches")
    assert hasattr(mod, "build_graph_bunches")
    assert hasattr(mod, "distributed_union_find")
    # Touching MPI-only path should raise with hpc install hint.
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ImportError, match=r"leanmap\[hpc\]"):
        mod.mpi_world_size()


def test_probe_reconcile_partition_halo_helpers():
    torch.manual_seed(1)
    X = torch.randn(30, 3)
    idx = probe_shards(X, n_probe=10, seed=3)
    assert idx.shape == (10,)
    assert int(idx.unique().numel()) == 10

    local = X[idx[:5]]
    out = reconcile_landmarks(local, world_size=1)
    assert torch.equal(out, local)

    bunch = partition_bunches(torch.arange(9), n_bunches=3)
    assert bunch.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    assign = torch.tensor([0, 0, 1, 1, 2, 2])
    owned, halo = margin_halo(assign, margin=1)
    assert owned.dtype == torch.bool
    assert halo.any()  # multi-bunch → boundary fringe

    # top-c path: foreign shortlist → halo
    topc = torch.tensor([[0, 1], [0, 0], [1, 0], [1, 1]])
    _, halo2 = margin_halo(topc, margin=1)
    assert halo2.tolist() == [True, False, True, False]


def test_owned_net_and_fill_knn_and_metrics():
    torch.manual_seed(2)
    X = torch.randn(20, 4)
    owned = torch.arange(10)
    reps, member_of = owned_net(X, owned, radius=0.5, seed=0)
    assert reps.ndim == 1 and member_of.shape == (10,)
    assert int(reps.numel()) >= 1

    knn_idx, knn_dist = fill_knn_rows(X, 0, 5, k=3)
    assert knn_idx.shape == (5, 3)
    assert knn_dist.shape == (5, 3)
    assert (knn_idx >= 0).all()
    assert knn_completeness_audit(3, knn_idx) == pytest.approx(0.0)

    assert halo_fraction(90, 10) == pytest.approx(0.1)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cut = torch.tensor([False, True, False, True])
    assert cut_mass(w, cut) == pytest.approx(6.0 / 10.0)

    stitched = stitch_graph(
        [
            {"rep_idx": torch.tensor([0, 1]), "edges": torch.tensor([[0, 1]]), "weights": torch.tensor([1.0])},
            {"rep_idx": torch.tensor([2]), "edges": torch.tensor([[0, 0]]), "weights": torch.tensor([0.5])},
        ],
        n_total=3,
    )
    assert stitched["rep_idx"].tolist() == [0, 1, 2]
    assert stitched["edges"].tolist() == [[0, 1], [2, 2]]

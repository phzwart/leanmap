"""Tests for leanmap.store (ptfile / dirstore / fingerprint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from leanmap.graph import build_graph_pyramid, load_graph_pyramid, save_graph_pyramid
from leanmap.metrics import wrap_metric
from leanmap.store import (
    DirStore,
    PtFileStore,
    fingerprint_array,
    needs_rebuild,
    open_graph_store,
    select_backend,
    verify_fingerprint,
)
from leanmap.store.schema import DIR_STORE_R_THRESHOLD, STORE_DIRS


def _tiny_pyramid(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(80, 4)
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=seed)
    graphs, M, top1, topc = build_graph_pyramid(
        X,
        metric,
        pyramid_scales=1,
        pyramid_rep_ratio=3.0,
        pyramid_min_reps=8,
        n_neighbors=8,
        n_landmarks=8,
        epsilon=0.0,
        seed=seed,
        knn_mode="brute",
    )
    n = int(X.shape[0])
    train_idx = torch.arange(n, dtype=torch.int64)
    calib_idx = torch.arange(0, dtype=torch.int64)
    return X, graphs, M, top1, topc, train_idx, calib_idx


def _graphs_equivalent(a_graphs, b_graphs) -> None:
    assert len(a_graphs) == len(b_graphs)
    for ga, gb in zip(a_graphs, b_graphs):
        assert torch.equal(ga.edges, gb.edges)
        assert torch.allclose(ga.weights, gb.weights)
        assert torch.equal(ga.knn_idx, gb.knn_idx)
        assert torch.equal(ga.reps.rep_idx, gb.reps.rep_idx)
        assert torch.equal(ga.reps.member_of, gb.reps.member_of)
        assert torch.allclose(ga.reps.weight, gb.reps.weight)
        assert torch.equal(ga.reps.offsets, gb.reps.offsets)
        assert torch.equal(ga.reps.values, gb.reps.values)


def test_select_backend_rules(tmp_path: Path):
    pt = tmp_path / "graph.pt"
    pt.write_bytes(b"x")
    assert select_backend(pt) == "ptfile"
    assert select_backend(tmp_path / "new.pt") == "ptfile"

    d = tmp_path / "store_dir"
    d.mkdir()
    assert select_backend(d) == "dirstore"

    assert select_backend(tmp_path / "fresh_store", n_reps=10) == "ptfile"
    assert (
        select_backend(tmp_path / "big_store", n_reps=DIR_STORE_R_THRESHOLD + 1)
        == "dirstore"
    )


def test_roundtrip_ptfile(tmp_path: Path):
    X, graphs, M, top1, topc, train_idx, calib_idx = _tiny_pyramid()
    path = tmp_path / "graph.pt"
    fp = {"shape": list(X.shape), "mean": float(X.mean()), "head": X.reshape(-1)[:8].tolist(), "tail": X.reshape(-1)[-8:].tolist()}
    save_graph_pyramid(
        path,
        graphs=graphs,
        M=M,
        assign_top1=top1,
        assign_topc=topc,
        train_idx=train_idx,
        calib_idx=calib_idx,
        fingerprint=fp,
        metric_name="l2",
        n_all=int(X.shape[0]),
        n_neighbors=8,
        epsilon=0.0,
        seed=0,
        dedup=True,
    )
    store = open_graph_store(path)
    assert isinstance(store, PtFileStore)
    loaded = store.load()
    _graphs_equivalent(graphs, loaded["graphs"])
    assert torch.equal(store.edges(0), graphs[0].edges)
    meta = store.meta()
    assert meta["metric_name"] == "l2"
    assert "diagnostics" in meta
    # Public shim still works.
    again = load_graph_pyramid(path)
    _graphs_equivalent(graphs, again["graphs"])


def test_roundtrip_dirstore_from_build(tmp_path: Path):
    X, graphs, M, top1, topc, train_idx, calib_idx = _tiny_pyramid(seed=1)
    # Save via pt first, then DirStore from that state (requirement: save from loaded pt).
    pt_path = tmp_path / "graph.pt"
    from leanmap.build.pipeline import tensor_fingerprint

    save_graph_pyramid(
        pt_path,
        graphs=graphs,
        M=M,
        assign_top1=top1,
        assign_topc=topc,
        train_idx=train_idx,
        calib_idx=calib_idx,
        fingerprint=tensor_fingerprint(X),
        metric_name="l2",
        n_all=int(X.shape[0]),
        n_neighbors=8,
        epsilon=float(graphs[0].stats.epsilon),
        seed=1,
        dedup=True,
    )
    pt_state = load_graph_pyramid(pt_path)

    root = tmp_path / "graph_store"
    store = DirStore(root)
    store.save_from_state(pt_state, X=X)
    for name in STORE_DIRS:
        assert (root / name).is_dir()
    assert (root / "meta.json").is_file()

    reopened = open_graph_store(root)
    assert isinstance(reopened, DirStore)
    loaded = reopened.load()
    _graphs_equivalent(pt_state["graphs"], loaded["graphs"])
    assert torch.equal(reopened.edges(0), pt_state["graphs"][0].edges)
    assert torch.equal(loaded["M"], pt_state["M"])
    assert torch.equal(loaded["train_idx"], pt_state["train_idx"])
    meta = reopened.meta()
    assert meta["schema_version"] == 1
    assert "diagnostics" in meta
    assert verify_fingerprint(X, meta, full=True)


@pytest.mark.parametrize(
    "mismatch",
    ["fingerprint", "seed", "metric_name", "n_neighbors", "n_landmarks", "dedup"],
)
def test_needs_rebuild_invalidation_matrix(mismatch: str):
    X = torch.randn(40, 3)
    fp = fingerprint_array(X)
    meta = {
        "fingerprint": fp,
        "metric_name": "l2",
        "epsilon": 0.1,
        "n_neighbors": 10,
        "n_landmarks": 16,
        "seed": 7,
        "dedup": True,
        "n_pyramid_levels": 2,
    }
    assert needs_rebuild(meta, X, {}) is False
    assert needs_rebuild(meta, X, {"seed": 7, "metric_name": "l2"}) is False

    if mismatch == "fingerprint":
        X2 = X + 1.0
        assert needs_rebuild(meta, X2, {}) is True
    elif mismatch == "seed":
        assert needs_rebuild(meta, X, {"seed": 99}) is True
    elif mismatch == "metric_name":
        assert needs_rebuild(meta, X, {"metric_name": "cosine"}) is True
    elif mismatch == "n_neighbors":
        assert needs_rebuild(meta, X, {"n_neighbors": 3}) is True
    elif mismatch == "n_landmarks":
        assert needs_rebuild(meta, X, {"n_landmarks": 4}) is True
    elif mismatch == "dedup":
        assert needs_rebuild(meta, X, {"dedup": False}) is True


def test_dirstore_save_accepts_delta_auto_token(tmp_path: Path):
    """CLI may pass config delta='auto'; store must persist the resolved radius."""
    X, graphs, M, top1, topc, train_idx, calib_idx = _tiny_pyramid(seed=2)
    graphs[0].stats.delta = 0.42
    store = DirStore(tmp_path / "store")
    store.save(
        graphs=graphs,
        M=M,
        assign_top1=top1,
        assign_topc=topc,
        train_idx=train_idx,
        calib_idx=calib_idx,
        fingerprint={"shape": list(X.shape)},
        metric_name="l2",
        n_all=int(X.shape[0]),
        n_neighbors=8,
        epsilon=0.1,
        seed=0,
        dedup=True,
        delta="auto",
    )
    meta = store.meta()
    assert meta["delta"] == pytest.approx(0.42)
    assert meta["diagnostics"]["delta"] == pytest.approx(0.42)


def test_fingerprint_sampled_vs_full():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 8)).astype(np.float32)
    fp = fingerprint_array(X)
    meta = {"fingerprint": fp}
    assert verify_fingerprint(X, meta, full=False)
    assert verify_fingerprint(X, meta, full=True)
    X_bad = X.copy()
    # Corrupt a region that may or may not be in a sample block; full must catch it.
    X_bad.reshape(-1)[123] = X_bad.reshape(-1)[123] + 10.0
    assert verify_fingerprint(X_bad, meta, full=True) is False

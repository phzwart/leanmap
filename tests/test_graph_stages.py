"""Zarr graph stage round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("zarr")

from leanmap.graph import Representatives
from leanmap.graph_stages import (
    create_knn_store,
    fingerprint_matches,
    init_meta,
    load_enet,
    load_knn,
    load_landmarks,
    mark_knn_complete,
    save_enet,
    save_landmarks,
)


def test_stages_landmarks_enet_knn_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    X = torch.randn(30, 4)
    root = tmp_path / "stages"
    init_meta(root, X, seed=0)
    assert fingerprint_matches(root, X)

    M = X[:5].contiguous()
    top1 = torch.randint(0, 5, (30,), dtype=torch.int64)
    topc = torch.randint(0, 5, (30, 3), dtype=torch.int64)
    save_landmarks(root, M, top1, topc)
    Ml, t1, tc = load_landmarks(root)
    assert torch.allclose(Ml, M)
    assert torch.equal(t1, top1)

    reps = Representatives(
        rep_idx=torch.arange(10, dtype=torch.int64),
        member_of=torch.randint(0, 10, (30,), dtype=torch.int64),
        weight=torch.ones(10),
        offsets=torch.arange(11, dtype=torch.int64),
        values=torch.arange(30, dtype=torch.int64),
    )
    save_enet(root, reps, 0.2, halo_done=True)
    loaded, eps, halo_done = load_enet(root)
    assert eps == pytest.approx(0.2)
    assert halo_done is True
    assert torch.equal(loaded.rep_idx, reps.rep_idx)

    # Pre-halo checkpoint round-trip
    save_enet(root, reps, 0.2, halo_done=False)
    _, _, halo_done2 = load_enet(root)
    assert halo_done2 is False

    store = create_knn_store(root, 10, 3)
    store.idx[:] = np.array([[(i + j + 1) % 10 for j in range(3)] for i in range(10)], dtype=np.int64)
    store.dist[:] = np.random.rand(10, 3).astype(np.float32)
    mark_knn_complete(root)
    knn_idx, knn_dist = load_knn(root)
    assert knn_idx.shape == (10, 3)
    assert knn_dist.shape == (10, 3)

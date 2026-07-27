"""End-to-end integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from leanmap import PLANEConfig, fit, knn_recall_out_of_sample, load_plane
from leanmap.config import AlignmentSpec
from leanmap.evaluate import trustworthiness_continuity
from leanmap.metrics import wrap_metric


def _swiss_roll(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    X = np.stack([x, y, z], axis=1).astype(np.float32)
    return X, t.astype(np.float32)


def test_swiss_roll_integration():
    X, t = _swiss_roll(1200, seed=0)  # smaller for speed
    cfg = PLANEConfig.for_scale(X.shape[0])
    cfg.epochs = 15
    cfg.batch_edges = 512
    cfg.n_landmarks = 32
    cfg.width = 64
    cfg.depth = 2
    cfg.spectral_norm = False
    cfg.epsilon = 0.0
    cfg.calib_max = 100
    cfg.device = "cpu"
    # hold out test
    rng = np.random.default_rng(0)
    perm = rng.permutation(X.shape[0])
    n_test = 200
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]
    result = fit(X_train, "l2", config=cfg)
    Z_train, _ = result.model.embed(
        torch.as_tensor(X_train), return_score=False
    )
    Z_test, _ = result.model.embed(
        torch.as_tensor(X_test), return_score=False
    )
    metric = wrap_metric("l2", X=torch.as_tensor(X_train), n_neighbors=15, seed=0)
    tw = trustworthiness_continuity(
        X_train, Z_train.numpy(), metric, k_list=(15,), n_sample=800
    )
    # Soft floor for short training
    assert tw.get("knn_overlap_15", 0) > 0.3 or tw.get("trust_15", 0) > 0.7
    recall = knn_recall_out_of_sample(
        X_train, Z_train.numpy(), X_test, Z_test.numpy(), metric, k=15
    )
    assert recall > 0.25  # short training; full 0.6 target needs more epochs


def test_save_load_roundtrip_no_N_arrays():
    X = np.random.randn(400, 8).astype(np.float32)
    cfg = PLANEConfig.for_scale(X.shape[0])
    cfg.epochs = 3
    cfg.batch_edges = 256
    cfg.n_landmarks = 16
    cfg.width = 32
    cfg.depth = 2
    cfg.spectral_norm = False
    cfg.epsilon = 0.0
    cfg.calib_max = 40
    cfg.device = "cpu"
    result = fit(X, "l2", config=cfg)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "plane.pt"
        result.save(path)
        payload = torch.load(path, weights_only=False)
        N = X.shape[0]
        R = result.graph_stats.get("n_reps", -1)
        for k, v in payload.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                assert v.shape[0] != N, k
                if R > 0:
                    assert v.shape[0] != R or k in ("s_calib",), k
        model2 = load_plane(path)
        z1, _ = result.model.embed(torch.as_tensor(X[:50]), return_score=False)
        z2, _ = model2.embed(torch.as_tensor(X[:50]), return_score=False)
        assert torch.allclose(z1, z2, atol=1e-4)


def test_pipeline_metrics_l1_cosine_composite():
    X = np.random.randn(300, 16).astype(np.float32)
    cfg = PLANEConfig(
        epochs=2,
        batch_edges=128,
        n_landmarks=8,
        width=32,
        depth=2,
        spectral_norm=False,
        epsilon=0.0,
        calib_max=30,
        n_neighbors=10,
        knn_mode="ivf",  # exercise landmark-IVF (needed for l1)
        device="cpu",
    )
    for m in ("l2", "l1", "cosine"):
        cfg.metric = m
        cfg.knn_mode = "ivf" if m == "l1" else "brute"
        fit(X, m, config=cfg)

    from leanmap.metrics import CompositeMetric

    comp = CompositeMetric(
        [(slice(0, 8), "l2", 1.0), (slice(8, 16), "correlation", 1.0)]
    )
    cfg.knn_mode = "ivf"
    fit(X, comp, config=cfg)

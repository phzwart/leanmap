"""Core capability API guard — must stay green on every refactor PR."""
from __future__ import annotations

import numpy as np
import torch

import leanmap


CORE_SYMBOLS = [
    "PLANEConfig",
    "PLANE",
    "fit",
    "PathConstraint",
    "PathTripletSampler",
    "path_constraint_loss",
    "build_path_triplets",
    "ClassAxis",
    "ClassOrderSampler",
    "ordinal_class_axis",
    "ConformalCalibrator",
    "MondrianCalibrator",
    "FactorStack",
    "build_graph",
    "build_graph_pyramid",
    "fit_negative_space",
]


def test_core_symbols_importable():
    for name in CORE_SYMBOLS:
        assert hasattr(leanmap, name), name
        assert getattr(leanmap, name) is not None


def test_path_smoke_fit():
    from leanmap import PLANEConfig, PathConstraint, build_path_triplets, fit

    rng = np.random.default_rng(0)
    n = 80
    t = np.arange(n, dtype=np.float32)
    X = np.stack([t, rng.normal(scale=0.2, size=n).astype(np.float32)], axis=1)
    group = np.zeros(n, dtype=np.int32)
    tri, dt = build_path_triplets(group, t, lag=4)
    assert tri.shape[0] > 0
    pc = PathConstraint("walk", tri, dt, weight=1.0)
    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = 3
    cfg.device = "cpu"
    cfg.seed = 0
    cfg.n_landmarks = 8
    cfg.lambda_path = 0.5
    result = fit(X, config=cfg, path_constraints=[pc])
    Z, _ = result.embed(X)
    assert Z.shape[0] == n
    assert np.isfinite(np.asarray(Z)).all()


def test_class_axis_smoke_fit():
    from leanmap import PLANEConfig, fit, ordinal_class_axis

    rng = np.random.default_rng(1)
    n = 60
    y = rng.integers(0, 3, size=n)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[:, 0] += y.astype(np.float32)
    axis = ordinal_class_axis(3, axis=None)
    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = 3
    cfg.device = "cpu"
    cfg.seed = 1
    cfg.n_landmarks = 8
    cfg.lambda_class = 0.5
    result = fit(X, config=cfg, class_labels=y, class_axes=[axis])
    Z, _ = result.embed(X)
    assert Z.shape[0] == n
    assert np.isfinite(np.asarray(Z)).all()

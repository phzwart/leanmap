"""Path constraint: triplet tables and scale-free ranking hinges."""

from __future__ import annotations

import numpy as np
import torch

from leanmap import PLANEConfig, fit
from leanmap.path import (
    PathConstraint,
    build_path_triplets,
    path_constraint_loss,
    remap_triplets,
)


def test_build_and_remap_triplets():
    group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    index = np.arange(10, dtype=np.float64)
    index[5:] = np.arange(5)
    tri, dt = build_path_triplets(group, index, lag=3)
    assert tri.shape[1] == 3
    assert np.all(dt[:, 0] == 1) and np.all(dt[:, 1] == 3)
    # first group rows 0-4: (0,1,3), (1,2,4)
    assert [0, 1, 3] in tri.tolist()
    keep = np.array([0, 1, 2, 3, 5, 6, 7, 8])  # drop 4 and 9
    tri2, dt2 = remap_triplets(tri, dt, keep)
    # (1,2,4) must drop because 4 is gone
    assert [1, 2, 3] not in tri2.tolist()  # remapped; original (0,1,3) -> (0,1,3)
    assert tri2.shape[0] < tri.shape[0]


def test_restrict_drops_calib_endpoints():
    tri = np.array([[0, 1, 3], [7, 8, 9]], dtype=np.int64)
    dt = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float32)
    pc = PathConstraint("c", tri, dt)
    kept = pc.restrict(np.arange(8), n_all=10)
    assert kept is not None
    assert kept.triplets.shape[0] == 1
    assert kept.triplets[0].tolist() == [0, 1, 3]


def test_loss_zero_when_ordered_and_lipschitz():
    # isometric walk: d(Δ) = Δ, far much farther
    z_a = torch.zeros(4, 2)
    z_n = torch.tensor([[1.0, 0.0]] * 4)
    z_m = torch.tensor([[8.0, 0.0]] * 4)
    z_f = torch.tensor([[40.0, 0.0]] * 4)
    dt_n = torch.ones(4)
    dt_m = torch.full((4,), 8.0)
    loss, st, ord_frac = path_constraint_loss(
        z_a, z_n, z_m, z_f, dt_n, dt_m, scale_state={}
    )
    assert ord_frac == 1.0
    assert float(loss) < 1e-6
    assert st["s"] > 0


def test_loss_scale_free():
    z_a = torch.zeros(8, 2)
    z_n = torch.tensor([[1.0, 0.0]] * 8)
    z_m = torch.tensor([[8.0, 0.0]] * 8)
    z_f = torch.tensor([[40.0, 0.0]] * 8)
    dt_n = torch.ones(8)
    dt_m = torch.full((8,), 8.0)
    l1, _, _ = path_constraint_loss(z_a, z_n, z_m, z_f, dt_n, dt_m, scale_state={})
    l2, _, _ = path_constraint_loss(
        z_a * 5, z_n * 5, z_m * 5, z_f * 5, dt_n, dt_m, scale_state={}
    )
    assert abs(float(l1) - float(l2)) < 1e-5


def test_fit_consumes_triplets_without_graph_key_change():
    rng = np.random.default_rng(0)
    n = 80
    t = np.arange(n, dtype=np.float32)
    X = np.stack([t, rng.normal(scale=0.2, size=n).astype(np.float32)], axis=1)
    group = np.zeros(n, dtype=np.int32)
    tri, dt = build_path_triplets(group, t, lag=4)
    pc = PathConstraint("walk", tri, dt)
    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = 2
    cfg.n_landmarks = 16
    cfg.dedup = False
    cfg.lambda_geo = 0.0
    cfg.lambda_density = 0.0
    cfg.lambda_path = 1.0
    cfg.path_ramp = (0.0, 0.05)
    cfg.batch_edges = 64
    res = fit(
        X,
        dist_fn="l2",
        config=cfg,
        X_calib=X[:8],
        path_constraints=[pc],
    )
    assert res.config.n_landmarks == 16
    Z, _ = res.embed(X)
    # successor should not be farther than a random pair on average
    d_succ = np.linalg.norm(Z[1:] - Z[:-1], axis=1).mean()
    i = rng.integers(0, n, 200)
    j = rng.integers(0, n, 200)
    d_rand = np.linalg.norm(Z[i] - Z[j], axis=1).mean()
    assert d_succ < d_rand

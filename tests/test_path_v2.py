"""Path module v2: vectorized build, tie policies, log-space loss."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.diagnostics.record import DiagnosticsRecord
from leanmap.path import (
    build_path_triplets,
    build_path_triplets_with_stats,
    path_constraint_loss,
)


def _legacy_dict_build(group, index, lag=8):
    """Reference dict-lookup build (pre-PR-5) for bit-compat checks."""
    group = np.asarray(group)
    index = np.asarray(index, dtype=np.float64)
    lag = int(lag)
    rows: list[list[int]] = []
    dts: list[list[float]] = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        t = index[idx]
        order = np.argsort(t, kind="mergesort")
        t_s = t[order]
        r_s = idx[order]
        lookup = {}
        for tt, rr in zip(t_s, r_s):
            lookup[float(tt)] = int(rr)
        for tt, rr in zip(t_s, r_s):
            nkey = float(tt) + 1.0
            mkey = float(tt) + float(lag)
            if nkey in lookup and mkey in lookup:
                rows.append([int(rr), lookup[nkey], lookup[mkey]])
                dts.append([1.0, float(lag)])
    if not rows:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )
    return np.asarray(rows, dtype=np.int64), np.asarray(dts, dtype=np.float32)


def test_vectorized_equals_legacy_no_duplicate_t():
    group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    index = np.arange(10, dtype=np.float64)
    index[5:] = np.arange(5)
    tri_new, dt_new = build_path_triplets(group, index, lag=3)
    tri_old, dt_old = _legacy_dict_build(group, index, lag=3)
    assert tri_new.shape == tri_old.shape
    # Same multiset of rows (order may differ by group traversal)
    assert sorted(map(tuple, tri_new.tolist())) == sorted(map(tuple, tri_old.tolist()))
    assert np.allclose(dt_new, dt_old)


def test_tie_policy_first_last_drop_differ():
    # Duplicate t=2 appears twice; lag=3 needs t+1 and t+3 present.
    # index: 0,1,2,2,3,5  → unique with first/last keep one of the t=2 rows;
    # drop removes both t=2 rows so fewer / different anchors.
    group = np.zeros(6, dtype=np.int32)
    index = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 5.0])
    tri_first, _, st_first = build_path_triplets_with_stats(
        group, index, lag=3, tie_policy="first"
    )
    tri_last, _, st_last = build_path_triplets_with_stats(
        group, index, lag=3, tie_policy="last"
    )
    tri_drop, _, st_drop = build_path_triplets_with_stats(
        group, index, lag=3, tie_policy="drop"
    )
    assert st_first["path_tie_values"] >= 1
    assert st_drop["path_tie_values"] >= 1
    # first uses row 2 as the t=2 representative; last uses row 3
    assert tri_first.tolist() != tri_last.tolist() or tri_first.shape[0] != tri_last.shape[0]
    # Explicit: anchors that use the tied t differ
    first_anchors = set(tri_first[:, 0].tolist()) if tri_first.size else set()
    last_anchors = set(tri_last[:, 0].tolist()) if tri_last.size else set()
    # At least one of {2,3} appears as near/mid/anchor differently across policies
    assert first_anchors != last_anchors or sorted(map(tuple, tri_first.tolist())) != sorted(
        map(tuple, tri_last.tolist())
    )
    assert sorted(map(tuple, tri_drop.tolist())) != sorted(map(tuple, tri_first.tolist()))


def test_tie_stats_into_diagnostics_extra():
    group = np.zeros(6, dtype=np.int32)
    index = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 5.0])
    D = DiagnosticsRecord()
    build_path_triplets(group, index, lag=3, tie_policy="first", diagnostics=D)
    assert "path_tie_policy" in D.extra
    assert D.extra["path_tie_policy"] == "first"
    assert D.extra["path_tie_values"] >= 1


def test_eps_filter_drops_near_duplicates():
    group = np.zeros(5, dtype=np.int32)
    index = np.arange(5, dtype=np.float64)
    # Rows 0 and 1 are identical in X → φ(x_0,x_1)=0 ≤ eps
    X = np.zeros((5, 2), dtype=np.float32)
    X[:, 0] = np.array([0.0, 0.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tri_all, _ = build_path_triplets(group, index, lag=3)
    assert tri_all.shape[0] > 0
    tri_f, _, st = build_path_triplets_with_stats(
        group, index, lag=3, eps=1e-6, X=X, dist_fn="l2"
    )
    assert st["path_eps_dropped"] >= 1
    assert tri_f.shape[0] < tri_all.shape[0]


def test_log_space_floor_bounds_near_duplicate_grads():
    # Sliding-window near-duplicate: tiny d_n, moderate d_m → ratio hinge spikes.
    z_a = torch.zeros(8, 2, requires_grad=True)
    z_n = torch.full((8, 2), 1e-8)  # almost coincident
    z_m = torch.tensor([[1.0, 0.0]] * 8)
    z_f = torch.tensor([[10.0, 0.0]] * 8)
    dt_n = torch.ones(8)
    dt_m = torch.full((8,), 8.0)

    z_a_r = z_a.detach().clone().requires_grad_(True)
    loss_ratio, _, ord_frac = path_constraint_loss(
        z_a_r, z_n, z_m, z_f, dt_n, dt_m, scale_state={}, log_space=False
    )
    loss_ratio.backward()
    grad_ratio = float(z_a_r.grad.norm().item())

    z_a_l = z_a.detach().clone().requires_grad_(True)
    loss_log, _, ord_frac_log = path_constraint_loss(
        z_a_l,
        z_n,
        z_m,
        z_f,
        dt_n,
        dt_m,
        scale_state={},
        log_space=True,
        distance_floor_kappa=1e-3,
    )
    loss_log.backward()
    grad_log = float(z_a_l.grad.norm().item())

    assert ord_frac == ord_frac_log  # same distance ordering
    assert float(loss_log.detach()) < float(loss_ratio.detach()) or grad_log < grad_ratio
    assert np.isfinite(grad_log)
    assert grad_log < 1e6


def test_ord_frac_returned_as_today():
    z_a = torch.zeros(4, 2)
    z_n = torch.tensor([[1.0, 0.0]] * 4)
    z_m = torch.tensor([[8.0, 0.0]] * 4)
    z_f = torch.tensor([[40.0, 0.0]] * 4)
    dt_n = torch.ones(4)
    dt_m = torch.full((4,), 8.0)
    loss, st, ord_frac = path_constraint_loss(
        z_a, z_n, z_m, z_f, dt_n, dt_m, scale_state={}
    )
    assert isinstance(ord_frac, float)
    assert ord_frac == 1.0
    assert float(loss) < 1e-6
    assert st["s"] > 0

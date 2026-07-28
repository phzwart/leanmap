"""Warm start and the coarse-to-fine step schedule."""

import numpy as np
import torch

from leanmap.metrics import get_metric
from leanmap.train import _split_budget, coarse_to_fine_plan
from leanmap.warmstart import nystrom_targets

L2 = get_metric("l2").fn


def _landmarks_on_a_line(n=200, L=16):
    """Points on a 1-D curve in 3-D, with landmarks sampled along it."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    X = np.stack([t, t**2, np.zeros_like(t)], axis=1)
    idx = np.linspace(0, n - 1, L).astype(int)
    return torch.as_tensor(X), torch.as_tensor(X[idx]), torch.as_tensor(t[idx, None])


def test_nystrom_targets_recover_a_monotone_coordinate():
    """Interpolating a 1-D MDS coordinate must order points along the curve.

    Rank correlation rather than strict monotonicity: the weight set changes
    discontinuously wherever the c-th nearest landmark switches, so the
    interpolant is only piecewise smooth and can wobble by a hair at those seams.
    """
    from scipy.stats import spearmanr

    X, X_lm, Z_lm = _landmarks_on_a_line()
    T = nystrom_targets(X, X_lm, Z_lm, L2, min_dist=0.5)
    assert T.shape == (X.shape[0], 1)
    rho = spearmanr(T[:, 0].numpy(), np.arange(X.shape[0])).correlation
    assert rho > 0.999


def test_nystrom_targets_scaled_to_min_dist():
    """Whatever units the MDS came in, neighbours start at ``min_dist``."""
    X, X_lm, Z_lm = _landmarks_on_a_line()
    for scale in (1e-3, 1.0, 1e4):
        T = nystrom_targets(X, X_lm, Z_lm * scale, L2, min_dist=0.25)
        d = torch.cdist(T, T)
        d.fill_diagonal_(float("inf"))
        assert abs(float(d.min(dim=1).values.median()) - 0.25) < 1e-3


def test_nystrom_targets_land_on_coincident_landmarks():
    """A point sitting on a landmark takes that landmark's coordinate."""
    X, X_lm, Z_lm = _landmarks_on_a_line()
    T = nystrom_targets(X_lm, X_lm, Z_lm, L2, min_dist=1.0)
    # up to the uniform rescaling the ordering and relative spacing must match
    ref = (Z_lm[:, 0] - Z_lm[:, 0].mean()) / Z_lm[:, 0].std()
    got = (T[:, 0] - T[:, 0].mean()) / T[:, 0].std()
    assert torch.allclose(ref, got, atol=1e-3)


def _ring_graph(n=60):
    """A cycle graph: edges, unit weights, and the ambient kNN of its embedding."""
    lo = np.arange(n)
    hi = (lo + 1) % n
    edges = torch.as_tensor(np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], 1))
    weights = torch.ones(edges.shape[0])
    theta = 2 * np.pi * np.arange(n) / n
    X = torch.as_tensor(
        np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32)
    )
    return edges, weights, X


def test_spectral_layout_recovers_a_ring():
    """The two leading eigenvectors of a cycle are its circular coordinates."""
    from leanmap.warmstart import spectral_layout

    edges, weights, X = _ring_graph()
    Z = spectral_layout(edges, weights, X.shape[0], 2, seed=0).numpy()
    ang = np.arctan2(Z[:, 1], Z[:, 0])
    # consecutive nodes must be adjacent in angle, i.e. one full turn, no folding
    step = np.diff(np.unwrap(ang))
    assert abs(abs(step.sum()) - 2 * np.pi) < 0.2
    assert np.all(np.sign(step) == np.sign(step[0]))


def test_spectral_layout_is_reproducible():
    """ARPACK starts from a random vector unless told otherwise."""
    from leanmap.warmstart import spectral_layout

    edges, weights, X = _ring_graph()
    a = spectral_layout(edges, weights, X.shape[0], 2, seed=0)
    b = spectral_layout(edges, weights, X.shape[0], 2, seed=0)
    assert torch.equal(a, b)
    # signs pinned on the largest-magnitude entry, which is well away from zero
    pivot = a.abs().argmax(dim=0)
    assert bool((a[pivot, torch.arange(a.shape[1])] > 0).all())


def test_rank_inits_prefers_the_layout_that_keeps_neighbours():
    """A faithful layout must outrank a scrambled one."""
    from leanmap.warmstart import rank_inits

    edges, weights, X = _ring_graph()
    n = X.shape[0]
    good = X.clone()
    rng = np.random.default_rng(0)
    bad = torch.as_tensor(rng.standard_normal((n, 2)).astype(np.float32))
    knn = torch.as_tensor(
        np.stack(
            [(np.arange(n) - 1) % n, (np.arange(n) + 1) % n], axis=1
        ).astype(np.int64)
    )
    ranking = rank_inits(
        {"scrambled": (X, bad), "faithful": (X, good)},
        X,
        knn,
        L2,
        min_dist=0.5,
    )
    assert ranking[0][0] == "faithful"
    assert ranking[0][1] > ranking[1][1]


def test_rank_inits_refuses_empty_reference_neighbours():
    """Better an error than silently ranking on nothing."""
    import pytest

    from leanmap.warmstart import rank_inits

    _, _, X = _ring_graph()
    with pytest.raises(ValueError, match="reference neighbours"):
        rank_inits(
            {"a": (X, X)},
            X,
            torch.zeros((X.shape[0], 0), dtype=torch.int64),
            L2,
            min_dist=0.5,
        )


def test_named_layout_that_is_unavailable_does_not_silently_train():
    """Asking for a layout that was not built must not quietly skip the warm start."""
    from leanmap.warmstart import warm_start

    _, _, X = _ring_graph()
    knn = torch.as_tensor(
        np.stack(
            [(np.arange(X.shape[0]) - 1) % X.shape[0], (np.arange(X.shape[0]) + 1) % X.shape[0]],
            axis=1,
        ).astype(np.int64)
    )
    info = warm_start(
        torch.nn.Linear(2, 2),  # never reached
        X,
        {"spectral": (X, X)},
        L2,
        layout="isomap",
        X_ref=X,
        reference_knn=knn,
        steps=5,
        batch=8,
        lr=1e-3,
        min_dist=0.5,
    )
    assert info == {}  # signalled, not silently substituted


def test_split_budget_conserves_edges_and_zeroes_inactive():
    counts = _split_budget(4096, [1.0, 2.0, 4.0], [0, 1, 2])
    assert sum(counts) == 4096
    assert counts[2] > counts[1] > counts[0]  # follows the weights
    coarse_only = _split_budget(4096, [1.0, 2.0, 4.0], [2])
    assert coarse_only == [0, 0, 4096]
    assert _split_budget(4096, [1.0], []) == [0]


def test_coarse_frac_zero_reproduces_flat_schedule():
    edges = [40000, 10000, 2600]
    flat = -(-edges[0] // 4096)
    plan = coarse_to_fine_plan(20, edges, 4096, [1.0, 1.0, 1.0], 0.0)
    assert [s for _, s in plan] == [flat] * 20
    assert all(all(c > 0 for c in counts) for counts, _ in plan)


def test_coarse_first_costs_fewer_steps_and_ends_on_all_levels():
    edges = [40000, 10000, 2600]
    flat = 20 * -(-edges[0] // 4096)
    plan = coarse_to_fine_plan(20, edges, 4096, [1.0, 2.0, 4.0], 0.5)
    assert sum(s for _, s in plan) < flat
    first_active = [i for i, c in enumerate(plan[0][0]) if c > 0]
    assert first_active == [2]  # coarsest alone
    assert all(c > 0 for c in plan[-1][0])  # finishes at the full mix
    # step count is monotone as finer levels are admitted
    steps = [s for _, s in plan]
    assert steps == sorted(steps)


def test_single_level_pyramid_is_unaffected():
    plan = coarse_to_fine_plan(4, [40000], 4096, [1.0], 0.5)
    assert [s for _, s in plan] == [-(-40000 // 4096)] * 4

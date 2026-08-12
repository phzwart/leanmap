"""Stage-1 tests: metric registry properties."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.spatial.distance import correlation as scipy_corr

from leanmap.distance import chunked_cdist
from leanmap.metrics import CompositeMetric, get_metric, wrap_metric


REGISTRY = [
    "l2",
    "sqeuclidean",
    "frobenius",
    "cosine",
    "correlation",
    "correlation_sqrt",
    "l1",
    "linf",
    "canberra",
    "braycurtis",
    "jensenshannon",
    "wasserstein1d",
]


def _check_basic(fn, A, B):
    d = fn(A, B)
    assert d.shape == (A.shape[0], B.shape[0])
    assert torch.isfinite(d).all()
    assert (d >= -1e-5).all()
    # symmetry on square
    if A.shape == B.shape and torch.allclose(A, B):
        assert torch.allclose(d, d.T, atol=1e-4)
        assert torch.allclose(torch.diag(d), torch.zeros(A.shape[0]), atol=1e-4)


def test_registry_basic_properties():
    torch.manual_seed(0)
    A = torch.randn(20, 12).abs() + 0.1  # positive for JS / canberra
    B = torch.randn(15, 12).abs() + 0.1
    A0 = torch.zeros(5, 12)
    Aeq = torch.ones(5, 12)
    for name in REGISTRY:
        spec = get_metric(name)
        _check_basic(spec.fn, A, B)
        _check_basic(spec.fn, A0, A0)
        _check_basic(spec.fn, Aeq, Aeq)
        d = spec.fn(A, A)
        assert torch.allclose(d, d.T, atol=1e-4)
        assert torch.allclose(torch.diag(d), torch.zeros(A.shape[0]), atol=1e-4)


def test_l2_exact_knn_sets_identical():
    torch.manual_seed(2)
    X = torch.randn(500, 10)
    for name in REGISTRY:
        spec = get_metric(name)
        if not spec.l2_exact or spec.l2_transform is None:
            continue
        Xt = spec.l2_transform(X)
        k = 5
        # exact metric top-k
        d_exact, idx_exact = chunked_cdist(spec.fn, X, X, topk=k + 1)
        # L2 on transform
        from leanmap.distance import EuclideanDistance

        d_l2, idx_l2 = chunked_cdist(EuclideanDistance(), Xt, Xt, topk=k + 1)
        # Drop self (col 0 typically)
        # Compare neighbor *sets* excluding self index
        for i in range(X.shape[0]):
            se = set(idx_exact[i].tolist()) - {i}
            sl = set(idx_l2[i].tolist()) - {i}
            # take k nearest non-self
            se = set(list(se)[:k]) if len(se) > k else se
            # Better: sort by distance excluding self
            pass
        # Robust compare: for each row, top-k non-self from exact vs transform L2
        for i in range(min(100, X.shape[0])):
            # full row
            de = spec.fn(X[i : i + 1], X)[0]
            de[i] = float("inf")
            _, ie = torch.topk(de, k=k, largest=False)
            dl = torch.cdist(Xt[i : i + 1], Xt)[0]
            dl[i] = float("inf")
            _, il = torch.topk(dl, k=k, largest=False)
            assert set(ie.tolist()) == set(il.tolist()), name


def test_cosine_transform_relation():
    torch.manual_seed(3)
    X = torch.randn(50, 16)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(40, 16)
    Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)
    spec = get_metric("cosine")
    d_cos = spec.fn(X, Y)
    Xt = spec.l2_transform(X)
    Yt = spec.l2_transform(Y)
    d2 = torch.cdist(Xt, Yt, p=2) ** 2
    assert torch.allclose(d2, 2.0 * d_cos, atol=1e-5)


def test_correlation_matches_scipy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 20)).astype(np.float32)
    B = rng.standard_normal((25, 20)).astype(np.float32)
    spec = get_metric("correlation")
    d = spec.fn(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    # scipy pairwise
    ref = np.empty_like(d)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            ref[i, j] = scipy_corr(A[i], B[j])
    assert np.allclose(d, ref, atol=1e-5)


def test_wasserstein1d_matches_scipy():
    from scipy.stats import wasserstein_distance

    rng = np.random.default_rng(1)
    A = rng.random((12, 16), dtype=np.float32)
    B = rng.random((9, 16), dtype=np.float32)
    pos = np.arange(A.shape[1], dtype=np.float64)
    spec = get_metric("wasserstein1d")
    d = spec.fn(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    ref = np.empty_like(d)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            ref[i, j] = wasserstein_distance(pos, pos, A[i], B[j])
    assert np.allclose(d, ref, atol=1e-5)


def test_composite_capability_and_scales():
    torch.manual_seed(4)
    # Block0 huge magnitude, block1 small
    X = torch.randn(200, 20)
    X[:, :10] *= 1000.0
    comp = CompositeMetric(
        [
            (slice(0, 10), "l2", 1.0),
            (slice(10, 20), "l2", 1.0),
        ]
    )
    assert comp.l2_exact is True
    assert comp.is_true_metric is True
    comp.fit_scales(X, n_sample=200, seed=0)
    assert comp.scales is not None
    assert comp.scales[0] / comp.scales[1] > 10  # scales absorb magnitude
    # Equal weight contributions on same random pairs should be comparable
    A, B = X[:20], X[20:40]
    d0 = get_metric("l2").fn(A[:, :10], B[:, :10]) / comp.scales[0]
    d1 = get_metric("l2").fn(A[:, 10:], B[:, 10:]) / comp.scales[1]
    ratio = (d0.mean() / d1.mean()).item()
    assert 0.2 < ratio < 5.0

    comp_bad = CompositeMetric(
        [(slice(0, 10), "l1", 1.0), (slice(10, 20), "l2", 1.0)]
    )
    assert comp_bad.l2_transform is None
    assert comp_bad.l2_exact is False


def test_natural_scale_invariance_of_wrapped_distances():
    torch.manual_seed(5)
    X = torch.randn(500, 8)
    m1 = wrap_metric("l2", X=X, n_neighbors=10, seed=0)
    # 100 * l2 via custom
    from leanmap.distance import CallableDistance, EuclideanDistance

    base = EuclideanDistance()

    def scaled(A, B):
        return 100.0 * base(A, B)

    m2 = wrap_metric(
        get_metric("custom", fn=CallableDistance(scaled), differentiable=True),
        X=X,
        n_neighbors=10,
        seed=0,
    )
    A, B = X[:30], X[30:60]
    d1 = m1(A, B)
    d2 = m2(A, B)
    # After natural_scale both should be on comparable scale
    assert torch.allclose(d1, d2, atol=0.15, rtol=0.15)

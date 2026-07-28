"""Factored conditioning: migration gate + §12 tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from leanmap.conditioning import (
    ConditioningFactor,
    FactorHyper,
    FactorStack,
    Role,
    build_factor_stack,
    identity_view,
    metric_from_factors,
    scale_quotient_factorization,
    validate_factors,
)
from leanmap.distance import EuclideanDistance
from leanmap.landmarks import AnchorAffinity, fps_init, quantile_init
from leanmap.model import FiLMEncoder, PLANE, fit_pca_weight
from leanmap.train import fit, load_plane
from leanmap.config import PLANEConfig


def _legacy_pair(D=8, L=4, d_out=2, width=32, depth=2, hyper_width=16):
    """Build legacy Affinity+FiLMEncoder and equivalent single-PRIMARY FactorStack."""
    torch.manual_seed(0)
    X = torch.randn(80, D)
    M = fps_init(X, EuclideanDistance(), L=L, seed=0)
    aff = AnchorAffinity(M, EuclideanDistance(), probe_differentiable=False)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    enc = FiLMEncoder(
        D,
        d_out,
        width=width,
        depth=depth,
        L=L,
        hyper_width=hyper_width,
        pca_skip=False,
    )
    enc.set_normalization(mean, std)
    legacy = PLANE(aff, enc)

    factor = ConditioningFactor(
        name="primary",
        view=identity_view,
        metric=EuclideanDistance(),
        n_anchors=L,
        role=Role.PRIMARY,
        learn_anchors=False,
        learn_temperature=False,
    )
    aff2 = AnchorAffinity(
        M.clone(),
        EuclideanDistance(),
        tau_init=aff.tau().detach().clone(),
        learn_anchors=False,
        learn_tau=False,
        probe_differentiable=False,
    )
    aff2.log_tau.data.copy_(aff.log_tau.data)
    stack = FactorStack(
        [factor],
        [aff2],
        width=width,
        depth=depth,
        hyper_width=hyper_width,
        d_out=d_out,
    )
    # Copy PRIMARY hyper weights from legacy encoder hyper
    hyp = stack.hypers[0]
    assert isinstance(hyp, FactorHyper)
    with torch.no_grad():
        for p_new, p_old in zip(hyp.hyper.parameters(), enc.hyper.parameters()):
            p_new.copy_(p_old)
        for p_new, p_old in zip(stack.affinities[0].parameters(), aff.parameters()):
            p_new.copy_(p_old)
    enc2 = FiLMEncoder(
        D,
        d_out,
        width=width,
        depth=depth,
        L=L,
        affinity_dim=L,
        hyper_width=hyper_width,
        pca_skip=False,
    )
    enc2.set_normalization(mean, std)
    with torch.no_grad():
        enc2.backbone.load_state_dict(enc.backbone.state_dict())
        enc2.norms.load_state_dict(enc.norms.state_dict())
        enc2.head.load_state_dict(enc.head.state_dict())
        enc2.hyper.load_state_dict(enc.hyper.state_dict())
    factored = PLANE(stack, enc2)
    return legacy, factored, X


def test_single_primary_bit_exact_migration():
    """Single PRIMARY FactorStack matches legacy Affinity+FiLMEncoder."""
    legacy, factored, X = _legacy_pair()
    xb = X[:16]
    with torch.no_grad():
        z0, a0, d0 = legacy(xb)
        z1, a1, d1 = factored(xb)
        g0, b0 = legacy.encoder.film_params(a0)
        a_map, _, a_list = factored.factors.affinities_forward(xb)
        g1, b1, _, _, _ = factored.factors.film_params_from_affinities(a_list)
    assert torch.equal(a0, a1)
    assert torch.equal(d0, d1)
    assert torch.equal(g0, g1)
    assert torch.equal(b0, b1)
    assert torch.equal(z0, z1)


def test_zero_init_gamma_beta_for_n_factors():
    for n in (1, 2, 4):
        torch.manual_seed(1)
        X = torch.randn(40, 6)
        factors = []
        for i in range(n):
            role = Role.PRIMARY if i == 0 else Role.MODULATOR
            factors.append(
                ConditioningFactor(
                    name=f"f{i}",
                    view=identity_view if i == 0 else (lambda x, j=i: x[:, :2]),
                    metric=EuclideanDistance(),
                    n_anchors=4 if i == 0 else 3,
                    role=role,
                    learn_anchors=False,
                    learn_temperature=False,
                )
            )
        # Fix non-primary views to 2-d slices with enough anchors
        stack = build_factor_stack(X, factors, width=16, depth=2, hyper_width=8, d_out=2)
        xb = X[:8]
        a_map, _, a_list = stack.affinities_forward(xb)
        gamma, beta, _, _, _ = stack.film_params_from_affinities(a_list)
        assert torch.allclose(gamma, torch.ones_like(gamma))
        assert torch.allclose(beta, torch.zeros_like(beta))


def test_order_independence():
    torch.manual_seed(2)
    X = torch.randn(30, 5)
    f_primary = ConditioningFactor(
        "p", identity_view, EuclideanDistance(), 4, Role.PRIMARY, learn_anchors=False, learn_temperature=False
    )
    f_mod = ConditioningFactor(
        "m",
        lambda x: x[:, :2],
        EuclideanDistance(),
        3,
        Role.MODULATOR,
        learn_anchors=False,
        learn_temperature=False,
    )
    s1 = build_factor_stack(X, [f_primary, f_mod], width=16, depth=2, hyper_width=8)
    s2 = build_factor_stack(X, [f_mod, f_primary], width=16, depth=2, hyper_width=8)
    with torch.no_grad():
        for n1, a1, h1 in zip(s1.names, s1.affinities, s1.hypers):
            i2 = s2.names.index(n1)
            s2.affinities[i2].load_state_dict(a1.state_dict())
            s2.hypers[i2].load_state_dict(h1.state_dict())
    xb = X[:5]
    a_map1, _, _ = s1.affinities_forward(xb)
    a_map2, _, _ = s2.affinities_forward(xb)
    # Compose in name order for both stacks
    def compose(stack, a_map):
        B = xb.shape[0]
        gamma = torch.ones(B, stack.depth, stack.width)
        beta = torch.zeros(B, stack.depth, stack.width)
        for f, hyp in zip(stack.factor_defs, stack.hypers):
            if f.role == Role.AXIS:
                continue
            assert isinstance(hyp, FactorHyper)
            g, b = hyp(a_map[f.name])
            gamma = gamma * g
            if b is not None:
                beta = beta + b
        return gamma, beta

    g1, b1 = compose(s1, a_map1)
    g2, b2 = compose(s2, a_map2)
    assert torch.allclose(g1, g2, atol=1e-6)
    assert torch.allclose(b1, b2, atol=1e-6)


def test_modulator_gain_gamma_shapes():
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    factors = [
        ConditioningFactor("p", identity_view, EuclideanDistance(), 4, Role.PRIMARY),
        ConditioningFactor("m", lambda x: x[:, :1], EuclideanDistance(), 3, Role.MODULATOR),
        ConditioningFactor("g", lambda x: x[:, 1:2], EuclideanDistance(), 3, Role.GAIN),
    ]
    stack = build_factor_stack(X, factors, width=16, depth=2, hyper_width=8)
    xb = X[:4]
    _, _, alist = stack.affinities_forward(xb)
    for f, a, hyp in zip(stack.factor_defs, alist, stack.hypers):
        if f.role == Role.AXIS:
            continue
        assert isinstance(hyp, FactorHyper)
        g, b = hyp(a)
        if f.role == Role.PRIMARY:
            assert g.shape == (4, 2, 16)
            assert b.shape == (4, 2, 16)
        elif f.role == Role.MODULATOR:
            assert g.shape == (4, 2, 1)
            assert b.shape == (4, 2, 16)
        elif f.role == Role.GAIN:
            assert g.shape == (4, 2, 1)
            assert b is None


def test_gain_role_reaches_the_output():
    """A GAIN factor emits a scalar gamma only; it must still change z.

    Under modulate-then-normalize this role was a no-op, because LayerNorm
    cancels any positive scalar rescale. This is the end-to-end guard.
    """
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    factors = [
        ConditioningFactor("p", identity_view, EuclideanDistance(), 4, Role.PRIMARY),
        ConditioningFactor("g", lambda x: x[:, 1:2], EuclideanDistance(), 3, Role.GAIN),
    ]
    stack = build_factor_stack(X, factors, width=16, depth=2, hyper_width=8)
    enc = FiLMEncoder(4, 2, width=16, depth=2, L=4, affinity_dim=7, pca_skip=False)
    model = PLANE(stack, enc)

    z_before, _, _ = model(X[:8])
    gain_hyper = stack.hypers[1]
    assert isinstance(gain_hyper, FactorHyper)
    with torch.no_grad():
        # Non-zero bias on the last hyper layer => gamma departs from 1.
        gain_hyper.hyper[-1].bias.fill_(1.5)
    z_after, _, _ = model(X[:8])
    assert not torch.allclose(z_before, z_after, atol=1e-6)


def test_axis_monotonicity():
    from leanmap.conditioning import Monotone1D

    m = Monotone1D(width=16)
    v = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = m(v).squeeze(-1)
    assert torch.all(y[1:] >= y[:-1] - 1e-5)


def test_metric_weight_none_separation():
    torch.manual_seed(0)
    X = torch.randn(50, 4)
    pos = torch.randn(50, 2)

    def view_pos(x):
        # x is ambient; attach position via closure
        return pos[: x.shape[0]]

    factors = [
        ConditioningFactor(
            "spec", identity_view, EuclideanDistance(), 8, Role.PRIMARY, metric_weight=1.0
        ),
        ConditioningFactor(
            "position",
            view_pos,
            EuclideanDistance(),
            4,
            Role.MODULATOR,
            metric_weight=None,
        ),
    ]
    metric = metric_from_factors(factors, X=X, n_neighbors=5, seed=0)
    assert metric is not None
    d0 = metric(X[:10], X[10:20])
    # Non-isometric change (translation leaves Euclidean affinities unchanged).
    pos2 = pos.clone() * 3.0 + torch.randn_like(pos)
    factors2 = [
        factors[0],
        ConditioningFactor(
            "position",
            lambda x, p=pos2: p[: x.shape[0]],
            EuclideanDistance(),
            4,
            Role.MODULATOR,
            metric_weight=None,
        ),
    ]
    metric2 = metric_from_factors(factors2, X=X, n_neighbors=5, seed=0)
    d1 = metric2(X[:10], X[10:20])
    assert torch.allclose(d0, d1, atol=1e-5)

    # Encoder output should change when position view changes
    # Encoder output should change when position view changes
    stack = build_factor_stack(X, factors, width=16, depth=2, hyper_width=8)
    with torch.no_grad():
        hyp_m = stack.hypers[1]
        assert isinstance(hyp_m, FactorHyper)
        hyp_m.hyper[-1].weight.normal_(0, 0.5)
        hyp_m.hyper[-1].bias.normal_(0, 0.1)
    _, _, alist0 = stack.affinities_forward(X[:8])
    g0, _, _, _, _ = stack.film_params_from_affinities(alist0)
    stack2 = build_factor_stack(X, factors2, width=16, depth=2, hyper_width=8)
    with torch.no_grad():
        stack2.hypers[1].load_state_dict(stack.hypers[1].state_dict())
    _, _, alist1 = stack2.affinities_forward(X[:8])
    # Align alist1 to stack factor order (same names)
    a_by = {n: a for n, a in zip(stack2.names, alist1)}
    alist1_ord = [a_by[n] for n in stack.names]
    g1, _, _, _, _ = stack.film_params_from_affinities(alist1_ord)
    assert not torch.allclose(g0, g1, atol=1e-5)


def test_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_factors(
            [
                ConditioningFactor("a", identity_view, EuclideanDistance(), 2, Role.MODULATOR),
            ]
        )
        assert any("PRIMARY" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_factors(
            [
                ConditioningFactor(
                    "p", identity_view, EuclideanDistance(), 2, Role.PRIMARY, metric_weight=0.0
                ),
            ]
        )
        assert any("metric_weight=0.0" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_factors(
            [
                ConditioningFactor("p", identity_view, EuclideanDistance(), 2, Role.PRIMARY),
                ConditioningFactor(
                    "ax", lambda x: x[:, :1], EuclideanDistance(), 2, Role.AXIS, axis=None
                ),
            ]
        )
        assert any("axis" in str(x.message).lower() for x in w)


def test_quantile_init_1d():
    v = torch.randn(100, 1)
    M = quantile_init(v, 16)
    assert M.shape == (16, 1)
    assert torch.all(M[1:] >= M[:-1] - 1e-6)


def test_scale_quotient_factory():
    factors = scale_quotient_factorization(norm="l2", n_shape_anchors=8, n_scale_anchors=4)
    assert factors[0].role == Role.PRIMARY
    assert factors[1].role == Role.AXIS
    X = torch.randn(20, 6).abs() + 0.1
    stack = build_factor_stack(X, factors, width=16, depth=2, hyper_width=8, d_out=2)
    z_enc = FiLMEncoder(6, 2, width=16, depth=2, L=8, affinity_dim=12, pca_skip=False)
    model = PLANE(stack, z_enc)
    z, a, dm = model(X[:5])
    assert z.shape == (5, 2)


def test_artefact_roundtrip_single_primary(tmp_path):
    torch.manual_seed(0)
    X = np.random.randn(400, 6).astype(np.float32)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.epochs = 1
    cfg.dedup = False
    cfg.n_landmarks = 16
    cfg.width = 32
    cfg.depth = 2
    cfg.device = "cpu"
    result = fit(X, dist_fn="l2", config=cfg)
    path = tmp_path / "plane.pt"
    result.save(path)
    payload = torch.load(path, weights_only=False)
    for k, v in payload.items():
        if torch.is_tensor(v):
            assert v.numel() < 400 * 6  # no N-sized arrays for N=400 D=6 full
    model2 = load_plane(path)
    z1, _ = result.model.embed(torch.as_tensor(X[:50]), return_score=False)
    z2, _ = model2.embed(torch.as_tensor(X[:50]), return_score=False)
    assert torch.allclose(z1, z2, atol=1e-4)


def test_synthetic_two_factor_retention():
    """Independent shape/scale generators → high retention_f, low affinity corr."""
    rng = np.random.default_rng(0)
    n = 600
    # shape on sphere-ish, scale independent
    shape = rng.normal(size=(n, 4)).astype(np.float32)
    shape /= np.linalg.norm(shape, axis=1, keepdims=True).clip(1e-3)
    scale = rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float32)
    X = shape * scale
    factors = scale_quotient_factorization(
        norm="l2",
        n_shape_anchors=32,
        n_scale_anchors=16,
        scale_role=Role.GAIN,
        shape_metric_weight=1.0,
        scale_metric_weight=0.3,
    )
    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = 3
    cfg.dedup = False
    cfg.width = 64
    cfg.depth = 2
    cfg.device = "cpu"
    cfg.batch_edges = 1024
    rets = {f.name: [] for f in factors}

    def cb(ep, model, metrics):
        for f in factors:
            key = "retention" if f.name == "shape" or f.role == Role.PRIMARY else f"retention_{f.name}"
            # primary key is retention; shape is primary named shape
            if f.role == Role.PRIMARY:
                rets[f.name].append(metrics.get("retention", 0.0))
            else:
                rets[f.name].append(metrics.get(f"retention_{f.name}", 0.0))

    result = fit(X, dist_fn="l2", config=cfg, factors=factors, callbacks=[cb])
    # Short smoke: retention is logged and affinity correlation is defined
    assert all(len(v) == 3 for v in rets.values())
    xb = torch.as_tensor(X[:256])
    _, a_map, _, _, _, _, _ = result.model.forward_detailed(xb)
    names = list(a_map.keys())
    assert len(names) == 2
    # Correlate mean affinity vectors after padding to equal length
    v0 = a_map[names[0]].mean(0)
    v1 = a_map[names[1]].mean(0)
    L = max(v0.numel(), v1.numel())
    p0 = torch.zeros(L)
    p1 = torch.zeros(L)
    p0[: v0.numel()] = v0
    p1[: v1.numel()] = v1
    vi, vj = p0 - p0.mean(), p1 - p1.mean()
    corr = float(((vi @ vj) / (vi.norm() * vj.norm() + 1e-8)).detach())
    assert np.isfinite(corr)

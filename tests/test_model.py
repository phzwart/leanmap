"""Tests for FiLM encoder and PLANE.embed."""

from __future__ import annotations

import torch

from leanmap.distance import EuclideanDistance
from leanmap.landmarks import LandmarkAffinity, fps_init
from leanmap.model import FiLMEncoder, PLANE, fit_pca_weight


def _tiny_model(D=8, L=4, d_out=2, pca_skip=True, pca_center=True):
    torch.manual_seed(0)
    X = torch.randn(100, D)
    M = fps_init(X, EuclideanDistance(), L=L, seed=0)
    aff = LandmarkAffinity(M, EuclideanDistance(), probe_differentiable=True)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    X_n = (X - mean) / std
    w = fit_pca_weight(X_n, d_out, center=pca_center) if pca_skip else None
    enc = FiLMEncoder(
        D,
        d_out,
        width=32,
        depth=2,
        L=L,
        hyper_width=16,
        spectral_norm_flag=False,
        pca_skip=pca_skip,
        pca_weight=w,
    )
    enc.set_normalization(mean, std)
    return PLANE(aff, enc), X


def test_gamma_beta_init():
    model, X = _tiny_model()
    a = torch.softmax(torch.randn(16, 4), dim=1)
    gamma, beta = model.encoder.film_params(a)
    assert torch.allclose(gamma, torch.ones_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))


def test_pca_skip_init_matches_pca():
    """With near-zero residual + identity FiLM, embed ≈ PCA(x_n)."""
    for center in (True, False):
        model, X = _tiny_model(pca_skip=True, pca_center=center)
        enc = model.encoder
        with torch.no_grad():
            enc.head.weight.zero_()
            enc.head.bias.zero_()
        x_n = (X - enc.x_mean) / enc.x_std
        expected = x_n @ enc.pca.weight.T
        a = torch.zeros(X.shape[0], enc.L)
        a[:, 0] = 1.0
        z = enc(X, a)
        assert torch.allclose(z, expected, atol=1e-4), center


def test_pca_skip_off_no_pca_module():
    model, _ = _tiny_model(pca_skip=False)
    assert model.encoder.pca is None


def test_embed_after_deleting_training_data():
    model, X = _tiny_model()
    del X
    X_new = torch.randn(50, 8)
    z, score = model.embed(X_new, return_score=True, tau_embed=1.0)
    assert z.shape == (50, 2)
    assert score is not None and score.shape == (50,)


def test_embed_far_ood_finite():
    model, X = _tiny_model()
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    X_far = mean + 100 * std * torch.randn(20, 8)
    z, _ = model.embed(X_far, return_score=False)
    assert torch.isfinite(z).all()


def test_gradient_wrt_input():
    model, X = _tiny_model()
    x = X[:8].clone().requires_grad_(True)
    z, _, _ = model(x)
    z.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0

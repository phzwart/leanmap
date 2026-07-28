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


def test_scalar_gamma_changes_output():
    """A per-layer scalar gamma must be visible in the forward pass.

    LayerNorm is exactly scale-invariant, so under the old modulate-then-norm
    ordering a scalar gamma with beta=0 was a no-op and the GAIN role could
    never do anything. Modulating after the norm is what makes it real.
    """
    model, X = _tiny_model(pca_skip=False)
    enc = model.encoder
    B, x = 16, X[:16]
    beta = torch.zeros(B, enc.depth, enc.width)
    z1 = enc(x, gamma=torch.ones(B, enc.depth, 1).expand(B, enc.depth, enc.width), beta=beta)
    z2 = enc(x, gamma=torch.full((B, enc.depth, 1), 3.0).expand(B, enc.depth, enc.width), beta=beta)
    assert not torch.allclose(z1, z2, atol=1e-6)


def test_film_applied_after_layernorm():
    """Pin the ordering: h = gelu(gamma * LN(Wh) + beta)."""
    model, X = _tiny_model(pca_skip=False)
    enc = model.encoder
    B, x = 4, X[:4]
    gamma = 1.0 + 0.5 * torch.randn(B, enc.depth, enc.width)
    beta = 0.3 * torch.randn(B, enc.depth, enc.width)
    z = enc(x, gamma=gamma, beta=beta)

    h = (x - enc.x_mean) / enc.x_std
    for k, (lin, norm) in enumerate(zip(enc.backbone, enc.norms)):
        h = torch.nn.functional.gelu(gamma[:, k, :] * norm(lin(h)) + beta[:, k, :])
    assert torch.allclose(z, enc.head(h), atol=1e-6)


def test_hidden_taps_are_post_film():
    """negative_space hooks the taps; they must carry gamma/beta."""
    model, X = _tiny_model(pca_skip=False)
    enc = model.encoder
    captured = {}
    handle = enc.taps[0].register_forward_hook(lambda _m, _i, out: captured.__setitem__(0, out))
    B, x = 4, X[:4]
    gamma = torch.full((B, enc.depth, enc.width), 2.0)
    beta = torch.full((B, enc.depth, enc.width), 0.5)
    enc(x, gamma=gamma, beta=beta)
    handle.remove()
    expected = 2.0 * enc.norms[0](enc.backbone[0]((x - enc.x_mean) / enc.x_std)) + 0.5
    assert torch.allclose(captured[0], expected, atol=1e-6)


def test_gradient_wrt_input():
    model, X = _tiny_model()
    x = X[:8].clone().requires_grad_(True)
    z, _, _ = model(x)
    z.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# AdamW parameter groups for the PCA skip
# ---------------------------------------------------------------------------


def _pg_model_and_cfg(pca_skip=True, mult=1.0):
    from leanmap.config import PLANEConfig
    from leanmap.distance import EuclideanDistance
    from leanmap.landmarks import LandmarkAffinity, fps_init
    from leanmap.model import PLANE

    torch.manual_seed(0)
    X = torch.randn(200, 6)
    M = fps_init(X, EuclideanDistance(), L=8, seed=0)
    aff = LandmarkAffinity(M, EuclideanDistance(), probe_differentiable=False)
    enc = FiLMEncoder(6, 2, width=16, depth=2, L=8, hyper_width=8, pca_skip=pca_skip)
    cfg = PLANEConfig(pca_skip=pca_skip, pca_lr_mult=mult, lr=1e-3)
    return PLANE(aff, enc), cfg


def test_default_multiplier_keeps_a_single_flat_group():
    from leanmap.train import _param_groups

    model, cfg = _pg_model_and_cfg(mult=1.0)
    groups = _param_groups(model, cfg)
    assert len(groups) == 1
    assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_multiplier_splits_head_and_hypers_from_the_skip():
    from leanmap.train import _param_groups

    model, cfg = _pg_model_and_cfg(mult=15.0)
    groups = _param_groups(model, cfg)
    assert len(groups) == 2
    slow, fast = groups
    assert fast["lr"] == 15.0 * slow["lr"]
    slow_ids = {id(p) for p in slow["params"]}
    # The PCA skip and backbone stay slow; the residual head runs fast.
    assert id(model.encoder.pca.weight) in slow_ids
    assert id(model.encoder.backbone[0].weight) in slow_ids
    assert id(model.encoder.head.weight) not in slow_ids
    # Every parameter lands in exactly one group.
    n = len(slow["params"]) + len(fast["params"])
    assert n == len([p for p in model.parameters() if p.requires_grad])


def test_multiplier_is_ignored_without_the_skip():
    """With pca_skip=False there is no skip to hold back."""
    from leanmap.train import _param_groups

    model, cfg = _pg_model_and_cfg(pca_skip=False, mult=15.0)
    assert len(_param_groups(model, cfg)) == 1


def test_group_lr_ratio_survives_the_schedule():
    """Warmup + cosine must scale groups proportionally, not flatten them."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    from leanmap.train import _param_groups

    model, cfg = _pg_model_and_cfg(mult=10.0)
    opt = AdamW(_param_groups(model, cfg), lr=cfg.lr)
    sched = SequentialLR(
        opt,
        [LinearLR(opt, start_factor=0.01, total_iters=10), CosineAnnealingLR(opt, T_max=90)],
        milestones=[10],
    )
    for _ in range(60):
        opt.step()
        sched.step()
        lrs = [g["lr"] for g in opt.param_groups]
        assert abs(lrs[1] / max(lrs[0], 1e-18) - 10.0) < 1e-6

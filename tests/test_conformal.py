"""Tests for conformal exchangeability."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.conformal import ConformalCalibrator, bh_reject, model_weight_hash
from leanmap.distance import EuclideanDistance
from leanmap.landmarks import LandmarkAffinity, fps_init
from leanmap.model import FiLMEncoder, PLANE


def _model(n=400, D=6, L=8):
    torch.manual_seed(0)
    X = torch.randn(n, D)
    M = fps_init(X, EuclideanDistance(), L=L, seed=0)
    aff = LandmarkAffinity(M, EuclideanDistance(), probe_differentiable=False)
    enc = FiLMEncoder(D, 2, width=32, depth=2, L=L, hyper_width=16, spectral_norm_flag=False)
    enc.set_normalization(X.mean(0), X.std(0).clamp_min(1e-6))
    return PLANE(aff, enc), X


def test_pvalue_coverage():
    model, X = _model()
    cal, hold = X[:200], X[200:]
    calibrator = ConformalCalibrator()
    calibrator.fit(model, cal)
    # Monte Carlo over random exchangeable scores by resampling holdout
    alphas = [0.01, 0.05, 0.1]
    counts = {a: 0 for a in alphas}
    trials = 200  # keep under 30s; slightly looser than 2000
    g = torch.Generator().manual_seed(1)
    for _ in range(trials):
        idx = torch.randint(0, hold.shape[0], (1,), generator=g)
        from leanmap.conformal import geometry_consistency_score

        sc, _ = geometry_consistency_score(
            model, hold[idx], tau_embed=calibrator.tau_embed
        )
        p = calibrator.p_value(sc, model=model)
        for a in alphas:
            if float(p) <= a:
                counts[a] += 1
    for a in alphas:
        rate = counts[a] / trials
        # Monte Carlo error band
        assert abs(rate - a) < 0.08 + 2 * np.sqrt(a * (1 - a) / trials)


def test_ks_uniform_exchangeable():
    from scipy.stats import kstest

    model, X = _model(n=600)
    cal, hold = X[:200], X[200:400]
    calibrator = ConformalCalibrator()
    calibrator.fit(model, cal)
    from leanmap.conformal import geometry_consistency_score

    sc, _ = geometry_consistency_score(model, hold, tau_embed=calibrator.tau_embed)
    p = calibrator.p_value(sc, model=model).numpy()
    stat = kstest(p, "uniform")
    assert stat.pvalue > 0.01


def test_batch_test_detects_shift():
    model, X = _model(n=500)
    cal = X[:200]
    calibrator = ConformalCalibrator()
    calibrator.fit(model, cal)
    # Directly feed stochastically larger scores (drift in score space)
    s_batch = calibrator.s_calib[-1] + 0.5 + 0.1 * torch.rand(80)
    out = calibrator.batch_test(s_batch, n_perm=2000, seed=0)
    assert out["p_global"] < 0.01


def test_bh_reject_null():
    torch.manual_seed(0)
    # Under the global null, BH should not reject more than ~alpha of tests.
    trials = 100
    alpha = 0.1
    rates = []
    for _ in range(trials):
        p = torch.rand(50)
        rej = bh_reject(p, alpha=alpha)
        rates.append(float(rej.float().mean()))
    assert np.mean(rates) <= alpha + 0.05


def test_pvalue_raises_on_weight_change():
    model, X = _model()
    calibrator = ConformalCalibrator()
    calibrator.fit(model, X[:100])
    # mutate weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
            break
    sc = torch.rand(10)
    try:
        calibrator.p_value(sc, model=model)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "hash" in str(e).lower() or "match" in str(e).lower()


def test_cover_ood_score_separates():
    """Landmark cover (primary OOD score) is larger off-support than on-support."""
    model, X = _model(n=500, D=6, L=16)
    cal = X[:200]
    calibrator = ConformalCalibrator()
    calibrator.fit(model, cal)
    from leanmap.conformal import geometry_consistency_score

    on = X[200:350]
    # Far off the Gaussian cloud used for training.
    ood = X.mean(0) + 8.0 * X.std(0).clamp_min(1e-3) * torch.randn(150, X.shape[1])
    cover_on, _ = geometry_consistency_score(model, on, tau_embed=calibrator.tau_embed)
    cover_ood, _ = geometry_consistency_score(model, ood, tau_embed=calibrator.tau_embed)
    assert float(cover_ood.median().detach()) > float(cover_on.median().detach()) * 2.0
    p_on = calibrator.p_value(cover_on, model=model)
    p_ood = calibrator.p_value(cover_ood, model=model)
    assert float(p_ood.median().detach()) < float(p_on.median().detach())


def test_embed_score_is_cover():
    model, X = _model(n=300, D=6, L=8)
    calibrator = ConformalCalibrator()
    calibrator.fit(model, X[:100])
    from leanmap.conformal import geometry_consistency_score

    xb = X[100:150]
    z, score = model.embed(xb, return_score=True)
    cover, _ = geometry_consistency_score(model, xb, tau_embed=calibrator.tau_embed)
    assert z.shape == (50, 2)
    assert torch.allclose(score, cover.cpu(), rtol=1e-5, atol=1e-5)
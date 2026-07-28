"""Tests for conformal exchangeability."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.conformal import (
    ConformalCalibrator,
    LandmarkSupport,
    bh_reject,
    model_weight_hash,
)
from leanmap.distance import EuclideanDistance
from leanmap.landmarks import LandmarkAffinity, fps_init
from leanmap.model import FiLMEncoder, PLANE


def _model(n=400, D=6, L=8):
    torch.manual_seed(0)
    X = torch.randn(n, D)
    M = fps_init(X, EuclideanDistance(), L=L, seed=0)
    aff = LandmarkAffinity(M, EuclideanDistance(), probe_differentiable=False)
    enc = FiLMEncoder(D, 2, width=32, depth=2, L=L, hyper_width=16)
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


# ---------------------------------------------------------------------------
# Per-landmark support: local scale, tangent charts, heterogeneous repair
# ---------------------------------------------------------------------------


def _thin_sheet(n=1200, D=8, thickness=0.02, seed=0):
    """A 2-D sheet embedded in D dims with a tiny normal thickness."""
    g = torch.Generator().manual_seed(seed)
    tangent = torch.rand(n, 2, generator=g) * 4.0 - 2.0
    X = torch.zeros(n, D)
    X[:, :2] = tangent
    X[:, 2:] = thickness * torch.randn(n, D - 2, generator=g)
    return X


def test_local_radii_track_local_density():
    """r_l is a *local* scale, so a sparse region gets a larger radius."""
    g = torch.Generator().manual_seed(0)
    dense = 0.05 * torch.randn(400, 3, generator=g)
    sparse = torch.tensor([10.0, 0.0, 0.0]) + 1.0 * torch.randn(400, 3, generator=g)
    X = torch.cat([dense, sparse])
    M = torch.stack([torch.zeros(3), torch.tensor([10.0, 0.0, 0.0])])
    sup = LandmarkSupport.fit(M, X, mode="ball")
    assert float(sup.r[1]) > 5.0 * float(sup.r[0])


def test_local_scale_equalises_sparse_and_dense_regions():
    """The point of local scale: a typical point scores ~1 in either region."""
    g = torch.Generator().manual_seed(1)
    dense = 0.05 * torch.randn(400, 3, generator=g)
    sparse = torch.tensor([10.0, 0.0, 0.0]) + 1.0 * torch.randn(400, 3, generator=g)
    X = torch.cat([dense, sparse])
    M = torch.stack([torch.zeros(3), torch.tensor([10.0, 0.0, 0.0])])
    sup = LandmarkSupport.fit(M, X, mode="ball")
    s_dense = sup.score(dense).median()
    s_sparse = sup.score(sparse).median()
    assert abs(float(s_dense) - float(s_sparse)) < 0.25
    # A global scale would rate the whole sparse region as far more anomalous.
    d_glob = torch.cdist(X, M).min(dim=1).values
    g_dense = d_glob[:400].median()
    g_sparse = d_glob[400:].median()
    assert float(g_sparse) > 5.0 * float(g_dense)


def _off_vs_along(sup, X, step=0.5):
    """Ratio of the score for a normal move to a tangent move of equal length."""
    base = X[:200]
    along = base.clone()
    along[:, 0] += step
    off = base.clone()
    off[:, 3] += step
    return float(sup.score(off).median()) / float(sup.score(along).median())


def test_charts_beat_balls_on_a_thin_sheet():
    """Off-sheet displacement must cost more than the same move along it."""
    X = _thin_sheet(thickness=0.02)
    M = fps_init(X, EuclideanDistance(), L=12, seed=0)
    ball = LandmarkSupport.fit(M, X, mode="ball")
    chart = LandmarkSupport.fit(M, X, mode="chart", m_tangent=2)
    r_ball = _off_vs_along(ball, X)
    r_chart = _off_vs_along(chart, X)
    # An isotropic ball barely tells the two apart: both are a distance `step`.
    assert r_ball < 2.0
    assert r_chart > 3.0 * r_ball


def test_chart_discrimination_scales_as_the_sheet_thins():
    """The volume argument: balls are *increasingly* wrong as thickness drops.

    A union of balls of radius tau has volume ~ L tau^D against a support of
    ~ L tau^m t^(D-m). Charts track t, so their off-sheet sensitivity grows as
    t shrinks; isotropic balls are indifferent to it.
    """
    ratios = {}
    for thickness in (0.05, 0.005):
        X = _thin_sheet(thickness=thickness)
        M = fps_init(X, EuclideanDistance(), L=12, seed=0)
        ratios[thickness] = (
            _off_vs_along(LandmarkSupport.fit(M, X, mode="ball"), X),
            _off_vs_along(LandmarkSupport.fit(M, X, mode="chart", m_tangent=2), X),
        )
    ball_thick, chart_thick = ratios[0.05]
    ball_thin, chart_thin = ratios[0.005]
    # Balls are essentially unchanged; charts sharpen by roughly 10x.
    assert ball_thin < 1.5 * ball_thick
    assert chart_thin > 5.0 * chart_thick


def test_chart_falls_back_to_ball_when_bucket_is_tiny():
    """A landmark with too few training points degrades to an isotropic ball."""
    g = torch.Generator().manual_seed(2)
    X = torch.randn(60, 5, generator=g)
    M = fps_init(X, EuclideanDistance(), L=20, seed=0)
    sup = LandmarkSupport.fit(M, X, mode="chart", m_tangent=2, min_bucket=1000)
    assert torch.allclose(sup.sigma_par, sup.sigma_perp)
    ball = LandmarkSupport.fit(M, X, mode="ball")
    assert torch.allclose(sup.score(X), ball.score(X), atol=1e-5)


def test_repair_prefers_the_cheaper_landmark_not_the_nearest():
    """With heterogeneous radii the nearest centre is the wrong target."""
    # Landmark 0 is closer but tight; landmark 1 is farther but generous.
    M = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    g = torch.Generator().manual_seed(3)
    tight = 0.1 * torch.randn(300, 2, generator=g)
    loose = torch.tensor([10.0, 0.0]) + 3.0 * torch.randn(300, 2, generator=g)
    sup = LandmarkSupport.fit(M, torch.cat([tight, loose]), mode="ball")
    x = torch.tensor([[4.6, 0.0]])
    d = torch.cdist(x, M)[0]
    assert float(d[0]) < float(d[1])  # landmark 0 is nearest
    tau = 1.0
    out = sup.repair(x, tau=tau)
    # Cheaper to fall into the generous ball around landmark 1.
    assert float(out[0, 0]) > 5.0
    assert float(sup.score(out).max()) <= tau + 1e-4


def test_repair_lands_inside_the_acceptance_region():
    X = _thin_sheet(n=600, D=6)
    M = fps_init(X, EuclideanDistance(), L=10, seed=0)
    sup = LandmarkSupport.fit(M, X, mode="ball")
    g = torch.Generator().manual_seed(4)
    far = 6.0 * torch.randn(50, 6, generator=g)
    tau = 1.5
    out = sup.repair(far, tau=tau)
    assert float(sup.score(out).max()) <= tau + 1e-4
    # Points already inside are untouched.
    inside = X[:40]
    assert torch.allclose(sup.repair(inside, tau=10.0), inside)


def test_support_scores_stay_conformally_valid():
    """Exchangeable calib/test split ⇒ p-values are super-uniform."""
    X = _thin_sheet(n=1500, D=6, seed=5)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(6))
    X = X[perm]
    train, calib, test = X[:700], X[700:1100], X[1100:]
    model, _ = _model(n=64, D=6, L=10)
    sup = LandmarkSupport.fit(
        fps_init(train, EuclideanDistance(), L=10, seed=0), train, mode="chart"
    )
    cal = ConformalCalibrator(support=sup)
    cal.fit(model, calib)
    p = cal.p_value(cal.cover_score(model, test))
    for alpha in (0.05, 0.1, 0.2):
        assert float((p <= alpha).float().mean()) <= alpha + 0.05


def test_cover_score_routes_through_the_support():
    X = _thin_sheet(n=400, D=6, seed=7)
    model, _ = _model(n=64, D=6, L=8)
    sup = LandmarkSupport.fit(fps_init(X, EuclideanDistance(), L=8, seed=0), X)
    cal = ConformalCalibrator(support=sup)
    cal.fit(model, X[:200])
    assert torch.allclose(cal.cover_score(model, X[200:]), sup.score(X[200:]))

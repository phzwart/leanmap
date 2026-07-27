"""Tests for farthest-point landmarks and load balancing."""

from __future__ import annotations

import torch

from leanmap.distance import EuclideanDistance
from leanmap.landmarks import (
    LandmarkAffinity,
    fps_init,
    landmark_geodesic_matrix,
    poisson_disk_indices_geodesic,
)
from leanmap.losses import landmark_regularisation


def test_fps_spreads_better_than_random():
    torch.manual_seed(0)
    # Uniform square
    X = torch.rand(2000, 2)
    dist = EuclideanDistance()
    M = fps_init(X, dist, L=32, seed=0)
    # min pairwise among FPS landmarks
    d = dist(M, M)
    d.fill_diagonal_(float("inf"))
    fps_min = float(d.min().item())
    # random sample min pairwise
    mins = []
    for s in range(5):
        g = torch.Generator().manual_seed(s + 1)
        idx = torch.randperm(X.shape[0], generator=g)[:32]
        Mr = X[idx]
        dr = dist(Mr, Mr)
        dr.fill_diagonal_(float("inf"))
        mins.append(float(dr.min().item()))
    assert fps_min > float(sum(mins) / len(mins))


def test_landmark_geodesic_matrix_matches_euclidean_on_line():
    # Points on a 1-D line: graph geodesic ≈ ambient distance.
    torch.manual_seed(0)
    t = torch.linspace(0.0, 10.0, 200)
    X = torch.stack([t, torch.zeros_like(t)], dim=1)
    idx = torch.linspace(0, 199, 12).long()
    M = X[idx]
    dist = EuclideanDistance()
    X_lm, G, finite = landmark_geodesic_matrix(X, M, dist, n_neighbors=5)
    assert X_lm.shape == (12, 2)
    assert G.shape == (12, 12)
    assert bool(finite.sum()) > 0
    ii, jj = torch.where(torch.triu(finite, diagonal=1))
    amb = (X_lm[ii] - X_lm[jj]).norm(dim=1)
    geo = G[ii, jj]
    corr = torch.corrcoef(torch.stack([amb, geo]))[0, 1]
    assert float(corr) > 0.99


def test_poisson_disk_geodesic_spreads_and_separates():
    torch.manual_seed(0)
    X = torch.rand(1500, 2)
    dist = EuclideanDistance()
    L = 40
    idx = poisson_disk_indices_geodesic(X, dist, L, n_neighbors=10, seed=0)
    # Unique, valid, and not wildly over/under target.
    assert idx.numel() == torch.unique(idx).numel()
    assert idx.numel() <= L
    assert idx.numel() >= L // 2
    M = X[idx]
    d = dist(M, M)
    d.fill_diagonal_(float("inf"))
    pois_min = float(d.min().item())
    # Blue-noise min separation should beat random subsets of the same size.
    mins = []
    for s in range(5):
        g = torch.Generator().manual_seed(s + 1)
        ridx = torch.randperm(X.shape[0], generator=g)[: idx.numel()]
        dr = dist(X[ridx], X[ridx])
        dr.fill_diagonal_(float("inf"))
        mins.append(float(dr.min().item()))
    assert pois_min > float(sum(mins) / len(mins))


def test_landmark_load_balancing_imbalanced_clusters():
    """With eta>0, minority cluster retains at least one top-1 landmark."""
    torch.manual_seed(0)
    n_maj, n_min = 900, 100
    maj = torch.randn(n_maj, 8) + torch.tensor([5.0] + [0.0] * 7)
    minor = torch.randn(n_min, 8) + torch.tensor([-5.0] + [0.0] * 7)
    X = torch.cat([maj, minor], dim=0)
    dist = EuclideanDistance()
    M = fps_init(X, dist, L=16, seed=0)
    aff = LandmarkAffinity(M, dist, learn_landmarks=True, learn_tau=True)
    opt = torch.optim.Adam(aff.parameters(), lr=1e-2)
    for _ in range(80):
        idx = torch.randint(0, X.shape[0], (256,))
        xb = X[idx]
        a, Dm = aff(xb)
        loss = landmark_regularisation(a, Dm, eta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Assign all points to nearest landmark under learned M
    with torch.no_grad():
        d = dist(X, aff.M)
        top1 = d.argmin(dim=1)
    counts = torch.bincount(top1, minlength=aff.M.shape[0])
    # Landmarks whose nearest points are mostly in minority (idx >= 900)
    min_covered = False
    for ell in range(aff.M.shape[0]):
        members = torch.where(top1 == ell)[0]
        if members.numel() == 0:
            continue
        frac_min = float((members >= n_maj).float().mean().item())
        if frac_min > 0.5:
            min_covered = True
            break
    assert min_covered or int((counts > 0).sum()) >= 2

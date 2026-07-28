"""Tests for loss terms."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.losses import (
    find_ab_params,
    fuzzy_cross_entropy,
    geodesic_stress_loss,
    local_rigidity_loss,
    ordinal_triplet_loss,
    procrustes_anchor_loss,
)


def test_find_ab_params():
    a, b = find_ab_params(1.0, 0.1)
    assert abs(a - 1.577) < 0.1
    assert abs(b - 0.895) < 0.1


def test_fuzzy_ce_finite_at_zero_separation():
    z = torch.randn(32, 2)
    w = torch.ones(32)
    z_neg = torch.randn(32, 5, 2)
    loss = fuzzy_cross_entropy(z, z.clone(), w, z_neg, a=1.577, b=0.895)
    assert torch.isfinite(loss)


def test_ordinal_decreases_when_corrected():
    torch.manual_seed(0)
    # Bad order: near far, mid mid, far near in embedding
    za = torch.zeros(16, 2)
    zn_bad = torch.ones(16, 2) * 3
    zm = torch.ones(16, 2) * 2
    zf_bad = torch.ones(16, 2) * 1
    mask = torch.ones(16, dtype=torch.bool)
    bad, _ = ordinal_triplet_loss(za, zn_bad, zm, zf_bad, mask)
    zn_good = torch.ones(16, 2) * 1
    zf_good = torch.ones(16, 2) * 3
    good, _ = ordinal_triplet_loss(za, zn_good, zm, zf_good, mask)
    assert float(good) < float(bad)


def _rigidity_batch(seed=0, B=8, m=6, D=3):
    """A batch of stars whose ambient neighbourhoods live in a 2-D plane."""
    torch.manual_seed(seed)
    x_c = torch.randn(B, D)
    # Neighbour offsets confined to the first two ambient axes (2-D manifold).
    off2 = torch.randn(B, m, 2)
    off = torch.zeros(B, m, D)
    off[:, :, :2] = off2
    x_nbr = x_c.unsqueeze(1) + off
    return x_c, x_nbr, off2


def test_rigidity_zero_for_similarity_map():
    # Embedding = scaled rotation of the (2-D) ambient neighbourhood => ~0 loss.
    x_c, x_nbr, off2 = _rigidity_batch()
    theta = 0.7
    R = torch.tensor([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    s = 2.5
    z_c = torch.randn(x_c.shape[0], 2)  # centre offset is arbitrary (uses differences)
    v = s * (off2 @ R.T)  # (B, m, 2)
    z_nbr = z_c.unsqueeze(1) + v
    loss = local_rigidity_loss(z_c, z_nbr, x_c, x_nbr)
    assert float(loss) < 1e-6


def test_rigidity_positive_for_twist():
    # Per-neighbour rotation (shear/twist of the frame) preserves lengths but
    # breaks relative orientation => strictly positive loss.
    x_c, x_nbr, off2 = _rigidity_batch()
    B, m, _ = off2.shape
    z_c = torch.zeros(B, 2)
    # Rotate each neighbour by a different angle (a twist), not a single R.
    ang = np.linspace(0.0, 1.2, m)
    v = torch.empty(B, m, 2)
    for k in range(m):
        c, s = float(np.cos(ang[k])), float(np.sin(ang[k]))
        Rk = torch.tensor([[c, -s], [s, c]])
        v[:, k, :] = off2[:, k, :] @ Rk.T
    z_nbr = z_c.unsqueeze(1) + v
    twist = local_rigidity_loss(z_c, z_nbr, x_c, x_nbr)
    # Sanity: a single global rotation of the same magnitudes stays ~0.
    c, s = float(np.cos(0.6)), float(np.sin(0.6))
    Rg = torch.tensor([[c, -s], [s, c]])
    z_nbr_rigid = z_c.unsqueeze(1) + (off2 @ Rg.T)
    rigid = local_rigidity_loss(z_c, z_nbr_rigid, x_c, x_nbr)
    assert float(twist) > 1e-3
    assert float(twist) > 10.0 * float(rigid)


def test_rigidity_scale_invariant():
    # Loss must not change when the whole embedding is globally rescaled.
    x_c, x_nbr, off2 = _rigidity_batch(seed=1)
    z_c = torch.randn(x_c.shape[0], 2)
    z_nbr = z_c.unsqueeze(1) + torch.randn_like(off2)  # arbitrary embedding
    base = float(local_rigidity_loss(z_c, z_nbr, x_c, x_nbr))
    scaled = float(local_rigidity_loss(10.0 * z_c, 10.0 * z_nbr, x_c, x_nbr))
    assert abs(base - scaled) < 1e-5


def test_rigidity_mask_ignores_padding():
    # Padded (masked-out) neighbours must not change the loss.
    x_c, x_nbr, off2 = _rigidity_batch(seed=2)
    B, m, D = x_nbr.shape
    z_c = torch.randn(B, 2)
    z_nbr = z_c.unsqueeze(1) + torch.randn(B, m, 2)
    mask_full = torch.ones(B, m)
    ref = float(local_rigidity_loss(z_c, z_nbr, x_c, x_nbr, mask_full))
    # Append two junk padded neighbours with mask=0.
    x_pad = torch.cat([x_nbr, 99.0 * torch.randn(B, 2, D)], dim=1)
    z_pad = torch.cat([z_nbr, 99.0 * torch.randn(B, 2, 2)], dim=1)
    mask_pad = torch.cat([mask_full, torch.zeros(B, 2)], dim=1)
    padded = float(local_rigidity_loss(z_c, z_pad, x_c, x_pad, mask_pad))
    assert abs(ref - padded) < 1e-5


def test_rigidity_tangent_drops_across_sheet_neighbor():
    # A star with 6 in-sheet neighbours (z=0 plane) + 1 across-sheet shortcut
    # (mostly normal, large z). The embedding is a faithful similarity of the
    # in-plane coords but puts the shortcut neighbour in the wrong place.
    ang = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    inplane = np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], axis=1)
    across = np.array([[0.1, 0.1, 1.0]])  # offset dominated by the normal (z) dir
    U = np.concatenate([inplane, across], axis=0).astype(np.float32)  # (7, 3)
    x_c = torch.zeros(1, 3)
    x_nbr = torch.tensor(U).unsqueeze(0)  # (1, 7, 3)

    th, s = 0.5, 2.0
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    v_in = s * (inplane[:, :2] @ R.T)
    v_across = np.array([[5.0, -5.0]])  # deliberately wrong for the shortcut
    V = np.concatenate([v_in, v_across], axis=0).astype(np.float32)
    z_c = torch.zeros(1, 2)
    z_nbr = torch.tensor(V).unsqueeze(0)
    mask = torch.ones(1, 7)

    amb = float(local_rigidity_loss(z_c, z_nbr, x_c, x_nbr, mask, tangent=False))
    tan = float(
        local_rigidity_loss(
            z_c, z_nbr, x_c, x_nbr, mask, tangent=True, normal_thresh=0.5
        )
    )
    # Tangent mode drops the shortcut => remaining in-plane star is rigid (~0).
    assert tan < 1e-4
    # Ambient mode is polluted by the mis-placed shortcut neighbour.
    assert amb > 1e-2
    assert amb > 100.0 * tan


def test_rigidity_tangent_zero_for_similarity():
    # Tangent mode must still be ~0 for a genuine similarity (no shortcuts).
    x_c, x_nbr, off2 = _rigidity_batch()
    th, s = 0.7, 2.5
    R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]],
                     dtype=torch.float32)
    z_c = torch.randn(x_c.shape[0], 2)
    z_nbr = z_c.unsqueeze(1) + s * (off2 @ R.T)
    loss = local_rigidity_loss(z_c, z_nbr, x_c, x_nbr, tangent=True)
    assert float(loss) < 1e-5


def test_geodesic_stress_zero_for_scaled_isometry():
    # Embedding distances = s * geodesic targets => ~0 loss.
    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Place points on a line so ||z_a - z_b|| = 2.5 * g
    za = torch.stack([torch.zeros(4), torch.zeros(4)], dim=1)
    zb = torch.stack([2.5 * g, torch.zeros(4)], dim=1)
    loss = geodesic_stress_loss(za, zb, g)
    assert float(loss) < 1e-6


def test_geodesic_stress_positive_for_banana():
    # Same geodesics, but embedding bends so chord lengths underestimate long
    # geodesics (classic banana): long pairs too short relative to short ones.
    g = torch.linspace(1.0, 10.0, 32)
    # Short pairs nearly correct; long pairs compressed (nonlinear map).
    dz = torch.sqrt(g)  # concave: long distances relatively too short
    za = torch.zeros(32, 2)
    zb = torch.stack([dz, torch.zeros(32)], dim=1)
    banana = float(geodesic_stress_loss(za, zb, g))
    # Rigid scaled embedding of the same g is ~0.
    za_r = torch.zeros(32, 2)
    zb_r = torch.stack([1.7 * g, torch.zeros(32)], dim=1)
    rigid = float(geodesic_stress_loss(za_r, zb_r, g))
    assert banana > 1e-3
    assert banana > 10.0 * rigid


def test_geodesic_stress_scale_invariant():
    g = torch.rand(64) + 0.5
    za = torch.randn(64, 2)
    zb = torch.randn(64, 2)
    base = float(geodesic_stress_loss(za, zb, g))
    scaled = float(geodesic_stress_loss(7.0 * za, 7.0 * zb, g))
    assert abs(base - scaled) < 1e-5


def test_procrustes_anchor_zero_for_similarity():
    torch.manual_seed(0)
    target = torch.randn(40, 2)
    # similarity: rotate + scale + translate
    th = 0.8
    R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]],
                     dtype=torch.float32)
    z = 3.0 * (target @ R.T) + torch.tensor([1.5, -2.0])
    assert float(procrustes_anchor_loss(z, target)) < 1e-5


def test_procrustes_anchor_positive_for_twist():
    torch.manual_seed(0)
    # Rectangle of landmarks; twist the right half.
    xs = torch.linspace(-2, 2, 20)
    ys = torch.linspace(-1, 1, 8)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    target = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    z = target.clone()
    right = z[:, 0] > 0
    z[right, 1] = -z[right, 1]  # flip width on one side = twist
    assert float(procrustes_anchor_loss(z, target)) > 1e-2

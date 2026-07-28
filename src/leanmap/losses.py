"""PLANE loss terms."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

from .utils import get_logger


def _fit_ab(spread: float, min_dist: float) -> Tuple[float, float]:
    """Least-squares fit of ``1/(1+a x^{2b})`` to the piecewise UMAP target."""
    x = np.linspace(0.0, 3.0 * spread, 300, dtype=np.float64)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))

    def curve(xv: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * np.power(xv, 2.0 * b))

    try:
        params, _ = curve_fit(
            curve, x, y, p0=(1.0, 1.0), bounds=(0.0, np.inf), maxfev=10000
        )
        return float(params[0]), float(params[1])
    except Exception:  # noqa: BLE001
        return 1.577, 0.895


def min_dist_for_b(target_b: float = 1.0, spread: float = 1.0) -> float:
    """Smallest ``min_dist`` whose fitted curve reaches ``target_b``.

    ``b`` rises monotonically with ``min_dist`` over most of the range, so a
    bisection works, but the fit degenerates once ``min_dist`` approaches the
    ``3 * spread`` fit domain and the target becomes all-ones -- hence the
    bracket is grown from below and capped well short of it. At the default
    ``spread=1`` the answer for ``target_b=1`` is ~0.199.
    """
    cap = 2.0 * spread
    lo, hi = 0.0, 0.25 * spread
    while _fit_ab(spread, hi)[1] < target_b:
        hi *= 2.0
        if hi > cap:
            return cap
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _fit_ab(spread, mid)[1] < target_b:
            lo = mid
        else:
            hi = mid
    return hi


def find_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> Tuple[float, float]:
    """Fit ``1/(1+a x^{2b})`` to the UMAP target curve.

    ``b`` is not a free knob but it governs stability: the attractive gradient
    near contact goes as ``d^(2b-1)``, so for ``b < 1`` the force decays more
    slowly than the separation and a pair that is already close gets pulled
    proportionally harder. That positive feedback collapses neighbourhoods into
    knots separated by voids. UMAP ships ``min_dist=0.1`` (``b=0.895``) and gets
    away with it because its SGD kernel clips gradients; this implementation
    differentiates the loss directly, so ``b >= 1`` is a floor rather than a
    guideline. Measured on a uniform manifold the clumping only stops growing
    with training at ``b ~ 1.3``, which is why the shipped default sits at
    ``min_dist=0.5`` rather than at the boundary; see ``PLANEConfig.min_dist``.

    Parameters
    ----------
    spread, min_dist : float

    Returns
    -------
    a, b : float
    """
    a, b = _fit_ab(spread, min_dist)
    if b < 1.0:
        get_logger().warning(
            "min_dist=%.3g (spread=%.3g) gives b=%.3f < 1: attraction is not "
            "self-limiting and the layout will clump, more so the longer it "
            "trains. Use min_dist >= %.3g for b >= 1, or %.3g (the default "
            "0.5*spread) to stop the clumping growing with training.",
            min_dist,
            spread,
            b,
            min_dist_for_b(1.0, spread),
            0.5 * spread,
        )
    return a, b


def _clamp_prob(p: torch.Tensor) -> torch.Tensor:
    return p.clamp(1e-7, 1.0 - 1e-7)


def fuzzy_cross_entropy(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    w: torch.Tensor,
    z_neg: torch.Tensor,
    a: float,
    b: float,
) -> torch.Tensor:
    """Fuzzy CE over positive edges and negatives.

    Parameters
    ----------
    z_i, z_j : (B, d) float32
    w : (B,) float32
    z_neg : (B, n_neg, d) float32
    a, b : float

    Returns
    -------
    scalar float32
    """
    def q(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d2 = ((u - v) ** 2).sum(dim=-1).clamp_min(0.0)
        return 1.0 / (1.0 + a * (d2 + 1e-10) ** b)

    q_pos = _clamp_prob(q(z_i, z_j))
    pos = -(w * torch.log(q_pos)).mean()
    # negatives: z_neg (B, n_neg, d)
    zi = z_i.unsqueeze(1).expand_as(z_neg)
    q_neg = _clamp_prob(q(zi, z_neg))
    neg = -torch.log(_clamp_prob(1.0 - q_neg)).mean()
    return (pos + neg).float()


def ordinal_triplet_loss(
    z_a: torch.Tensor,
    z_n: torch.Tensor,
    z_m: torch.Tensor,
    z_f: torch.Tensor,
    mask: torch.Tensor,
    scale_state: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Ordering-only ordinal loss (no margin).

    Parameters
    ----------
    z_a, z_n, z_m, z_f : (B, d) float32
    mask : (B,) bool — triplets that passed distance verification
    scale_state : dict with running mean of ||z_a - z_f||

    Returns
    -------
    loss : scalar
    scale_state : updated
    """
    if scale_state is None:
        scale_state = {"mean_af": 1.0}
    if not mask.any():
        return z_a.sum() * 0.0, scale_state

    za, zn, zm, zf = z_a[mask], z_n[mask], z_m[mask], z_f[mask]
    d_an = (za - zn).norm(dim=-1)
    d_am = (za - zm).norm(dim=-1)
    d_af = (za - zf).norm(dim=-1)
    # running mean of ||za-zf|| detached
    with torch.no_grad():
        batch_mean = float(d_af.mean().item())
        scale_state["mean_af"] = 0.9 * scale_state["mean_af"] + 0.1 * batch_mean
    s = max(scale_state["mean_af"], 1e-6)
    t1 = -F.logsigmoid((d_am - d_an) / s)
    t2 = -F.logsigmoid((d_af - d_am) / s)
    return (t1 + t2).mean(), scale_state



def landmark_regularisation(
    a: torch.Tensor,
    Dm: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """Soft quantisation error minus batch-level affinity entropy.

    Parameters
    ----------
    a : (B, L) float32
    Dm : (B, L) float32
    eta : float

    Returns
    -------
    scalar
    """
    quant = (a * Dm).sum(dim=1).mean()
    mean_a = a.mean(dim=0).clamp_min(1e-12)
    ent = -(mean_a * mean_a.log()).sum()
    return (quant - eta * ent).float()


def local_isometry_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Graph-coupled local isometry on (fine) neighbour edges.

    Penalises deviation from a *proportional* (locally isometric) map: embedding
    edge lengths ``||z_i - z_j||`` should track ambient edge lengths
    ``||x_i - x_j||`` up to a single global scale. A pinch (embedding length → 0
    while ambient length > 0) is a gross local-isometry violation and is heavily
    penalised, so this term directly opposes the frame-rotation pinch while still
    permitting the ribbon to bend.

    The scale is a detached least-squares fit, so the term is invariant to the
    (arbitrary) global embedding scale; the result is normalised by mean squared
    embedding edge length to stay dimensionless across training.

    Intended for the finest graph level only (true kNN neighbours), where ambient
    distance is a good local geodesic proxy.
    """
    dz = (z_i - z_j).norm(dim=1)
    dx = (x_i - x_j).norm(dim=1)
    # Detached least-squares scale s = <dz, dx> / <dx, dx>.
    s = (dz.detach() * dx).sum() / (dx * dx).sum().clamp_min(eps)
    resid = dz - s * dx
    denom = (dz.detach() ** 2).mean().clamp_min(eps)
    return ((resid ** 2).mean() / denom).float()


def local_rigidity_loss(
    z_c: torch.Tensor,
    z_nbr: torch.Tensor,
    x_c: torch.Tensor,
    x_nbr: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tangent: bool = False,
    tangent_dim: Optional[int] = None,
    normal_thresh: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """As-rigid-as-possible neighbourhood loss (Gram-domain, SVD-free grads).

    For each fine node ``c`` with neighbours ``{k}``, the neighbourhood offsets
    ``u_k = x_k - x_c`` (ambient) should map to ``v_k = z_k - z_c`` (embedding)
    by a single local rotation + uniform scale. A similarity transform preserves
    every pairwise inner product up to one scalar, so we penalise the mismatch
    between the embedding and ambient neighbour Gram matrices::

        Gx[k, l] = <u_k, u_l>          Gz[k, l] = <v_k, v_l>
        min_s || Gz - s^2 Gx ||^2      (s detached, per star)

    The diagonal reproduces the edge-length isometry term; the off-diagonals
    encode *relative orientation*, so a frame twist (a rotation that preserves
    lengths but shears the local frame relative to its neighbours) is penalised.
    The objective is rotation/reflection-invariant and needs no SVD gradient, so
    it is fully batched and MPS-safe. Normalised by mean squared embedding Gram
    entry to stay dimensionless and invariant to the (arbitrary) global scale.

    Geodesic / tangent-aware mode (``tangent=True``)
    ------------------------------------------------
    Raw ambient offsets are unsafe when the manifold folds back on itself: the
    fine kNN graph then has across-sheet "shortcut" neighbours whose ambient
    offset points *off* the local surface, and enforcing ambient rigidity glues
    the sheets together (this breaks the swiss roll). With ``tangent=True`` we
    estimate the local tangent frame per star (top-``tangent_dim`` principal
    directions of the offset cloud, computed under ``no_grad`` on the ambient
    coords, so no SVD backward) and (i) drop neighbours whose offset lies mostly
    *off* that tangent plane (normal residual fraction ``> normal_thresh`` — the
    across-sheet shortcuts) and (ii) build the target Gram from the tangent-
    projected offsets, which also removes extrinsic-curvature bias. This makes
    the term a first-order *geodesic* rigidity that is safe on folded manifolds.

    Parameters
    ----------
    z_c : (B, d) embedding of the centre points
    z_nbr : (B, m, d) embedding of the (padded) neighbours
    x_c : (B, D) ambient centre points
    x_nbr : (B, m, D) ambient (padded) neighbours
    mask : (B, m) bool/float — valid neighbours (padding => 0). None => all valid.
    tangent : project offsets onto the local tangent plane + drop off-tangent
        (across-sheet) neighbours before matching Grams.
    tangent_dim : tangent dimensionality (default: embedding dim ``d``).
    normal_thresh : drop a neighbour if ``||u - proj_tan(u)|| / ||u|| >`` this.

    Returns
    -------
    scalar float32
    """
    if z_c.shape[0] == 0 or z_nbr.shape[1] == 0:
        return z_c.sum() * 0.0

    u = x_nbr - x_c.unsqueeze(1)  # (B, m, D)
    v = z_nbr - z_c.unsqueeze(1)  # (B, m, d)
    if mask is not None:
        m_f = mask.to(u.dtype)
    else:
        m_f = torch.ones(u.shape[0], u.shape[1], device=u.device, dtype=u.dtype)

    if tangent:
        d = z_nbr.shape[-1] if tangent_dim is None else int(tangent_dim)
        with torch.no_grad():
            # Robust local tangent: distance-weight the PCA so the *nearest*
            # neighbours (almost always same-sheet) dominate the frame estimate.
            # Otherwise several across-sheet shortcut neighbours would tilt the
            # plane toward the radial (normal) direction and defeat filtering.
            dnbr = u.norm(dim=-1)  # (B, m)
            cnt = m_f.sum(dim=1).clamp_min(1.0)
            h = (dnbr * m_f).sum(dim=1) / cnt  # (B,) mean neighbour distance
            h = h.clamp_min(eps).unsqueeze(1)  # (B, 1)
            w = torch.exp(-0.5 * (dnbr / h) ** 2) * m_f  # (B, m)
            uw = u * w.sqrt().unsqueeze(-1)  # rows scaled => weighted covariance
            # Right singular vectors span the offset cloud; top-d = tangent.
            k = min(uw.shape[1], uw.shape[2])
            d_eff = max(1, min(d, k))
            _, _, Vh = torch.linalg.svd(uw, full_matrices=False)  # Vh: (B, k, D)
            basis = Vh[:, :d_eff, :]  # (B, d_eff, D)
            T = basis.transpose(1, 2)  # (B, D, d_eff)
            u_proj = (u @ T) @ basis  # (B, m, D) reconstruction in tangent plane
            unorm = dnbr.clamp_min(eps)
            normal_frac = (u - u_proj).norm(dim=-1) / unorm
            keep = (normal_frac <= normal_thresh).to(u.dtype)  # (B, m)
        m_f = m_f * keep
        # Tangent-projected ambient offsets (intrinsic / unrolled local coords).
        u = u @ T  # (B, m, d_eff)

    u = u * m_f.unsqueeze(-1)
    v = v * m_f.unsqueeze(-1)
    # Pair mask: entry (k, l) valid iff both neighbours are valid.
    M = m_f.unsqueeze(2) * m_f.unsqueeze(1)  # (B, m, m)

    Gx = u @ u.transpose(1, 2)  # (B, m, m)
    Gz = v @ v.transpose(1, 2)  # (B, m, m)
    # Per-star detached least-squares scale s^2 = <Gz, Gx> / <Gx, Gx>.
    num = (Gz.detach() * Gx * M).sum(dim=(1, 2))
    den = (Gx * Gx * M).sum(dim=(1, 2)).clamp_min(eps)
    s2 = (num / den).view(-1, 1, 1)
    resid = Gz - s2 * Gx
    loss = (resid ** 2 * M).sum() / ((Gz.detach() ** 2 * M).sum().clamp_min(eps))
    return loss.float()


def geodesic_stress_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    g: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale-invariant Isomap / MDS stress on landmark pairs.

    Matches embedding Euclidean distances ``||z_a - z_b||`` to graph-geodesic
    targets ``g`` up to one global scale. Unlike fuzzy neighbour attraction
    (which only cares about *who* is near whom), this constrains *how far*
    landmarks sit — the global metric gauge that affinity losses leave free
    (the "banana" bend of an otherwise isometric unrolling).

    The scale is a detached least-squares fit so the term is invariant to the
    arbitrary global embedding scale; the result is normalised by mean squared
    embedding distance to stay dimensionless.

    Parameters
    ----------
    z_a, z_b : (B, d) embedding of the two landmark ends
    g : (B,) finite graph-geodesic distances between those landmarks

    Returns
    -------
    scalar float32
    """
    if z_a.shape[0] == 0:
        return z_a.sum() * 0.0
    dz = (z_a - z_b).norm(dim=-1)
    g = g.to(dtype=dz.dtype, device=dz.device).clamp_min(eps)
    s = (dz.detach() * g).sum() / (g * g).sum().clamp_min(eps)
    resid = dz - s * g
    denom = (dz.detach() ** 2).mean().clamp_min(eps)
    return ((resid ** 2).mean() / denom).float()


def procrustes_anchor_loss(
    z: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Soft pull of landmark embeddings toward a classical-MDS layout.

    Computes the optimal similarity (rotation + uniform scale + translation)
    aligning ``target`` to the *current* ``z`` under ``no_grad``, then returns
    ``MSE(z, aligned_target)``. Gradients flow only into ``z``, so the loss
    pins the global metric *gauge* (including untwisting a banana / bowtie)
    without fighting the arbitrary rigid motion of the embedding.

    Stronger than pairwise geodesic stress alone: stress is invariant to any
    local frame twist that approximately preserves landmark–landmark distances,
    while this absolute-position prior (from classical MDS of the geodesic
    matrix) selects the untwisted developable layout.

    Parameters
    ----------
    z : (L, d) current landmark embeddings
    target : (L, d) frozen classical-MDS coordinates of the same landmarks
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    with torch.no_grad():
        zc = z - z.mean(dim=0, keepdim=True)
        yc = target.to(device=z.device, dtype=z.dtype)
        yc = yc - yc.mean(dim=0, keepdim=True)
        # Orthogonal Procrustes: R = U V^T from SVD(Y^T Z)
        M = yc.transpose(0, 1) @ zc  # (d, d)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh
        if torch.det(R) < 0:
            U = U.clone()
            U[:, -1] = -U[:, -1]
            R = U @ Vh
            S = S.clone()
            S[-1] = -S[-1]
        denom = (yc * yc).sum().clamp_min(eps)
        scale = S.sum() / denom
        t = z.mean(dim=0) - scale * (target.to(z.device, z.dtype).mean(dim=0) @ R)
    aligned = scale * (target.to(device=z.device, dtype=z.dtype) @ R) + t
    return ((z - aligned) ** 2).mean().float()


def alignment_ramp(
    t: float,
    start: float = 0.3,
    end: float = 0.6,
    *,
    down: bool = False,
) -> float:
    """Piecewise-linear weight schedule on training fraction ``t ∈ [0, 1]``.

    Ramp **up** (default): 0 until ``start``, linear to 1 by ``end``, then 1.
    ``(0, 0)`` => on from the start (always 1).

    Ramp **down** (``down=True``): 1 until ``start``, linear to 0 by ``end``,
    then 0. Use e.g. ``(0.0, 0.25)`` to front-load a term for the first
    quarter of training and then shut it off.
    """
    if down:
        if end <= start:
            return 1.0 if (start == 0.0 and end == 0.0) else 0.0
        if t <= start:
            return 1.0
        if t >= end:
            return 0.0
        return float(1.0 - (t - start) / (end - start))
    if start == 0.0 and end == 0.0:
        return 1.0
    if t < start:
        return 0.0
    if t >= end:
        return 1.0
    return float((t - start) / (end - start))

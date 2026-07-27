"""Conformal exchangeability test on OOD scores (landmark cover)."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .distance import EuclideanDistance
from .model import PLANE
from .utils import get_logger


def geometry_consistency_score(
    model: PLANE,
    x: torch.Tensor,
    tau_embed: float,
    z_M: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Primary OOD score (landmark cover) + secondary affinity-consistency.

    **Primary — cover.** ``min_l ||x - M_l||`` in input space. Points far from
    every landmark are off the support the model was trained to chart. This is
    what conformal calibration / ``embed`` scores use for OOD gating.

    **Secondary — consistency.** ``0.5 ||a - a_embed||_1`` compares input-space
    PRIMARY affinity to embedding-space affinity to the same anchors. Useful as
    a *chart-quality* diagnostic (conditioning vs metric disagreement) but
    **not** a reliable OOD gate: off-manifold points can look spuriously
    consistent.

    Parameters
    ----------
    model : PLANE
    x : (B, D) float32
    tau_embed : float
        Embedding-space softmax temperature (stored in the artefact).
    z_M : (L, d_out) | None
        Cached PRIMARY landmark embeddings.

    Returns
    -------
    ood_score : (B,) float32
        Landmark cover distance (higher ⇒ more OOD).
    consistency : (B,) float32 in [0, 1]
        Affinity L1 chart diagnostic.
    """
    z, a, Dm = model(x)
    if z_M is None:
        z_M = model._primary_anchor_embeddings(z.device)
    d_emb = EuclideanDistance()(z, z_M.to(z.device))
    a_embed = F.softmax(-d_emb / float(tau_embed), dim=1)
    L = min(a.shape[1], a_embed.shape[1])
    consistency = 0.5 * (a[:, :L] - a_embed[:, :L]).abs().sum(dim=1)
    cover = Dm.min(dim=1).values
    return cover, consistency


def model_weight_hash(model: torch.nn.Module) -> str:
    """Stable hash of model parameters for calibration invalidation."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items(), key=lambda kv: kv[0]):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bh_reject(p: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Benjamini–Hochberg FDR rejection mask.

    Parameters
    ----------
    p : (m,) float32
    alpha : float

    Returns
    -------
    reject : (m,) bool
    """
    m = p.numel()
    if m == 0:
        return torch.zeros(0, dtype=torch.bool)
    order = torch.argsort(p)
    p_sorted = p[order]
    thresh = alpha * (torch.arange(1, m + 1, device=p.device, dtype=p.dtype) / m)
    below = p_sorted <= thresh
    reject = torch.zeros(m, dtype=torch.bool, device=p.device)
    if below.any():
        j = int(torch.where(below)[0].max().item())
        reject[order[: j + 1]] = True
    return reject


def _mannwhitneyu_statistic(x: torch.Tensor, y: torch.Tensor) -> float:
    """Two-sample Mann–Whitney U (x vs y), larger means x tends larger."""
    nx, ny = x.numel(), y.numel()
    pool = torch.cat([x, y])
    order = torch.argsort(pool)
    ranks = torch.empty_like(pool, dtype=torch.float64)
    ranks[order] = torch.arange(1, pool.numel() + 1, dtype=torch.float64, device=pool.device)
    rx = ranks[:nx].sum()
    U = float(rx.item() - nx * (nx + 1) / 2.0)
    return U


class ConformalCalibrator:
    """Hold-out calibration of landmark-cover OOD scores.

    Caveats (also in the README):

    1. The test is on the **cover** distribution. A shift that leaves cover
       unchanged (e.g. sliding along the manifold) is invisible.
    2. It answers "is this point near the landmark support", not "have I seen
       this exact point before". A novel point sitting on the manifold will pass.

    Retraining or updating landmarks invalidates calibration: ``weight_hash``
    must match or ``p_value`` raises.
    """

    def __init__(self):
        self.s_calib: Optional[torch.Tensor] = None  # sorted cover scores
        self.tau_embed: Optional[float] = None
        self.weight_hash: Optional[str] = None
        self.cover_calib: Optional[torch.Tensor] = None  # alias of s_calib
        self.consistency_calib: Optional[torch.Tensor] = None  # diagnostic only

    @torch.no_grad()
    def fit(self, model: PLANE, X_calib: torch.Tensor, batch_size: int = 1024) -> None:
        """Calibrate on raw held-out points (never epsilon-netted).

        Parameters
        ----------
        model : PLANE
        X_calib : (n, D) float32 — raw calibration array
        batch_size : int
        """
        log = get_logger()
        model.eval()
        device = next(model.parameters()).device
        z_M = model._primary_anchor_embeddings(device)
        # tau_embed = median ||z - z_M|| over calib (for consistency diagnostic)
        dists = []
        for s in range(0, X_calib.shape[0], batch_size):
            e = min(X_calib.shape[0], s + batch_size)
            xb = X_calib[s:e].to(device)
            z, _, _ = model(xb)
            d_emb = EuclideanDistance()(z, z_M)
            dists.append(d_emb.reshape(-1).cpu())
        all_d = torch.cat(dists)
        self.tau_embed = float(all_d.median().item())

        covers = []
        consistencies = []
        for s in range(0, X_calib.shape[0], batch_size):
            e = min(X_calib.shape[0], s + batch_size)
            xb = X_calib[s:e].to(device)
            cover, consistency = geometry_consistency_score(
                model, xb, tau_embed=self.tau_embed, z_M=z_M
            )
            covers.append(cover.cpu())
            consistencies.append(consistency.cpu())
        cover_all = torch.cat(covers)
        self.s_calib = torch.sort(cover_all).values
        self.cover_calib = self.s_calib
        self.consistency_calib = torch.cat(consistencies)
        self.weight_hash = model_weight_hash(model)
        n = self.s_calib.numel()
        if n < 200:
            log.warning(
                "n_calib=%d < 200: alpha values below 1/(n+1)=%.4f are unreachable",
                n,
                1.0 / (n + 1),
            )

    def _check_hash(self, model: PLANE) -> None:
        if self.weight_hash is None:
            raise RuntimeError("ConformalCalibrator.fit has not been called")
        h = model_weight_hash(model)
        if h != self.weight_hash:
            raise RuntimeError(
                "model weights do not match calibration hash — recalibrate"
            )

    def p_value(self, scores: torch.Tensor, model: Optional[PLANE] = None) -> torch.Tensor:
        """``p(x) = (1 + #{s_calib >= s(x)}) / (n+1)`` via searchsorted.

        ``scores`` should be landmark **cover** distances (higher = more OOD).
        Small ``p`` ⇒ cover is large relative to the calibration set ⇒ OOD.

        Parameters
        ----------
        scores : (B,) float32
        model : PLANE | None
            If given, verify weight hash.

        Returns
        -------
        p : (B,) float32
        """
        if model is not None:
            self._check_hash(model)
        assert self.s_calib is not None
        n = self.s_calib.numel()
        idx = torch.searchsorted(self.s_calib, scores.cpu(), right=False)
        count_ge = n - idx
        p = (1 + count_ge.float()) / (n + 1)
        return p.to(scores.device)

    def is_exchangeable(
        self, scores: torch.Tensor, alpha: float = 0.05, model: Optional[PLANE] = None
    ) -> torch.Tensor:
        """Per-point flag: ``p > alpha`` (True = looks exchangeable / in-support)."""
        p = self.p_value(scores, model=model)
        return p > alpha

    def batch_test(
        self,
        scores: torch.Tensor,
        n_perm: int = 10_000,
        seed: int = 0,
    ) -> dict:
        """Permutation Mann–Whitney test on raw cover scores.

        Returns
        -------
        dict with keys p_global, statistic, n_calib, n_batch, median_shift
        """
        assert self.s_calib is not None
        s_c = self.s_calib.double()
        s_b = scores.detach().cpu().double().reshape(-1)
        observed = _mannwhitneyu_statistic(s_b, s_c)
        pool = torch.cat([s_c, s_b])
        n = s_c.numel()
        g = torch.Generator().manual_seed(seed)
        ge = 0
        for _ in range(n_perm):
            perm = pool[torch.randperm(pool.numel(), generator=g)]
            stat = _mannwhitneyu_statistic(perm[n:], perm[:n])
            if stat >= observed:
                ge += 1
        p_global = (1 + ge) / (n_perm + 1)
        return {
            "p_global": float(p_global),
            "statistic": float(observed),
            "n_calib": int(n),
            "n_batch": int(s_b.numel()),
            "median_shift": float(s_b.median().item() - s_c.median().item()),
        }

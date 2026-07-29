"""Conformal exchangeability test on OOD scores (landmark cover).

Validity note
-------------
The conformal guarantee here rests on exchangeability of the *calibration*
scores with the test scores, and on the score function being fixed before
calibration. Quantities that only rescale the score are therefore free to be
estimated on all of ``X``:

* the metric natural scale ``s_nat`` and any global ``(mu, sigma)`` are strictly
  monotone rescalings of the cover, and :func:`ConformalCalibrator.p_value`
  depends on scores only through their **ranks** against the calibration set;
  a common monotone map applied to both sides leaves every p-value unchanged;
* the per-landmark radii and tangent charts in :class:`LandmarkSupport` are
  *not* rank-preserving — they reorder points — so they must be fit on data
  disjoint from the calibration split. :meth:`LandmarkSupport.fit` therefore
  takes **training** points, which are plentiful, and never the calibration
  set.

So there is no leak in either case, but for different reasons: the first is
harmless because it cannot change ranks, the second is handled by the split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F

from .distance import EuclideanDistance
from .model import PLANE
from .utils import get_logger

# Nonconformity scorers: (model, x, **ctx) -> (B,) higher ⇒ more nonconforming.
NonconformityFn = Callable[..., torch.Tensor]
ScoreSpec = Union[str, NonconformityFn]

# Mondrian taxonomy used by :class:`MondrianCalibrator` by default.
MONDRIAN_GROUPS: Tuple[str, ...] = ("digit", "gauss", "shuffle")


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


@dataclass
class LandmarkSupport:
    """Per-landmark support model: locally-scaled balls or tangent charts.

    The plain cover :math:`\\min_\\ell d(x, M_\\ell) / s_{\\mathrm{nat}}` uses one
    global scale, so it flags the sparse tail of the training distribution as
    readily as anything genuinely off-manifold. Two refinements, both fit from
    **training** points (never the calibration split — see the module docstring):

    ``mode="ball"``
        :math:`\\mathrm{cover}(x) = \\min_\\ell d(x, M_\\ell) / r_\\ell`, with
        :math:`r_\\ell` the median distance-to-landmark among training points
        whose nearest landmark is :math:`\\ell`. This buys approximate
        *conditional* validity while keeping a single pooled calibration set.

    ``mode="chart"`` (default)
        Residual to a per-landmark local PCA chart. Isotropic balls in
        :math:`D \\gg m` are exponentially too generous: a union of balls has
        volume :math:`\\sim L\\tau^D` against a support of
        :math:`\\sim L\\tau^m t^{D-m}` for sheet thickness :math:`t \\ll \\tau`.
        Splitting each landmark's neighbourhood into tangent and normal
        directions and scaling them separately replaces balls with ellipsoids
        that track the sheet.

    Attributes
    ----------
    M : (L, D) landmark coordinates in the PRIMARY view.
    r : (L,) local radii (``mode="ball"``, and the repair target).
    V : (L, D, m) | None orthonormal tangent bases (``mode="chart"``).
    sigma_par, sigma_perp : (L,) tangent / normal scales.
    mode : ``"ball"`` or ``"chart"``.

    Notes
    -----
    Charts are linear, so they assume the PRIMARY view is a vector space. For a
    non-linear view, use ``mode="ball"``.
    """

    M: torch.Tensor
    r: torch.Tensor
    sigma_par: torch.Tensor
    sigma_perp: torch.Tensor
    V: Optional[torch.Tensor] = None
    mode: str = "chart"

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        M: torch.Tensor,
        X_train: torch.Tensor,
        mode: str = "chart",
        m_tangent: int = 2,
        min_bucket: int = 8,
        floor_frac: float = 1e-3,
    ) -> "LandmarkSupport":
        """Fit radii and charts from training points.

        Parameters
        ----------
        M : (L, D) landmarks.
        X_train : (n, D) training points — must be disjoint from calibration.
        mode : ``"ball"`` or ``"chart"``.
        m_tangent : intrinsic dimension used for the local chart.
        min_bucket : buckets with fewer points fall back to an isotropic ball,
            which the chart score reproduces exactly when ``V_l = 0`` and the
            two scales are equal.
        floor_frac : scales are floored at this fraction of the global median
            distance, so a degenerate direction cannot produce an infinite
            score.
        """
        log = get_logger()
        M = M.detach().float()
        X = X_train.detach().float().to(M.device)
        L, D = M.shape
        d = torch.cdist(X, M)
        owner = d.argmin(dim=1)
        d_own = d.gather(1, owner[:, None]).squeeze(1)
        global_med = float(d_own.median().item()) if d_own.numel() else 1.0
        floor = max(global_med * floor_frac, torch.finfo(torch.float32).tiny)

        r = torch.full((L,), global_med, device=M.device)
        sigma_par = torch.full((L,), global_med, device=M.device)
        sigma_perp = torch.full((L,), global_med, device=M.device)
        want_chart = mode == "chart"
        m_eff = max(1, min(int(m_tangent), D - 1)) if D > 1 else 1
        V = torch.zeros(L, D, m_eff, device=M.device) if want_chart else None

        n_chart = 0
        for ell in range(L):
            sel = owner == ell
            n_l = int(sel.sum().item())
            if n_l == 0:
                continue
            pts = X[sel]
            u = pts - M[ell]
            r_l = float(u.norm(dim=1).median().item())
            r[ell] = max(r_l, floor)
            if not want_chart or n_l < max(min_bucket, m_eff + 2):
                sigma_par[ell] = r[ell]
                sigma_perp[ell] = r[ell]
                continue
            # Tangent directions from the centred bucket scatter (a proper
            # local PCA); the chart origin stays at the landmark.
            centred = pts - pts.mean(dim=0, keepdim=True)
            try:
                _, _, Vh = torch.linalg.svd(centred, full_matrices=False)
            except RuntimeError:
                sigma_par[ell] = r[ell]
                sigma_perp[ell] = r[ell]
                continue
            basis = Vh[:m_eff].T.contiguous()
            V[ell] = basis
            par = u @ basis
            perp = u - par @ basis.T
            sigma_par[ell] = max(float(par.norm(dim=1).median().item()), floor)
            sigma_perp[ell] = max(float(perp.norm(dim=1).median().item()), floor)
            n_chart += 1

        if want_chart:
            log.info(
                "LandmarkSupport: charts on %d/%d landmarks (m=%d); "
                "%d fell back to isotropic balls",
                n_chart,
                L,
                m_eff,
                L - n_chart,
            )
        return cls(
            M=M,
            r=r,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            V=V,
            mode=mode,
        )

    @classmethod
    def from_model(
        cls, model: PLANE, X_train: torch.Tensor, **kwargs
    ) -> "LandmarkSupport":
        """Fit against a trained model's PRIMARY landmarks.

        ``X_train`` must be the training split — passing the calibration split
        would make the score depend on the data it is calibrated against.
        """
        M = model.affinity.M.detach()
        if X_train.shape[1] != M.shape[1]:
            raise ValueError(
                f"X_train has {X_train.shape[1]} columns but PRIMARY landmarks "
                f"have {M.shape[1]}; charts need the PRIMARY view coordinates"
            )
        return cls.fit(M, X_train, **kwargs)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Nonconformity score (higher ⇒ more OOD), minimised over landmarks."""
        return self.score_per_landmark(x).min(dim=1).values

    @torch.no_grad()
    def score_per_landmark(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) per-landmark scores; the cover score is the row-wise min."""
        x = x.detach().float().to(self.M.device)
        d2 = torch.cdist(x, self.M).pow(2)
        if self.mode != "chart" or self.V is None:
            return d2.clamp_min(0).sqrt() / self.r[None, :]
        # ||par||^2 without materialising (B, L, D): project onto each basis.
        proj = torch.einsum("bd,ldm->blm", x, self.V) - torch.einsum(
            "ld,ldm->lm", self.M, self.V
        )[None]
        par2 = proj.pow(2).sum(dim=-1)
        perp2 = (d2 - par2).clamp_min(0)
        s2 = par2 / self.sigma_par[None, :].pow(2) + perp2 / self.sigma_perp[
            None, :
        ].pow(2)
        return s2.clamp_min(0).sqrt()

    @torch.no_grad()
    def repair(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Minimum-norm move into ``{score <= tau}``, with heterogeneous radii.

        Projection onto a union of balls equals projection onto the ball of the
        *nearest centre* only when the radii are equal. With per-landmark radii
        the required travel to ball :math:`\\ell` is :math:`d_\\ell - \\tau
        r_\\ell`, so the correct target minimises **that**, not :math:`d_\\ell`.
        A distant landmark with a generous radius can be the cheaper repair.

        In ``mode="chart"`` the target landmark is chosen the same way, on the
        chart score, and the point is then moved onto that landmark's isotropic
        ball of radius :math:`\\tau r_\\ell` — a conservative inner move, since
        an exact ellipsoid projection is a 2-D subproblem not needed here.
        """
        x = x.detach().float().to(self.M.device)
        d = torch.cdist(x, self.M)
        travel = d - float(tau) * self.r[None, :]
        star = travel.argmin(dim=1)
        d_star = d.gather(1, star[:, None]).squeeze(1)
        radius = float(tau) * self.r[star]
        need = d_star > radius
        out = x.clone()
        if not bool(need.any()):
            return out
        idx = torch.where(need)[0]
        centre = self.M[star[idx]]
        scale = (radius[idx] / d_star[idx].clamp_min(1e-12))[:, None]
        out[idx] = centre + scale * (x[idx] - centre)
        return out


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

    def __init__(self, support: Optional[LandmarkSupport] = None):
        self.s_calib: Optional[torch.Tensor] = None  # sorted cover scores
        self.tau_embed: Optional[float] = None
        self.weight_hash: Optional[str] = None
        self.cover_calib: Optional[torch.Tensor] = None  # alias of s_calib
        self.consistency_calib: Optional[torch.Tensor] = None  # diagnostic only
        self.support = support

    @torch.no_grad()
    def fit(self, model: PLANE, X_calib: torch.Tensor, batch_size: int = 1024) -> None:
        """Calibrate on raw held-out points (never epsilon-netted).

        If a :class:`LandmarkSupport` was supplied, its score replaces the plain
        global-scale cover. The support must have been fit on training points;
        fitting it on ``X_calib`` would break exchangeability.

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
            if self.support is not None:
                cover = self.support.score(xb)
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

    @torch.no_grad()
    def cover_score(self, model: PLANE, x: torch.Tensor) -> torch.Tensor:
        """Score ``x`` with the same function used at calibration time.

        Calibrating with a :class:`LandmarkSupport` and then scoring with the
        plain cover silently compares two different score functions, which
        voids the guarantee. Route both through here.
        """
        if self.support is not None:
            return self.support.score(x)
        tau = self.tau_embed if self.tau_embed is not None else 1.0
        cover, _ = geometry_consistency_score(model, x, tau_embed=float(tau))
        return cover

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


# ---------------------------------------------------------------------------
# Pluggable nonconformity scores + Mondrian (category-conditional) calibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def affinity_entropy_score(
    model: PLANE,
    x: torch.Tensor,
    **_ctx,
) -> torch.Tensor:
    """Primary affinity entropy ``H(a) = -∑ a_ℓ log a_ℓ`` (higher ⇒ flatter)."""
    _z, a, _Dm = model(x)
    return -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=1)


@torch.no_grad()
def cover_score(model: PLANE, x: torch.Tensor, **_ctx) -> torch.Tensor:
    """Ambient landmark cover ``min_ℓ ‖x - M_ℓ‖``."""
    _z, _a, Dm = model(x)
    return Dm.min(dim=1).values


@torch.no_grad()
def soft_cover_score(
    model: PLANE,
    x: torch.Tensor,
    *,
    tau_embed: float = 1.0,
    **_ctx,
) -> torch.Tensor:
    """Softmin landmark cover ``-τ logsumexp(-Dm / τ)``."""
    _z, _a, Dm = model(x)
    tau = max(float(tau_embed), 1e-6)
    return -tau * torch.logsumexp(-Dm / tau, dim=1)


@torch.no_grad()
def affinity_max_neg_score(model: PLANE, x: torch.Tensor, **_ctx) -> torch.Tensor:
    """``-max a_ℓ``: low peak affinity ⇒ more nonconforming."""
    _z, a, _Dm = model(x)
    return -a.max(dim=1).values


@torch.no_grad()
def emb_cover_score(
    model: PLANE,
    x: torch.Tensor,
    *,
    z_M: Optional[torch.Tensor] = None,
    **_ctx,
) -> torch.Tensor:
    """Embedding-space cover ``min_ℓ ‖z - z(M_ℓ)‖``."""
    z, _a, _Dm = model(x)
    if z_M is None:
        z_M = model._primary_anchor_embeddings(z.device)
    return EuclideanDistance()(z, z_M.to(z.device)).min(dim=1).values


@torch.no_grad()
def cover_plus_entropy_score(
    model: PLANE,
    x: torch.Tensor,
    *,
    cover_scale: float = 1.0,
    ent_scale: float = 1.0,
    **_ctx,
) -> torch.Tensor:
    """``cover / cover_scale + H(a) / ent_scale`` (scales from digit calib)."""
    _z, a, Dm = model(x)
    cover = Dm.min(dim=1).values
    ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=1)
    return cover / max(float(cover_scale), 1e-8) + ent / max(float(ent_scale), 1e-8)


NONCONFORMITY_SCORES: Dict[str, NonconformityFn] = {
    "affinity_entropy": affinity_entropy_score,
    "a_ent": affinity_entropy_score,
    "cover": cover_score,
    "dm_min": cover_score,
    "soft_cover": soft_cover_score,
    "a_max_neg": affinity_max_neg_score,
    "emb_cover": emb_cover_score,
    "dm_min+a_ent": cover_plus_entropy_score,
}


def list_nonconformity_scores() -> Tuple[str, ...]:
    """Registered nonconformity score names (canonical first)."""
    # Dedup aliases while keeping a stable preferred order.
    preferred = (
        "affinity_entropy",
        "cover",
        "soft_cover",
        "a_max_neg",
        "emb_cover",
        "dm_min+a_ent",
        "lda",  # CoverEntropyLDA instance — see that class
    )
    return preferred


def resolve_nonconformity(score: ScoreSpec) -> Tuple[str, NonconformityFn]:
    """Return ``(name, fn)`` for a registry key or callable."""
    if isinstance(score, CoverEntropyLDA):
        return "lda", score
    if callable(score) and not isinstance(score, str):
        name = getattr(score, "__name__", "custom")
        return str(name), score
    key = str(score)
    if key == "lda":
        raise ValueError(
            "score='lda' needs a fitted CoverEntropyLDA instance, e.g. "
            "MondrianCalibrator(score=CoverEntropyLDA().fit(model, X_in, X_ood))"
        )
    if key not in NONCONFORMITY_SCORES:
        known = ", ".join(list_nonconformity_scores())
        raise ValueError(f"unknown nonconformity score {key!r}; choose from: {known}")
    return key, NONCONFORMITY_SCORES[key]


@torch.no_grad()
def cover_entropy_features(
    model: PLANE,
    x: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """``(N, 2)`` features ``[min_ℓ ‖x−M_ℓ‖, H(a)]`` on CPU."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for s in range(0, x.shape[0], batch_size):
        e = min(x.shape[0], s + batch_size)
        xb = x[s:e].to(device)
        _z, a, Dm = model(xb)
        cover = Dm.min(dim=1).values
        ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=1)
        outs.append(torch.stack([cover, ent], dim=1).cpu())
    return torch.cat(outs, dim=0)


@dataclass
class CoverEntropyLDA:
    """Fisher LDA on ``(cover, affinity_entropy)`` → signed-distance nonconformity.

    Fit on in-support points vs an OOD pool (gauss, shuffle, or both). The score
    is the signed distance to the separating hyperplane in feature space,
    oriented so **higher ⇒ more OOD**. Pass a fitted instance as
    ``MondrianCalibrator(score=lda)``.

    Attributes
    ----------
    weight : (2,) unit normal of the hyperplane in ``(cover, H)`` space
    bias : plane offset; score = ``phi · weight + bias``
    """

    weight: Optional[torch.Tensor] = None
    bias: float = 0.0
    batch_size: int = 1024

    @torch.no_grad()
    def fit(
        self,
        model: PLANE,
        X_in: torch.Tensor,
        X_ood: torch.Tensor,
    ) -> "CoverEntropyLDA":
        """Fit binary Fisher LDA: ``X_in`` (digit) vs ``X_ood`` (noise)."""
        import numpy as np

        Phi0 = cover_entropy_features(model, X_in.float(), self.batch_size).numpy()
        Phi1 = cover_entropy_features(model, X_ood.float(), self.batch_size).numpy()
        mu0, mu1 = Phi0.mean(0), Phi1.mean(0)
        n0, n1 = max(len(Phi0) - 1, 1), max(len(Phi1) - 1, 1)
        Sw = n0 * np.cov(Phi0.T) + n1 * np.cov(Phi1.T)
        Sw = Sw + 1e-6 * np.eye(2)
        w = np.linalg.solve(Sw, mu1 - mu0)
        w = w / (np.linalg.norm(w) + 1e-12)
        mid = 0.5 * (mu0 + mu1)
        bias = -float(mid @ w)
        # Orient: OOD mean must score higher than in-support mean.
        if float(mu1 @ w + bias) < float(mu0 @ w + bias):
            w = -w
            bias = -bias
        self.weight = torch.as_tensor(w, dtype=torch.float32)
        self.bias = bias
        return self

    def __call__(self, model: PLANE, x: torch.Tensor, **_ctx) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("CoverEntropyLDA.fit has not been called")
        phi = cover_entropy_features(model, x, self.batch_size)  # (B, 2) CPU
        s = phi @ self.weight + float(self.bias)
        return s.to(dtype=torch.float32)

    def hyperplane(self) -> Tuple[torch.Tensor, float]:
        """Return ``(weight, bias)`` for plotting ``w·phi + b = 0``."""
        if self.weight is None:
            raise RuntimeError("CoverEntropyLDA.fit has not been called")
        return self.weight.detach().clone(), float(self.bias)

    def state_dict(self) -> dict:
        if self.weight is None:
            raise RuntimeError("CoverEntropyLDA.fit has not been called")
        return {
            "weight": self.weight.cpu(),
            "bias": float(self.bias),
            "batch_size": int(self.batch_size),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "CoverEntropyLDA":
        obj = cls(batch_size=int(state.get("batch_size", 1024)))
        obj.weight = torch.as_tensor(state["weight"], dtype=torch.float32)
        obj.bias = float(state["bias"])
        return obj


def conformal_threshold(s_calib: torch.Tensor, alpha: float) -> float:
    """One-sided conformal threshold: reject when ``score > q`` at level ``alpha``.

    With sorted calibration scores ``s_(1) ≤ … ≤ s_(n)``,
    ``q = s_(k)`` for ``k = ⌈(n+1)(1-α)⌉``. If ``k > n``, no finite threshold
    can guarantee the level (returns ``+inf``).
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = torch.sort(s_calib.detach().float().reshape(-1)).values
    n = int(s.numel())
    if n == 0:
        raise ValueError("empty calibration scores")
    k = int(torch.ceil(torch.tensor((n + 1) * (1.0 - float(alpha)))).item())
    if k > n:
        return float("inf")
    return float(s[k - 1].item())


def make_mondrian_groups(
    X_digit: torch.Tensor,
    *,
    n_gauss: Optional[int] = None,
    n_shuffle: Optional[int] = None,
    seed: int = 0,
    digit_key: str = "digit",
) -> Dict[str, torch.Tensor]:
    """Build the default Mondrian taxonomy from a digit calibration pool.

    ``gauss``
        i.i.d. Gaussian with the same per-feature mean/std as ``X_digit``.
    ``shuffle``
        pixel-permuted copies of random digit rows (exact intensity multiset).
    """
    X = X_digit.detach().float()
    n, D = X.shape
    n_g = int(n if n_gauss is None else n_gauss)
    n_s = int(n if n_shuffle is None else n_shuffle)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    mu = X.mean(dim=0)
    sig = X.std(dim=0).clamp_min(1e-8)
    gauss = mu + sig * torch.randn(n_g, D, generator=g)
    parents = torch.randint(0, n, (n_s,), generator=g)
    shuffle = X[parents].clone()
    for i in range(n_s):
        shuffle[i] = shuffle[i][torch.randperm(D, generator=g)]
    return {
        digit_key: X.cpu(),
        "gauss": gauss.cpu(),
        "shuffle": shuffle.cpu(),
    }


@dataclass
class MondrianCalibrator:
    """Mondrian (category-conditional) conformal thresholds / p-values.

    Calibrate separately on each group of a taxonomy — by default
    ``digit``, ``gauss`` (μ/σ-matched noise), and ``shuffle`` (pixel-permuted
    digits). A test point then gets one p-value **per group**; the prediction
    set at level ``α`` is ``{g : p_g > α}``.

    The nonconformity score defaults to affinity entropy (best unsupervised
    separator in the digits hunt); pass ``score=`` to choose another registered
    name or a custom ``(model, x, **ctx) -> (B,)`` callable.

    Parameters
    ----------
    score : str | callable
        Nonconformity function. Registered names from
        :func:`list_nonconformity_scores`; default ``"affinity_entropy"``.
    batch_size : int
        Forward-pass batching for scoring.
    """

    score: ScoreSpec = "affinity_entropy"
    batch_size: int = 1024
    # Filled by fit:
    score_name: str = ""
    s_calib: Dict[str, torch.Tensor] = field(default_factory=dict)
    weight_hash: Optional[str] = None
    tau_embed: Optional[float] = None
    cover_scale: float = 1.0
    ent_scale: float = 1.0
    _score_fn: Optional[NonconformityFn] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        name, fn = resolve_nonconformity(self.score)
        self.score_name = name
        self._score_fn = fn

    @torch.no_grad()
    def score_points(self, model: PLANE, X: torch.Tensor) -> torch.Tensor:
        """Score ``X`` with the configured nonconformity function → ``(N,)``."""
        assert self._score_fn is not None
        model.eval()
        device = next(model.parameters()).device
        outs = []
        ctx = {
            "tau_embed": float(self.tau_embed) if self.tau_embed is not None else 1.0,
            "cover_scale": self.cover_scale,
            "ent_scale": self.ent_scale,
            "z_M": None,
        }
        z_M = None
        if self.score_name in ("emb_cover",):
            z_M = model._primary_anchor_embeddings(device)
            ctx["z_M"] = z_M
        for s in range(0, X.shape[0], self.batch_size):
            e = min(X.shape[0], s + self.batch_size)
            xb = X[s:e].to(device)
            outs.append(self._score_fn(model, xb, **ctx).detach().cpu())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def fit(
        self,
        model: PLANE,
        groups: Mapping[str, torch.Tensor],
        *,
        digit_key: str = "digit",
    ) -> "MondrianCalibrator":
        """Calibrate on a mapping ``{group_name: X_group}``.

        Use :func:`make_mondrian_groups` for the default digit/gauss/shuffle
        taxonomy. ``digit_key`` selects which group supplies the monotone
        scale stats for composite scores (e.g. ``dm_min+a_ent``).
        """
        if not groups:
            raise ValueError("groups must be a non-empty mapping")
        model.eval()
        device = next(model.parameters()).device

        # Embedding temperature from the digit (or first) group — monotone.
        key0 = digit_key if digit_key in groups else next(iter(groups))
        X0 = groups[key0].float()
        z_M = model._primary_anchor_embeddings(device)
        dists = []
        for s in range(0, X0.shape[0], self.batch_size):
            e = min(X0.shape[0], s + self.batch_size)
            z, _, _ = model(X0[s:e].to(device))
            dists.append(EuclideanDistance()(z, z_M).reshape(-1).cpu())
        self.tau_embed = float(torch.cat(dists).median().item())

        # Composite-score scales from the digit group (rank-free within group
        # once fixed; estimated before any other group is scored).
        if self.score_name in ("dm_min+a_ent",) and key0 in groups:
            raw_cover = []
            raw_ent = []
            for s in range(0, X0.shape[0], self.batch_size):
                e = min(X0.shape[0], s + self.batch_size)
                xb = X0[s:e].to(device)
                _z, a, Dm = model(xb)
                raw_cover.append(Dm.min(dim=1).values.cpu())
                raw_ent.append(
                    -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(1).cpu()
                )
            c = torch.cat(raw_cover)
            h = torch.cat(raw_ent)
            self.cover_scale = float(c.median().clamp_min(1e-8).item())
            self.ent_scale = float(h.median().clamp_min(1e-8).item())

        self.s_calib = {}
        log = get_logger()
        for name, Xg in groups.items():
            Xg = Xg.detach().float()
            if Xg.ndim != 2:
                raise ValueError(f"group {name!r} must be (n, D), got {tuple(Xg.shape)}")
            s = self.score_points(model, Xg)
            self.s_calib[str(name)] = torch.sort(s).values
            n = int(s.numel())
            if n < 50:
                log.warning(
                    "Mondrian group %r has n_calib=%d; alphas below 1/(n+1)=%.4f "
                    "are unreachable",
                    name,
                    n,
                    1.0 / (n + 1),
                )
        self.weight_hash = model_weight_hash(model)
        return self

    def fit_from_digits(
        self,
        model: PLANE,
        X_digit: torch.Tensor,
        *,
        n_gauss: Optional[int] = None,
        n_shuffle: Optional[int] = None,
        seed: int = 0,
    ) -> "MondrianCalibrator":
        """Convenience: build default noise groups from digits, then :meth:`fit`."""
        groups = make_mondrian_groups(
            X_digit, n_gauss=n_gauss, n_shuffle=n_shuffle, seed=seed
        )
        return self.fit(model, groups)

    def group_names(self) -> Tuple[str, ...]:
        return tuple(self.s_calib.keys())

    def _check(self, model: Optional[PLANE] = None) -> None:
        if not self.s_calib:
            raise RuntimeError("MondrianCalibrator.fit has not been called")
        if model is not None:
            if self.weight_hash is None:
                raise RuntimeError("MondrianCalibrator.fit has not been called")
            h = model_weight_hash(model)
            if h != self.weight_hash:
                raise RuntimeError(
                    "model weights do not match calibration hash — recalibrate"
                )

    def p_value(
        self,
        scores: torch.Tensor,
        group: str,
        model: Optional[PLANE] = None,
        *,
        sided: str = "upper",
    ) -> torch.Tensor:
        """Category-conditional p-value under Mondrian group ``group``.

        ``sided="upper"`` (default)
            ``p = (1 + #{s_calib ≥ s}) / (n+1)``. Small ``p`` ⇒ score is in the
            upper tail of that group (classical OOD / "more nonconforming").
            Use with :meth:`threshold` / :meth:`levels`.

        ``sided="two"``
            Two-sided rank p-value
            ``p = min(1, 2 · min(p_lo, p_hi))``. Prefer this for
            :meth:`prediction_set` when groups sit at different score levels
            (digits vs noise): a typical digit is *not* in the upper tail of
            the gauss pool, but it is also not typical *of* gauss.
        """
        self._check(model)
        if group not in self.s_calib:
            raise KeyError(f"unknown group {group!r}; have {list(self.s_calib)}")
        if sided not in ("upper", "two"):
            raise ValueError("sided must be 'upper' or 'two'")
        s_c = self.s_calib[group]
        n = s_c.numel()
        s = scores.detach().cpu().float()
        # right=False → index of first s_c >= s; #{s_c >= s} = n - idx
        idx_lo = torch.searchsorted(s_c, s, right=False)
        idx_hi = torch.searchsorted(s_c, s, right=True)
        count_ge = n - idx_lo
        count_le = idx_hi
        p_hi = (1 + count_ge.float()) / (n + 1)
        if sided == "upper":
            return p_hi.to(scores.device)
        p_lo = (1 + count_le.float()) / (n + 1)
        p = torch.minimum(p_lo, p_hi).mul(2).clamp_max(1.0)
        return p.to(scores.device)

    def p_values(
        self,
        scores: torch.Tensor,
        model: Optional[PLANE] = None,
        *,
        sided: str = "upper",
    ) -> Dict[str, torch.Tensor]:
        """p-value under every calibrated Mondrian group."""
        return {
            g: self.p_value(scores, g, model=model, sided=sided) for g in self.s_calib
        }

    def threshold(self, group: str, alpha: float = 0.05) -> float:
        """Upper-tailed conformal score threshold for ``group`` at level ``alpha``.

        Reject "looks like ``group`` under an upper-tailed score" when
        ``score > threshold``. This is the Mondrian *level* used for OOD
        gating within a category.
        """
        self._check()
        if group not in self.s_calib:
            raise KeyError(f"unknown group {group!r}; have {list(self.s_calib)}")
        return conformal_threshold(self.s_calib[group], alpha)

    def levels(
        self,
        alphas: Sequence[float] = (0.01, 0.05, 0.1),
    ) -> Dict[str, Dict[float, float]]:
        """``{group: {alpha: threshold}}`` — Mondrian calibration levels."""
        self._check()
        return {
            g: {float(a): self.threshold(g, float(a)) for a in alphas}
            for g in self.s_calib
        }

    def prediction_set(
        self,
        scores: torch.Tensor,
        alpha: float = 0.05,
        model: Optional[PLANE] = None,
        *,
        sided: str = "two",
    ) -> list:
        """Per-point Mondrian prediction sets ``{g : p_g > α}``.

        Defaults to two-sided p-values so categories at different score
        levels (digit vs noise) separate. Pass ``sided="upper"`` for
        classical upper-tailed OOD sets.

        Returns a list of length ``B``; each entry is a tuple of accepted
        group names (empty ⇒ rejected by every calibrated category).
        """
        pv = self.p_values(scores, model=model, sided=sided)
        groups = list(pv.keys())
        B = int(scores.reshape(-1).numel())
        out = []
        for i in range(B):
            accepted = tuple(g for g in groups if float(pv[g][i]) > float(alpha))
            out.append(accepted)
        return out

    def state_dict(self) -> dict:
        """CPU tensors + metadata for ``torch.save``."""
        out = {
            "score_name": self.score_name,
            "s_calib": {g: s.cpu() for g, s in self.s_calib.items()},
            "weight_hash": self.weight_hash,
            "tau_embed": self.tau_embed,
            "cover_scale": self.cover_scale,
            "ent_scale": self.ent_scale,
            "batch_size": self.batch_size,
        }
        if isinstance(self._score_fn, CoverEntropyLDA):
            out["lda"] = self._score_fn.state_dict()
        return out

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "MondrianCalibrator":
        """Restore a calibrator saved by :meth:`state_dict`."""
        score_name = str(state["score_name"])
        if score_name == "lda" or "lda" in state:
            score: ScoreSpec = CoverEntropyLDA.from_state_dict(state["lda"])
        else:
            score = score_name
        cal = cls(score=score, batch_size=int(state.get("batch_size", 1024)))
        cal.s_calib = {
            str(g): torch.as_tensor(s).float().cpu()
            for g, s in dict(state["s_calib"]).items()
        }
        cal.weight_hash = state.get("weight_hash")
        cal.tau_embed = state.get("tau_embed")
        cal.cover_scale = float(state.get("cover_scale", 1.0))
        cal.ent_scale = float(state.get("ent_scale", 1.0))
        return cal

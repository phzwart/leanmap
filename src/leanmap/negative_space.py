"""Amortized, calibrated distance-to-manifold from the encoder's internal state.

Idea (the "negative space" map): freeze a trained :class:`~leanmap.model.PLANE`,
push *perturbed* points through it, capture internal states (landmark distances
and affinities, FiLM ``gamma``/``beta``, the ``gamma``-clamp hit rate, the
post-activation hidden layers, and the embedding), and fit a **quantile
regression head** on those features that predicts a distance-to-support with
lower / median / upper bounds.

Design choices (fixed):

* **Target** ``y`` is the *empirical distance-to-support*
  ``min_j ||x_tilde - x_j||`` over the training set — the honest
  distance-to-manifold, cheap via :func:`~leanmap.distance.chunked_cdist`.
* **No quantile crossing.** The head predicts a median plus two non-negative
  offsets (``softplus``), so ``q_lo <= q_med <= q_hi`` by construction. The
  median is detached in the offset (pinball) terms by default, so it is learned
  purely by its own 0.5-pinball loss and the offsets only shape the width.
* **CQR.** After fitting, conformalized quantile regression widens the interval
  by the ``(1 - alpha)`` quantile of the calibration conformity scores
  ``E = max(q_lo - y, y - q_hi)``, giving finite-sample marginal coverage.

The result reuses the frozen network as a learned probe; inference is a single
forward pass and needs no training set or landmark cloud beyond that pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distance import DistanceFn, EuclideanDistance, chunked_cdist
from .model import PLANE
from .utils import get_logger

# Per-point internal-state feature groups the extractor can emit.
# NOTE: ``hit`` (the gamma-clamp fraction) is intentionally NOT here — it is a
# per-*batch* scalar, degenerate as a per-point feature (zero within-batch
# variance breaks standardization), and redundant with per-point ``gamma``.
# It remains available for explicit requests / diagnostics.
ALL_FEATURES: Tuple[str, ...] = (
    "dm_min",  # (1,)   min landmark distance — the current cover score
    "dm",      # (L,)   raw distances to primary anchors
    "a",       # (L,)   primary anchor affinity (attention over landmarks)
    "gamma",   # (depth*width,) FiLM scale (clamped gammas flag extrapolation)
    "beta",    # (depth*width,) FiLM shift
    "hidden",  # (depth*width,) post-activation backbone states
    "z",       # (d_out,) embedding
)
# Ablation baseline: only the ambient landmark-distance signal.
DM_ONLY_FEATURES: Tuple[str, ...] = ("dm_min", "dm")


def _primary_name(model: PLANE) -> str:
    if model.factors is not None:
        return model.factors.primary_factor.name
    return "primary"


class _HiddenCapture:
    """Forward-hook context manager capturing post-activation backbone states.

    ``FiLMEncoder`` applies ``gelu`` functionally after the FiLM modulation; we
    hook the parameter-free taps that sit at exactly that point (post-FiLM,
    pre-gelu) and apply ``gelu`` when assembling features, so the captured value
    matches the true hidden activation. Hooking the ``LayerNorm`` instead would
    miss ``gamma``/``beta``, which are applied after it.
    """

    def __init__(self, model: PLANE):
        self.norms = list(getattr(model.encoder, "taps", model.encoder.norms))
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._store: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "_HiddenCapture":
        for i, mod in enumerate(self.norms):
            self._handles.append(mod.register_forward_hook(self._make_hook(i)))
        return self

    def _make_hook(self, i: int):
        def hook(_module, _inp, out):
            self._store[i] = out
        return hook

    def stacked(self) -> torch.Tensor:
        """Concatenate ``gelu(h_k)`` across layers → (B, depth*width)."""
        parts = [F.gelu(self._store[i]) for i in range(len(self.norms))]
        return torch.cat(parts, dim=1)

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._store.clear()


def _assemble_features(
    model: PLANE,
    x: torch.Tensor,
    feature_groups: Sequence[str],
) -> torch.Tensor:
    """Core feature assembly on ``x.device`` (grad follows the caller's context)."""
    name = _primary_name(model)
    with _HiddenCapture(model) as cap:
        z, a_map, dm_map, gamma, beta, _g_by, hit = model.forward_detailed(x)
        hidden = cap.stacked() if "hidden" in feature_groups else None

    a = a_map[name]
    dm = dm_map[name]
    B = x.shape[0]
    comps: Dict[str, torch.Tensor] = {
        "dm_min": dm.min(dim=1, keepdim=True).values,
        "dm": dm,
        "a": a,
        "gamma": gamma.reshape(B, -1),
        "beta": beta.reshape(B, -1),
        "hit": torch.full((B, 1), float(hit), device=x.device, dtype=z.dtype),
        "z": z,
    }
    if hidden is not None:
        comps["hidden"] = hidden

    missing = [g for g in feature_groups if g not in comps]
    if missing:
        raise ValueError(f"unknown feature groups: {missing}")
    return torch.cat([comps[g] for g in feature_groups], dim=1).float()


@torch.no_grad()
def extract_features(
    model: PLANE,
    x: torch.Tensor,
    feature_groups: Sequence[str] = ALL_FEATURES,
) -> torch.Tensor:
    """Concatenate the selected internal-state groups (frozen, no grad → CPU).

    Parameters
    ----------
    model : PLANE (frozen)
    x : (B, D) float32
    feature_groups : which of :data:`ALL_FEATURES` to include, in order.

    Returns
    -------
    phi : (B, F) float32 (on CPU)
    """
    model.eval()
    device = next(model.parameters()).device
    return _assemble_features(model, x.to(device), feature_groups).cpu()


@torch.no_grad()
def distance_to_support(
    X_query: torch.Tensor,
    X_support: torch.Tensor,
    dist_fn: Optional[DistanceFn] = None,
    chunk_a: int = 4096,
) -> torch.Tensor:
    """Empirical distance-to-support ``min_j d(x_i, X_support_j)`` → (n,)."""
    dist_fn = dist_fn if dist_fn is not None else EuclideanDistance()
    vals, _ = chunked_cdist(
        dist_fn, X_query, X_support, chunk_a=chunk_a, topk=1, out_device="cpu"
    )
    return vals[:, 0]


@dataclass
class PerturbationConfig:
    """Controls the perturb-and-label sampling that builds the training set."""

    n_base: int = 4000          # base points drawn (with replacement) from train
    radii_per_base: int = 6     # shells sampled per base point
    r_min_mult: float = 0.25    # min shell radius, in units of median 1-NN dist
    r_max_mult: float = 25.0    # max shell radius, in units of median 1-NN dist
    include_onmanifold: bool = True  # also emit r=0 base points (y≈0)
    n_uniform_far: int = 2000   # extra uniform samples in a padded data box
    far_box_pad: float = 0.5    # box padding as a fraction of per-dim range
    seed: int = 0


def _median_nn_scale(X: torch.Tensor, dist_fn: DistanceFn, n_sample: int = 4096) -> float:
    n = X.shape[0]
    idx = torch.randperm(n)[: min(n_sample, n)]
    vals, _ = chunked_cdist(dist_fn, X[idx], X, topk=2, out_device="cpu")
    nn1 = vals[:, 1]
    return float(nn1.median().clamp_min(1e-9).item())


def sample_perturbations(
    X_train: torch.Tensor,
    cfg: Optional[PerturbationConfig] = None,
    dist_fn: Optional[DistanceFn] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perturb train points at multiple radii; label by distance-to-support.

    Distribution: (optional) on-manifold base points, ``radii_per_base``
    log-spaced isotropic shells, and (optional) uniform far-box coverage. Any
    two draws with the same ``cfg`` (different seeds) are exchangeable — use a
    fresh seed for a valid CQR holdout.

    Returns
    -------
    X_pert : (M, D) float32 perturbed points
    y : (M,) float32 empirical distance-to-support targets
    """
    cfg = cfg or PerturbationConfig()
    dist_fn = dist_fn if dist_fn is not None else EuclideanDistance()
    X_train = X_train.float()
    n, D = X_train.shape
    g = torch.Generator().manual_seed(cfg.seed)

    nn_scale = _median_nn_scale(X_train, dist_fn)
    radii = torch.logspace(
        np.log10(cfg.r_min_mult * nn_scale),
        np.log10(cfg.r_max_mult * nn_scale),
        cfg.radii_per_base,
    )

    base = X_train[torch.randint(0, n, (cfg.n_base,), generator=g)]
    samples: List[torch.Tensor] = []
    if cfg.include_onmanifold:
        samples.append(base.clone())
    for r in radii.tolist():
        dirs = torch.randn(cfg.n_base, D, generator=g)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        samples.append(base + r * dirs)
    if cfg.n_uniform_far > 0:
        lo = X_train.min(dim=0).values
        hi = X_train.max(dim=0).values
        rng = (hi - lo).clamp_min(1e-9)
        lo_pad = lo - cfg.far_box_pad * rng
        hi_pad = hi + cfg.far_box_pad * rng
        u = torch.rand(cfg.n_uniform_far, D, generator=g)
        samples.append(lo_pad + u * (hi_pad - lo_pad))

    X_pert = torch.cat(samples, dim=0)
    y = distance_to_support(X_pert, X_train, dist_fn=dist_fn)
    return X_pert, y


def build_labeled_set(
    model: PLANE,
    X_train: torch.Tensor,
    feature_groups: Sequence[str] = ALL_FEATURES,
    cfg: Optional[PerturbationConfig] = None,
    dist_fn: Optional[DistanceFn] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample perturbations and extract frozen features + distance labels.

    Returns
    -------
    phi : (M, F) float32 features
    y : (M,) float32 empirical distance-to-support targets
    """
    X_pert, y = sample_perturbations(X_train, cfg=cfg, dist_fn=dist_fn)
    phi = extract_features(model, X_pert, feature_groups=feature_groups)
    return phi, y


class DistanceQuantileHead(nn.Module):
    """Median + two non-negative offsets → non-crossing (lo, med, hi).

    ``q_med = softplus(trunk_med)`` (distances are non-negative);
    ``q_lo = clamp_min(q_med - softplus(off_lo), 0)``;
    ``q_hi = q_med + softplus(off_hi)``.
    """

    def __init__(self, in_dim: int, width: int = 128, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        self.trunk = nn.Sequential(*layers)
        self.med = nn.Linear(d, 1)
        self.off_lo = nn.Linear(d, 1)
        self.off_hi = nn.Linear(d, 1)

    def forward(
        self, phi: torch.Tensor, detach_median: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_lo, q_med, q_hi)``.

        When ``detach_median``, the median is detached in the offset branches so
        the offset (interval-width) grads do not flow into the median value —
        the median is then shaped only by its own 0.5-pinball term.
        """
        h = self.trunk(phi)
        q_med = F.softplus(self.med(h)).squeeze(1)
        base = q_med.detach() if detach_median else q_med
        d_lo = F.softplus(self.off_lo(h)).squeeze(1)
        d_hi = F.softplus(self.off_hi(h)).squeeze(1)
        q_lo = (base - d_lo).clamp_min(0.0)
        q_hi = base + d_hi
        return q_lo, q_med, q_hi


def pinball_loss(y: torch.Tensor, yhat: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile (pinball) loss at level ``tau``."""
    e = y - yhat
    return torch.maximum(tau * e, (tau - 1.0) * e).mean()


def _cqr_offset(lo: torch.Tensor, hi: torch.Tensor, y: torch.Tensor, alpha: float) -> float:
    """CQR conformity offset: (1-alpha)(1+1/n) quantile of ``max(lo-y, y-hi)``.

    May be negative (tightens the interval when the base quantiles over-cover).
    """
    E = torch.maximum(lo - y, y - hi)
    n = int(E.numel())
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(1, n)))
    return float(torch.quantile(E, level).item())


@dataclass
class NegativeSpaceModel:
    """Frozen probe: feature spec + standardizer + quantile head + CQR offset.

    Predicts a calibrated distance-to-support interval from a single forward
    pass of the underlying :class:`PLANE`.
    """

    model: PLANE
    head: DistanceQuantileHead
    feature_groups: Tuple[str, ...]
    feat_mean: torch.Tensor
    feat_std: torch.Tensor
    alpha: float = 0.1
    cqr_offset: float = 0.0
    dist_fn: DistanceFn = field(default_factory=EuclideanDistance)

    def _standardize(self, phi: torch.Tensor) -> torch.Tensor:
        return (phi - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def predict(
        self, X: torch.Tensor, batch_size: int = 8192, apply_cqr: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Return ``{lo, med, hi}`` distance-to-support estimates for ``X``.

        ``lo``/``hi`` are the CQR-widened bounds when ``apply_cqr`` (default).
        """
        self.head.eval()
        X = torch.as_tensor(X).float()
        los, meds, his = [], [], []
        q = self.cqr_offset if apply_cqr else 0.0
        for s in range(0, X.shape[0], batch_size):
            xb = X[s : s + batch_size]
            phi = extract_features(self.model, xb, self.feature_groups)
            phi = self._standardize(phi)
            lo, med, hi = self.head(phi)
            # Median-bracketed CQR adjustment: widen (q>0) or tighten (q<0) but
            # always keep lo <= med <= hi, so tightening can never collapse or
            # invert the interval.
            lo_adj = torch.minimum((lo - q).clamp_min(0.0), med)
            hi_adj = torch.maximum(hi + q, med)
            los.append(lo_adj)
            meds.append(med)
            his.append(hi_adj)
        return {
            "lo": torch.cat(los),
            "med": torch.cat(meds),
            "hi": torch.cat(his),
        }

    @torch.no_grad()
    def score(self, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        """Point estimate of distance-to-manifold: the (raw) median head output.

        This is a pointwise function of the input's internal states — it does
        not depend on any assumed test distribution, so it transfers across
        covariate shift (unlike the CQR *interval*). It is the nonconformity
        score used by :class:`NoveltyDetector`.
        """
        return self.predict(X, batch_size=batch_size, apply_cqr=False)["med"]


def _coverage(y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> float:
    return float(((y >= lo) & (y <= hi)).float().mean().item())


@dataclass
class NoveltyDetector:
    """One-class conformal novelty test calibrated on positives (the null).

    We never model the negatives (their distribution is unknowable). Instead we
    calibrate the *null*: the score distribution of held-out on-manifold points.
    For a query the conformal p-value is

        p(x) = (1 + #{cal : s_cal >= s(x)}) / (n_cal + 1),

    with ``s = ns.score`` the median distance-to-manifold. Under exchangeability
    (the query is on-manifold), ``p`` is super-uniform, so flagging ``p <= alpha``
    controls the on-manifold false-alarm rate at ``alpha`` — finite-sample,
    distribution-free, and *independent of whatever negatives arrive*.
    """

    ns: NegativeSpaceModel
    cal_scores: torch.Tensor  # ascending-sorted on-manifold null scores

    @torch.no_grad()
    def pvalue(self, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        s = self.ns.score(X, batch_size=batch_size)
        n = int(self.cal_scores.numel())
        # #{cal >= s} = n - #{cal < s}; ties count toward >= (conservative).
        idx = torch.searchsorted(self.cal_scores, s.contiguous(), right=False)
        ge = n - idx
        return (1.0 + ge.to(torch.float64)) / (n + 1.0)

    @torch.no_grad()
    def is_novel(
        self, X: torch.Tensor, alpha: Optional[float] = None, batch_size: int = 8192
    ) -> torch.Tensor:
        alpha = self.ns.alpha if alpha is None else alpha
        return self.pvalue(X, batch_size=batch_size) <= alpha


def calibrate_novelty(
    ns: NegativeSpaceModel,
    X_on_manifold: torch.Tensor,
    batch_size: int = 8192,
) -> NoveltyDetector:
    """Build a :class:`NoveltyDetector` from held-out on-manifold points.

    ``X_on_manifold`` must be genuine in-distribution samples (e.g. a held-out
    slice of the training manifold), NOT perturbations — they define the null.
    """
    s = ns.score(X_on_manifold, batch_size=batch_size)
    return NoveltyDetector(ns=ns, cal_scores=torch.sort(s).values)


def fit_negative_space(
    model: PLANE,
    X_train: torch.Tensor,
    feature_groups: Sequence[str] = ALL_FEATURES,
    *,
    alpha: float = 0.1,
    perturb: Optional[PerturbationConfig] = None,
    dist_fn: Optional[DistanceFn] = None,
    head_width: int = 128,
    head_depth: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 2048,
    calib_frac: float = 0.3,
    detach_median: bool = True,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[NegativeSpaceModel, Dict[str, float]]:
    """Fit the quantile head on frozen-encoder features + conformalize (CQR).

    Parameters
    ----------
    model : PLANE (frozen)
    X_train : (N, D) — same support the model was trained on.
    feature_groups : internal-state groups to probe (:data:`ALL_FEATURES` or
        :data:`DM_ONLY_FEATURES` for the ablation baseline).
    alpha : miscoverage level; the CQR interval targets ``1 - alpha`` coverage.
    detach_median : learn the median only from its own 0.5-pinball term.

    Returns
    -------
    ns_model : NegativeSpaceModel
    stats : dict with calibration coverage / median interval width.
    """
    log = get_logger()
    dist_fn = dist_fn if dist_fn is not None else EuclideanDistance()
    feature_groups = tuple(feature_groups)
    X_train = torch.as_tensor(X_train).float()
    torch.manual_seed(seed)

    phi, y = build_labeled_set(
        model, X_train, feature_groups=feature_groups, cfg=perturb, dist_fn=dist_fn
    )
    if verbose:
        log.info(
            "negative-space set: M=%d features=%d groups=%s y[med=%.4g max=%.4g]",
            phi.shape[0], phi.shape[1], list(feature_groups),
            float(y.median()), float(y.max()),
        )

    # Split: head-train vs conformal calibration.
    m = phi.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(m, generator=g)
    n_cal = max(1, int(calib_frac * m))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]

    feat_mean = phi[tr_idx].mean(dim=0)
    feat_std = phi[tr_idx].std(dim=0).clamp_min(1e-6)
    phi_n = (phi - feat_mean) / feat_std

    dev = torch.device(device) if device is not None else torch.device("cpu")
    head = DistanceQuantileHead(phi.shape[1], width=head_width, depth=head_depth).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)

    tau_lo, tau_hi = alpha / 2.0, 1.0 - alpha / 2.0
    Xtr = phi_n[tr_idx].to(dev)
    ytr = y[tr_idx].to(dev)
    n_tr = Xtr.shape[0]

    head.train()
    for ep in range(epochs):
        order = torch.randperm(n_tr, generator=g)
        tot = 0.0
        for s in range(0, n_tr, batch_size):
            b = order[s : s + batch_size]
            xb, yb = Xtr[b], ytr[b]
            lo, med, hi = head(xb, detach_median=detach_median)
            loss = (
                pinball_loss(yb, med, 0.5)
                + pinball_loss(yb, lo, tau_lo)
                + pinball_loss(yb, hi, tau_hi)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            log.info("  head epoch %d/%d pinball=%.4f", ep + 1, epochs, tot)

    # CQR: widen by the (1-alpha) quantile of calibration conformity scores.
    head.eval()
    with torch.no_grad():
        lo_c, med_c, hi_c = head(phi_n[cal_idx].to(dev))
        lo_c, med_c, hi_c = lo_c.cpu(), med_c.cpu(), hi_c.cpu()
        y_c = y[cal_idx]
        cqr_offset = _cqr_offset(lo_c, hi_c, y_c, alpha)

    ns_model = NegativeSpaceModel(
        model=model,
        head=head.to(dev),
        feature_groups=feature_groups,
        feat_mean=feat_mean,
        feat_std=feat_std,
        alpha=alpha,
        cqr_offset=cqr_offset,
        dist_fn=dist_fn,
    )

    # Calibration-set diagnostics (coverage before/after CQR + interval width).
    with torch.no_grad():
        cov_raw = _coverage(y_c, lo_c, hi_c)
        lo_adj = torch.minimum((lo_c - cqr_offset).clamp_min(0.0), med_c)
        hi_adj = torch.maximum(hi_c + cqr_offset, med_c)
        cov_cqr = _coverage(y_c, lo_adj, hi_adj)
        width_cqr = float((hi_adj - lo_adj).median())
    stats = {
        "n_calib": int(n_cal),
        "cqr_offset": cqr_offset,
        "coverage_raw": cov_raw,
        "coverage_cqr": cov_cqr,
        "target_coverage": 1.0 - alpha,
        "median_interval_width": width_cqr,
    }
    if verbose:
        log.info(
            "CQR: offset=%.4g coverage raw=%.3f -> cqr=%.3f (target=%.3f) "
            "median width=%.4g",
            cqr_offset, cov_raw, cov_cqr, 1.0 - alpha, width_cqr,
        )
    return ns_model, stats

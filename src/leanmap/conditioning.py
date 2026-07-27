"""Factored conditioning: ConditioningFactor, Role, FactorStack, helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distance import DistanceFn, EuclideanDistance, chunked_cdist
from .landmarks import AnchorAffinity, init_anchors
from .utils import get_logger

ViewFn = Callable[[torch.Tensor], torch.Tensor]

GAMMA_MIN = 0.1
GAMMA_MAX = 10.0
RETENTION_CHANCE = 0.475
RETENTION_WARN = 0.55


class Role(str, Enum):
    PRIMARY = "primary"
    MODULATOR = "modulator"
    GAIN = "gain"
    AXIS = "axis"


@dataclass(frozen=True)
class ConditioningFactor:
    """One independent conditioning factor (§2).

    ``metric_weight`` is independent of ``role`` — a factor may condition only,
    score only, both, or neither (diagnostics).
    """

    name: str
    view: ViewFn
    metric: DistanceFn
    n_anchors: int
    role: Role
    metric_weight: Optional[float] = None
    learn_anchors: bool = True
    learn_temperature: bool = True
    axis: Optional[int] = None
    view_grad_from_geom: bool = False


def identity_view(x: torch.Tensor) -> torch.Tensor:
    return x


def validate_factors(factors: Sequence[ConditioningFactor]) -> None:
    """Emit warnings for malformed factor lists."""
    n_primary = sum(1 for f in factors if f.role == Role.PRIMARY)
    if n_primary != 1:
        warnings.warn(
            f"exactly one PRIMARY factor is recommended; got {n_primary}",
            UserWarning,
            stacklevel=2,
        )
    for f in factors:
        if f.metric_weight is not None and float(f.metric_weight) == 0.0:
            warnings.warn(
                f"factor {f.name!r} has metric_weight=0.0 — declares a metric "
                "block with no influence (usually a mistake; use None to omit)",
                UserWarning,
                stacklevel=2,
            )
        if f.role == Role.AXIS and f.axis is None:
            warnings.warn(
                f"AXIS factor {f.name!r} requires axis= to be set",
                UserWarning,
                stacklevel=2,
            )


class Monotone1D(nn.Module):
    """Small monotone scalar MLP (non-negative weights + softplus output)."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, width, bias=True)
        self.fc2 = nn.Linear(width, width, bias=True)
        self.fc3 = nn.Linear(width, 1, bias=True)
        # Start near identity-ish: small weights
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, std=1e-3)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, 1) -> (B, 1) non-decreasing in v."""
        if v.ndim == 1:
            v = v.unsqueeze(1)
        h = F.softplus(v @ F.softplus(self.fc1.weight).T + self.fc1.bias)
        h = F.softplus(h @ F.softplus(self.fc2.weight).T + self.fc2.bias)
        return h @ F.softplus(self.fc3.weight).T + self.fc3.bias


class FactorHyper(nn.Module):
    """Per-factor FiLM hypernetwork sized by role."""

    def __init__(
        self,
        role: Role,
        L: int,
        width: int,
        depth: int,
        hyper_width: int = 128,
    ):
        super().__init__()
        self.role = role
        self.L = L
        self.width = width
        self.depth = depth
        if role == Role.AXIS:
            self.hyper = None
            return
        if role == Role.PRIMARY:
            out = 2 * width * depth
        elif role == Role.MODULATOR:
            # gamma (depth,) + beta (depth * width)
            out = depth + depth * width
        elif role == Role.GAIN:
            out = depth  # gamma only, shape (B, depth, 1)
        else:
            raise ValueError(role)
        self.hyper = nn.Sequential(
            nn.Linear(L, hyper_width),
            nn.GELU(),
            nn.Linear(hyper_width, out),
        )
        last = self.hyper[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self, a: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """a: (B, L). Returns gamma_f, beta_f (broadcastable to (B, depth, width))."""
        if self.role == Role.AXIS or self.hyper is None:
            return None, None
        B = a.shape[0]
        raw = self.hyper(a)
        if self.role == Role.PRIMARY:
            raw = raw.view(B, self.depth, 2, self.width)
            gamma_raw, beta = raw[:, :, 0, :], raw[:, :, 1, :]
            return 1.0 + gamma_raw, beta
        if self.role == Role.MODULATOR:
            g = raw[:, : self.depth].view(B, self.depth, 1)
            b = raw[:, self.depth :].view(B, self.depth, self.width)
            return 1.0 + g, b
        # GAIN
        g = raw.view(B, self.depth, 1)
        return 1.0 + g, None


class FactorStack(nn.Module):
    """Owns per-factor anchors, hypers, and FiLM composition."""

    def __init__(
        self,
        factors: Sequence[ConditioningFactor],
        affinities: Sequence[AnchorAffinity],
        width: int,
        depth: int,
        hyper_width: int = 128,
        gamma_min: float = GAMMA_MIN,
        gamma_max: float = GAMMA_MAX,
        d_out: int = 2,
    ):
        super().__init__()
        if len(factors) != len(affinities):
            raise ValueError("factors and affinities length mismatch")
        validate_factors(factors)
        self.factor_defs: List[ConditioningFactor] = list(factors)
        self.affinities = nn.ModuleList(list(affinities))
        self.width = width
        self.depth = depth
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.d_out = int(d_out)

        hypers: List[nn.Module] = []
        axes: List[nn.Module] = []
        for f, aff in zip(self.factor_defs, self.affinities):
            L = aff.M.shape[0]
            if f.role == Role.AXIS:
                hypers.append(nn.Identity())
                axes.append(Monotone1D())
            else:
                hypers.append(
                    FactorHyper(f.role, L, width, depth, hyper_width=hyper_width)
                )
                axes.append(nn.Identity())
        self.hypers = nn.ModuleList(hypers)
        self.axis_mlps = nn.ModuleList(axes)

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.factor_defs]

    @property
    def primary_index(self) -> int:
        for i, f in enumerate(self.factor_defs):
            if f.role == Role.PRIMARY:
                return i
        return 0

    @property
    def primary_affinity(self) -> AnchorAffinity:
        return self.affinities[self.primary_index]  # type: ignore[return-value]

    @property
    def primary_factor(self) -> ConditioningFactor:
        return self.factor_defs[self.primary_index]

    def view_batch(
        self, x: torch.Tensor, for_geom: bool = True
    ) -> List[torch.Tensor]:
        views = []
        for f in self.factor_defs:
            v = f.view(x)
            if for_geom and not f.view_grad_from_geom:
                v = v.detach()
            views.append(v)
        return views

    def affinities_forward(
        self, x: torch.Tensor, for_geom: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[torch.Tensor]]:
        views = self.view_batch(x, for_geom=for_geom)
        a_map: Dict[str, torch.Tensor] = {}
        dm_map: Dict[str, torch.Tensor] = {}
        a_list: List[torch.Tensor] = []
        for f, aff, v in zip(self.factor_defs, self.affinities, views):
            # Anchors always see the view; detach only for geom path on view params
            a, dm = aff(v)
            a_map[f.name] = a
            dm_map[f.name] = dm
            a_list.append(a)
        return a_map, dm_map, a_list

    def film_params_from_affinities(
        self, a_list: Sequence[torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
        float,
    ]:
        """Compose gamma/beta. Returns gamma, beta, gamma_by_name, beta_by_name, clamp_hit_rate."""
        B = a_list[0].shape[0]
        device = a_list[0].device
        dtype = a_list[0].dtype
        gamma = torch.ones(B, self.depth, self.width, device=device, dtype=dtype)
        beta = torch.zeros(B, self.depth, self.width, device=device, dtype=dtype)
        gamma_by: Dict[str, torch.Tensor] = {}
        beta_by: Dict[str, Optional[torch.Tensor]] = {}
        for f, a, hyp in zip(self.factor_defs, a_list, self.hypers):
            if f.role == Role.AXIS:
                continue
            assert isinstance(hyp, FactorHyper)
            g_f, b_f = hyp(a)
            assert g_f is not None
            gamma_by[f.name] = g_f
            beta_by[f.name] = b_f
            gamma = gamma * g_f
            if b_f is not None:
                beta = beta + b_f
        before = gamma
        gamma = gamma.clamp(self.gamma_min, self.gamma_max)
        hit = float(((before <= self.gamma_min) | (before >= self.gamma_max)).float().mean().item())
        return gamma, beta, gamma_by, beta_by, hit

    def apply_axis_skips(
        self, z: torch.Tensor, x: torch.Tensor, for_geom: bool = True
    ) -> torch.Tensor:
        views = self.view_batch(x, for_geom=for_geom)
        out = z
        for f, mlp, v in zip(self.factor_defs, self.axis_mlps, views):
            if f.role != Role.AXIS or f.axis is None:
                continue
            assert isinstance(mlp, Monotone1D)
            m = mlp(v if v.shape[-1] == 1 else v[..., :1])
            out = out.clone()
            out[:, f.axis] = out[:, f.axis] + m.squeeze(-1)
        return out

    def concat_affinity(self, a_list: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(a_list), dim=1)


def default_primary_factor(
    n_anchors: int,
    metric: Optional[DistanceFn] = None,
    name: str = "primary",
) -> ConditioningFactor:
    return ConditioningFactor(
        name=name,
        view=identity_view,
        metric=metric if metric is not None else EuclideanDistance(),
        n_anchors=n_anchors,
        role=Role.PRIMARY,
        metric_weight=None,
    )


def build_factor_stack(
    X: torch.Tensor,
    factors: Sequence[ConditioningFactor],
    width: int,
    depth: int,
    hyper_width: int = 128,
    d_out: int = 2,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> FactorStack:
    """FPS/quantile-init anchors on each view and construct a FactorStack."""
    validate_factors(factors)
    device = device or X.device
    affs: List[AnchorAffinity] = []
    for i, f in enumerate(factors):
        v = f.view(X).to(device)
        M = init_anchors(v, f.metric, f.n_anchors, seed=seed + i)
        affs.append(
            AnchorAffinity(
                M.to(device),
                f.metric,
                learn_anchors=f.learn_anchors,
                learn_tau=f.learn_temperature,
            )
        )
    return FactorStack(
        factors,
        affs,
        width=width,
        depth=depth,
        hyper_width=hyper_width,
        d_out=d_out,
    ).to(device)


def scale_quotient_factorization(
    norm: str = "l1",
    n_shape_anchors: int = 256,
    n_scale_anchors: int = 16,
    shape_metric_weight: float = 1.0,
    scale_metric_weight: float = 0.3,
    scale_role: Role = Role.AXIS,
    scale_axis: int = 0,
    shape_metric: Optional[DistanceFn] = None,
) -> List[ConditioningFactor]:
    """Direction (PRIMARY) + log-magnitude (AXIS/GAIN) group-quotient factors."""
    if norm not in ("l1", "l2"):
        raise ValueError(f"norm must be l1 or l2, got {norm!r}")

    def direction(x: torch.Tensor) -> torch.Tensor:
        if norm == "l1":
            s = x.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            s = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return x / s

    def log_scale(x: torch.Tensor) -> torch.Tensor:
        if norm == "l1":
            s = x.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            s = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.log(s)

    shape_m = shape_metric if shape_metric is not None else EuclideanDistance()
    factors = [
        ConditioningFactor(
            name="shape",
            view=direction,
            metric=shape_m,
            n_anchors=n_shape_anchors,
            role=Role.PRIMARY,
            metric_weight=shape_metric_weight,
        ),
        ConditioningFactor(
            name="log_scale",
            view=log_scale,
            metric=EuclideanDistance(),
            n_anchors=n_scale_anchors,
            role=scale_role,
            metric_weight=scale_metric_weight,
            axis=scale_axis if scale_role == Role.AXIS else None,
        ),
    ]
    return factors


class FactorViewMetric:
    """Composite scoring metric from factors with ``metric_weight`` set.

    ``d(x,y)^2 = sum_f w_f * (d_f(view_f(x), view_f(y)) / scale_f)^2``
    """

    def __init__(self, factors: Sequence[ConditioningFactor]):
        self.blocks = [
            f for f in factors if f.metric_weight is not None
        ]
        self.scales: Dict[str, float] = {f.name: 1.0 for f in self.blocks}
        self.natural_scale: Optional[float] = None
        self.name = "factor_view_metric"
        self.l2_transform = None
        self.l2_exact = False
        self.is_true_metric = all(
            getattr(f.metric, "is_true_metric", True) for f in self.blocks
        )
        self.differentiable = True

    def fit_scales(
        self, X: torch.Tensor, n_sample: int = 10_000, seed: int = 0
    ) -> None:
        n = X.shape[0]
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        take = min(n_sample, n)
        idx = torch.randperm(n, generator=g)[:take]
        Xs = X[idx]
        for f in self.blocks:
            v = f.view(Xs)
            # median 1-NN within block
            if v.shape[0] < 2:
                self.scales[f.name] = 1.0
                continue
            vals, _ = chunked_cdist(f.metric, v, v, topk=2, out_device=v.device)
            nn1 = vals[:, 1]
            med = float(nn1.median().item())
            self.scales[f.name] = med if med > 0 else 1.0

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not self.blocks:
            raise RuntimeError("FactorViewMetric has no scoring blocks")
        acc = None
        for f in self.blocks:
            va, vb = f.view(A), f.view(B)
            d = f.metric(va, vb)
            sc = max(float(self.scales.get(f.name, 1.0)), 1e-12)
            w = float(f.metric_weight)  # type: ignore[arg-type]
            term = w * (d / sc) ** 2
            acc = term if acc is None else acc + term
        assert acc is not None
        d_out = acc.clamp_min(0.0).sqrt()
        if self.natural_scale is not None and self.natural_scale > 0:
            d_out = d_out / float(self.natural_scale)
        return d_out


def metric_from_factors(
    factors: Sequence[ConditioningFactor],
    X: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    seed: int = 0,
) -> Optional[FactorViewMetric]:
    """Build a FactorViewMetric if any factor has metric_weight; else None."""
    if not any(f.metric_weight is not None for f in factors):
        return None
    validate_factors(factors)
    m = FactorViewMetric(factors)
    if X is not None:
        m.fit_scales(X, seed=seed)
        n = X.shape[0]
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        take = min(10_000, n)
        idx = torch.randperm(n, generator=g)[:take]
        Xs = X[idx]
        k = min(n_neighbors + 1, take)
        vals, _ = chunked_cdist(m, Xs, Xs, topk=k, out_device=Xs.device)
        nn_k = vals[:, min(n_neighbors, vals.shape[1] - 1)]
        scale = float(nn_k.median().item())
        m.natural_scale = scale if scale > 0 else 1.0
    return m

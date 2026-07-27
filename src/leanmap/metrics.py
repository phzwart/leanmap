"""Metric registry, capability flags, and L2-reduction transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from .distance import (
    BUILTIN_FNS,
    CallableDistance,
    DistanceFn,
    EuclideanDistance,
    chunked_cdist,
    is_differentiable,
)
from .utils import as_float32_tensor, ensure_2d_float32, get_logger

L2Transform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MetricSpec:
    """Capability-bearing description of a distance metric.

    Attributes
    ----------
    name : str
    fn : DistanceFn
        Exact, batched, authoritative distance.
    l2_transform : Callable | None
        ``x -> x'`` such that ``||x'-y'||_2`` is monotone in ``d(x,y)``.
    l2_exact : bool
        True if the monotone map preserves kNN sets exactly.
    is_true_metric : bool
        Satisfies the triangle inequality.
    differentiable : bool
        Differentiable w.r.t. the second argument.
    natural_scale : float | None
        Median n_neighbors-th NN distance; filled at fit time (see §3.4).
    """

    name: str
    fn: DistanceFn
    l2_transform: Optional[L2Transform]
    l2_exact: bool
    is_true_metric: bool
    differentiable: bool
    natural_scale: Optional[float] = None

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Apply exact ``fn`` then divide by ``natural_scale`` if set.

        A: (n, D) float32. B: (m, D) float32. Returns: (n, m) float32.
        """
        d = self.fn(A, B)
        if self.natural_scale is not None and self.natural_scale > 0:
            d = d / float(self.natural_scale)
        return d


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _center_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(dim=1, keepdim=True)
    return xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _mahalanobis_fn(inv_cov: torch.Tensor) -> DistanceFn:
    # L from chol(Σ^{-1}) so ||L(x-y)|| = mahalanobis
    # Prefer chol of Σ then solve; here we take chol of precision if SPD.
    try:
        L = torch.linalg.cholesky(inv_cov)
    except RuntimeError:
        # Regularise
        eye = torch.eye(inv_cov.shape[0], device=inv_cov.device, dtype=inv_cov.dtype)
        L = torch.linalg.cholesky(inv_cov + 1e-6 * eye)

    def fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        AL = A @ L.T
        BL = B @ L.T
        return EuclideanDistance()(AL, BL)

    return CallableDistance(fn)


def _mahalanobis_transform(inv_cov: torch.Tensor) -> L2Transform:
    try:
        L = torch.linalg.cholesky(inv_cov)
    except RuntimeError:
        eye = torch.eye(inv_cov.shape[0], device=inv_cov.device, dtype=inv_cov.dtype)
        L = torch.linalg.cholesky(inv_cov + 1e-6 * eye)

    def transform(x: torch.Tensor) -> torch.Tensor:
        return x @ L.T

    return transform


def get_metric(
    name: str,
    *,
    cov: Optional[torch.Tensor] = None,
    fn: Optional[DistanceFn] = None,
    l2_transform: Optional[L2Transform] = None,
    l2_exact: bool = False,
    is_true_metric: bool = False,
    differentiable: Optional[bool] = None,
    D: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> MetricSpec:
    """Look up a built-in metric or wrap a custom callable.

    Parameters
    ----------
    name : str
        Registry name, or ``"custom"`` / ``"mahalanobis"``.
    cov : (D, D) tensor, optional
        Covariance for Mahalanobis (Σ); precision = Σ^{-1}.
    fn : DistanceFn, optional
        Required for ``custom``.
    """
    if name == "mahalanobis":
        if cov is None:
            raise ValueError("mahalanobis requires cov=(D,D)")
        inv = torch.linalg.inv(cov.to(dtype=torch.float32))
        return MetricSpec(
            name="mahalanobis",
            fn=_mahalanobis_fn(inv),
            l2_transform=_mahalanobis_transform(inv),
            l2_exact=True,
            is_true_metric=True,
            differentiable=True,
        )
    if name == "custom":
        if fn is None:
            raise ValueError("custom metric requires fn=")
        diff = (
            is_differentiable(fn, D or 8, device)
            if differentiable is None
            else bool(differentiable)
        )
        return MetricSpec(
            name="custom",
            fn=fn,
            l2_transform=l2_transform,
            l2_exact=l2_exact if l2_transform is not None else False,
            is_true_metric=is_true_metric,
            differentiable=diff,
        )

    table: dict[str, MetricSpec] = {
        "l2": MetricSpec("l2", BUILTIN_FNS["l2"], _identity, True, True, True),
        "sqeuclidean": MetricSpec(
            "sqeuclidean", BUILTIN_FNS["sqeuclidean"], _identity, True, False, True
        ),
        "frobenius": MetricSpec(
            "frobenius", BUILTIN_FNS["frobenius"], _identity, True, True, True
        ),
        "cosine": MetricSpec(
            "cosine", BUILTIN_FNS["cosine"], _l2_normalize, True, False, True
        ),
        "correlation": MetricSpec(
            "correlation",
            BUILTIN_FNS["correlation"],
            _center_l2_normalize,
            True,
            False,
            True,
        ),
        "correlation_sqrt": MetricSpec(
            "correlation_sqrt",
            BUILTIN_FNS["correlation_sqrt"],
            _center_l2_normalize,
            True,
            True,
            True,
        ),
        "l1": MetricSpec("l1", BUILTIN_FNS["l1"], None, False, True, True),
        "linf": MetricSpec("linf", BUILTIN_FNS["linf"], None, False, True, True),
        "canberra": MetricSpec(
            "canberra", BUILTIN_FNS["canberra"], None, False, False, True
        ),
        "braycurtis": MetricSpec(
            "braycurtis", BUILTIN_FNS["braycurtis"], None, False, False, True
        ),
        "jensenshannon": MetricSpec(
            "jensenshannon",
            BUILTIN_FNS["jensenshannon"],
            None,
            False,
            True,
            True,
        ),
    }
    if name not in table:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(table)}")
    return table[name]


BlockSpec = Tuple[slice, str, float]


class CompositeMetric:
    """Weighted sum of per-block metrics with median scale normalisation.

    ``d(x,y) = Σ_b w_b · d_b(x_b, y_b) / scale_b``
    """

    def __init__(self, blocks: Sequence[BlockSpec]):
        if not blocks:
            raise ValueError("blocks must be non-empty")
        self.blocks = list(blocks)
        self._block_specs: List[MetricSpec] = [
            get_metric(name) for _, name, _ in self.blocks
        ]
        self.scales: Optional[List[float]] = None  # filled by fit_scales
        self.natural_scale: Optional[float] = None

    @property
    def name(self) -> str:
        return "composite"

    @property
    def l2_exact(self) -> bool:
        # Only if every block is l2_exact and weights fold into diagonal scaling
        if any(s.l2_transform is None or not s.l2_exact for s in self._block_specs):
            return False
        return True

    @property
    def l2_transform(self) -> Optional[L2Transform]:
        if not self.l2_exact:
            return None
        scales = self.scales or [1.0] * len(self.blocks)
        blocks = self.blocks
        specs = self._block_specs

        def transform(x: torch.Tensor) -> torch.Tensor:
            parts = []
            for (sl, _, w), spec, sc in zip(blocks, specs, scales):
                xb = x[:, sl]
                assert spec.l2_transform is not None
                tb = spec.l2_transform(xb)
                factor = (float(w) / max(float(sc), 1e-12)) ** 0.5
                parts.append(tb * factor)
            return torch.cat(parts, dim=1)

        return transform

    @property
    def is_true_metric(self) -> bool:
        return all(s.is_true_metric for s in self._block_specs)

    @property
    def differentiable(self) -> bool:
        return all(s.differentiable for s in self._block_specs)

    def fit_scales(
        self,
        X: torch.Tensor,
        n_sample: int = 10_000,
        seed: int = 0,
    ) -> None:
        """Estimate per-block median pairwise distance on a subsample.

        Parameters
        ----------
        X : (N, D) float32
        n_sample : int
        seed : int
        """
        n = X.shape[0]
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        take = min(n_sample, n)
        idx = torch.randperm(n, generator=g)[:take]
        Xs = X[idx].to(device=X.device)
        scales: List[float] = []
        for (sl, _, _), spec in zip(self.blocks, self._block_specs):
            xb = Xs[:, sl]
            # Pairwise among subsample — use upper triangle sample if large
            d = chunked_cdist(spec.fn, xb, xb, chunk_a=512, chunk_b=512, out_device=xb.device)
            assert isinstance(d, torch.Tensor)
            mask = ~torch.eye(take, dtype=torch.bool, device=d.device)
            vals = d[mask]
            med = float(vals.median().item()) if vals.numel() else 1.0
            scales.append(max(med, 1e-12))
        self.scales = scales

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        scales = self.scales or [1.0] * len(self.blocks)
        total = None
        for (sl, _, w), spec, sc in zip(self.blocks, self._block_specs, scales):
            db = spec.fn(A[:, sl], B[:, sl]) / max(float(sc), 1e-12)
            term = float(w) * db
            total = term if total is None else total + term
        assert total is not None
        if self.natural_scale is not None and self.natural_scale > 0:
            total = total / float(self.natural_scale)
        return total

    def as_metric_spec(self) -> MetricSpec:
        """Expose composite as a ``MetricSpec`` for the rest of the pipeline."""
        return MetricSpec(
            name="composite",
            fn=self,
            l2_transform=self.l2_transform,
            l2_exact=self.l2_exact,
            is_true_metric=self.is_true_metric,
            differentiable=self.differentiable,
            natural_scale=self.natural_scale,
        )


def fit_natural_scale(
    metric: MetricSpec,
    X: torch.Tensor,
    n_neighbors: int = 15,
    n_sample: int = 10_000,
    seed: int = 0,
) -> MetricSpec:
    """Set ``natural_scale`` = median of the n_neighbors-th NN distance.

    Parameters
    ----------
    metric : MetricSpec
        Spec *without* natural_scale applied (raw ``fn``).
    X : (N, D) float32
    n_neighbors : int
    n_sample : int
    seed : int

    Returns
    -------
    MetricSpec
        Copy with ``natural_scale`` filled.
    """
    log = get_logger()
    n = X.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    take = min(n_sample, n)
    idx = torch.randperm(n, generator=g)[:take]
    Xs = X[idx]
    # Use raw fn (bypass natural_scale) — metric may already wrap
    raw_fn = metric.fn
    k = min(n_neighbors + 1, take)  # +1 self
    vals, _ = chunked_cdist(  # type: ignore[misc]
        raw_fn, Xs, Xs, topk=k, out_device=Xs.device
    )
    # Self is typically the nearest; take the n_neighbors-th non-self ≈ col k-1
    # After sorting ascending, col 0 is self (~0); col n_neighbors is the k-th NN
    nn_k = vals[:, min(n_neighbors, vals.shape[1] - 1)]
    scale = float(nn_k.median().item())
    if scale <= 0:
        scale = 1.0
        log.warning("natural_scale estimated as <=0; using 1.0")
    return replace(metric, natural_scale=scale)


def wrap_metric(
    metric: Union[str, MetricSpec, CompositeMetric],
    *,
    X: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    seed: int = 0,
    **kwargs,
) -> MetricSpec:
    """Resolve a metric name/object and optionally fit natural_scale / block scales.

    Parameters
    ----------
    metric : str | MetricSpec | CompositeMetric
    X : (N, D), optional
        If given, fit natural_scale (and composite block scales).
    """
    if isinstance(metric, CompositeMetric):
        if X is not None:
            metric.fit_scales(X, seed=seed)
        spec = metric.as_metric_spec()
        if X is not None:
            fitted = fit_natural_scale(spec, X, n_neighbors=n_neighbors, seed=seed)
            metric.natural_scale = fitted.natural_scale
            return replace(spec, natural_scale=fitted.natural_scale)
        return spec
    if isinstance(metric, str):
        spec = get_metric(metric, **kwargs)
    else:
        spec = metric
    if X is not None:
        return fit_natural_scale(spec, X, n_neighbors=n_neighbors, seed=seed)
    return spec

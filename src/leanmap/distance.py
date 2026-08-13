"""Distance function protocol, built-ins, and chunked evaluation."""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Tuple, Union

import torch

from .utils import chunk_ranges, resolve_device


class DistanceFn(Protocol):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D) float32. B: (m, D) float32.

        Returns: (n, m) float32, non-negative, no NaN/Inf.
        Must be symmetric: d(A,B) == d(B,A).T
        Must satisfy d(a,a) == 0.
        """


class SquaredEuclideanDistance:
    """``‖x−y‖₂²`` via matmul identity (MPS/CUDA-friendly fwd+bwd)."""

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a·b  — avoids torch.cdist (no MPS bwd).
        aa = (A * A).sum(dim=1, keepdim=True)
        bb = (B * B).sum(dim=1, keepdim=True).T
        d2 = aa + bb - 2.0 * (A @ B.T)
        return d2.clamp_min(0.0)


class EuclideanDistance:
    """``‖x−y‖₂`` via matmul identity + sqrt (not ``torch.cdist``).

    Same distances as ``torch.cdist(..., p=2)``, but backward uses GEMM ops that
    MPS implements — important for learnable landmarks in ``LandmarkAffinity``.
    """

    def __init__(self) -> None:
        self._sq = SquaredEuclideanDistance()

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        d2 = self._sq(A, B)
        # Floor before sqrt so coincidence (d2→0) has finite grads — raw
        # sqrt'(0)=∞ and NaNs landmark updates; torch.cdist uses a 0 subgradient.
        d = torch.sqrt(d2 + 1e-8)
        # Keep exact ties at 0 so ε-net / dedup still see d(a,a')==0.
        d = torch.where(d2 <= 0.0, torch.zeros_like(d), d)
        if A is B:
            d = d.clone()
            d.fill_diagonal_(0.0)
        return d


class CosineDistance:
    """``1 − ⟨x,y⟩ / (‖x‖‖y‖)``."""

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        a_norm = A.norm(dim=1, keepdim=True)
        b_norm = B.norm(dim=1, keepdim=True)
        sim = (A / a_norm.clamp_min(1e-12)) @ (B / b_norm.clamp_min(1e-12)).T
        d = (1.0 - sim).clamp_min(0.0)
        # Both near-zero → define d=0 (d(a,a)==0 on the origin).
        both_zero = (a_norm < 1e-12) & (b_norm.T < 1e-12)
        return torch.where(both_zero, torch.zeros_like(d), d)


class ManhattanDistance:
    """``‖x−y‖₁``.

    ``torch.cdist(..., p=1)`` materialises an ``(n, m, D)`` workspace, which
    blows up for landmark batches (e.g. 10k × 1024 × 4096). Chunk over A.
    """

    _max_workspace = 64 * 1024 * 1024  # bytes

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        n, d = A.shape
        m = int(B.shape[0])
        n_chunk = max(1, min(n, self._max_workspace // max(m * d * 4, 1)))
        if n_chunk >= n:
            return torch.cdist(A, B, p=1)
        parts = [
            torch.cdist(A[s:e], B, p=1) for s, e in chunk_ranges(n, n_chunk)
        ]
        return torch.cat(parts, dim=0)


class CallableDistance:
    """Wrap a user callable as a ``DistanceFn``."""

    def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.fn = fn

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """A: (n, D). B: (m, D). Returns: (n, m) float32."""
        out = self.fn(A, B)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=torch.float32, device=A.device)
        return out.to(dtype=torch.float32)


def _linf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.cdist(A, B, p=float("inf"))


def _canberra(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Σ |x_i - y_i| / (|x_i| + |y_i|); 0/0 -> 0
    n, d = A.shape
    m = B.shape[0]
    # chunk over B to limit memory for large m; callers usually chunk anyway
    out = torch.empty(n, m, dtype=torch.float32, device=A.device)
    for s, e in chunk_ranges(m, max(1, 65536 // max(d, 1))):
        Bb = B[s:e]
        num = (A.unsqueeze(1) - Bb.unsqueeze(0)).abs()
        den = A.unsqueeze(1).abs() + Bb.unsqueeze(0).abs()
        frac = torch.where(den > 0, num / den, torch.zeros_like(num))
        out[:, s:e] = frac.sum(dim=-1)
    return out


def _braycurtis(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    n, d = A.shape
    m = B.shape[0]
    out = torch.empty(n, m, dtype=torch.float32, device=A.device)
    for s, e in chunk_ranges(m, max(1, 65536 // max(d, 1))):
        Bb = B[s:e]
        diff = (A.unsqueeze(1) - Bb.unsqueeze(0)).abs().sum(dim=-1)
        tot = (A.unsqueeze(1).abs() + Bb.unsqueeze(0).abs()).sum(dim=-1).clamp_min(1e-12)
        out[:, s:e] = diff / tot
    return out


def _jensenshannon(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Jensen–Shannon distance (square-root of JS divergence).

    Treats rows as non-negative discrete distributions after L1 normalisation.
    All-zero rows are treated as the uniform distribution so ``d(a,a)==0``.
    """
    eps = 1e-12
    A_n = A.clamp_min(0.0)
    B_n = B.clamp_min(0.0)
    a_sum = A_n.sum(dim=1, keepdim=True)
    b_sum = B_n.sum(dim=1, keepdim=True)
    D = A_n.shape[1]
    A_n = torch.where(a_sum > eps, A_n / a_sum.clamp_min(eps), torch.full_like(A_n, 1.0 / D))
    B_n = torch.where(b_sum > eps, B_n / b_sum.clamp_min(eps), torch.full_like(B_n, 1.0 / D))
    n, d = A_n.shape
    m = B_n.shape[0]
    out = torch.empty(n, m, dtype=torch.float32, device=A.device)
    for s, e in chunk_ranges(m, max(1, 65536 // max(d, 1))):
        Bb = B_n[s:e]
        M = 0.5 * (A_n.unsqueeze(1) + Bb.unsqueeze(0))
        # KL(p||M) = Σ p log(p/M)
        kl_am = (A_n.unsqueeze(1) * (A_n.unsqueeze(1).clamp_min(eps).log() - M.clamp_min(eps).log())).sum(-1)
        kl_bm = (Bb.unsqueeze(0) * (Bb.unsqueeze(0).clamp_min(eps).log() - M.clamp_min(eps).log())).sum(-1)
        js = 0.5 * (kl_am + kl_bm).clamp_min(0.0)
        out[:, s:e] = js.sqrt()
    return out


def _unit_mass(X: torch.Tensor) -> torch.Tensor:
    """Non-negative L1-normalised rows; all-zero → uniform (so ``d(a,a)==0``)."""
    eps = 1e-12
    Xn = X.clamp_min(0.0)
    total = Xn.sum(dim=1, keepdim=True)
    D = Xn.shape[1]
    return torch.where(
        total > eps, Xn / total.clamp_min(eps), torch.full_like(Xn, 1.0 / D)
    )


def _wasserstein1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """1-D Wasserstein-1 between rows as histograms on a uniform bin grid.

    After unit-mass normalisation, ``W₁(p,q) = Σ_{k=0}^{D-2} |F_p(k) − F_q(k)|``
    with bin centres at ``0, 1, …, D−1`` (unit spacing). This is the closed form
    for discrete measures on the line, and matches ``scipy.stats.wasserstein_distance``
    on the same support. Natural for equal-width P(r) / density profiles.
    """
    A_n = _unit_mass(A)
    B_n = _unit_mass(B)
    # Drop the final CDF entry (always 1); spacing between consecutive bins is 1.
    Fa = A_n.cumsum(dim=1)[:, :-1]
    Fb = B_n.cumsum(dim=1)[:, :-1]
    n, d = Fa.shape
    m = Fb.shape[0]
    out = torch.empty(n, m, dtype=torch.float32, device=A.device)
    for s, e in chunk_ranges(m, max(1, 65536 // max(d, 1))):
        out[:, s:e] = (Fa.unsqueeze(1) - Fb[s:e].unsqueeze(0)).abs().sum(dim=-1)
    return out


def _correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``1 − pearson(x, y)`` per pair (scipy.spatial.distance.correlation)."""
    # float64 for self-distance numerical zeros under float32 pearson.
    Ad = A.double()
    Bd = B.double()
    A_c = Ad - Ad.mean(dim=1, keepdim=True)
    B_c = Bd - Bd.mean(dim=1, keepdim=True)
    a_norm = A_c.norm(dim=1, keepdim=True)
    b_norm = B_c.norm(dim=1, keepdim=True)
    sim = (A_c / a_norm.clamp_min(1e-12)) @ (B_c / b_norm.clamp_min(1e-12)).T
    sim = sim.clamp(-1.0, 1.0)
    d = (1.0 - sim).clamp_min(0.0)
    both_flat = (a_norm < 1e-12) & (b_norm.T < 1e-12)
    d = torch.where(both_flat, torch.zeros_like(d), d)
    return d.float()


def _correlation_sqrt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``√(2(1−ρ))`` — a true metric variant of correlation distance."""
    return (2.0 * _correlation(A, B)).clamp_min(0.0).sqrt()


def _frobenius(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix Frobenius distance = Euclidean on flattened rows."""
    return EuclideanDistance()(A, B)


BUILTIN_FNS: dict[str, DistanceFn] = {
    "l2": EuclideanDistance(),
    "sqeuclidean": SquaredEuclideanDistance(),
    "frobenius": CallableDistance(_frobenius),
    "cosine": CosineDistance(),
    "correlation": CallableDistance(_correlation),
    "correlation_sqrt": CallableDistance(_correlation_sqrt),
    "l1": ManhattanDistance(),
    "linf": CallableDistance(_linf),
    "canberra": CallableDistance(_canberra),
    "braycurtis": CallableDistance(_braycurtis),
    "jensenshannon": CallableDistance(_jensenshannon),
    "wasserstein1d": CallableDistance(_wasserstein1d),
}


def is_differentiable(
    dist_fn: DistanceFn,
    D: int,
    device: Optional[Union[str, torch.device]] = None,
) -> bool:
    """Probe whether ``d`` yields a finite non-zero grad w.r.t. its second arg.

    Parameters
    ----------
    dist_fn : DistanceFn
    D : int
        Feature dimension.
    device : str | torch.device | None

    Returns
    -------
    bool
    """
    dev = resolve_device(str(device) if device is not None else None)
    A = torch.randn(4, D, device=dev, dtype=torch.float32)
    B = torch.randn(4, D, device=dev, dtype=torch.float32, requires_grad=True)
    try:
        d = dist_fn(A, B)
        loss = d.sum()
        loss.backward()
        if B.grad is None:
            return False
        g = B.grad
        if not torch.isfinite(g).all():
            return False
        return bool(g.abs().sum().item() > 0.0)
    except Exception:  # noqa: BLE001
        return False


TopKResult = Tuple[torch.Tensor, torch.Tensor]


def chunked_cdist(
    dist_fn: DistanceFn,
    A: torch.Tensor,
    B: torch.Tensor,
    chunk_a: int = 4096,
    chunk_b: int = 65536,
    out_device: Union[str, torch.device] = "cpu",
    topk: Optional[int] = None,
) -> Union[torch.Tensor, TopKResult]:
    """Compute ``d(A, B)`` in tiles to bound peak memory.

    Parameters
    ----------
    dist_fn : DistanceFn
    A : (n, D) float32
    B : (m, D) float32
    chunk_a, chunk_b : int
        Tile sizes along A and B.
    out_device : str | torch.device
        Device for the returned tensors.
    topk : int | None
        If None: return full ``(n, m)`` matrix on ``out_device``.
        If int: return ``(values (n, topk), indices (n, topk))`` of the
        smallest ``topk`` entries per row, never materialising the full matrix.

    Returns
    -------
    Tensor (n, m) or (values (n, topk), indices (n, topk))
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must share feature dimension")
    n, m = A.shape[0], B.shape[0]
    out_dev = torch.device(out_device)

    if topk is None:
        out = torch.empty(n, m, dtype=torch.float32, device=out_dev)
        for sa, ea in chunk_ranges(n, chunk_a):
            for sb, eb in chunk_ranges(m, chunk_b):
                block = dist_fn(A[sa:ea], B[sb:eb])
                out[sa:ea, sb:eb] = block.to(out_dev)
        return out

    if topk <= 0:
        raise ValueError("topk must be positive")
    k = min(topk, m)
    # Running top-k (smallest). Initialise with +inf / -1.
    best_vals = torch.full((n, k), float("inf"), dtype=torch.float32, device=A.device)
    best_idx = torch.full((n, k), -1, dtype=torch.int64, device=A.device)

    for sa, ea in chunk_ranges(n, chunk_a):
        row_vals = best_vals[sa:ea]
        row_idx = best_idx[sa:ea]
        for sb, eb in chunk_ranges(m, chunk_b):
            block = dist_fn(A[sa:ea], B[sb:eb])  # (ba, bb)
            bb = eb - sb
            take = min(k, bb)
            chunk_vals, chunk_local = torch.topk(block, k=take, dim=1, largest=False)
            chunk_idx = chunk_local.to(torch.int64) + sb
            # Concatenate current best with chunk top-k and re-reduce
            merged_vals = torch.cat([row_vals, chunk_vals], dim=1)
            merged_idx = torch.cat([row_idx, chunk_idx], dim=1)
            new_vals, order = torch.topk(merged_vals, k=k, dim=1, largest=False)
            new_idx = torch.gather(merged_idx, 1, order)
            row_vals = new_vals
            row_idx = new_idx
        best_vals[sa:ea] = row_vals
        best_idx[sa:ea] = row_idx

    return best_vals.to(out_dev), best_idx.to(out_dev)

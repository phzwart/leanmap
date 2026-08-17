"""FiLM encoder backbone and top-level PLANE model."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conditioning import FactorStack, Role
from ..landmarks import AnchorAffinity, LandmarkAffinity


def fit_pca_weight(
    X: Union[torch.Tensor, np.ndarray],
    n_components: int,
    center: bool = True,
) -> torch.Tensor:
    """Top-``n_components`` principal axes of ``X``, shape ``(n_components, D)``.

    Parameters
    ----------
    X : (N, D)
        Typically the encoder-normalized training matrix.
    n_components : int
    center : bool
        If True (classical PCA), subtract the column mean before SVD. If False,
        SVD on ``X`` as-is (uncentered).
    """
    Xn = np.asarray(
        X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X,
        dtype=np.float64,
        order="C",
    )
    if center:
        Xn = Xn - Xn.mean(axis=0, keepdims=True)
    if Xn.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {Xn.shape}")
    n_components = int(n_components)
    if n_components < 1 or n_components > Xn.shape[1]:
        raise ValueError(
            f"n_components must be in [1, {Xn.shape[1]}], got {n_components}"
        )
    _, _, vt = np.linalg.svd(Xn, full_matrices=False)
    w = np.ascontiguousarray(vt[:n_components], dtype=np.float32)
    return torch.as_tensor(w, dtype=torch.float32)


class FiLMEncoder(nn.Module):
    """FiLM-conditioned MLP backbone (gamma/beta supplied externally).

    Parameters
    ----------
    D, d_out, width, depth : architecture
    affinity_dim : int
        Total affinity dims across factors (sum of L_f); used for legacy hyper.
    pca_skip : bool
        If True, output is ``pca(x_n) + residual`` with near-zero residual head.
    pca_weight : (d_out, D) | None
    """

    def __init__(
        self,
        D: int,
        d_out: int,
        width: int = 384,
        depth: int = 3,
        affinity_dim: int = 0,
        pca_skip: bool = True,
        pca_weight: Optional[torch.Tensor] = None,
        *,
        L: Optional[int] = None,
        hyper_width: int = 128,
    ):
        super().__init__()
        self.D = D
        self.d_out = d_out
        self.width = width
        self.depth = depth
        # Legacy: L was both #landmarks and concat dim; keep attribute for tests.
        self.L = int(L) if L is not None else int(affinity_dim)
        self.affinity_dim = int(affinity_dim) if affinity_dim else self.L
        self.pca_skip = bool(pca_skip)
        self.hyper_width = hyper_width

        self.register_buffer("x_mean", torch.zeros(D))
        self.register_buffer("x_std", torch.ones(D))

        layers = []
        for i in range(depth):
            lin = nn.Linear(D if i == 0 else width, width, bias=False)
            layers.append(lin)
        self.backbone = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        # Parameter-free tap sitting after FiLM and before the activation, so
        # feature extraction can hook the true pre-activation state.
        self.taps = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.head = nn.Linear(width, d_out)

        if self.pca_skip:
            self.pca = nn.Linear(D, d_out, bias=False)
            if pca_weight is not None:
                with torch.no_grad():
                    self.pca.weight.copy_(pca_weight.float())
            nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.head.bias)
        else:
            self.pca = None

        # Optional legacy single-factor hyper (used only if FactorStack not attached
        # and film_params(a) is called). Kept for migration / tiny tests.
        self.hyper = nn.Sequential(
            nn.Linear(self.L if self.L > 0 else 1, hyper_width),
            nn.GELU(),
            nn.Linear(hyper_width, 2 * width * depth),
        )
        last = self.hyper[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store training-split mean/std. mean/std: (D,)."""
        self.x_mean.copy_(mean.float().view(-1))
        self.x_std.copy_(std.float().view(-1).clamp_min(1e-6))

    def set_pca_weight(self, weight: torch.Tensor) -> None:
        """Copy ``(d_out, D)`` axes into the PCA skip (no-op if ``pca_skip=False``)."""
        if self.pca is None:
            return
        with torch.no_grad():
            self.pca.weight.copy_(weight.float())

    def film_params(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy single-hyper path. a: (B, L). Returns gamma, beta (B, depth, width)."""
        raw = self.hyper(a)
        B = a.shape[0]
        raw = raw.view(B, self.depth, 2, self.width)
        gamma_raw, beta = raw[:, :, 0, :], raw[:, :, 1, :]
        return 1.0 + gamma_raw, beta

    def forward(
        self,
        x: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, D). Provide either ``a`` (legacy) or ``gamma``/``beta``."""
        x_n = (x - self.x_mean) / self.x_std
        if gamma is None or beta is None:
            if a is None:
                raise ValueError("FiLMEncoder.forward requires a= or gamma=/beta=")
            gamma, beta = self.film_params(a)
        h = x_n
        for k, (lin, norm, tap) in enumerate(zip(self.backbone, self.norms, self.taps)):
            h = lin(h)
            # Normalize *then* modulate. LayerNorm is exactly invariant to a
            # positive scalar rescale, so modulating first made a per-layer
            # scalar gamma a no-op (GAIN) and left a scalar-gamma factor acting
            # only through the size of beta relative to gamma * h (MODULATOR).
            h = norm(h)
            h = tap(gamma[:, k, :] * h + beta[:, k, :])
            h = F.gelu(h)
        residual = self.head(h)
        if self.pca is not None:
            return self.pca(x_n) + residual
        return residual


class ConcatEncoder(nn.Module):
    """Plain MLP on ``[x_n, a(x)]`` — the baseline FiLM has to beat.

    FiLM adds no information: ``a(x)`` is a deterministic function of ``x``, so
    the roles, temperatures, gamma clamps, and perplexity calibration buy an
    inductive bias (a soft partition-of-unity mixture of experts), not capacity.
    The honest control is to hand the same affinity vector to an ordinary
    network as extra input columns and keep everything else — width, depth,
    head, PCA skip — identical.

    The interface matches :class:`FiLMEncoder` closely enough for
    :class:`PLANE` to switch on ``conditioning``.
    """

    conditioning = "concat"

    def __init__(
        self,
        D: int,
        d_out: int,
        width: int = 384,
        depth: int = 3,
        affinity_dim: int = 0,
        pca_skip: bool = True,
        pca_weight: Optional[torch.Tensor] = None,
        *,
        L: Optional[int] = None,
    ):
        super().__init__()
        self.D = D
        self.d_out = d_out
        self.width = width
        self.depth = depth
        self.L = int(L) if L is not None else int(affinity_dim)
        self.affinity_dim = int(affinity_dim) if affinity_dim else self.L
        self.pca_skip = bool(pca_skip)

        self.register_buffer("x_mean", torch.zeros(D))
        self.register_buffer("x_std", torch.ones(D))

        in_dim = D + self.affinity_dim
        self.backbone = nn.ModuleList(
            [nn.Linear(in_dim if i == 0 else width, width, bias=False) for i in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.taps = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.head = nn.Linear(width, d_out)

        if self.pca_skip:
            self.pca = nn.Linear(D, d_out, bias=False)
            if pca_weight is not None:
                with torch.no_grad():
                    self.pca.weight.copy_(pca_weight.float())
            nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.head.bias)
        else:
            self.pca = None

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.copy_(mean.float().view(-1))
        self.x_std.copy_(std.float().view(-1).clamp_min(1e-6))

    def set_pca_weight(self, weight: torch.Tensor) -> None:
        if self.pca is None:
            return
        with torch.no_grad():
            self.pca.weight.copy_(weight.float())

    def forward(
        self,
        x: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, D), a: (B, affinity_dim). ``gamma``/``beta`` are ignored."""
        if a is None:
            raise ValueError("ConcatEncoder.forward requires a=")
        x_n = (x - self.x_mean) / self.x_std
        if a.shape[1] != self.affinity_dim:
            raise ValueError(
                f"affinity width {a.shape[1]} != expected {self.affinity_dim}"
            )
        h = torch.cat([x_n, a], dim=1)
        for lin, norm, tap in zip(self.backbone, self.norms, self.taps):
            h = F.gelu(tap(norm(lin(h))))
        residual = self.head(h)
        if self.pca is not None:
            return self.pca(x_n) + residual
        return residual


class PLANE(nn.Module):
    """Factor-conditioned parametric embedder.

    Ambient ``x`` may pack multiple vectors per item (e.g. metric features and
    a conditioning view). ``encoder_view`` selects the backbone input; factor
    ``view`` callables select landmark / FiLM inputs.
    """

    def __init__(
        self,
        factors: Union[FactorStack, AnchorAffinity, LandmarkAffinity],
        encoder: FiLMEncoder,
        encoder_view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        if isinstance(factors, FactorStack):
            self.factors: Optional[FactorStack] = factors
            self.affinity: AnchorAffinity = factors.primary_affinity
        else:
            # Legacy: single AnchorAffinity — wrap is caller's responsibility for new code
            self.factors = None
            self.affinity = factors  # type: ignore[assignment]
        self.encoder = encoder
        self.encoder_view = encoder_view  # None → identity

    def _x_enc(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_view(x) if self.encoder_view is not None else x

    @property
    def is_concat(self) -> bool:
        return getattr(self.encoder, "conditioning", "film") == "concat"

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, D_ambient). Returns ``z, a_primary, Dm_primary``."""
        x_enc = self._x_enc(x)
        if self.factors is not None:
            a_map, dm_map, a_list = self.factors.affinities_forward(x, for_geom=True)
            if self.is_concat:
                z = self.encoder(x_enc, a=self.factors.concat_affinity(a_list))
            else:
                gamma, beta, _, _, _ = self.factors.film_params_from_affinities(a_list)
                z = self.encoder(x_enc, gamma=gamma, beta=beta)
            z = self.factors.apply_axis_skips(z, x, for_geom=True)
            name = self.factors.primary_factor.name
            return z, a_map[name], dm_map[name]
        a, Dm = self.affinity(x)
        z = self.encoder(x_enc, a)
        return z, a, Dm

    def forward_detailed(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        float,
    ]:
        """Full factor outputs for diagnostics."""
        x_enc = self._x_enc(x)
        if self.factors is None:
            a, Dm = self.affinity(x)
            gamma, beta = self.encoder.film_params(a)
            z = self.encoder(x_enc, a=a)
            return z, {"primary": a}, {"primary": Dm}, gamma, beta, {"primary": gamma}, 0.0
        a_map, dm_map, a_list = self.factors.affinities_forward(x, for_geom=True)
        if self.is_concat:
            z = self.encoder(x_enc, a=self.factors.concat_affinity(a_list))
            z = self.factors.apply_axis_skips(z, x, for_geom=True)
            # No FiLM parameters exist on this path; report the identity so the
            # trainer's gamma diagnostics stay well-defined.
            shape = (z.shape[0], self.encoder.depth, self.encoder.width)
            ones = torch.ones(shape, device=z.device, dtype=z.dtype)
            return z, a_map, dm_map, ones, torch.zeros_like(ones), {}, 0.0
        gamma, beta, g_by, _, hit = self.factors.film_params_from_affinities(a_list)
        z = self.encoder(x_enc, gamma=gamma, beta=beta)
        z = self.factors.apply_axis_skips(z, x, for_geom=True)
        return z, a_map, dm_map, gamma, beta, g_by, hit

    @torch.no_grad()
    def _primary_anchor_embeddings(self, device: torch.device) -> torch.Tensor:
        """Embed PRIMARY anchors for OOD affinity comparison.

        Anchors ``M`` live in PRIMARY *view* coordinates. ``forward(M)`` is only
        valid when that view is ambient identity and there is no encoder slice.
        Otherwise: FiLM from ``a(M)`` on zero content features.
        """
        from ..conditioning import FactorHyper, identity_view

        M = self.affinity.M.to(device)
        primary_is_ambient = (
            self.encoder_view is None
            and (
                self.factors is None
                or self.factors.primary_factor.view is identity_view
            )
            and M.shape[1] == self.encoder.D
        )
        if primary_is_ambient:
            z_M, _, _ = self.forward(M)
            return z_M

        a_m, _ = self.affinity(M)
        if self.is_concat:
            x0 = torch.zeros(M.shape[0], self.encoder.D, device=device)
            width = int(self.encoder.affinity_dim)
            a_full = torch.zeros(M.shape[0], width, device=device)
            take = min(width, a_m.shape[1])
            a_full[:, :take] = a_m[:, :take]
            return self.encoder(x0, a=a_full)
        if self.factors is not None and len(self.factors.factor_defs) == 1:
            hyp = self.factors.hypers[0]
            assert isinstance(hyp, FactorHyper)
            g, b = hyp(a_m)
            assert g is not None and b is not None
            x0 = torch.zeros(M.shape[0], self.encoder.D, device=device)
            return self.encoder(x0, gamma=g, beta=b)
        if self.factors is not None:
            # Multi-factor: PRIMARY from a(M); other roles contribute identity FiLM
            from ..conditioning import GAMMA_MAX, GAMMA_MIN

            gamma = None
            beta = None
            for f, hyp in zip(self.factors.factor_defs, self.factors.hypers):
                if f.role == Role.AXIS:
                    continue
                assert isinstance(hyp, FactorHyper)
                if f is self.factors.primary_factor:
                    g, b = hyp(a_m)
                else:
                    B = M.shape[0]
                    if f.role == Role.GAIN:
                        g = torch.ones(B, hyp.depth, 1, device=device)
                        b = None
                    elif f.role == Role.MODULATOR:
                        g = torch.ones(B, hyp.depth, 1, device=device)
                        b = torch.zeros(B, hyp.depth, hyp.width, device=device)
                    else:
                        g = torch.ones(B, hyp.depth, hyp.width, device=device)
                        b = torch.zeros(B, hyp.depth, hyp.width, device=device)
                if g is not None:
                    gamma = g if gamma is None else gamma * g
                if b is not None:
                    beta = b if beta is None else beta + b
            assert gamma is not None
            if beta is None:
                beta = torch.zeros(
                    M.shape[0], self.encoder.depth, self.encoder.width, device=device
                )
            gamma = gamma.clamp(GAMMA_MIN, GAMMA_MAX)
            x0 = torch.zeros(M.shape[0], self.encoder.D, device=device)
            return self.encoder(x0, gamma=gamma, beta=beta)
        x0 = torch.zeros(M.shape[0], self.encoder.D, device=device)
        L = self.encoder.L
        if a_m.shape[1] != L:
            a2 = torch.zeros(a_m.shape[0], L, device=device)
            take = min(L, a_m.shape[1])
            a2[:, :take] = a_m[:, :take]
            a_m = a2
        return self.encoder(x0, a=a_m)

    @torch.no_grad()
    def embed(
        self,
        X: torch.Tensor,
        batch_size: int = 8192,
        return_score: bool = True,
        tau_embed: Optional[float] = None,
        z_M: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batched inference. Cost independent of training-set size.

        When ``return_score=True``, the second return is the **landmark cover**
        OOD score ``min_l ||x - M_l||`` (higher ⇒ farther from support). Pass
        scores through ``ConformalCalibrator.p_value`` for exchangeability
        p-values. ``tau_embed`` / ``z_M`` are accepted for API compatibility
        but are not used for the cover score.
        """
        self.eval()
        device = next(self.parameters()).device
        N = X.shape[0]
        outs = []
        scores = [] if return_score else None
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            xb = X[s:e].to(device)
            z, a, Dm = self.forward(xb)
            outs.append(z.cpu())
            if return_score:
                # Primary OOD score: ambient distance to nearest landmark.
                scores.append(Dm.min(dim=1).values.cpu())
        Z = torch.cat(outs, dim=0)
        S = torch.cat(scores, dim=0) if scores is not None else None
        return Z, S

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PLANE":
        """Load a saved artefact and return a model ready for ``embed()``."""
        from ..train import load_plane

        return load_plane(path, device=device)

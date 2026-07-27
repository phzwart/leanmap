"""FiLM encoder backbone, optional decoder, and top-level PLANE model."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .conditioning import FactorStack, Role
from .landmarks import AnchorAffinity, LandmarkAffinity


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
        Total affinity dims when ``concat_affinity`` (sum of L_f).
    spectral_norm : bool
        Wrap backbone Linears only (not head).
    concat_affinity : bool
        Concatenate concatenated affinities to the input before the first layer.
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
        spectral_norm_flag: bool = True,
        concat_affinity: bool = False,
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
        self.concat_affinity = concat_affinity
        self.pca_skip = bool(pca_skip)
        self.hyper_width = hyper_width
        in_dim = D + (self.affinity_dim if concat_affinity else 0)

        self.register_buffer("x_mean", torch.zeros(D))
        self.register_buffer("x_std", torch.ones(D))

        layers = []
        for i in range(depth):
            lin = nn.Linear(in_dim if i == 0 else width, width, bias=False)
            if spectral_norm_flag:
                lin = spectral_norm(lin)
            layers.append(lin)
        self.backbone = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
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
        a_concat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, D). Provide either ``a`` (legacy) or ``gamma``/``beta`` (+ optional concat)."""
        x_n = (x - self.x_mean) / self.x_std
        if gamma is None or beta is None:
            if a is None:
                raise ValueError("FiLMEncoder.forward requires a= or gamma=/beta=")
            gamma, beta = self.film_params(a)
            a_cat = a
        else:
            a_cat = a_concat if a_concat is not None else a
        h = torch.cat([x_n, a_cat], dim=1) if self.concat_affinity and a_cat is not None else x_n
        for k, (lin, norm) in enumerate(zip(self.backbone, self.norms)):
            h = lin(h)
            h = gamma[:, k, :] * h + beta[:, k, :]
            h = norm(h)
            h = F.gelu(h)
        residual = self.head(h)
        if self.pca is not None:
            return self.pca(x_n) + residual
        return residual


class Decoder(nn.Module):
    """Plain MLP ``d_out -> width -> width -> D`` (no FiLM / spectral norm)."""

    def __init__(self, d_out: int, D: int, width: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_out, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, D),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_out). Returns x_hat: (B, D)."""
        return self.net(z)


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
        decoder: Optional[Decoder] = None,
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
        self.decoder = decoder
        self.encoder_view = encoder_view  # None → identity

    def _x_enc(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_view(x) if self.encoder_view is not None else x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, D_ambient). Returns ``z, a_primary, Dm_primary``."""
        x_enc = self._x_enc(x)
        if self.factors is not None:
            a_map, dm_map, a_list = self.factors.affinities_forward(x, for_geom=True)
            gamma, beta, _, _, _ = self.factors.film_params_from_affinities(a_list)
            a_cat = self.factors.concat_affinity(a_list)
            z = self.encoder(x_enc, gamma=gamma, beta=beta, a_concat=a_cat)
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
        gamma, beta, g_by, _, hit = self.factors.film_params_from_affinities(a_list)
        a_cat = self.factors.concat_affinity(a_list)
        z = self.encoder(x_enc, gamma=gamma, beta=beta, a_concat=a_cat)
        z = self.factors.apply_axis_skips(z, x, for_geom=True)
        return z, a_map, dm_map, gamma, beta, g_by, hit

    @torch.no_grad()
    def _primary_anchor_embeddings(self, device: torch.device) -> torch.Tensor:
        """Embed PRIMARY anchors for OOD affinity comparison.

        Anchors ``M`` live in PRIMARY *view* coordinates. ``forward(M)`` is only
        valid when that view is ambient identity and there is no encoder slice.
        Otherwise: FiLM from ``a(M)`` on zero content features.
        """
        from .conditioning import FactorHyper, identity_view

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
        if self.factors is not None and len(self.factors.factor_defs) == 1:
            hyp = self.factors.hypers[0]
            assert isinstance(hyp, FactorHyper)
            g, b = hyp(a_m)
            assert g is not None and b is not None
            x0 = torch.zeros(M.shape[0], self.encoder.D, device=device)
            return self.encoder(x0, gamma=g, beta=b, a_concat=a_m)
        if self.factors is not None:
            # Multi-factor: PRIMARY from a(M); other roles contribute identity FiLM
            from .conditioning import GAMMA_MAX, GAMMA_MIN

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
            return self.encoder(x0, gamma=gamma, beta=beta, a_concat=a_m)
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
        from .train import load_plane

        return load_plane(path, device=device)

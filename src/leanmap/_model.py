"""The parametric encoder: PCA-anchored linear map plus a nonlinear residual."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pca_components(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Top-`n_components` PCA directions of X, shape (n_components, d)."""
    covariance = (X.astype(np.float64).T @ X.astype(np.float64)) / max(
        1, X.shape[0] - 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, order].T.astype(np.float32)


class ParametricMapper(nn.Module):
    """Encoder mapping standardized input -> 2D.

    Output is a linear PCA projection (initialized to the true top-2 PCA
    components) plus a near-zero-initialized nonlinear residual MLP, so training
    begins at a PCA embedding and refines from there.
    """

    def __init__(
        self,
        input_dim: int,
        pca_weight: np.ndarray,
        hidden_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        # output dimensionality is inferred from the PCA weight (n_components, d)
        self.out_dim = int(np.asarray(pca_weight).shape[0])

        self.linear = nn.Linear(input_dim, self.out_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(pca_weight))

        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, self.out_dim))
        self.residual = nn.Sequential(*layers)
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual(x)


class FiLMMapper(nn.Module):
    """PCA-anchored encoder whose hidden activations are FiLM-modulated.

    Same PCA skip + residual MLP as :class:`ParametricMapper`, but each hidden
    layer's activations are feature-wise modulated by ``(1 + gamma(z)) * h +
    beta(z)``, where ``z`` is a per-point conditioning vector summarizing the
    point's membership to the stored landmarks (see ``_inducing``). The FiLM
    generator's final layer is zero-initialized, so at the start of training the
    model is *exactly* a PCA + MLP (gamma=beta=0) and learns to exploit the
    landmark structure from there.

    ``forward`` takes both the standardized input ``x`` and the (standardized)
    conditioning vector ``z``.
    """

    def __init__(
        self,
        input_dim: int,
        pca_weight: np.ndarray,
        cond_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.out_dim = int(np.asarray(pca_weight).shape[0])

        self.linear = nn.Linear(input_dim, self.out_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(pca_weight))

        self.hidden_layers = nn.ModuleList()
        width = input_dim
        for hidden in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(width, hidden))
            width = hidden
        self.out = nn.Linear(width, self.out_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out.bias)

        # FiLM generator: z -> concatenated (gamma, beta) for every hidden layer.
        self.film = nn.Sequential(
            nn.Linear(self.cond_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * sum(self.hidden_dims)),
        )
        nn.init.zeros_(self.film[-1].weight)  # start as identity modulation
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self.film(z)
        chunks = torch.split(params, [2 * h for h in self.hidden_dims], dim=1)
        h = x
        for layer, gb, size in zip(self.hidden_layers, chunks, self.hidden_dims):
            h = layer(h)
            gamma, beta = gb[:, :size], gb[:, size:]
            h = torch.nn.functional.silu((1.0 + gamma) * h + beta)
        return self.linear(x) + self.out(h)


class AttentionMapper(nn.Module):
    """Encoder that cross-attends to landmark tokens, then FiLM-modulates an MLP.

    A per-point query attends to a fixed set of landmark tokens (Set-Transformer
    style). Keys come from landmark high-D coordinates; values carry both the
    high-D coordinates and the landmark's reference 2D coordinate, so the
    attention readout expresses *where on the manifold* the point sits. The
    attention logits are biased by ``-beta * ||x - landmark||^2`` (a learnable
    locality prior, playing the role of UMAP's fuzzy kernel), which prevents the
    otherwise-global attention from overfitting.

    The attention readout ``c`` then generates FiLM ``(gamma, beta)`` that
    modulate the hidden activations of an MLP driven by ``x`` — modulation
    conditions more cleanly than concatenation. The FiLM head is zero-init, so
    training starts as a plain PCA + MLP and learns to use landmark structure.

    Attention keeps the landmark keys/values shared across the batch as
    (M, H, Dh) tensors (never expanded per-point), so the readout is two batched
    matmuls with the distance bias added to the logits. This is far cheaper than
    broadcasting the landmark set to (B, H, M, Dh) and calling a generic
    attention kernel, since there is a single query token per point.

    ``L_hd`` (M, d) and ``L_emb`` (M, 2) are registered as buffers so they
    travel with the model on ``save``/``load`` and ``.to(device)``.
    """

    def __init__(
        self,
        input_dim: int,
        pca_weight: np.ndarray,
        landmark_hd: np.ndarray,
        landmark_emb: np.ndarray,
        hidden_dims: tuple[int, ...] = (128, 128),
        attn_dim: int = 64,
        heads: int = 4,
        attn_layers: int = 2,
        film_hidden: int = 64,
        learn_landmarks: bool = False,
        distance_kernel: str = "linear",
        attend_top_p: int | None = None,
    ) -> None:
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError("attn_dim must be divisible by heads")
        if distance_kernel not in ("linear", "squared", "constant"):
            raise ValueError(
                "distance_kernel must be 'linear', 'squared', or 'constant'"
            )
        if attend_top_p is not None and int(attend_top_p) < 1:
            raise ValueError("attend_top_p must be >= 1 or None (dense)")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.attn_dim = int(attn_dim)
        self.heads = int(heads)
        self.attn_layers = int(attn_layers)
        self.learn_landmarks = bool(learn_landmarks)
        # Locality prior added to the attention logits. 'linear' (-beta*||x-l||)
        # is a Laplacian/heavy-tailed kernel that generalizes to held-out points
        # in sparse regions materially better than the Gaussian 'squared'
        # (-beta*||x-l||^2); 'constant' disables the distance bias (pure content
        # attention) and collapses held-out quality, so it exists only for
        # ablation. Default 'linear'.
        self.distance_kernel = str(distance_kernel)
        # Sparse attention: when set, each point attends only to its P nearest
        # landmarks (gathered, so the key/value projections and attention einsum
        # run over P instead of M). Gives O(N*P) attention cost independent of M,
        # matching dense held-out accuracy at P~=20 while ~1.5-2x faster at
        # M=2000 (the O(M) nearest-P search is not eliminated, so the speedup is
        # bounded and grows with M). None => dense attention over all M.
        self.attend_top_p = None if attend_top_p is None else int(attend_top_p)

        hd = torch.as_tensor(landmark_hd, dtype=torch.float32).clone()
        # High-D landmark positions: a free parameter (Titsias-style inducing
        # inputs) when ``learn_landmarks`` is set, otherwise a fixed buffer. The
        # anchor positions are always kept as a buffer so a Gram-anchor penalty
        # can reference the initial geometry during training.
        self.register_buffer("landmark_hd0", hd.clone())
        if self.learn_landmarks:
            self.landmark_hd = nn.Parameter(hd)
        else:
            self.register_buffer("landmark_hd", hd)
        self.register_buffer(
            "landmark_emb", torch.as_tensor(landmark_emb, dtype=torch.float32).clone()
        )
        # output dimensionality inferred from the PCA weight (n_components, d);
        # the landmark embedding must match.
        self.out_dim = int(np.asarray(pca_weight).shape[0])

        self.linear = nn.Linear(input_dim, self.out_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(pca_weight))

        self.q_proj = nn.Linear(input_dim, attn_dim)
        self.k_proj = nn.Linear(input_dim, attn_dim)
        self.v_proj = nn.Linear(input_dim + self.out_dim, attn_dim)
        self.q_update = nn.ModuleList(
            [nn.Linear(attn_dim, attn_dim) for _ in range(attn_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(attn_dim) for _ in range(attn_layers)]
        )
        self.log_beta = nn.Parameter(torch.zeros(()))

        self.hidden_layers = nn.ModuleList()
        width = input_dim
        for hidden in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(width, hidden))
            width = hidden
        self.out = nn.Linear(width, self.out_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out.bias)

        self.film = nn.Sequential(
            nn.Linear(attn_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * sum(self.hidden_dims)),
        )
        nn.init.zeros_(self.film[-1].weight)  # start as identity modulation
        nn.init.zeros_(self.film[-1].bias)

    def gram_penalty(self) -> torch.Tensor:
        """Squared deviation of the landmark distance-Gram matrix from its anchor.

        Penalizes collapse and shear of the learnable landmark cloud while
        leaving rigid motion (translation/rotation) free, so landmarks can move
        to informative positions without running off the manifold. Returns 0 when
        landmarks are frozen.
        """
        if not self.learn_landmarks:
            return self.landmark_hd.new_zeros(())

        def sqdist(points: torch.Tensor) -> torch.Tensor:
            sq = (points * points).sum(1)
            return (sq[:, None] + sq[None, :] - 2.0 * points @ points.T).clamp_min(0.0)

        g_now = sqdist(self.landmark_hd)
        g_ref = sqdist(self.landmark_hd0)
        scale = g_ref.mean().clamp_min(1e-6)
        return ((g_now - g_ref).square().mean()) / (scale * scale)

    def _conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Attention readout c for each point (B, attn_dim).

        The landmark set is shared across the batch, so keys/values are computed
        once as (M, dim) and never expanded per-point. With a single query token
        per point, attention is a batched matmul ``(B,H,Dh) x (H,M,Dh) -> (B,H,M)``
        followed by a weighted sum ``(B,H,M) x (H,M,Dh) -> (B,H,Dh)`` — the same
        math as scaled_dot_product_attention but without materializing a
        per-batch (B,H,M,Dh) key/value tensor (which is what makes the naive
        expand-then-SDPA form slow on CPU). The distance bias is added to the
        logits directly.
        """
        B = x.shape[0]
        H, Dh, M = self.heads, self.attn_dim // self.heads, self.landmark_hd.shape[0]
        scale = Dh ** -0.5
        kernel = getattr(self, "distance_kernel", "squared")  # pre-upgrade default
        top_p = getattr(self, "attend_top_p", None)

        def _dist(d2: torch.Tensor) -> torch.Tensor:
            if kernel == "squared":
                return d2                                          # Gaussian falloff
            if kernel == "linear":
                return (d2 + 1e-8).sqrt()                          # Laplacian falloff
            return torch.zeros_like(d2)                            # "constant" (ablation)

        if top_p is not None and top_p < M:
            # --- gathered sparse: attend only to the P nearest landmarks ---
            # O(M) distance to find the P nearest, then key/value projections and
            # the attention einsum run over P instead of M.
            d2 = (x[:, None, :] - self.landmark_hd[None, :, :]).square().sum(-1)  # (B,M)
            topd, topi = torch.topk(d2, top_p, dim=1, largest=False)             # (B,P)
            lhd = self.landmark_hd[topi]                                          # (B,P,d)
            lemb = self.landmark_emb[topi]                                        # (B,P,e)
            keys = self.k_proj(lhd).view(B, top_p, H, Dh)                         # (B,P,H,Dh)
            vals = self.v_proj(torch.cat([lhd, lemb], dim=-1)).view(B, top_p, H, Dh)
            bias = -torch.nn.functional.softplus(self.log_beta) * _dist(topd)     # (B,P)
            q = self.q_proj(x)
            for layer, norm in zip(self.q_update, self.norms):
                qh = q.view(B, H, Dh)
                scores = torch.einsum("bhd,bphd->bhp", qh, keys) * scale
                scores = scores + bias[:, None, :]
                attn = scores.softmax(dim=-1)
                ctx = torch.einsum("bhp,bphd->bhd", attn, vals).reshape(B, self.attn_dim)
                q = norm(q + torch.nn.functional.silu(layer(ctx)))
            return q

        # --- dense: attend to all M shared landmarks (keys/values built once) ---
        keys = self.k_proj(self.landmark_hd).view(M, H, Dh)         # (M, H, Dh)
        vals = self.v_proj(
            torch.cat([self.landmark_hd, self.landmark_emb], dim=1)
        ).view(M, H, Dh)                                           # (M, H, Dh)
        d2 = (x[:, None, :] - self.landmark_hd[None, :, :]).square().sum(-1)
        bias = -torch.nn.functional.softplus(self.log_beta) * _dist(d2)   # (B, M)
        q = self.q_proj(x)
        for layer, norm in zip(self.q_update, self.norms):
            qh = q.view(B, H, Dh)
            # logits: (B,H,Dh) . (M,H,Dh) -> (B,H,M)
            scores = torch.einsum("bhd,mhd->bhm", qh, keys) * scale
            scores = scores + bias[:, None, :]
            attn = scores.softmax(dim=-1)
            # context: (B,H,M) . (M,H,Dh) -> (B,H,Dh)
            ctx = torch.einsum("bhm,mhd->bhd", attn, vals).reshape(B, self.attn_dim)
            q = norm(q + torch.nn.functional.silu(layer(ctx)))
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self._conditioning(x)
        params = self.film(c)
        chunks = torch.split(params, [2 * h for h in self.hidden_dims], dim=1)
        h = x
        for layer, gb, size in zip(self.hidden_layers, chunks, self.hidden_dims):
            h = layer(h)
            gamma, beta = gb[:, :size], gb[:, size:]
            h = torch.nn.functional.silu((1.0 + gamma) * h + beta)
        return self.linear(x) + self.out(h)


class DeployableMapper(nn.Module):
    """Encoder plus stored standardization stats; accepts RAW (unscaled) input."""

    def __init__(
        self,
        encoder: ParametricMapper,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # .clone() so the two buffers never alias each other's memory even when
        # the caller passes the same array object for both (e.g. dummy stats at
        # load time). Without it, load_state_dict's in-order copy_ would let the
        # second buffer's values overwrite the first through shared storage.
        self.register_buffer(
            "input_mean", torch.as_tensor(mean, dtype=torch.float32).clone()
        )
        self.register_buffer(
            "input_scale", torch.as_tensor(scale, dtype=torch.float32).clone()
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        return self.encoder((raw_x - self.input_mean) / self.input_scale)

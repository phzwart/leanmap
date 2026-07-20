"""Step-level leanmap pipeline: PCA preprocess -> fuzzy graph -> train.

This packages the step-based training loop (checkpoint/plot every N *steps*,
not epochs) used for large embeddings, and adds two features that the plain
epoch trainer does not have:

* ``pca_preprocess`` -- reduce the (standardized) input to a fixed number of
  PCA components before building the k-NN graph. Essential for high-dimensional
  pixel data (e.g. raw MNIST 784-d), where Euclidean distances concentrate and
  the raw-pixel k-NN graph is too noisy to embed.

* ``pca_graph_only`` -- when ``True`` (default), PCA defines *only the geometry*
  (the k-NN graph, the candidate-ranking distances, and the global Gram-anchor
  reference), while the encoder still maps the FULL, unreduced input to the
  embedding. This is the recommended MNIST-style setup: neighbors are found in
  the denoised PCA space, but the network keeps full pixel expressiveness and
  raw images pass straight through at inference. When ``False``, PCA is applied
  everywhere (encoder input included), the classic "reduce then embed" recipe.

* ``global_gram_weight`` -- global Gram anchoring. Each step, penalize the 2D
  embedding's normalized pairwise-distance matrix (on a random subsample)
  against the high-dimensional (PCA-reduced) distance matrix. This anchors the
  embedding's *global* geometry to the true manifold and suppresses the
  runaway "streaking" that unanchored graph loss can produce. It is DISTINCT
  from the landmark ``gram_anchor_weight`` used in attention/inducing mode:
  there is no landmark set here -- the reference geometry is the encoder's own
  high-D input space.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._graph import build_fuzzy_graph
from ._model import ParametricMapper, pca_components
from ._train import resolve_device, umap_pair_nll


def pca_reduce(x_scaled: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Project standardized data onto its top-``k`` PCA directions.

    Returns ``(Z, components)`` where ``Z`` is ``(n, k)`` and ``components`` is
    ``(k, d)`` so new data can be projected the same way via ``x @ components.T``.
    """
    k = int(min(k, x_scaled.shape[1]))
    components = pca_components(x_scaled, n_components=k)  # (k, d)
    z = (x_scaled.astype(np.float32) @ components.T).astype(np.float32)
    return z, components


def run_pipeline(
    X: np.ndarray,
    *,
    # --- preprocessing / graph ---
    scale_mode: str = "center",
    pca_preprocess: int | None = None,
    pca_graph_only: bool = True,
    n_neighbors: int = 15,
    # --- embedding / model ---
    n_components: int = 2,
    hidden_dims: tuple[int, ...] = (128, 128),
    # --- optimization ---
    n_steps: int = 2000,
    batch_size: int = 4096,
    negative_sample_rate: int = 10,
    candidate_count: int = 6,
    repulsion_strength: float = 1.0,
    rank_weight: float = 0.05,
    rank_margin: float = 0.05,
    rank_temperature: float = 0.20,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-5,
    # --- global gram anchoring (distinct from landmark gram) ---
    global_gram_weight: float = 0.0,
    global_gram_subsample: int = 512,
    # --- bookkeeping ---
    checkpoint_every: int = 100,
    on_checkpoint: Callable[[int, np.ndarray, float, ParametricMapper], None]
    | None = None,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
    init_encoder_state: dict | None = None,
) -> dict:
    """Run the step-level leanmap pipeline and return the fitted embedding.

    Parameters mirror the fuzzy-graph / attract-repel-rank loss used elsewhere
    in the package. Set ``pca_preprocess`` (e.g. 50) for high-dimensional input
    and ``global_gram_weight`` (e.g. 0.1-1.0) to anchor global geometry.

    ``on_checkpoint(step, embedding, loss, encoder)`` is called every
    ``checkpoint_every`` steps (and at the final step) -- use it to save
    weights / embeddings / scatter plots without baking I/O into the package.

    Returns a dict with ``embedding`` (n, n_components), ``encoder``,
    ``history`` (per-step loss), ``graph``, ``pca_components`` (or ``None``),
    ``input_dim`` (working dimensionality after any PCA), and ``transform``
    (a callable mapping raw new data -> embedding through the same pipeline).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = resolve_device(device)

    X = np.asarray(X, dtype=np.float32)

    from ._graph import standardize

    # Two spaces:
    #   geom -- defines the geometry: k-NN graph, candidate ranking, gram ref.
    #   enc  -- the encoder's input (what the network maps to the embedding).
    # With ``pca_graph_only`` these differ: geom is PCA-reduced (denoised
    # neighbors) while enc is the full standardized input (full expressiveness).
    comps = None
    x_scaled_full, train_mean, train_scale = standardize(X, mode=scale_mode)

    if pca_preprocess is not None and pca_preprocess > 0:
        z_reduced, comps = pca_reduce(x_scaled_full, pca_preprocess)
        # Graph on the reduced (geometry) space; scale_mode="none" because
        # z_reduced is already a linear map of standardized data.
        graph = build_fuzzy_graph(
            z_reduced, n_neighbors=n_neighbors, scale_mode="none"
        )
        geom = graph.x_scaled                       # == z_reduced
        enc_in = x_scaled_full if pca_graph_only else geom
    else:
        graph = build_fuzzy_graph(
            X, n_neighbors=n_neighbors, scale_mode=scale_mode
        )
        geom = graph.x_scaled
        enc_in = geom
        pca_graph_only = False                       # nothing to decouple

    n = geom.shape[0]
    input_dim = enc_in.shape[1]
    encoder = ParametricMapper(
        input_dim, pca_components(enc_in, n_components), hidden_dims
    ).to(device_obj)
    if init_encoder_state is not None:
        # Warm start: load a pre-trained encoder (e.g. from Stage-1 regression
        # to reference coordinates) before the graph-loss fine-tune. Requires
        # the same architecture (input_dim / hidden_dims / n_components).
        encoder.load_state_dict(
            {k: v.to(device_obj) for k, v in init_encoder_state.items()}
        )

    # xt_geom drives neighbor/rank/gram distances; xt is the encoder input.
    xt_geom = torch.from_numpy(np.ascontiguousarray(geom)).to(device_obj)
    xt = torch.from_numpy(np.ascontiguousarray(enc_in)).to(device_obj)
    eh = torch.from_numpy(graph.head)
    et = torch.from_numpy(graph.tail)
    ew = torch.from_numpy(graph.weight)
    ec = eh.numel()
    if ec == 0:
        raise ValueError("The fuzzy graph has no edges")
    mew = float(graph.weight.mean())
    A, B_ab = graph.a, graph.b

    opt = torch.optim.AdamW(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    @torch.no_grad()
    def embed_all(bs: int = 8192) -> np.ndarray:
        encoder.eval()
        out = [
            encoder(xt[i : i + bs]).cpu().numpy() for i in range(0, n, bs)
        ]
        encoder.train()
        return np.concatenate(out)

    history: list[float] = []
    for step in range(1, n_steps + 1):
        eid = torch.randint(ec, (batch_size,), generator=gen)
        anc = eh[eid].to(device_obj, dtype=torch.long)
        nr = et[eid].to(device_obj, dtype=torch.long)
        evw = ew[eid].to(device_obj) / mew

        cand = torch.randint(n, (batch_size, candidate_count), device=device_obj)
        cand = torch.where(cand == anc[:, None], (cand + 1) % n, cand)
        # ranking distances live in the geometry (PCA) space
        hd2 = (xt_geom[cand] - xt_geom[anc, None, :]).square().sum(-1)
        order = hd2.argsort(1)
        mid = cand.gather(1, order[:, 1:2]).squeeze(1)
        rf = cand.gather(1, order[:, -1:]).squeeze(1)

        neg = torch.randint(
            n, (batch_size, negative_sample_rate), device=device_obj
        )
        neg = torch.where(neg == anc[:, None], (neg + 1) % n, neg)

        ai = torch.cat((anc, nr, mid, rf, neg.reshape(-1)))
        az = encoder(xt[ai])
        Bz = batch_size
        za, zn, zm, zf = az[:Bz], az[Bz : 2 * Bz], az[2 * Bz : 3 * Bz], az[3 * Bz : 4 * Bz]
        zneg = az[4 * Bz :].reshape(Bz, negative_sample_rate, encoder.out_dim)

        nd2 = (za - zn).square().sum(-1)
        md2 = (za - zm).square().sum(-1)
        fd2 = (za - zf).square().sum(-1)
        ngd2 = (za[:, None, :] - zneg).square().sum(-1)

        pos = (evw * umap_pair_nll(nd2, a=A, b=B_ab, positive=True)).mean()
        negl = (
            evw[:, None]
            * umap_pair_nll(ngd2, a=A, b=B_ab, positive=False)
        ).sum(1).mean()
        rn, rm, rff = (
            0.5 * torch.log1p(nd2),
            0.5 * torch.log1p(md2),
            0.5 * torch.log1p(fd2),
        )
        rank = (
            rank_temperature
            * (
                F.softplus((rn - rm + rank_margin) / rank_temperature)
                + F.softplus((rm - rff + rank_margin) / rank_temperature)
            )
        ).mean()

        loss = pos + repulsion_strength * negl + rank_weight * rank

        # --- global Gram anchoring ---------------------------------------
        # Penalize the mismatch between the embedding's normalized pairwise
        # distances and the high-D (PCA-reduced) input's normalized pairwise
        # distances on a random subsample. Normalizing each matrix by its own
        # mean makes this anchor the *shape* of the global geometry, not its
        # absolute scale -- the same "fix the shape, not the distance"
        # principle as the landmark Gram anchor, but with no landmarks.
        if global_gram_weight > 0.0:
            m = min(global_gram_subsample, n)
            sub = torch.randint(n, (m,), device=device_obj)
            with torch.no_grad():
                # reference geometry = PCA (geom) space distances
                H = torch.cdist(xt_geom[sub], xt_geom[sub])
                H = H / H.mean().clamp_min(1e-9)
            zc = encoder(xt[sub])
            D = torch.cdist(zc, zc)
            D = D / D.mean().clamp_min(1e-9)
            gram = (D - H).square().mean()
            loss = loss + global_gram_weight * gram

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        opt.step()
        history.append(float(loss))

        if verbose and step % 25 == 0:
            print(
                f"step {step:5d}/{n_steps} loss={history[-1]:.4f} "
                f"avg25={np.mean(history[-25:]):.4f}"
            )
        if on_checkpoint is not None and (
            step % checkpoint_every == 0 or step == n_steps
        ):
            on_checkpoint(step, embed_all(), history[-1], encoder)

    def transform(X_new: np.ndarray) -> np.ndarray:
        # The encoder input space must match training: full standardized pixels
        # when pca_graph_only (PCA was geometry-only), else the PCA projection.
        X_new = np.asarray(X_new, dtype=np.float32)
        # apply the *training* standardization (not re-fit on X_new)
        xs_new = (X_new - train_mean) / train_scale
        z = xs_new if (comps is None or pca_graph_only) else (xs_new @ comps.T)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.from_numpy(z.astype(np.float32)).to(device_obj))
        encoder.train()
        return out.cpu().numpy()

    return {
        "embedding": embed_all(),
        "encoder": encoder,
        "history": history,
        "graph": graph,
        "pca_components": comps,
        "input_dim": input_dim,
        "geom_dim": xt_geom.shape[1],
        "pca_graph_only": bool(comps is not None and pca_graph_only),
        "transform": transform,
    }

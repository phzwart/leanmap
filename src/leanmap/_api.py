"""High-level, scikit-learn-flavored estimator wrapping the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ._graph import build_fuzzy_graph, standardize
from ._inducing import induce_embed, select_landmarks
from ._model import AttentionMapper, DeployableMapper, ParametricMapper
from ._train import train_attention_mapper, train_parametric_mapper, transform

FORMAT_VERSION = 3


@dataclass
class MapperConfig:
    """All knobs for the fit. Serialized alongside the weights."""

    # graph stage
    n_neighbors: int = 50
    min_dist: float = 0.1
    spread: float = 1.0
    index_kind: Literal["flat", "hnsw"] = "hnsw"
    use_gpu_for_flat: bool = True
    local_connectivity: float = 1.0
    prune_below: float = 1e-4
    scale_mode: Literal["zscore", "center", "none"] = "zscore"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    # train stage
    epochs: int = 25
    batch_size: int = 4096
    pairs_per_epoch: int | None = None
    negative_sample_rate: int = 5
    candidate_count: int = 6
    repulsion_strength: float = 1.0
    mid_weight_start: float = 1.0
    mid_weight_end: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (64, 64)
    n_components: int = 2                      # embedding dimensionality
    seed: int = 42
    # inducing-point (landmark) extension
    n_inducing: int = 0                       # 0 disables; e.g. 300 to enable
    landmark_method: Literal["fps", "kmeans", "hexgrid"] = "fps"
    induce_k: int = 5
    # conditioning mode for the encoder
    #   "none"      -> plain PCA+MLP (train_parametric_mapper)
    #   "attention" -> landmark cross-attention -> FiLM-modulated MLP
    #                  (requires n_inducing > 0)
    conditioning: Literal["none", "attention"] = "none"
    attn_dim: int = 64
    attn_heads: int = 4
    attn_layers: int = 2
    # learnable inducing points (attention mode). Landmarks start at the
    # data-anchored positions and are refined by the graph loss; the Gram-anchor
    # penalty keeps their relative geometry coherent to prevent runoff.
    learn_landmarks: bool = True
    landmark_lr_mult: float = 1.0
    gram_anchor_weight: float = 1.0
    # distance-bias kernel for landmark attention: "linear" (default, Laplacian
    # falloff, best held-out generalization), "squared" (Gaussian), or
    # "constant" (no distance prior; ablation only).
    distance_kernel: str = "linear"
    # sparse attention: each point attends only to its P nearest landmarks
    # (gathered, O(N*P) attention independent of M). None => dense over all M.
    # P~=20 matches dense held-out accuracy while ~1.5-2x faster at M=2000
    # (measured MNIST-5k, held-out; the O(M) nearest-P search is not eliminated,
    # so the speedup is bounded and grows with M).
    attend_top_p: int | None = None


class LeanMap:
    """Parametric UMAP: fit once, then embed new points forever.

    Example
    -------
    >>> mapper = LeanMap(n_neighbors=15, epochs=30)
    >>> emb = mapper.fit_transform(X_train)      # (n, 2)
    >>> emb_new = mapper.transform(X_test)       # any new data
    >>> mapper.save("model.mmap")
    >>> emb2 = LeanMap.load("model.mmap").transform(X_other)
    """

    def __init__(self, device: str | None = None, verbose: bool = True, **config: Any):
        self.config = MapperConfig(**config)
        self.device = device
        self.verbose = verbose
        self._mapper: DeployableMapper | None = None
        self.n_features_in_: int | None = None
        # inducing-point state (populated when config.n_inducing > 0)
        self.landmark_hd_: np.ndarray | None = None   # standardized landmark coords
        self.landmark_emb_: np.ndarray | None = None  # landmark 2D coordinates
        self._decoder = None                          # optional GenerativeDecoder
        self._discriminator = None                    # optional LeanmapDiscriminator

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        reference_coords: np.ndarray | None = None,
        order_constraints: list[dict] | None = None,
    ) -> "LeanMap":
        """Fit the parametric network and, if ``n_inducing > 0``, build landmarks.

        reference_coords : optional (n, 2) target embedding for the landmarks
            (e.g. coordinates from a reference ``umap-learn`` run). When omitted,
            landmarks inherit the trained network's own embedding of ``X``.
        order_constraints : optional supervised axis-ordering (attention mode).
            A list of dicts, each aligning an embedding axis to a label gradient::

                [{"axis": "x", "kind": "ordinal", "labels": y, "weight": 2.0},
                 {"axis": "y", "kind": "separate", "labels": is_prime,
                  "order": [0, 1], "weight": 2.0}]

            ``kind="ordinal"`` enforces a total order of class centroids along the
            axis (numeric or ordinal labels); ``kind="separate"`` pushes one group
            above another (binary/categorical). ``order`` gives the desired
            low->high id sequence (default: sorted unique labels).
        """
        c = self.config
        graph = build_fuzzy_graph(
            X,
            n_neighbors=c.n_neighbors,
            min_dist=c.min_dist,
            spread=c.spread,
            index_kind=c.index_kind,
            use_gpu_for_flat=c.use_gpu_for_flat,
            local_connectivity=c.local_connectivity,
            prune_below=c.prune_below,
            scale_mode=c.scale_mode,
            hnsw_m=c.hnsw_m,
            hnsw_ef_construction=c.hnsw_ef_construction,
            hnsw_ef_search=c.hnsw_ef_search,
        )
        self.n_features_in_ = int(np.asarray(X).shape[1])

        if c.conditioning == "attention":
            # Landmarks must exist BEFORE training (they are baked into the
            # attention encoder), so reference coordinates are required here.
            if c.n_inducing <= 0:
                raise ValueError("conditioning='attention' requires n_inducing > 0")
            if reference_coords is None:
                raise ValueError(
                    "conditioning='attention' requires reference_coords "
                    "(the network cannot self-derive landmark coordinates before "
                    "it is trained). Pass e.g. umap-learn coordinates for X."
                )
            xs, _, _ = standardize(np.asarray(X), mode=c.scale_mode)
            ref = np.asarray(reference_coords, dtype=np.float32)
            if ref.shape != (len(xs), c.n_components):
                raise ValueError(
                    "reference_coords must have shape "
                    f"(n_samples, n_components) = ({len(xs)}, {c.n_components})"
                )
            idx = select_landmarks(
                xs, ref, c.n_inducing, method=c.landmark_method, seed=c.seed
            )
            self.landmark_hd_ = xs[idx].astype(np.float32)
            self.landmark_emb_ = ref[idx].astype(np.float32)
            self._mapper = train_attention_mapper(
                graph,
                self.landmark_hd_,
                self.landmark_emb_,
                epochs=c.epochs,
                batch_size=c.batch_size,
                pairs_per_epoch=c.pairs_per_epoch,
                negative_sample_rate=c.negative_sample_rate,
                repulsion_strength=c.repulsion_strength,
                learning_rate=c.learning_rate,
                weight_decay=c.weight_decay,
                hidden_dims=tuple(c.hidden_dims),
                attn_dim=c.attn_dim,
                heads=c.attn_heads,
                attn_layers=c.attn_layers,
                learn_landmarks=c.learn_landmarks,
                landmark_lr_mult=c.landmark_lr_mult,
                gram_anchor_weight=c.gram_anchor_weight,
                distance_kernel=c.distance_kernel,
                attend_top_p=c.attend_top_p,
                order_constraints=order_constraints,
                seed=c.seed,
                device=self.device,
                verbose=self.verbose,
            )
            # persist the (possibly moved) learned landmark positions
            enc = self._mapper.encoder
            self.landmark_hd_ = enc.landmark_hd.detach().cpu().numpy().astype(np.float32)
            return self

        # default: plain PCA + MLP encoder
        self._mapper = train_parametric_mapper(
            graph,
            epochs=c.epochs,
            batch_size=c.batch_size,
            pairs_per_epoch=c.pairs_per_epoch,
            negative_sample_rate=c.negative_sample_rate,
            candidate_count=c.candidate_count,
            repulsion_strength=c.repulsion_strength,
            mid_weight_start=c.mid_weight_start,
            mid_weight_end=c.mid_weight_end,
            learning_rate=c.learning_rate,
            weight_decay=c.weight_decay,
            hidden_dims=tuple(c.hidden_dims),
            n_components=c.n_components,
            seed=c.seed,
            device=self.device,
            verbose=self.verbose,
        )

        if c.n_inducing and c.n_inducing > 0:
            # Standardize with the same stats the encoder uses, then pick
            # landmarks and give them reference (or network-derived) 2D coords.
            xs, _, _ = standardize(np.asarray(X), mode=c.scale_mode)
            if reference_coords is not None:
                ref = np.asarray(reference_coords, dtype=np.float32)
                if ref.shape != (len(xs), c.n_components):
                    raise ValueError(
                        "reference_coords must have shape "
                        f"(n_samples, n_components) = ({len(xs)}, {c.n_components})"
                    )
            else:
                ref = self.transform(X)
            idx = select_landmarks(
                xs, ref, c.n_inducing, method=c.landmark_method, seed=c.seed
            )
            self.landmark_hd_ = xs[idx].astype(np.float32)
            self.landmark_emb_ = ref[idx].astype(np.float32)
        return self

    def fit_transform(self, X: np.ndarray, reference_coords: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, reference_coords=reference_coords).transform(X)

    # -- inference ---------------------------------------------------------
    def transform(self, X: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
        if self._mapper is None:
            raise RuntimeError("Call fit() before transform().")
        return transform(self._mapper, X, batch_size=batch_size)

    __call__ = transform

    def induce_transform(self, X: np.ndarray, *, k: int | None = None) -> np.ndarray:
        """Embed new points via the stored landmarks (training-free extension).

        Places each query by its high-dimensional fuzzy membership to the
        nearest landmarks, generalizing like ``umap.transform``. Requires the
        model to have been fit with ``n_inducing > 0``.
        """
        if self.landmark_hd_ is None or self.landmark_emb_ is None:
            raise RuntimeError(
                "No landmarks. Fit with n_inducing > 0 to use induce_transform()."
            )
        xs, _, _ = standardize(np.asarray(X), mode=self.config.scale_mode)
        # reuse the stored standardization stats via the deployable buffers so
        # query scaling matches the landmarks exactly
        if self._mapper is not None:
            mean = self._mapper.input_mean.cpu().numpy()
            scale = self._mapper.input_scale.cpu().numpy()
            xs = ((np.asarray(X, dtype=np.float32) - mean) / scale).astype(np.float32)
        return induce_embed(
            xs,
            self.landmark_hd_,
            self.landmark_emb_,
            k=k if k is not None else self.config.induce_k,
            local_connectivity=self.config.local_connectivity,
        )

    # -- generative decoding (inverse: embedding -> image) -----------------
    def fit_decoder(
        self,
        X: np.ndarray,
        *,
        residual_dim: int = 15,
        flow_layers: int = 10,
        residual_mode: str = "pca",
        dec_epochs: int = 500,
        flow_epochs: int = 600,
        seed: int = 0,
        verbose: bool = False,
    ) -> "LeanMap":
        """Fit a generative decoder p(x|z) on this model's embedding of ``X``.

        Trains a mean decoder E[x|z] plus a conditional normalizing flow over the
        residuals, so ``decode_sample`` draws sharp, varied reconstructions at a
        manifold coordinate (not just the blurred mean). Attach after ``fit``.

        ``residual_mode='pca'`` models the top-``residual_dim`` principal
        directions of the residual (compact); ``'full'`` models every residual
        dimension (higher fidelity, removes the low-rank tell a discriminator
        exploits, at the cost of a larger flow).
        """
        from ._decoder import GenerativeDecoder

        if self._mapper is None:
            raise RuntimeError("Call fit() before fit_decoder().")
        Z = self.transform(np.asarray(X, dtype=np.float32))
        self._decoder = GenerativeDecoder(
            n_components=self.config.n_components,
            residual_dim=residual_dim,
            flow_layers=flow_layers,
            residual_mode=residual_mode,
            device=self.device,
        ).fit(
            Z,
            np.asarray(X, dtype=np.float32),
            dec_epochs=dec_epochs,
            flow_epochs=flow_epochs,
            seed=seed,
            verbose=verbose,
        )
        return self

    def _require_decoder(self):
        if getattr(self, "_decoder", None) is None:
            raise RuntimeError("No decoder. Call fit_decoder() first.")
        return self._decoder

    def decode_mean(self, Z: np.ndarray) -> np.ndarray:
        """Conditional mean image E[x|z] for embedding coordinates ``Z``."""
        return self._require_decoder().mean(np.asarray(Z, dtype=np.float32))

    def decode_sample(
        self, Z: np.ndarray, *, n_per: int = 1, temperature: float = 1.0, seed: int | None = None
    ) -> np.ndarray:
        """Draw (n_query, n_per, n_features) samples from p(x|z).

        ``temperature`` scales the latent noise: <1 sharper/tamer, >1 more varied.
        """
        return self._require_decoder().sample(
            np.asarray(Z, dtype=np.float32), n_per=n_per, temperature=temperature, seed=seed
        )

    def decode_logprob(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Residual-space log p(x|z): a manifold-consistency / novelty score."""
        return self._require_decoder().log_prob(
            np.asarray(X, dtype=np.float32), np.asarray(Z, dtype=np.float32)
        )

    # -- conformalized 'could be real' discriminator -----------------------
    def fit_discriminator(
        self,
        X: np.ndarray,
        *,
        calib_fraction: float = 0.28,
        n_regions: int = 4,
        min_regional_pool: int = 10,
        temperature: float = 1.2,
        input_scale: float = 16.0,
        epochs: int = 300,
        seed: int = 0,
    ) -> "LeanMap":
        """Train a LeanmapDiscriminator on real ``X`` vs this model's own samples.

        Requires ``fit_decoder`` first. Splits ``X`` into a classifier-training
        pool and a held-out calibration pool (``calib_fraction``); generated
        negatives come from ``decode_sample``. The result is a calibrated
        'could this be real?' test, stored on the model and persisted by ``save``.
        """
        from ._discriminator import LeanmapDiscriminator

        self._require_decoder()
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        n_cal = max(int(len(X) * calib_fraction), n_regions * n_regions)
        i_cal, i_disc = perm[:n_cal], perm[n_cal:]
        z_disc = self.transform(X[i_disc])
        gen = np.clip(
            np.round(
                self.decode_sample(z_disc, n_per=1, temperature=temperature, seed=seed).reshape(
                    len(i_disc), -1
                )
            ),
            0,
            input_scale,
        )
        self._discriminator = LeanmapDiscriminator(
            n_regions=n_regions,
            min_regional_pool=min_regional_pool,
            input_scale=input_scale,
            device=self.device,
        ).fit(
            X_real=X[i_disc],
            X_generated=gen,
            X_calib=X[i_cal],
            Z_calib=self.transform(X[i_cal]),
            epochs=epochs,
            seed=seed,
        )
        return self

    def _require_discriminator(self):
        if getattr(self, "_discriminator", None) is None:
            raise RuntimeError("No discriminator. Call fit_discriminator() first.")
        return self._discriminator

    def could_be_real(self, X: np.ndarray, *, alpha: float = 0.1, mondrian: bool = True) -> np.ndarray:
        """Boolean 'could this be real?' gate (conformal p >= alpha) for images ``X``."""
        disc = self._require_discriminator()
        Z = self.transform(np.asarray(X, dtype=np.float32)) if mondrian else None
        return disc.could_be_real(np.asarray(X, dtype=np.float32), Z, alpha=alpha, mondrian=mondrian)

    def realness_pvalue(self, X: np.ndarray, *, mondrian: bool = True) -> np.ndarray:
        """Conformal p-value that image ``X`` came from the training distribution."""
        disc = self._require_discriminator()
        Z = self.transform(np.asarray(X, dtype=np.float32)) if mondrian else None
        return disc.p_value(np.asarray(X, dtype=np.float32), Z, mondrian=mondrian)

    def sample_real(
        self, Z: np.ndarray, *, n_want: int = 8, alpha: float = 0.1, temperature: float = 1.2,
        mondrian: bool = True, seed: int = 0
    ):
        """Rejection-sample p(x|z), keeping only draws that pass the discriminator.

        Returns (kept_images, kept_pvalues, n_drawn). Needs both a decoder and a
        discriminator. Yield ~= accept rate; fidelity of kept samples is gated at
        ``alpha`` with a calibrated false-keep rate.
        """
        self._require_decoder()
        disc = self._require_discriminator()

        def _gen(z_batch, s):
            return np.clip(
                np.round(self.decode_sample(z_batch, n_per=1, temperature=temperature, seed=s).reshape(len(z_batch), -1)),
                0,
                disc.input_scale,
            )

        return disc.rejection_sample(
            _gen, Z, n_want=n_want, alpha=alpha, mondrian=mondrian, seed=seed
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize config + weights + standardization stats to one file."""
        if self._mapper is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        enc = self._mapper.encoder
        payload = {
            "format_version": FORMAT_VERSION,
            "config": {**asdict(self.config), "hidden_dims": list(self.config.hidden_dims)},
            "n_features_in": self.n_features_in_,
            "input_dim": enc.input_dim,
            "hidden_dims": list(enc.hidden_dims),
            "state_dict": self._mapper.state_dict(),
            "landmark_hd": self.landmark_hd_,
            "landmark_emb": self.landmark_emb_,
            "decoder": self._decoder.state_dict() if self._decoder is not None else None,
            "discriminator": (
                self._discriminator.state_dict() if self._discriminator is not None else None
            ),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "LeanMap":
        payload = torch.load(str(path), map_location=device or "cpu", weights_only=False)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {payload.get('format_version')}"
            )
        cfg = dict(payload["config"])
        cfg["hidden_dims"] = tuple(cfg.get("hidden_dims", (64, 64)))
        obj = cls(device=device, **cfg)

        input_dim = int(payload["input_dim"])
        hidden_dims = tuple(payload["hidden_dims"])
        n_components = int(cfg.get("n_components", 2))
        dummy_pca = np.zeros((n_components, input_dim), dtype=np.float32)
        lhd = payload.get("landmark_hd")
        lemb = payload.get("landmark_emb")

        if cfg.get("conditioning") == "attention":
            # AttentionMapper bakes the landmarks in as buffers; build with the
            # stored landmark shapes so load_state_dict restores everything.
            encoder = AttentionMapper(
                input_dim,
                dummy_pca,
                np.asarray(lhd, dtype=np.float32),
                np.asarray(lemb, dtype=np.float32),
                hidden_dims=hidden_dims,
                attn_dim=int(cfg.get("attn_dim", 64)),
                heads=int(cfg.get("attn_heads", 4)),
                attn_layers=int(cfg.get("attn_layers", 2)),
                learn_landmarks=bool(cfg.get("learn_landmarks", True)),
                distance_kernel=str(cfg.get("distance_kernel", "squared")),
                attend_top_p=cfg.get("attend_top_p", None),
            )
        else:
            encoder = ParametricMapper(input_dim, dummy_pca, hidden_dims)
        dummy_stat = np.zeros(input_dim, dtype=np.float32)
        mapper = DeployableMapper(encoder, dummy_stat, dummy_stat)
        mapper.load_state_dict(payload["state_dict"])
        obj._mapper = mapper.to(device or "cpu").eval()
        obj.n_features_in_ = payload.get("n_features_in")
        obj.landmark_hd_ = None if lhd is None else np.asarray(lhd, dtype=np.float32)
        obj.landmark_emb_ = None if lemb is None else np.asarray(lemb, dtype=np.float32)
        dec_state = payload.get("decoder")
        if dec_state is not None:
            from ._decoder import GenerativeDecoder

            obj._decoder = GenerativeDecoder.load_state(dec_state, device=device)
        disc_state = payload.get("discriminator")
        if disc_state is not None:
            from ._discriminator import LeanmapDiscriminator

            obj._discriminator = LeanmapDiscriminator.load_state(disc_state, device=device)
        return obj

    @property
    def torch_module(self) -> DeployableMapper:
        """The underlying nn.Module (raw-input -> 2D) for export/embedding."""
        if self._mapper is None:
            raise RuntimeError("Call fit() before accessing torch_module.")
        return self._mapper

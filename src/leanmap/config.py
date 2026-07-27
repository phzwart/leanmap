"""Dataclasses for all PLANE hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple


@dataclass
class PLANEConfig:
    """Hyperparameters for graph construction, model, losses, and training.

    At ``N <= 5k`` a parametric map has enough capacity to memorise, which is
    why :meth:`for_scale` shrinks the model and raises ``lambda_lip``. The
    advantage at that scale is a reusable, differentiable, out-of-sample-
    capable map — not better fidelity than UMAP.
    """

    # geometry
    n_neighbors: int = 15
    local_connectivity: int = 1
    # Reaches the loss only through ``find_ab_params``, which fits 1/(1+a d^2b);
    # the attractive force near contact then goes as d^(2b-1), so below b = 1 it
    # decays slower than the separation, a pair already close is pulled
    # proportionally harder, and neighbourhoods run away into knots. UMAP's 0.1
    # gives b = 0.895 and is safe there only because its SGD kernel clips
    # gradients, which this implementation does not do.
    #
    # Raised from 0.1 to 0.5 on measurement, not on that argument alone. A
    # 0.1-0.8 ladder on a uniformly sampled s-curve shows no knee where b
    # crosses 1 (at 0.199 * spread) -- kNN-spacing CV falls smoothly, 0.56 ->
    # 0.29 -- but the sign of its DRIFT during training does flip, at b ~ 1.3:
    # at 0.1-0.3 the layout keeps clumping the longer it trains (+0.05..0.08 CV
    # per 100 epochs), while from 0.5 up it is flat. 0.5 is the smallest value
    # that stops the runaway, and it lands below UMAP on uniformity (0.35 vs
    # 0.37). It is not free on clustered data: over 3 seeds on 8x8 digits it
    # costs 5-NN label accuracy 0.941 -> 0.915 (seed sd 0.016-0.024) while
    # halving the area distortion, 0.56 -> 0.34. Lower it toward 0.2 when class
    # separation matters more than an undistorted layout, accepting that the
    # layout there keeps clumping with the epoch budget. Below 0.2 is b < 1 and
    # is never safe.
    min_dist: float = 0.5
    spread: float = 1.0
    beta_multiplicity: float = 0.5
    hub_correction: bool = False
    lambda_backbone: float = 0.01
    # Near-duplicate collapse via ε-net before kNN. Default on — much faster at
    # large N and principled for tied / near-tied rows. Set False (or pass
    # epsilon=0) to keep every point as its own graph node.
    dedup: bool = True
    epsilon: Optional[float] = None  # None => estimate when dedup; ignored if dedup=False

    # Cohesive multi-scale graph pyramid (coarse levels supply long-range /
    # geodesic attraction so far regions do not drift apart). 0 => single-scale.
    # Default: 3 coarsenings (4 levels) with coarse-heavy weights + MST backbone.
    pyramid_scales: int = 3
    pyramid_rep_ratio: float = 4.0
    # Coarsening stops once a level reaches this size, so the number of levels
    # actually built is often < pyramid_scales + 1: with these defaults 4 levels
    # need N >~ 17k, and smaller N yields 3.
    pyramid_min_reps: int = 256
    # Per-level attraction weights (finest first); None => equal weight 1.0 each.
    # The last entry is the coarse/global weight and is the strongest lever in
    # this config for global (geodesic) fidelity and density preservation. If
    # more weights are given than levels were built, the coarsest weight is kept
    # and middle entries are dropped (with a warning) -- pass a tuple matching
    # the real level count to control it exactly, e.g. (1, 4, 16) at N=5000.
    pyramid_level_weights: Optional[Sequence[float]] = (1.0, 1.0, 2.0, 4.0)
    # Weight of bridge edges added to the COARSEST level to join disconnected
    # regions (0 = off); a no-op when that level is already connected. Keeps the
    # embedding from splitting into drifting islands without perturbing the
    # weights of edges that already exist.
    pyramid_coarse_backbone: float = 1.0

    # landmarks
    n_landmarks: int = 256
    c_buckets: int = 8
    # Select landmarks by farthest-point sampling on geodesic (kNN shortest-
    # path) distances instead of the ambient metric. Spreads anchors uniformly
    # over the intrinsic manifold; helps folded manifolds (S-curve, swiss roll).
    landmark_geodesic: bool = False
    # Geodesic Poisson-disk (blue-noise / Delone) landmark sampling: a maximal
    # set with a guaranteed minimum *geodesic* separation, auto-calibrated to
    # ~n_landmarks. More uniform interior coverage than FPS (which over-samples
    # boundaries/tips). Takes precedence over ``landmark_geodesic`` when set.
    landmark_poisson: bool = False
    # kNN neighbors for the geodesic graph (None => use ``n_neighbors``). Shared
    # by geodesic FPS and geodesic Poisson-disk sampling.
    landmark_geodesic_k: Optional[int] = None
    # Conditioning pyramid: extra COARSE anchor sets (each a frozen MODULATOR
    # FiLM factor) added on top of the fine PRIMARY anchors. Anchor counts,
    # coarsest-first, e.g. (32, 96). Each level's temperature auto-scales with
    # its spacing (coarse => large tau / broad modulation; fine => sharp), so a
    # single ``tau_scale`` yields genuinely multi-resolution tau. None => the
    # single-resolution conditioning (current default).
    conditioning_pyramid_levels: Optional[Sequence[int]] = None
    learn_landmarks: bool = True
    learn_tau: bool = True
    # Multiplies the default per-anchor temperature at init. >1 spreads each
    # point's affinity over more surrounding landmarks (softer, less clumpy);
    # <1 sharpens toward single-landmark assignment.
    tau_scale: float = 1.0
    # If set, ignore ``_default_tau`` / ``tau_scale`` and initialise every
    # anchor temperature to this constant (still subject to ``tau_min`` floor
    # inside ``AnchorAffinity.tau()``).
    tau_init: Optional[float] = None

    # model
    width: int = 384
    depth: int = 3
    d_out: int = 2
    hyper_width: int = 128
    spectral_norm: bool = True
    use_decoder: bool = False
    concat_affinity: bool = False
    # Linear PCA skip: output is ``pca(x_n) + residual`` with the residual head
    # initialized near zero, so training starts AT plain PCA. Convenient, but the
    # unconstrained linear path can end up supplying most of the layout: on 8x8
    # digits it pins 5-NN label accuracy to ~0.65 against PCA-2D's 0.60 no matter
    # how the graph or losses are tuned. Turning it off frees the map but leaves
    # the 1e-4-initialized head undertrained at the default ``lr``, which scores
    # WORSE (~0.41) -- so pca_skip=False needs ``lr`` raised with it (5e-3..2e-2
    # reached ~0.94, matching UMAP). Treat the two as one decision, not two.
    pca_skip: bool = True
    # classical PCA centers before SVD; False = uncentered (ablation)
    pca_center: bool = True

    # losses
    n_negatives: int = 5
    lambda_ord: float = 0.1
    lambda_rec: float = 0.1
    lambda_lip: float = 0.0
    # Local-rigidity (as-rigid-as-possible) loss on fine-graph neighbourhoods:
    # matches each node's neighbour Gram matrix (all pairwise offset inner
    # products) up to one scale, so it constrains edge length *and* relative
    # orientation and opposes the parametric frame-rotation twist/pinch. This
    # replaces the older length-only ``lambda_iso`` term. 0 = off.
    lambda_frame: float = 0.0
    # Neighbours per star (padded) and centres sampled per step for the frame
    # loss. ``frame_centers=None`` => 128. Kept small so the extra forward is
    # cheap relative to ``batch_edges``.
    frame_neighbors: int = 6
    frame_centers: Optional[int] = None
    # Geodesic/tangent-aware rigidity: estimate each star's local tangent plane,
    # drop off-tangent (across-sheet shortcut) neighbours and match Grams in the
    # tangent frame. Required for manifolds that fold back on themselves (swiss
    # roll); harmless on non-folding ones (S-curve). ``frame_tangent_dim=None``
    # uses the embedding dim ``d_out``.
    frame_tangent: bool = True
    frame_tangent_dim: Optional[int] = None
    frame_normal_thresh: float = 0.5
    # Ramp (start, end) as fractions of training for the frame weight. Local
    # rigidity cannot distinguish a rolled manifold from its unrolled isometry
    # and penalises the *transient* stretch of unrolling, so on fold-back
    # manifolds it must switch on only AFTER the neighbour-embedding loss has
    # globally unrolled (e.g. (0.5, 0.75)); it then just cleans up width/twist.
    # (0.0, 0.0) => on from the start (fine for non-folding manifolds).
    frame_ramp: Tuple[float, float] = (0.0, 0.0)
    # Coarse geodesic (Isomap) backbone: classical MDS of landmark–landmark
    # graph geodesics + Procrustes pull of landmark embeddings toward that
    # layout (plus mild pairwise stress). Pins the global metric gauge —
    # straightens the banana and untwists frame flips that pairwise stress
    # alone cannot escape. 0 = off. Default 0.5: strong enough to unroll
    # S-curve / swiss-roll bananas without dominating local affinity.
    lambda_geo: float = 0.5
    # Landmark pairs sampled per step for the geodesic stress (None =>
    # min(L*(L-1)/2, 2048)).
    geo_pairs: Optional[int] = None
    # Ramp (start, end) as fractions of training for the geodesic weight.
    # Mild delay lets local affinity establish topology before the global
    # metric gauge locks in. (0.0, 0.0) => on from the start.
    # With ``geo_ramp_down=True`` the schedule is inverted: full weight until
    # ``start``, linear down to 0 by ``end``, then off — e.g. (0.0, 0.25)
    # front-loads the MDS gauge then releases it.
    geo_ramp: Tuple[float, float] = (0.2, 0.45)
    geo_ramp_down: bool = False
    # SIGReg (LeJEPA) isotropic-Gaussian anti-collapse regularizer
    lambda_sigreg: float = 0.0
    sigreg_slices: int = 256
    sigreg_points: int = 17
    sigreg_domain: float = 5.0
    sigreg_target_std: float = 1.0
    lambda_lm: float = 0.1
    eta_balance: float = 1.0
    whiten_multi_axis: bool = True

    # Negative-space co-training (opt-in). An auxiliary distance-to-support
    # quantile head is trained *jointly* with the encoder; its pinball loss also
    # back-props into the backbone, so the embedding learns to keep off-manifold
    # ("negative space") geometry legible in its internal states. The regression
    # target (ambient min-distance to the train support) is fixed, so this is a
    # stationary auxiliary task. 0 = off (default: purely two-stage frozen probe).
    # The head is re-calibrated (CQR) on the frozen model after training; use a
    # small, ramped weight so it does not distort the on-manifold chart.
    lambda_dist: float = 0.0
    # Ramp (start, end) as fractions of training — switch on only after the
    # chart has formed, like the geodesic/frame terms.
    dist_ramp: Tuple[float, float] = (0.5, 0.75)
    dist_alpha: float = 0.1              # miscoverage; head targets 1 - alpha
    dist_perturb_per_step: int = 256     # perturbations scored per step
    dist_r_min_mult: float = 0.25        # min perturbation radius / median 1-NN
    dist_r_max_mult: float = 25.0        # max perturbation radius / median 1-NN
    dist_head_width: int = 128
    dist_head_depth: int = 2
    dist_features: Optional[Sequence[str]] = None  # None => ALL_FEATURES

    # optimisation
    batch_edges: int = 4096
    epochs: int = 200
    # Tuned for the ``pca_skip=True`` path, where the head only has to refine an
    # already-sensible PCA layout. With ``pca_skip=False`` the head must build the
    # layout from a near-zero init and needs 5e-3..2e-2; see ``pca_skip``.
    lr: float = 1e-3
    # If set, hold ``lr`` for ``lr_switch_epochs``, then switch to ``lr_after``
    # (disables the default warmup+cosine schedule).
    lr_after: Optional[float] = None
    lr_switch_epochs: int = 0
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    align_ramp: Tuple[float, float] = (0.3, 0.6)

    # conformal
    calib_frac: float = 0.05
    calib_max: int = 2000

    # metric / knn
    metric: str = "l2"
    knn_mode: str = "auto"
    c_search: int = 8

    seed: int = 0
    device: Optional[str] = None

    @classmethod
    def for_scale(cls, N: int) -> "PLANEConfig":
        """Return scale-appropriate presets for dataset size ``N``."""
        base = cls()
        if N <= 5_000:
            return replace(
                base,
                width=128,
                depth=2,
                n_landmarks=32,
                n_neighbors=10,
                lambda_lip=0.1,
                epochs=500,
                calib_max=200,
            )
        if N <= 200_000:
            return replace(
                base,
                width=384,
                depth=3,
                n_landmarks=256,
                n_neighbors=15,
                lambda_lip=0.01,
                epochs=200,
                calib_max=2000,
            )
        return replace(
            base,
            width=512,
            depth=3,
            n_landmarks=512,
            n_neighbors=15,
            lambda_lip=0.0,
            epochs=50,
            calib_max=2000,
        )


@dataclass
class AlignmentSpec:
    """Steering constraint: axial property or regional label targets.

    Attributes
    ----------
    axis : int
        Embedding axis to align (axial kind).
    values : Tensor (N,)
        Per-point property values (axial).
    kind : {"axial", "regional"}
    weight : float
    sign : +1 | -1
    labels : Tensor (N,), optional
        For regional alignment.
    targets : dict[label -> (d_out,)], optional
        Target centroids per label.
    """

    axis: int = 0
    values: Optional[object] = None  # torch.Tensor (N,)
    kind: str = "axial"
    weight: float = 1.0
    sign: int = 1
    labels: Optional[object] = None
    targets: Optional[dict] = None

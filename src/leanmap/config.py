"""Dataclasses for all PLANE hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Plumbing constants — never varied as knobs; demoted from PLANEConfig.
# ---------------------------------------------------------------------------
BETA_MULTIPLICITY: float = 0.5
LAMBDA_BACKBONE: float = 0.01
C_BUCKETS: int = 8
C_SEARCH: int = 8
PYRAMID_REP_RATIO: float = 4.0
HYPER_WIDTH: int = 128
ETA_BALANCE: float = 1.0
WEIGHT_DECAY: float = 1e-4
WARMUP_FRAC: float = 0.05
# None => min(L*(L-1)/2, 2048) at the geodesic stress sample site.
GEO_PAIRS: Optional[int] = None
# None => fall back to ``n_neighbors``.
LANDMARK_GEODESIC_K: Optional[int] = None
PCA_CENTER: bool = True
CALIB_FRAC: float = 0.05
SPREAD: float = 1.0
LAMBDA_ORD: float = 0.1
# Frame sub-dials (frame-keep plan): only λ / ramp / tangent / neighbors stay
# on the config; these are fixed.
FRAME_CENTERS: int = 128
FRAME_TANGENT_DIM: Optional[int] = None  # None => embedding dim ``d_out``
FRAME_NORMAL_THRESH: float = 0.5
# Warn above this share of negative eigenvalue mass in the classical-MDS Gram.
# Diagnostic only -- no loss weight is adjusted automatically.
MDS_NEG_EIGEN_WARN: float = 0.10


@dataclass
class PLANEConfig:
    """Hyperparameters for graph construction, model, losses, and training.

    At ``N <= 5k`` :meth:`for_scale` ships the measured digits/s-curve recipe
    (no PCA skip, raised ``lr``, mid capacity, ``lambda_geo=0.15``). Larger
    ``N`` keeps a milder default schedule.
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
    #
    # Note what that measurement was made on: a UNIFORMLY sampled s-curve, whose
    # intrinsic dimension equals the embedding dimension and which therefore has
    # no density contrast to reproduce. The criterion it fixes -- spacing CV
    # stops drifting -- is a statement about training stability, not about
    # whether the layout's density matches the data's. On data with real density
    # structure, or with dimensions being discarded, this constant decides how
    # much contrast the map shows while knowing nothing about how much the data
    # has. ``lambda_density`` constrains which neighbourhoods end up crowded
    # regardless of where this constant puts the overall contrast.
    min_dist: float = 0.5
    # Near-duplicate collapse via ε-net before kNN. Default on — much faster at
    # large N and principled for tied / near-tied rows. Set False (or pass
    # epsilon=0) to keep every point as its own graph node.
    dedup: bool = True
    # None => estimate when dedup; ignored if dedup=False. ``fit`` writes the
    # resolved value back, so a saved artefact carries a fixed merge radius.
    epsilon: Optional[float] = None
    # Exponent on the cell-multiplicity reweighting (w_i w_j)^beta of edge
    # memberships. This is a statement about what a duplicate row *means*, so it
    # belongs to the dataset, not to the plumbing. Use beta -> 0 when repeated
    # rows are a deposition artefact (PDB resubmissions, redundant crystal
    # forms) and carry no density information; beta -> 1 when multiplicity is
    # genuine sampling density that the layout should reflect. 0.5 splits the
    # difference and is the historical default.
    beta_multiplicity: float = BETA_MULTIPLICITY

    # Cohesive multi-scale graph pyramid (coarse levels supply long-range /
    # geodesic attraction so far regions do not drift apart). 0 => single-scale.
    # Default: 3 coarsenings (4 levels) with coarse-heavy weights + MST backbone.
    pyramid_scales: int = 3
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

    # How aggregated coarse crossing weights are mapped into a membership.
    # "rational" is w/(w + median): monotone and unsaturating, so the strongest
    # coarse edges keep their ranking. "quantile_clamp" is the older
    # min(w/q99, 1), which flattens the top 1% to a common weight — the very
    # edges a (1, 2, 8) pyramid exists to exploit. The two differ in magnitude,
    # so pyramid_level_weights does not transfer between them.
    pyramid_squash: str = "rational_q99"

    # landmarks
    n_landmarks: int = 256
    # Select landmarks by farthest-point sampling on geodesic (kNN shortest-
    # path) distances instead of the ambient metric. Spreads anchors uniformly
    # over the intrinsic manifold; helps folded manifolds (S-curve, swiss roll).
    landmark_geodesic: bool = False
    # Geodesic Poisson-disk (blue-noise / Delone) landmark sampling: a maximal
    # set with a guaranteed minimum *geodesic* separation, auto-calibrated to
    # ~n_landmarks. More uniform interior coverage than FPS (which over-samples
    # boundaries/tips). Takes precedence over ``landmark_geodesic`` when set.
    landmark_poisson: bool = False
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
    # Linear PCA skip: output is ``pca(x_n) + residual`` with the residual head
    # initialized near zero, so training starts AT plain PCA. Convenient, but the
    # unconstrained linear path can end up supplying most of the layout: on 8x8
    # digits it pins 5-NN label accuracy to ~0.65 against PCA-2D's 0.60 no matter
    # how the graph or losses are tuned. Turning it off frees the map but leaves
    # the 1e-4-initialized head undertrained at the default ``lr``, which scores
    # WORSE (~0.41) -- so pca_skip=False needs ``lr`` raised with it (5e-3..2e-2
    # reached ~0.94, matching UMAP). Treat the two as one decision, not two.
    pca_skip: bool = True

    # Learning-rate multiplier for the residual head and FiLM hypernetworks
    # relative to the PCA skip and backbone. With pca_skip=True a single flat
    # rate couples two decisions -- how fast the skip drifts from its PCA seed
    # and how fast the head builds a residual on top of it. Raising this
    # (10-20x) decouples them. 1.0 reproduces the flat-rate behaviour exactly
    # and is the default until the ablation says otherwise. Ignored when
    # pca_skip=False, where there is no skip to hold back.
    pca_lr_mult: float = 1.0
    # "film" conditions the backbone through per-layer scale/shift produced from
    # a(x); "concat" appends a(x) to the input of an otherwise identical MLP.
    # Since a(x) is a deterministic function of x, concat is the control that
    # says whether the FiLM apparatus earns its complexity, not a weaker model.
    conditioning: str = "film"

    # losses
    n_negatives: int = 5
    # Local-rigidity (as-rigid-as-possible) loss on fine-graph neighbourhoods:
    # matches each node's neighbour Gram matrix (all pairwise offset inner
    # products) up to one scale, so it constrains edge length *and* relative
    # orientation and opposes the parametric frame-rotation twist/pinch. This
    # replaces the older length-only ``lambda_iso`` term. 0 = off.
    lambda_frame: float = 0.0
    # Neighbours per star (padded) for the frame loss.
    frame_neighbors: int = 6
    # Geodesic/tangent-aware rigidity: estimate each star's local tangent plane,
    # drop off-tangent (across-sheet shortcut) neighbours and match Grams in the
    # tangent frame. Required for manifolds that fold back on themselves (swiss
    # roll); harmless on non-folding ones (S-curve).
    frame_tangent: bool = True
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
    # Ramp (start, end) as fractions of training for the geodesic weight.
    # Mild delay lets local affinity establish topology before the global
    # metric gauge locks in. (0.0, 0.0) => on from the start.
    geo_ramp: Tuple[float, float] = (0.2, 0.45)
    # Split of the geodesic term: L_geo = lambda_anchor * L_anchor + 0.25 *
    # L_stress, the whole thing scaled by lambda_geo * ramp. The two halves do
    # different jobs — the Procrustes anchor pins an absolute gauge against a
    # classical-MDS layout, while stress only constrains pairwise landmark
    # distances and is blind to a global twist. Set lambda_anchor=0 to keep
    # metric fidelity while letting the local frame term choose the gauge; that
    # is the right test when the MDS negative-eigenvalue ratio is large and the
    # anchor target is not faithfully embeddable.
    lambda_anchor: float = 1.0
    lambda_lm: float = 0.1

    # Which neighbourhoods come out crowded. Left alone that is decided by
    # wherever the attraction/repulsion equilibrium lands, which is a property of
    # ``min_dist`` and not of the data -- so a map can invent clumps, or flatten
    # real ones, with nothing in the objective objecting. The density term
    # correlates each neighbourhood's log radius with the ambient graph's, which
    # constrains the *ordering* of density and deliberately not its magnitude
    # (see ``leanmap.density`` for why targeting magnitude cannot work when the
    # intrinsic dimension greatly exceeds ``d_out``).
    #
    # Being scale free removes a way to fail but does not make the term free:
    # honouring density costs local structure on data whose density signal is
    # mostly discrete cluster structure. Sweeping this on digits degrades
    # ``trust_5`` monotonically while s_curve improves on every axis, so the
    # tradeoff is a property of the data (see ``leanmap.density``). 1.0 puts the
    # term on the same footing as the other unit-scale terms, being the score of
    # a layout whose density is unrelated to the data's. Set 0 to opt out.
    lambda_density: float = 1.0
    # Ramp (start, end) as fractions of training, like ``geo_ramp``. Equal values
    # give a hard gate. Density is a refinement of a layout that already has the
    # right topology, and steering a forming one locks in wrong neighbourhoods,
    # so the term is held off until the tail -- densMAP's ``dens_frac=0.3``, i.e.
    # plain UMAP for the first 70% of epochs, expressed as a gate at 0.7.
    density_ramp: Tuple[float, float] = (0.7, 0.7)
    # Variance floor in the correlation denominator, ``density.DENSITY_VAR_SHIFT``
    # (densMAP's ``dens_var_shift``). Repeated rather than imported: ``density``
    # imports ``graph``, which imports this module.
    density_var_shift: float = 0.1
    # Stars per step for the density term (reuses the frame-loss sampler).
    density_centers: int = 256

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

    # Regression steps fitting the encoder to the landmark geodesic MDS, extended
    # to every point by its affinity barycentre, before the main loop starts (see
    # ``leanmap.warmstart``). A regression step costs about a ninth of a training
    # step -- one forward, no negatives or triplets -- so this is cheap next to
    # the epochs it aims to save. 0 disables it, which is the default until the
    # timing and quality comparison says otherwise. Requires ``lambda_geo > 0``,
    # since that is what builds the MDS.
    warm_start_steps: int = 0
    # Learning rate for those steps; None uses ``lr``. The regression is a much
    # easier problem than the embedding objective and tolerates more.
    warm_start_lr: Optional[float] = None
    # Which coarse layout to start from: "isomap" (classical MDS of the landmark
    # geodesics), "spectral" (leading eigenvectors of the fuzzy graph, UMAP's
    # init), "pca", or "auto". Neither of the first two dominates -- Isomap wins on
    # manifolds that are genuinely a sheet, spectral wins when the geodesics are
    # not realisable in ``d_out`` dimensions and Isomap's MDS spectrum goes
    # negative. "auto" does not guess from that diagnostic: it interpolates each
    # candidate onto the graph's representatives and scores neighbour agreement
    # against the kNN the build already computed, which is the same quantity the
    # layouts are being chosen for. See ``leanmap.warmstart.rank_inits``.
    warm_start_layout: str = "auto"

    # Fraction of epochs spent climbing the pyramid from the coarsest level up,
    # admitting one finer level at a time, before all levels run together. Steps
    # per epoch follow the *active* levels' edge count, so a coarse epoch costs
    # roughly ``PYRAMID_REP_RATIO`` times less per coarsening -- the early epochs
    # decide global layout, which coarse edges already carry, so paying
    # fine-graph prices for them is waste. 0 keeps every epoch at the full mix,
    # which is the behaviour every recipe here was tuned against.
    coarse_first_frac: float = 0.0

    # conformal
    calib_max: int = 2000

    # knn
    knn_mode: str = "auto"

    seed: int = 0
    device: Optional[str] = None

    @classmethod
    def for_scale(cls, N: int) -> "PLANEConfig":
        """Return scale-appropriate presets for dataset size ``N``."""
        base = cls()
        if N <= 5_000:
            # Measured DIGITS_MATCHED recipe (digits / s-curve scale).
            return replace(
                base,
                width=384,
                depth=3,
                n_landmarks=128,
                n_neighbors=15,
                epochs=240,
                calib_max=200,
                pca_skip=False,
                lr=2e-2,
                lambda_geo=0.15,
                pyramid_level_weights=(1.0, 2.0, 8.0),
            )
        if N <= 200_000:
            return replace(
                base,
                width=384,
                depth=3,
                n_landmarks=256,
                n_neighbors=15,
                epochs=200,
                calib_max=2000,
            )
        return replace(
            base,
            width=512,
            depth=3,
            n_landmarks=512,
            n_neighbors=15,
            epochs=50,
            calib_max=2000,
        )

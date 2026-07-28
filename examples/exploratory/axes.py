"""Named baselines and Phase-1 axis sweeps (data-agnostic config overlays)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RunSpec:
    """One training config in a sweep.

    ``overlay`` keys match ``fit_embed`` kwargs (and thus ``PLANEConfig`` fields).
    ``axis`` groups atlas rows; ``level`` is the human-readable level within that axis.
    """

    run_id: str
    axis: str
    level: str
    overlay: Dict[str, Any] = field(default_factory=dict)


# Data-agnostic Phase-1 baseline: frozen landmarks, soft tau, cohesive pyramid,
# geodesic backbone on (matches PLANEConfig default), local frame off.
BASELINE: Dict[str, Any] = {
    "pyramid_scales": 3,
    "pyramid_level_weights": (1.0, 1.0, 1.0, 1.0),
    "pyramid_coarse_backbone": 1.0,
    "n_landmarks": 128,
    "learn_landmarks": False,
    "learn_tau": False,
    "tau_scale": 1.0,
    "landmark_geodesic": False,
    "landmark_poisson": False,
    "n_neighbors": 10,
    "local_connectivity": 1,
    # Tracks the library default; 0.1 puts the attraction curve at b = 0.895,
    # which collapses neighbourhoods into knots (s-curve spacing_cv 0.58 vs
    # UMAP's 0.37). See PLANEConfig.min_dist.
    "min_dist": 0.5,
    "n_negatives": 5,
    "pca_skip": True,
    "lambda_geo": 0.5,
    "geo_ramp": (0.2, 0.45),
    "lambda_frame": 0.0,
    "frame_ramp": (0.0, 0.0),
    "frame_tangent": True,
    "lambda_lm": 0.1,
}


def _rid(axis: str, level: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{axis}__{level}")
    return safe


def _run(axis: str, level: str, **overlay: Any) -> RunSpec:
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=dict(overlay))


def phase1_runs() -> List[RunSpec]:
    """Lean Phase-1 sweep: baseline + 1D ablations + two interactions."""
    runs: List[RunSpec] = [
        _run("baseline", "default"),
    ]

    # A. Graph topology
    for k in (5, 10, 30):
        if k == BASELINE["n_neighbors"]:
            continue
        runs.append(_run("n_neighbors", str(k), n_neighbors=k))

    runs.append(_run("pyramid_scales", "0", pyramid_scales=0))
    # baseline already has scales=3; backbone ablation
    runs.append(
        _run(
            "pyramid_coarse_backbone",
            "0",
            pyramid_scales=3,
            pyramid_coarse_backbone=0.0,
        )
    )

    # B. Landmark system
    runs.append(
        _run(
            "landmark_mode",
            "geodesic",
            landmark_geodesic=True,
            landmark_poisson=False,
        )
    )
    runs.append(
        _run(
            "landmark_mode",
            "poisson",
            landmark_geodesic=False,
            landmark_poisson=True,
        )
    )
    for L in (32, 256):
        runs.append(_run("n_landmarks", str(L), n_landmarks=L))
    runs.append(_run("learn_landmarks", "on", learn_landmarks=True))
    for ts in (0.5, 2.0):
        runs.append(_run("tau_scale", str(ts), tau_scale=ts, learn_tau=False))
    runs.append(_run("learn_tau", "on", learn_tau=True, tau_scale=1.0))

    # C. Global structure losses — baseline has λ_geo=0.5; explore around it
    for w in (0.0, 0.1, 0.25, 1.0):
        runs.append(
            _run(
                "lambda_geo",
                str(w),
                lambda_geo=w,
                geo_ramp=(0.2, 0.45),
            )
        )
    runs.append(
        _run(
            "lambda_frame",
            "0.5_early",
            lambda_frame=0.5,
            frame_ramp=(0.0, 0.0),
        )
    )
    runs.append(
        _run(
            "lambda_frame",
            "0.5_delayed",
            lambda_frame=0.5,
            frame_ramp=(0.5, 0.75),
        )
    )

    # D. Packing
    runs.append(_run("min_dist", "0.3", min_dist=0.3))

    # E. Init
    runs.append(_run("pca_skip", "off", pca_skip=False))

    # Interactions
    runs.append(
        _run(
            "interaction",
            "geo_x_frame_delayed",
            lambda_geo=0.5,
            geo_ramp=(0.2, 0.45),
            lambda_frame=0.5,
            frame_ramp=(0.5, 0.75),
        )
    )
    runs.append(
        _run(
            "interaction",
            "poisson_x_geo",
            landmark_poisson=True,
            landmark_geodesic=False,
            lambda_geo=0.5,
            geo_ramp=(0.2, 0.45),
        )
    )

    return runs


def frame_weight_runs() -> List[RunSpec]:
    """λ_geo fixed at baseline (0.5); vary local-frame (ARAP) weight + ramp.

    Delayed ramp is the main series (frame after geo unrolls). One early-ramp
    point at 0.5 for comparison.
    """
    runs: List[RunSpec] = [
        _run("baseline", "geo_only"),  # λ_frame=0, λ_geo=0.5
    ]
    for w in (0.1, 0.25, 0.5, 1.0):
        runs.append(
            _run(
                "lambda_frame",
                f"{w}_delayed",
                lambda_frame=w,
                frame_ramp=(0.5, 0.75),
            )
        )
    runs.append(
        _run(
            "lambda_frame",
            "0.5_early",
            lambda_frame=0.5,
            frame_ramp=(0.0, 0.0),
        )
    )
    return runs


# Corrected baseline for matching UMAP. Differences from BASELINE that matter:
#
# - ``pyramid_level_weights`` is a 3-TUPLE. Only 3 levels are built at N in the
#   low thousands (coarsening floors at ``pyramid_min_reps=256``), and a 4th entry
#   never reaches a level. The last entry is the coarse/global weight.
# - ``n_landmarks`` / ``tau_scale`` come from ``calibrate.py`` for the dataset
#   rather than being inherited constants; ``--target-perp`` overrides tau_scale
#   per run so it tracks the anchor geometry.
# - ``pyramid_coarse_backbone`` is left on because it now self-disables when the
#   coarsest level is already connected.
UMAP_MATCH_BASELINE: Dict[str, Any] = {
    **BASELINE,
    "pyramid_level_weights": (1.0, 2.0, 8.0),
    "n_landmarks": 179,
    "tau_scale": 0.089,
    "n_neighbors": 15,  # UMAP's default; leanmap shipped 10
    "lambda_geo": 0.5,
}


def _mrun(axis: str, level: str, **overlay: Any) -> RunSpec:
    """Run spec whose overlay starts from ``UMAP_MATCH_BASELINE``."""
    ov = dict(UMAP_MATCH_BASELINE)
    ov.update(overlay)
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)


def umap_match_runs() -> List[RunSpec]:
    """Corrected baseline on 3 seeds: establishes the gap to the UMAP bar."""
    return [_mrun("baseline", "corrected")]


def weights_runs() -> List[RunSpec]:
    """Coarse-emphasis ladder on the 3 levels that actually exist."""
    return [
        _mrun("weights", "1_1_1", pyramid_level_weights=(1.0, 1.0, 1.0)),
        _mrun("weights", "1_1_2", pyramid_level_weights=(1.0, 1.0, 2.0)),
        _mrun("weights", "1_2_8", pyramid_level_weights=(1.0, 2.0, 8.0)),
        _mrun("weights", "1_4_16", pyramid_level_weights=(1.0, 4.0, 16.0)),
        _mrun("weights", "8_1_1", pyramid_level_weights=(8.0, 1.0, 1.0)),
    ]


def epochs_runs() -> List[RunSpec]:
    """Training length, pinned per run so the axis sweeps in one invocation.

    Every other axis is flat on digits while accuracy sits at PCA level, so the
    remaining suspect is simply that 60 epochs has not converged. UMAP itself
    runs 200-500 epochs.
    """
    return [_mrun("epochs", str(e), epochs=e) for e in (30, 60, 120, 240, 480)]


def neighbors_runs() -> List[RunSpec]:
    return [_mrun("n_neighbors", str(k), n_neighbors=k) for k in (10, 15, 30)]


def landmarks_runs() -> List[RunSpec]:
    return [_mrun("n_landmarks", str(L), n_landmarks=L) for L in (32, 90, 179, 450)]


NOSKIP_BASELINE: Dict[str, Any] = {
    **UMAP_MATCH_BASELINE,
    # Only this pair together escapes PCA: the skip must go so the layout is not
    # anchored to PCA, and lr must rise so the residual head (init std 1e-4) can
    # actually train. Either change alone scores WORSE than leaving both alone.
    "pca_skip": False,
    "lr": 2e-2,
}


def _nrun(axis: str, level: str, **overlay: Any) -> RunSpec:
    """Run spec whose overlay starts from ``NOSKIP_BASELINE``."""
    ov = dict(NOSKIP_BASELINE)
    ov.update(overlay)
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)


def refine_runs() -> List[RunSpec]:
    """Close the last 0.05 from the no-skip base (0.933) to the UMAP bar (0.987).

    Trustworthiness is the clearest remaining shortfall (0.937 vs 0.988) while
    kNN overlap already beats UMAP, so the layout is right and the residual gap
    is local tightening plus training length.
    """
    return [
        _nrun("ref", "base"),
        _nrun("ref", "e240", epochs=240),
        _nrun("ref", "mindist0", min_dist=0.0),
        _nrun("ref", "nn30", n_neighbors=30),
        _nrun("ref", "geo0", lambda_geo=0.0),
        _nrun("ref", "e240_mindist0", epochs=240, min_dist=0.0),
    ]


DIGITS_MATCHED: Dict[str, Any] = {
    **NOSKIP_BASELINE,
    # Best balanced point on digits: the only setting strong on all of label
    # accuracy, ARI, trustworthiness, kNN overlap, geodesic and density. Dropping
    # lambda_geo to 0 buys ~0.01 accuracy but costs geodesic fidelity (0.72 ->
    # 0.50, i.e. below PCA), so global structure is kept instead.
    "lambda_geo": 0.15,
    "epochs": 240,
}


PDB_MATCHED: Dict[str, Any] = {
    **DIGITS_MATCHED,
    # Re-derived for PDB (N=5000, D=9): calibrate.py gives L=450 at coverage 3.3.
    # It predicts 4 pyramid levels at the full N, but a 0.2 holdout fits only 4000
    # points and builds 3, so the tuple is written with 3 entries to avoid being
    # truncated. tau_scale is recalibrated per dataset by --target-perp.
    "n_landmarks": 450,
    "pyramid_level_weights": (1.0, 2.0, 8.0),
}


def _prun(axis: str, level: str, **overlay: Any) -> RunSpec:
    """Run spec whose overlay starts from ``PDB_MATCHED``."""
    ov = dict(PDB_MATCHED)
    ov.update(overlay)
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)


def pdb_runs() -> List[RunSpec]:
    """Transfer the digits winner to PDB."""
    return [_prun("pdb", "matched")]


def pdb_weights_runs() -> List[RunSpec]:
    """Settle the weights question on PDB, against a matched null.

    Never resolved before because the earlier PDB sweeps had no null, and here
    that is decisive: shuffled input already reaches trustworthiness 0.926, so
    only the null-corrected margin carries information (geodesic is the sharp
    one -- 0.857 real vs 0.460 shuffled for UMAP).

    Tuples are 3 entries, not 4: ``calibrate.py`` predicts 4 levels at the full
    N=5000, but with ``--holdout 0.2`` only 4000 points are fit and just 3 levels
    get built, so a 4-tuple would be silently truncated. Epochs are pinned to 120
    to keep 10 fits affordable; the digits sweep showed accuracy flat from 60 on.
    """
    ramp = (1.0, 2.0, 8.0)
    return [
        _prun("pdbw", "ramp", pyramid_level_weights=ramp, epochs=120),
        _prun("pdbw", "flat", pyramid_level_weights=(1.0, 1.0, 1.0), epochs=120),
        _prun("pdbw", "steep", pyramid_level_weights=(1.0, 4.0, 16.0), epochs=120),
        _prun("pdbw", "frontload", pyramid_level_weights=(8.0, 1.0, 1.0), epochs=120),
        _prun(
            "pdbw",
            "off",
            pyramid_scales=0,
            pyramid_level_weights=None,
            pyramid_coarse_backbone=0.0,
            epochs=120,
        ),
    ]


def s_curve_runs() -> List[RunSpec]:
    """Transfer the digits winner to the s-curve.

    The digits configuration already carries what calibrate.py recommends here
    (L=179, 3 pyramid levels), so ``matched`` is a straight transfer with only
    tau_scale re-derived by --target-perp. L=450 is the alternative that actually
    meets the coverage target (2.74 vs 4.74) at the cost of 4.4 points per anchor.

    Unlike digits, this feed is a smooth 2-D sheet (Levina-Bickel gives 1.94), so
    the question is unrolling rather than cluster separation: UMAP reaches kNN
    overlap 0.779 where PCA-2D gets 0.306, and lambda_geo is the term that pulls
    toward the globally unrolled solution -- hence bracketing it here.
    """
    def _srun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(DIGITS_MATCHED)
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("scurve", level), axis="scurve", level=level, overlay=ov
        )

    return [
        _srun("matched"),
        _srun("L450", n_landmarks=450),
        _srun("geo0.5", lambda_geo=0.5),
        _srun("geo0", lambda_geo=0.0),
        # kNN overlap rose monotonically with lambda_geo (0.709 / 0.710 / 0.744),
        # so push it to the ceiling and see whether the trend holds or the
        # Procrustes pull to landmark-geodesic MDS starts costing local fidelity.
        _srun("geo1.0", lambda_geo=1.0),
    ]


def uniform_runs() -> List[RunSpec]:
    """Attack the clumping: leanmap layouts are knots-and-voids, not evenly spread.

    On the uniformly sampled s-curve, kNN-spacing CV is 0.53-0.67 for every
    leanmap variant against 0.18 for the true flattening and a Poisson sample,
    0.30 for PCA-2D and 0.37 for UMAP. It is not a boundary effect -- restricting
    to the interior 64% barely moves it -- and no setting of ``lambda_geo`` fixes
    it. Candidate mechanisms, each isolated here:

    ``lambda_lm``
        Pulls points onto their landmark's position, i.e. L explicit attractors.
        Raising L from 179 to 450 made spacing_cv *worse* (0.53 -> 0.66), which
        is what more attractors would do. Prime suspect.
    ``min_dist``
        The direct anti-collapse term; 0.1 may simply be too permissive.
    ``n_negatives``
        Repulsion is what spreads a layout out; UMAP ships 5 as well, but UMAP
        does not also carry landmark attraction.
    """
    def _urun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(DIGITS_MATCHED)
        ov.update(lambda_geo=0.5, epochs=120)  # s-curve setting, halved for cost
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("uni", level), axis="uni", level=level, overlay=ov
        )

    return [
        _urun("base"),
        _urun("lm0", lambda_lm=0.0),
        _urun("mindist0.5", min_dist=0.5),
        _urun("neg20", n_negatives=20),
        _urun("lm0_mindist0.5", lambda_lm=0.0, min_dist=0.5),
        _urun("lm0_neg20_mindist0.5", lambda_lm=0.0, n_negatives=20, min_dist=0.5),
    ]


# Brackets the b = 1 boundary at min_dist = 0.197 (spread 1): 0.1 and 0.15 sit
# below it, 0.2 straddles it, and the rest are progressively deeper into the
# self-limiting regime.
MIN_DIST_LADDER = (0.1, 0.15, 0.2, 0.3, 0.5, 0.8)


def min_dist_scurve_runs() -> List[RunSpec]:
    """Does spacing_cv have a knee where the attraction exponent crosses b = 1?

    ``min_dist`` reaches the loss only through ``find_ab_params``, which fits
    ``1/(1 + a d^2b)``; the attractive force near contact then goes as
    ``d^(2b-1)``. Below ``b = 1`` that force decays more slowly than the
    separation, so a pair already close is pulled proportionally harder and
    neighbourhoods run away into knots. ``b`` crosses 1 at
    ``min_dist ~= 0.199 * spread``, which is why the ladder brackets 0.2.

    This is the load-bearing test of that story. A knee at 0.2 makes ``b >= 1``
    a principled default; a smooth decline means ``b`` is only a proxy and the
    default has to be picked empirically instead. Run on the s-curve because it
    is uniformly sampled, so spacing_cv has a known floor (~0.18) and the
    clumping is unambiguous rather than a property of the data.
    """
    def _srun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(DIGITS_MATCHED)
        ov.update(lambda_geo=0.5, epochs=120)  # s-curve setting, halved for cost
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("mdist", level), axis="mdist", level=level, overlay=ov
        )

    return [_srun(str(md), min_dist=md) for md in MIN_DIST_LADDER]


def min_dist_digits_runs() -> List[RunSpec]:
    """The same ladder on digits, where the trade-off may run the other way.

    Clustered data has no uniformity floor to hit, and separating classes wants
    tight neighbourhoods -- exactly what a large ``min_dist`` forbids. So this
    checks the cost side: whether moving into ``b >= 1`` for the sake of
    uniformity gives up label accuracy on the feed where leanmap matches UMAP.
    """
    return [_run_matched_mdist(md) for md in MIN_DIST_LADDER]


def _run_matched_mdist(md: float) -> RunSpec:
    ov = dict(DIGITS_MATCHED)
    ov["min_dist"] = md
    return RunSpec(
        run_id=_rid("mdist", str(md)), axis="mdist", level=str(md), overlay=ov
    )


def matched_runs() -> List[RunSpec]:
    """The winning digits configuration, for multi-seed / null confirmation."""
    ov = dict(DIGITS_MATCHED)
    return [RunSpec(run_id=_rid("matched", "digits"), axis="matched", level="digits", overlay=ov)]


def refine2_runs() -> List[RunSpec]:
    """Combine the two directions that each worked alone.

    ``lambda_geo=0`` gave the best label accuracy (0.964) but cost geodesic
    fidelity (0.503 vs 0.708), while 240 epochs gave the best trustworthiness
    and kNN overlap and KEPT geodesic above UMAP. A small non-zero lambda_geo is
    included to see whether global fidelity can be bought back cheaply.
    """
    return [
        _nrun("ref2", "geo0_e240", lambda_geo=0.0, epochs=240),
        _nrun("ref2", "geo0_mindist0", lambda_geo=0.0, min_dist=0.0),
        _nrun("ref2", "geo015_e240", lambda_geo=0.15, epochs=240),
        # The other end of the ladder: on the s-curve lambda_geo saturates by 0.5
        # and 1.0 costs nothing, but digits preferred 0.15 over 0.5, so the
        # optimum looks dataset-dependent. This pins down the clustered end.
        _nrun("ref2", "geo1_e240", lambda_geo=1.0, epochs=240),
        _nrun(
            "ref2",
            "geo0_e240_mindist0",
            lambda_geo=0.0,
            epochs=240,
            min_dist=0.0,
        ),
    ]


def optim_runs() -> List[RunSpec]:
    """Is the residual head simply not being trained hard enough?

    With ``pca_skip`` the head is initialized at std 1e-4 and the output is
    ``PCA(x) + residual``, so the layout starts AT plain PCA and has to climb
    away from it. Every data-side and objective-side axis is flat at PCA-level
    accuracy (0.59-0.71 vs PCA's 0.603), and turning the skip off drops to 0.412
    -- i.e. the learned part contributes almost nothing either way. That is the
    signature of an undertrained residual rather than a bad objective, and ``lr``
    is the one knob never varied on digits; a 5x raise mattered on PDB. Paired
    with the no-skip variants to see whether the learned map alone can stand up.
    """
    return [
        _mrun("lr", "5e-3", lr=5e-3),
        _mrun("lr", "2e-2", lr=2e-2),
        _mrun("lr", "5e-3_e240", lr=5e-3, epochs=240),
        _mrun("lr", "5e-3_noskip", lr=5e-3, pca_skip=False),
        _mrun("lr", "2e-2_noskip", lr=2e-2, pca_skip=False),
        _mrun("cap", "wide", width=768, depth=4, lr=5e-3),
    ]


def objective_runs() -> List[RunSpec]:
    """Which loss term is holding the classes together?

    Model capacity is ruled out: the backbone is an unconstrained 3x384 MLP
    (spectral norm is auto-disabled on MPS, so it was never active here), 240
    epochs is fully converged, and every local knob -- min_dist, n_negatives,
    n_neighbors, pyramid weights -- is flat at PCA-level label accuracy while
    kNN overlap sits well above PCA. Neighbourhoods survive but classes are
    never pulled apart, which is what a long-range ATTRACTION term does. The
    baseline carries three of them at once: the pyramid's coarse levels,
    ``lambda_geo`` Procrustes to landmark-geodesic MDS, and ``lambda_lm``. Only
    ``lambda_geo`` was tested alone, so this strips them individually and
    together down to a purely local, UMAP-like objective.
    """
    off_pyramid = dict(
        pyramid_scales=0,
        pyramid_level_weights=None,
        pyramid_coarse_backbone=0.0,
    )
    return [
        _mrun("obj", "pyramid_off", **off_pyramid),
        _mrun("obj", "lambda_lm_0", lambda_lm=0.0),
        _mrun("obj", "local_only", lambda_geo=0.0, lambda_lm=0.0, **off_pyramid),
        _mrun(
            "obj",
            "local_only_nn30",
            lambda_geo=0.0,
            lambda_lm=0.0,
            n_neighbors=30,
            **off_pyramid,
        ),
    ]


def packing_runs() -> List[RunSpec]:
    """Structural suspects for weak LOCAL fidelity.

    On digits the corrected baseline matches UMAP's geodesic correlation while
    losing badly on trustworthiness, kNN overlap and label separation, landing
    only just above PCA-2D. Two mechanisms could pin the layout near a linear /
    classical-MDS solution: the additive ``pca_skip`` path into a 2-D output, and
    ``lambda_geo``, which Procrustes-pulls toward classical MDS of landmark
    geodesics. ``min_dist`` / ``n_negatives`` control how hard neighbourhoods
    are allowed to tighten. Levels equal to the baseline are omitted.
    """
    return [
        _mrun("pca_skip", "off", pca_skip=False),
        _mrun("lambda_geo", "0.0", lambda_geo=0.0),
        _mrun("lambda_geo", "1.0", lambda_geo=1.0),
        _mrun("min_dist", "0.0", min_dist=0.0),
        _mrun("min_dist", "0.5", min_dist=0.5),
        _mrun("n_negatives", "15", n_negatives=15),
        _mrun(
            "interaction",
            "geo0_pcaskip_off",
            lambda_geo=0.0,
            pca_skip=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Canonical paper sweep — recommended config + the four ladders that carry
# real signal. Same arms on every dataset; on iris the pyramid ladder is a
# no-op (pyramid_min_reps=256 > N) and should be skipped via --only.
# ---------------------------------------------------------------------------
RECOMMENDED: Dict[str, Any] = {
    **DIGITS_MATCHED,
    "min_dist": 0.5,
    "pyramid_level_weights": (1.0, 2.0, 8.0),
    "lambda_frame": 0.0,
    "frame_ramp": (0.0, 0.0),
    "frame_tangent": True,
}


def _crun(axis: str, level: str, **overlay: Any) -> RunSpec:
    ov = dict(RECOMMENDED)
    ov.update(overlay)
    return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)


def canonical_runs() -> List[RunSpec]:
    """Recommended configuration plus min_dist / geo / frame / weights ladders."""
    runs: List[RunSpec] = [
        _crun("recommended", "default"),
    ]
    for md in (0.1, 0.2, 0.3, 0.5, 0.8):
        runs.append(_crun("min_dist", str(md), min_dist=md))
    for g in (0.0, 0.15, 0.5, 1.0):
        runs.append(_crun("lambda_geo", str(g), lambda_geo=g))
    # Frame: delayed ramp is the fold-back recipe; one early arm for contrast.
    for w in (0.0, 0.25, 0.5, 1.0):
        runs.append(
            _crun(
                "lambda_frame",
                f"{w}_delayed",
                lambda_frame=w,
                frame_ramp=(0.5, 0.75),
                frame_tangent=True,
            )
        )
    runs.append(
        _crun(
            "lambda_frame",
            "0.5_early",
            lambda_frame=0.5,
            frame_ramp=(0.0, 0.0),
            frame_tangent=True,
        )
    )
    for label, weights in (
        ("flat", (1.0, 1.0, 1.0)),
        ("ramp", (1.0, 2.0, 8.0)),
        ("steep", (1.0, 4.0, 16.0)),
        ("frontload", (8.0, 1.0, 1.0)),
    ):
        runs.append(_crun("weights", label, pyramid_level_weights=weights))
    runs.append(
        _crun(
            "weights",
            "off",
            pyramid_scales=0,
            pyramid_level_weights=None,
            pyramid_coarse_backbone=0.0,
        )
    )
    return runs


# Iris small-N recipe: pyramid is inert (min_reps=256 > 150), landmarks << N.
# calibrate.py at target-perp 8: L=64 gives coverage 2.36 (under the cover<=3
# target); L=32 is 4.86. Width/depth shrunk because N=150 memorizes easily.
IRIS_RECOMMENDED: Dict[str, Any] = {
    **RECOMMENDED,
    "n_landmarks": 64,
    "n_neighbors": 10,
    "pyramid_scales": 0,
    "pyramid_level_weights": None,
    "pyramid_coarse_backbone": 0.0,
    "lambda_geo": 0.5,
    "epochs": 240,
    "width": 128,
    "depth": 2,
}


def iris_canonical_runs() -> List[RunSpec]:
    """Canonical ladders under the iris small-N recipe (no pyramid arms)."""

    def _irun(axis: str, level: str, **overlay: Any) -> RunSpec:
        ov = dict(IRIS_RECOMMENDED)
        ov.update(overlay)
        return RunSpec(run_id=_rid(axis, level), axis=axis, level=level, overlay=ov)

    runs: List[RunSpec] = [_irun("recommended", "default")]
    for md in (0.1, 0.2, 0.3, 0.5, 0.8):
        runs.append(_irun("min_dist", str(md), min_dist=md))
    for g in (0.0, 0.15, 0.5, 1.0):
        runs.append(_irun("lambda_geo", str(g), lambda_geo=g))
    for w in (0.0, 0.25, 0.5, 1.0):
        runs.append(
            _irun(
                "lambda_frame",
                f"{w}_delayed",
                lambda_frame=w,
                frame_ramp=(0.5, 0.75),
            )
        )
    return runs


def iris_pyramid_weights_runs() -> List[RunSpec]:
    """Didactic iris panel: force multi-level pyramid so weight schedules show.

    Production iris sets ``pyramid_scales=0`` because default
    ``pyramid_min_reps=256`` never coarsens at N=150. Lowering ``min_reps`` to
    16 builds ~3 levels at train N≈120 so flat / ramp / steep / frontload are
    distinguishable. Not the small-N recipe — use only for the separate
    weights figure.
    """

    def _irun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(IRIS_RECOMMENDED)
        ov.update(
            pyramid_scales=3,
            pyramid_min_reps=16,
            pyramid_coarse_backbone=1.0,
            pyramid_level_weights=(1.0, 2.0, 8.0),
        )
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("weights", level), axis="weights", level=level, overlay=ov
        )

    return [
        _irun(
            "off",
            pyramid_scales=0,
            pyramid_level_weights=None,
            pyramid_coarse_backbone=0.0,
        ),
        _irun("flat", pyramid_level_weights=(1.0, 1.0, 1.0)),
        _irun("ramp", pyramid_level_weights=(1.0, 2.0, 8.0)),
        _irun("steep", pyramid_level_weights=(1.0, 4.0, 16.0)),
        _irun("frontload", pyramid_level_weights=(8.0, 1.0, 1.0)),
    ]


# Swiss-roll frame stress test under the recommended base with geo=0.5
# (fold-back manifolds want the global pull).
def swiss_roll_frame_runs() -> List[RunSpec]:
    def _srun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(RECOMMENDED)
        ov.update(lambda_geo=0.5)
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("frame", level), axis="frame", level=level, overlay=ov
        )

    return [
        _srun("0", lambda_frame=0.0, frame_ramp=(0.0, 0.0)),
        _srun("0.25_delayed", lambda_frame=0.25, frame_ramp=(0.5, 0.75)),
        _srun("0.5_delayed", lambda_frame=0.5, frame_ramp=(0.5, 0.75)),
        _srun("1.0_delayed", lambda_frame=1.0, frame_ramp=(0.5, 0.75)),
        _srun("0.5_early", lambda_frame=0.5, frame_ramp=(0.0, 0.0)),
    ]


def digits_geo_frame_runs() -> List[RunSpec]:
    """Clustered-data stress: geo=0.5 + frame=0.5 static vs geo-only / recommended.

    Docs say leave frame off on digits; this checks the cost of the swiss-roll
    recipe transferred literally (static ramps = on from epoch 0).
    """

    def _drun(level: str, **overlay: Any) -> RunSpec:
        ov = dict(RECOMMENDED)
        ov.update(overlay)
        return RunSpec(
            run_id=_rid("gf", level), axis="gf", level=level, overlay=ov
        )

    return [
        _drun("recommended"),  # geo=0.15, frame=0
        _drun("geo0.5", lambda_geo=0.5, lambda_frame=0.0, frame_ramp=(0.0, 0.0)),
        _drun(
            "geo0.5_frame0.5_static",
            lambda_geo=0.5,
            geo_ramp=(0.0, 0.0),
            lambda_frame=0.5,
            frame_ramp=(0.0, 0.0),
            frame_tangent=True,
        ),
        _drun(
            "geo0.5_frame0.5_delayed",
            lambda_geo=0.5,
            geo_ramp=(0.0, 0.0),
            lambda_frame=0.5,
            frame_ramp=(0.5, 0.75),
            frame_tangent=True,
        ),
    ]


SWEEPS = {
    "canonical": canonical_runs,
    "iris_canonical": iris_canonical_runs,
    "iris_pyramid_weights": iris_pyramid_weights_runs,
    "swiss_roll_frame": swiss_roll_frame_runs,
    "digits_geo_frame": digits_geo_frame_runs,
    "phase1": phase1_runs,
    "frame_weight": frame_weight_runs,
    "umap_match": umap_match_runs,
    "weights": weights_runs,
    "epochs": epochs_runs,
    "neighbors": neighbors_runs,
    "landmarks": landmarks_runs,
    "packing": packing_runs,
    "objective": objective_runs,
    "optim": optim_runs,
    "refine": refine_runs,
    "refine2": refine2_runs,
    "matched": matched_runs,
    "pdb": pdb_runs,
    "pdb_weights": pdb_weights_runs,
    "s_curve": s_curve_runs,
    "uniform": uniform_runs,
    "min_dist_scurve": min_dist_scurve_runs,
    "min_dist_digits": min_dist_digits_runs,
}


def list_sweeps() -> List[str]:
    return sorted(SWEEPS.keys())


def resolve_runs(
    sweep: str = "phase1",
    only_axis: Optional[str] = None,
    only_ids: Optional[Sequence[str]] = None,
) -> List[RunSpec]:
    """Return run specs for ``sweep``, optionally filtered by axis or run_id."""
    if sweep not in SWEEPS:
        raise KeyError(f"unknown sweep {sweep!r}; choose from {list_sweeps()}")
    runs = list(SWEEPS[sweep]())
    if only_axis:
        axis = only_axis.strip().lower()
        runs = [r for r in runs if r.axis.lower() == axis or r.run_id.lower() == axis]
    if only_ids:
        want = {s.strip() for s in only_ids if s.strip()}
        runs = [r for r in runs if r.run_id in want or r.level in want]
    return runs


def merged_overlay(run: RunSpec) -> Dict[str, Any]:
    """Baseline + per-run overlay (run wins)."""
    out = dict(BASELINE)
    out.update(run.overlay)
    return out

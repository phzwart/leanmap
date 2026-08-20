"""Training loop, checkpointing, and artefact I/O."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from ..conditioning import (
    RETENTION_CHANCE,
    RETENTION_WARN,
    ConditioningFactor,
    FactorStack,
    Role,
    build_factor_stack,
    default_primary_factor,
    metric_from_factors,
)
from ..classaxis import (
    CLASS_PAIRS_PER_STEP,
    ORDER_CHANCE,
    ORDER_WARN,
    ClassAxis,
    ClassOrderSampler,
    class_axis_report,
    class_direction_loss,
    class_order_loss,
    validate_class_axes,
)
from ..path import (
    PATH_PAIRS_PER_STEP,
    PathConstraint,
    PathTripletSampler,
    path_constraint_loss,
)
from ..config import (
    C_BUCKETS,
    C_SEARCH,
    CALIB_FRAC,
    ETA_BALANCE,
    FRAME_CENTERS,
    FRAME_NORMAL_THRESH,
    FRAME_TANGENT_DIM,
    GEO_PAIRS,
    HYPER_WIDTH,
    LAMBDA_BACKBONE,
    LAMBDA_ORD,
    LANDMARK_GEODESIC_K,
    MDS_NEG_EIGEN_WARN,
    PCA_CENTER,
    PLANEConfig,
    PYRAMID_REP_RATIO,
    SPREAD,
    WARMUP_FRAC,
    WEIGHT_DECAY,
    apply_scale_train_defaults,
)
from ..conformal import ConformalCalibrator, geometry_consistency_score, model_weight_hash
from ..density import density_budget, density_correlation_loss, star_log_radius
from ..distance import DistanceFn, chunked_cdist
from ..evaluate import geodesic_fidelity
from ..graph import (
    Graph,
    build_graph,
    build_graph_pyramid,
    check_tensor_fingerprint,
    load_graph_pyramid,
    save_graph_pyramid,
    tensor_fingerprint,
)
from ..warmstart import LAYOUTS as WARM_START_LAYOUTS, spectral_layout, warm_start
from ..landmarks import AnchorAffinity, LandmarkAffinity, classical_mds
from ..losses import (
    alignment_ramp,
    find_ab_params,
    fuzzy_cross_entropy,
    gauge_nu_diagnostic,
    geodesic_stress_loss,
    landmark_geodesics_on_level,
    landmark_regularisation,
    local_rigidity_loss,
    metric_edge_lengths,
    ordinal_triplet_loss,
    procrustes_anchor_loss,
    select_gauge_level,
)
from ..metrics import MetricSpec, wrap_metric
from ..model import ConcatEncoder, FiLMEncoder, PLANE, fit_pca_weight
from ..sampler import (
    EdgeSampler,
    NegativeSampler,
    OrdinalTripletSampler,
    StarSampler,
    estimate_retention_null,
)
from ..utils import ensure_2d_float32, get_logger, resolve_device, seed_everything


def _metric_name_from_dist_fn(dist_fn: Any) -> str:
    """Derive a saveable metric label from the ``fit(..., dist_fn=...)`` argument."""
    if isinstance(dist_fn, str):
        return dist_fn
    name = getattr(dist_fn, "name", None)
    return str(name) if name else "custom"


def _split_budget(
    batch_edges: int, lvl_w: Sequence[float], active: Sequence[int]
) -> List[int]:
    """Per-level edge counts for one step: ``batch_edges`` split over ``active``.

    Inactive levels get zero, so the per-step cost stays at ``batch_edges``
    however many levels are switched on.
    """
    counts = [0] * len(lvl_w)
    if not active:
        return counts
    wsum = sum(lvl_w[i] for i in active) or 1.0
    for i in active:
        counts[i] = int(round(batch_edges * (lvl_w[i] / wsum)))
    finest = min(active)
    counts[finest] += batch_edges - sum(counts)  # absorb rounding
    return [max(0, c) for c in counts]


def coarse_to_fine_plan(
    epochs: int,
    edges_per_level: Sequence[int],
    batch_edges: int,
    lvl_w: Sequence[float],
    coarse_frac: float,
    *,
    epoch_unit: str = "edges",
    n_landmarks: int = 1,
    landmark_epoch_samples: float = 128.0,
) -> List[Tuple[List[int], int]]:
    """Per-epoch ``(edge counts per level, steps)``, coarsest levels admitted first.

    Steps per epoch are set by the *active* levels' edge count when
    ``epoch_unit=\"edges\"`` (historical default). With
    ``epoch_unit=\"landmarks\"``, steps follow
    :func:`~leanmap.sampling.edges.landmark_epoch_steps` so the budget tracks
    landmark-basin cover rather than δ-net edge count.

    The first ``coarse_frac`` of epochs is divided evenly among the levels,
    starting from the coarsest alone and admitting one finer level at a time;
    after that every level is active with the configured weights.
    ``coarse_frac=0`` reproduces the old behaviour throughout.
    """
    from leanmap.sampling.edges import landmark_epoch_steps

    n_levels = len(edges_per_level)
    plan: List[Tuple[List[int], int]] = []
    n_warm = int(round(max(0.0, min(1.0, coarse_frac)) * epochs))
    unit = str(epoch_unit).lower()
    for epoch in range(epochs):
        if epoch < n_warm and n_levels > 1:
            phase = int(n_levels * epoch / max(n_warm, 1))
            finest = max(0, n_levels - 1 - phase)
            active = list(range(finest, n_levels))
        else:
            active = list(range(n_levels))
        counts = _split_budget(batch_edges, lvl_w, active)
        if unit in ("landmarks", "landmark", "basin"):
            steps = landmark_epoch_steps(
                n_landmarks,
                batch_edges,
                samples_per_landmark=landmark_epoch_samples,
            )
        else:
            edges = edges_per_level[min(active)]
            steps = max(1, math.ceil(edges / batch_edges))
        plan.append((counts, steps))
    return plan


def _param_groups(model: torch.nn.Module, config: "PLANEConfig") -> list:
    """Split parameters so the residual head can outrun the PCA skip.

    With ``pca_skip=True`` the map starts at plain PCA and the head only has to
    add a residual, so a single learning rate couples two decisions: raise it
    and the skip drifts away from the PCA solution it was seeded with, leave it
    low and the head never departs from PCA. ``pca_lr_mult`` decouples them by
    putting the head and the FiLM hypernetworks on a higher rate than the skip
    and backbone, which is the escape from that coupling.

    A multiplier of 1 (the default) reproduces a single flat rate exactly.
    """
    mult = float(getattr(config, "pca_lr_mult", 1.0))
    slow_prefixes = ("encoder.pca", "encoder.backbone", "encoder.norms")
    if mult == 1.0 or not getattr(config, "pca_skip", False):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    fast, slow = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (slow if name.startswith(slow_prefixes) else fast).append(p)
    if not fast or not slow:
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    get_logger().info(
        "param groups: %d slow tensors at lr, %d fast (head + FiLM hypers) at %.1fx",
        len(slow),
        len(fast),
        mult,
    )
    return [
        {"params": slow, "lr": float(config.lr)},
        {"params": fast, "lr": float(config.lr) * mult},
    ]


@dataclass
class PLANEResult:
    """Fitted artefact: model + calibration + provenance (no graph / no N-arrays)."""

    model: PLANE
    config: PLANEConfig
    calibrator: ConformalCalibrator
    a: float
    b: float
    graph_stats: dict
    metric_name: str = "l2"

    def save(self, path: Union[str, Path]) -> None:
        """Write a single ``.pt`` file containing only inference artefacts."""
        path = Path(path)
        enc = self.model.encoder
        aff = self.model.affinity
        factors = self.model.factors
        factor_payload = []
        factor_scales = getattr(self, "factor_scales", None) or {}
        if factors is not None:
            for f, a_mod in zip(factors.factor_defs, factors.affinities):
                assert isinstance(a_mod, AnchorAffinity)
                view_metric = f.metric
                factor_payload.append(
                    {
                        "name": f.name,
                        "role": f.role.value,
                        "n_anchors": f.n_anchors,
                        "metric_weight": f.metric_weight,
                        "axis": f.axis,
                        "learn_anchors": f.learn_anchors,
                        "learn_temperature": f.learn_temperature,
                        "view_grad_from_geom": f.view_grad_from_geom,
                        "M": a_mod.M.detach().cpu(),
                        "log_tau": a_mod.log_tau.detach().cpu(),
                        "view_metric_name": getattr(view_metric, "name", None),
                        "view_natural_scale": getattr(
                            view_metric, "natural_scale", None
                        ),
                        "scale_f": factor_scales.get(f.name),
                    }
                )
        payload = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "x_mean": enc.x_mean.cpu(),
            "x_std": enc.x_std.cpu(),
            "landmark_coordinates": aff.M.detach().cpu(),
            "log_tau": aff.log_tau.detach().cpu(),
            "factors": factor_payload,
            "a": self.a,
            "b": self.b,
            "tau_embed": self.calibrator.tau_embed,
            "s_calib": None
            if self.calibrator.s_calib is None
            else self.calibrator.s_calib.cpu(),
            "weight_hash": self.calibrator.weight_hash,
            "graph_stats": self.graph_stats,
            "D": enc.D,
            "L": aff.M.shape[0],
            "metric_name": self.metric_name,
            "natural_scale": getattr(self, "natural_scale", None),
            "factor_scales": getattr(self, "factor_scales", None),
        }
        torch.save(payload, str(path))

    def embed(self, X, **kwargs):
        return self.model.embed(torch.as_tensor(ensure_2d_float32(X)), **kwargs)


def load_plane(path: Union[str, Path], device: Optional[str] = None) -> PLANE:
    """Load a saved artefact into a ``PLANE`` ready for ``embed()``."""
    from ..metrics import get_metric

    payload = torch.load(str(path), map_location=device or "cpu", weights_only=False)
    cfg = PLANEConfig(**{k: v for k, v in payload["config"].items() if k in PLANEConfig.__dataclass_fields__})
    D = payload["D"]
    L = payload["L"]
    device_t = resolve_device(device)
    try:
        metric = get_metric(payload.get("metric_name", "l2"))
    except Exception:  # noqa: BLE001
        metric = get_metric("l2")
    from dataclasses import replace

    ns = payload.get("natural_scale")
    if ns is not None:
        metric = replace(metric, natural_scale=ns)
    dist_fn: DistanceFn = metric

    factor_payload = payload.get("factors") or []
    if factor_payload:
        # Rebuild FactorStack with identity views (callables are not serialised;
        # identity PRIMARY matches the default fit path).
        from ..conditioning import identity_view

        defs: List[ConditioningFactor] = []
        affs: List[AnchorAffinity] = []
        for fp in factor_payload:
            role = Role(fp["role"])
            mname = fp.get("view_metric_name") or payload.get("metric_name", "l2")
            try:
                f_metric = get_metric(mname)
            except Exception:  # noqa: BLE001
                f_metric = get_metric("l2")
            vns = fp.get("view_natural_scale")
            if vns is None and role == Role.PRIMARY:
                vns = payload.get("natural_scale")
            if vns is not None:
                f_metric = replace(f_metric, natural_scale=vns)
            defs.append(
                ConditioningFactor(
                    name=fp["name"],
                    view=identity_view,
                    metric=f_metric,
                    n_anchors=int(fp["n_anchors"]),
                    role=role,
                    metric_weight=fp.get("metric_weight"),
                    learn_anchors=False,
                    learn_temperature=False,
                    axis=fp.get("axis"),
                    view_grad_from_geom=bool(fp.get("view_grad_from_geom", False)),
                )
            )
            M = fp["M"].to(device_t)
            aff = AnchorAffinity(
                M,
                f_metric,
                tau_init=torch.exp(fp["log_tau"].to(device_t)),
                learn_anchors=False,
                learn_tau=False,
                probe_differentiable=False,
            )
            aff.log_tau.data.copy_(fp["log_tau"].to(device_t))
            affs.append(aff)
        stack = FactorStack(
            defs,
            affs,
            width=cfg.width,
            depth=cfg.depth,
            hyper_width=HYPER_WIDTH,
            d_out=cfg.d_out,
        ).to(device_t)
        affinity_dim = sum(a.M.shape[0] for a in affs)
        if getattr(cfg, "conditioning", "film") == "concat":
            enc: Union[FiLMEncoder, ConcatEncoder] = ConcatEncoder(
                D,
                cfg.d_out,
                width=cfg.width,
                depth=cfg.depth,
                L=L,
                affinity_dim=affinity_dim,
                pca_skip=cfg.pca_skip,
            )
        else:
            enc = FiLMEncoder(
                D,
                cfg.d_out,
                width=cfg.width,
                depth=cfg.depth,
                L=L,
                affinity_dim=affinity_dim,
                hyper_width=HYPER_WIDTH,
                pca_skip=cfg.pca_skip,
            )
        enc.set_normalization(payload["x_mean"], payload["x_std"])
        model = PLANE(stack, enc).to(device_t)
    else:
        M = payload["landmark_coordinates"].to(device_t)
        aff = LandmarkAffinity(
            M,
            dist_fn,
            tau_init=torch.exp(payload["log_tau"].to(device_t)),
            learn_landmarks=False,
            learn_tau=False,
            probe_differentiable=False,
        )
        aff.log_tau.data.copy_(payload["log_tau"].to(device_t))
        enc = FiLMEncoder(
            D,
            cfg.d_out,
            width=cfg.width,
            depth=cfg.depth,
            L=L,
            hyper_width=HYPER_WIDTH,
            pca_skip=cfg.pca_skip,
        )
        enc.set_normalization(payload["x_mean"], payload["x_std"])
        model = PLANE(aff, enc).to(device_t)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model


def fit(
    X: np.ndarray | torch.Tensor,
    dist_fn: Union[str, MetricSpec, DistanceFn] = "l2",
    config: Optional[PLANEConfig] = None,
    X_calib: Optional[np.ndarray | torch.Tensor] = None,
    callbacks: Optional[List[Callable]] = None,
    factors: Optional[Sequence[ConditioningFactor]] = None,
    encoder_view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    init_state_dict: Optional[dict] = None,
    precomputed_knn: Optional[Tuple[Any, Any]] = None,
    graph_path: Optional[Union[str, Path]] = None,
    rebuild_graph: bool = False,
    class_labels: Optional[np.ndarray | torch.Tensor] = None,
    class_axes: Optional[Sequence[ClassAxis]] = None,
    path_constraints: Optional[Sequence[PathConstraint]] = None,
) -> PLANEResult:
    """Fit PLANE. Calibration split is taken from raw ``X`` before the graph.

    Order of operations is fixed by the specification (§11).

    Parameters
    ----------
    factors : sequence of ConditioningFactor, optional
        If None, a single identity PRIMARY factor with ``config.n_landmarks``
        anchors is used (migration-compatible default).
    encoder_view : callable, optional
        Maps ambient ``x`` → backbone features. Use when ambient packs multiple
        vectors per item (e.g. metric features + conditioning view) and only a
        slice should enter the FiLM encoder.
    precomputed_knn : (knn_idx, knn_dist), optional
        Caller-supplied kNN over the **training** matrix (same rows as ``X``
        when ``X_calib`` is given, or as the train split otherwise). Shapes
        ``(N_train, k)`` int64 / float32. Edge distances may use a different
        metric than ``dist_fn`` (landmarks / ε-net still use ``dist_fn``).
        Requires ``config.dedup=False``. When set, pass ``X_calib`` explicitly
        so calib is not carved out of ``X`` — the caller owns the train-row
        indexing of the supplied graph.
    graph_path : path, optional
        If the file exists and ``rebuild_graph`` is false, skip kNN / ε-net /
        pyramid construction and reuse the cached split. After a fresh build,
        the pyramid is written here so later runs can change lr / epochs /
        pyramid weights without rebuilding. Metric, landmarks, neighbors,
        seed, and row identity must match.
    rebuild_graph : bool
        Ignore ``graph_path`` if present and rebuild (then overwrite).
    class_labels : (N,) int, optional
        Integer class codes ``0..K-1``, one per row of ``X`` — the same rows and
        the same order, *before* any calibration split. When the split is carved
        out of ``X`` internally, the labels are carved with it; supply
        ``X_calib`` explicitly if you need to control which rows train.
        Requires ``class_axes`` and ``config.lambda_class > 0`` to have an
        effect; labels alone change nothing.
    class_axes : sequence of ClassAxis, optional
        Which orderings of the classes the layout must respect (see
        :mod:`leanmap.classaxis`). An axis with ``axis=j`` pins coordinate ``j``,
        and at most ``d_out - 1`` may do so. An axis with ``axis=None`` asks only
        that its groups be ordered along *some* direction, which the fit chooses
        each step; that is the weaker request a secondary factor usually wants,
        and it cannot disturb the pinned coordinates. Per-axis ``weight`` scales
        ``lambda_class`` so a secondary factor can be applied gently.

        Labels never enter the graph, the metric or the conditioning — only these
        gauge terms — so inference needs no label and the neighbourhood target is
        unchanged.
    path_constraints : sequence of PathConstraint, optional
        Explicit ``(anchor, near, mid)`` index triples into the same rows as
        ``X`` (before any calibration split). Does not enter the neighbour
        graph. Requires ``config.lambda_path > 0`` to affect the layout.
    """
    from ..conditioning import identity_view
    from ..landmarks import assign_buckets, init_anchors

    log = get_logger()
    if config is None:
        X_tmp = ensure_2d_float32(X)
        config = PLANEConfig.for_scale(X_tmp.shape[0])
    seed_everything(config.seed)
    device = resolve_device(config.device)
    X_all = torch.as_tensor(ensure_2d_float32(X), dtype=torch.float32)
    # Optional large-N schedule fill-in (no-op unless apply_large_n_schedule).
    # for_scale(N>200k) already bakes warm start / coarse-first; this covers
    # blank configs that opt in at N>=50k without changing small-N goldens.
    apply_scale_train_defaults(config, int(X_all.shape[0]))
    enc_view = encoder_view if encoder_view is not None else (lambda t: t)

    if precomputed_knn is not None:
        if config.dedup:
            raise ValueError(
                "precomputed_knn requires config.dedup=False so neighbor indices "
                "align with ambient training rows"
            )
        if X_calib is None:
            raise ValueError(
                "precomputed_knn requires an explicit calibration matrix "
                "(X_calib=...); calib must not be carved out of X because the "
                "caller owns the train-row indexing of the supplied graph"
            )

    graph_path_p = Path(graph_path) if graph_path is not None else None
    loaded_pyramid: Optional[dict] = None
    if (
        graph_path_p is not None
        and graph_path_p.exists()
        and not rebuild_graph
        and precomputed_knn is None
    ):
        loaded_pyramid = load_graph_pyramid(graph_path_p)
        log.info("loading graph pyramid from %s", graph_path_p)

    # 2. Split calibration first
    N = X_all.shape[0]
    train_idx: Optional[torch.Tensor] = None
    if loaded_pyramid is not None:
        if X_calib is not None:
            raise ValueError("cached graph_path cannot be combined with X_calib")
        if int(loaded_pyramid["n_all"]) != int(N):
            raise ValueError(
                f"cached graph n_all={loaded_pyramid['n_all']} != X rows {N}; rebuild"
            )
        train_idx = loaded_pyramid["train_idx"].long()
        calib_idx = loaded_pyramid["calib_idx"].long()
        X_cal = X_all[calib_idx]
        X_train = X_all[train_idx]
        log.info(
            "reusing cached train/calib split (n_train=%d n_cal=%d)",
            int(X_train.shape[0]),
            int(X_cal.shape[0]),
        )
    elif X_calib is not None:
        X_cal = torch.as_tensor(ensure_2d_float32(X_calib), dtype=torch.float32)
        X_train = X_all
        calib_idx = None
    else:
        n_cal = min(int(CALIB_FRAC * N), config.calib_max)
        n_cal = max(n_cal, 1)
        g = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(N, generator=g)
        calib_idx = perm[:n_cal]
        train_idx = perm[n_cal:]
        X_cal = X_all[calib_idx]
        X_train = X_all[train_idx]

    # Labels follow the split rather than the caller's indexing: whichever rows
    # became calibration must take their labels with them, or the gauge term
    # would be trained against labels belonging to different points.
    labels_train: Optional[torch.Tensor] = None
    labels_calib: Optional[torch.Tensor] = None
    axes_list: List[ClassAxis] = list(class_axes) if class_axes else []
    if class_labels is not None:
        lab_all = torch.as_tensor(np.asarray(class_labels).reshape(-1), dtype=torch.int64)
        if lab_all.shape[0] != N:
            raise ValueError(
                f"class_labels has {lab_all.shape[0]} entries but X has {N} rows"
            )
        if int(lab_all.min()) < 0:
            raise ValueError("class_labels must be non-negative integer codes 0..K-1")
        if train_idx is not None:
            labels_train = lab_all[train_idx]
            labels_calib = lab_all[calib_idx] if calib_idx is not None else None
        else:
            labels_train = lab_all
        n_classes = int(lab_all.max().item()) + 1
        validate_class_axes(axes_list, config.d_out, n_classes)
    elif axes_list:
        raise ValueError(
            "class_axes was given without class_labels; the ordering has nothing "
            "to order"
        )

    # 3. normalisation stats on encoder features of the training split
    X_enc_train = enc_view(X_train)
    x_mean = X_enc_train.mean(dim=0)
    x_std = X_enc_train.std(dim=0).clamp_min(1e-6)

    factor_list: List[ConditioningFactor] = []
    # filled after scoring metric is resolved

    factor_metric = (
        metric_from_factors(
            list(factors), X=X_train, n_neighbors=config.n_neighbors, seed=config.seed
        )
        if factors is not None
        else None
    )
    if factor_metric is not None and factors is not None and dist_fn == "l2":
        metric: Any = factor_metric
    elif isinstance(dist_fn, str) or isinstance(dist_fn, MetricSpec) or hasattr(dist_fn, "blocks"):
        metric = wrap_metric(dist_fn, X=X_train, n_neighbors=config.n_neighbors, seed=config.seed)
    else:
        from ..metrics import get_metric

        metric = wrap_metric(
            get_metric("custom", fn=dist_fn, differentiable=True),
            X=X_train,
            n_neighbors=config.n_neighbors,
            seed=config.seed,
        )

    if factors is None:
        factor_list = [
            ConditioningFactor(
                name="primary",
                view=identity_view,
                metric=metric,
                n_anchors=config.n_landmarks,
                role=Role.PRIMARY,
                learn_anchors=config.learn_landmarks,
                learn_temperature=config.learn_tau,
            )
        ]
    else:
        factor_list = list(factors)

    n_graph_landmarks = config.n_landmarks
    for f in factor_list:
        if f.role == Role.PRIMARY:
            n_graph_landmarks = f.n_anchors
            break

    # Pre-init anchors; non-identity PRIMARY drives graph FPS + IVF knn tree.
    pre_M: List[Optional[torch.Tensor]] = []
    extra_ivf: List[Tuple[Any, Any, torch.Tensor]] = []
    fps_view = None
    fps_view_metric = None
    for i, f in enumerate(factor_list):
        use_graph_M = f.role == Role.PRIMARY and f.view is identity_view
        if use_graph_M:
            pre_M.append(None)
            continue
        if f.role == Role.PRIMARY:
            # Graph will FPS in this view; FiLM anchors = view(graph M)
            pre_M.append(None)
            fps_view = f.view
            fps_view_metric = f.metric
            continue
        v = f.view(X_train)
        Mf = init_anchors(v, f.metric, f.n_anchors, seed=config.seed + i)
        pre_M.append(Mf)
        extra_ivf.append((f.view, f.metric, Mf))

    # 4–8 graph (multi-scale pyramid; graphs[0] is the finest / legacy graph)
    if loaded_pyramid is not None:
        cached_metric = loaded_pyramid.get("metric_name")
        want_metric = _metric_name_from_dist_fn(dist_fn)
        if cached_metric is not None and str(cached_metric) != str(want_metric):
            raise ValueError(
                f"cached graph metric={cached_metric!r} != {want_metric!r}; "
                "pass rebuild_graph=True"
            )
        if int(loaded_pyramid["n_landmarks"]) != int(n_graph_landmarks):
            raise ValueError(
                f"cached graph n_landmarks={loaded_pyramid['n_landmarks']} != "
                f"{n_graph_landmarks}; pass rebuild_graph=True"
            )
        if int(loaded_pyramid.get("seed", config.seed)) != int(config.seed):
            log.warning(
                "cached graph seed=%s but config.seed=%s; reusing cached split/landmarks",
                loaded_pyramid.get("seed"),
                config.seed,
            )
        cached_k = loaded_pyramid.get("n_neighbors")
        if cached_k is not None and int(cached_k) != int(config.n_neighbors):
            raise ValueError(
                f"cached graph n_neighbors={cached_k} != {config.n_neighbors}; "
                "pass rebuild_graph=True"
            )
        from leanmap.store.fingerprint import verify_fingerprint

        fp = loaded_pyramid.get("fingerprint") or {}
        if "digest" in fp:
            if not verify_fingerprint(X_train, {"fingerprint": fp}, full=True):
                raise ValueError(
                    "X_train fingerprint digest does not match the cached graph; rebuild"
                )
        else:
            check_tensor_fingerprint(X_train, fp)
        graphs = loaded_pyramid["graphs"]
        M = loaded_pyramid["M"]
        assign_top1 = loaded_pyramid["assign_top1"]
        assign_topc = loaded_pyramid["assign_topc"]
        log.info(
            "using cached graph pyramid: %d level(s) R=%d L=%d",
            len(graphs),
            int(graphs[0].reps.rep_idx.shape[0]),
            int(M.shape[0]),
        )
    else:
        graphs, M, assign_top1, assign_topc = build_graph_pyramid(
            X_train,
            metric,
            pyramid_scales=config.pyramid_scales,
            pyramid_rep_ratio=PYRAMID_REP_RATIO,
            pyramid_min_reps=config.pyramid_min_reps,
            pyramid_coarse_backbone=config.pyramid_coarse_backbone,
            pyramid_squash=config.pyramid_squash,
            n_neighbors=config.n_neighbors,
            n_landmarks=n_graph_landmarks,
            c_buckets=C_BUCKETS,
            epsilon=config.epsilon,
            delta=config.delta,
            dedup=config.dedup,
            local_connectivity=config.local_connectivity,
            beta_multiplicity=config.beta_multiplicity,
            lambda_backbone=LAMBDA_BACKBONE,
            knn_mode=config.knn_mode,
            c_search=C_SEARCH,
            seed=config.seed,
            extra_ivf_anchors=extra_ivf or None,
            fps_view=fps_view,
            fps_view_metric=fps_view_metric,
            fps_geodesic=config.landmark_geodesic,
            fps_geodesic_k=LANDMARK_GEODESIC_K,
            fps_poisson=config.landmark_poisson,
            precomputed_knn=precomputed_knn,
            stages_dir=config.graph_stages_dir,
        )
    graph = graphs[0]  # finest graph: reps/negatives/knn_idx/stats live here

    if config.epsilon is None:
        # Freeze the resolved merge radius into the artefact. Re-estimating on a
        # refit would make the coarsening scale a function of how much data
        # happened to be on hand, which defeats "explored once and saved".
        config = replace(config, epsilon=float(graph.stats.epsilon))
        log.info(
            "epsilon=%.6g frozen into the saved config (pass epsilon=None to re-estimate)",
            config.epsilon,
        )
    if config.delta is None or config.delta == "auto" or config.delta == "eps":
        config = replace(config, delta=float(getattr(graph.stats, "delta", graph.stats.epsilon)))
        log.info(
            "delta=%.6g frozen into the saved config (mode=%s)",
            config.delta,
            graph.stats.extra.get("delta_mode", "eps"),
        )

    if (
        graph_path_p is not None
        and loaded_pyramid is None
        and train_idx is not None
        and calib_idx is not None
    ):
        save_graph_pyramid(
            graph_path_p,
            graphs=graphs,
            M=M,
            assign_top1=assign_top1,
            assign_topc=assign_topc,
            train_idx=train_idx,
            calib_idx=calib_idx,
            fingerprint=tensor_fingerprint(X_train),
            metric_name=_metric_name_from_dist_fn(dist_fn),
            n_all=int(N),
            n_neighbors=int(config.n_neighbors),
            epsilon=float(graph.stats.epsilon),
            seed=int(config.seed),
            dedup=bool(config.dedup),
        )

    if calib_idx is not None and int(X_cal.shape[0]) == 0:
        # Some frozen builds (e.g. bunches) store an empty calib split while
        # training on all rows. Carve a conformal holdout from ambient X only;
        # graph membership / train_idx stay unchanged.
        n_cal = min(int(CALIB_FRAC * N), int(config.calib_max))
        n_cal = max(n_cal, 1)
        g_cal = torch.Generator().manual_seed(int(config.seed) ^ 0xC411B)
        calib_idx = torch.randperm(N, generator=g_cal)[:n_cal]
        X_cal = X_all[calib_idx]
        log.info(
            "cached graph had empty calib; carved %d ambient rows for conformal only",
            n_cal,
        )
    elif calib_idx is not None:
        assert X_cal.shape[0] > 0

    # Epoch monitors can persist conformal scores into checkpoints.
    if callbacks:
        for cb in callbacks:
            set_calib = getattr(cb, "set_calib", None)
            if set_calib is not None and int(X_cal.shape[0]) > 0:
                set_calib(X_cal)

    # 9. FactorStack
    D = int(X_enc_train.shape[1])
    affs: List[AnchorAffinity] = []
    for i, f in enumerate(factor_list):
        if pre_M[i] is not None:
            M_init = pre_M[i]
        elif fps_view is not None and f.role == Role.PRIMARY:
            M_init = fps_view(M).clone()
        else:
            M_init = M.clone()
        assert M_init is not None
        if config.tau_init is not None:
            L_i = int(M_init.shape[0])
            tau_init = torch.full(
                (L_i,), float(config.tau_init), dtype=torch.float32, device=device
            )
        else:
            tau_init = None
        affs.append(
            AnchorAffinity(
                M_init.to(device),
                f.metric,
                tau_init=tau_init,
                learn_anchors=f.learn_anchors,
                learn_tau=f.learn_temperature,
                tau_scale=config.tau_scale,
            )
        )
    stack = FactorStack(
        factor_list,
        affs,
        width=config.width,
        depth=config.depth,
        hyper_width=HYPER_WIDTH,
        d_out=config.d_out,
    ).to(device)
    L = stack.primary_affinity.M.shape[0]
    affinity_dim = sum(a.M.shape[0] for a in affs)
    pca_weight = None
    if config.pca_skip:
        X_n = (X_enc_train - x_mean) / x_std
        pca_weight = fit_pca_weight(X_n, config.d_out, center=PCA_CENTER)
        log.info(
            "PCA skip: d_out=%d pca_center=%s (fit on encoder-normalized train)",
            config.d_out,
            PCA_CENTER,
        )
    if config.conditioning not in ("film", "concat"):
        raise ValueError(
            f"conditioning must be 'film' or 'concat', got {config.conditioning!r}"
        )
    if config.conditioning == "concat":
        encoder: Union[FiLMEncoder, ConcatEncoder] = ConcatEncoder(
            D,
            config.d_out,
            width=config.width,
            depth=config.depth,
            L=L,
            affinity_dim=affinity_dim,
            pca_skip=config.pca_skip,
            pca_weight=pca_weight,
        )
        log.info(
            "conditioning=concat: a(x) enters as %d extra input columns; "
            "FiLM hypernetworks are unused",
            affinity_dim,
        )
    else:
        encoder = FiLMEncoder(
            D,
            config.d_out,
            width=config.width,
            depth=config.depth,
            L=L,
            affinity_dim=affinity_dim,
            hyper_width=HYPER_WIDTH,
            pca_skip=config.pca_skip,
            pca_weight=pca_weight,
        )
    encoder.set_normalization(x_mean, x_std)
    model = PLANE(
        stack, encoder, encoder_view=encoder_view
    ).to(device)
    if init_state_dict is not None:
        missing, unexpected = model.load_state_dict(init_state_dict, strict=False)
        log.info(
            "warm-start: loaded init_state_dict (missing=%d unexpected=%d)",
            len(missing),
            len(unexpected),
        )

    # Measure from the ambient graph which neighbourhoods are crowded, before
    # anything is optimised. The layout is later held in correspondence with this
    # ordering; nothing here sets a contrast magnitude for it to reach.
    budget = None
    density_info: Dict[str, Any] = {}
    if config.lambda_density > 0:
        budget = density_budget(
            X_train,
            graph,
            metric,
            d_out=config.d_out,
            dim=graph.stats.extra.get("epsilon_intrinsic_dim"),
            seed=config.seed,
        )
        density_info.update(
            intrinsic_dim=budget.dim, ambient_log_r_sd=budget.ambient_sd
        )
        log.info("density budget: %s", budget.describe())

    a_param, b_param = find_ab_params(SPREAD, config.min_dist)
    # One EdgeSampler per pyramid level; per-step batch budget is split across
    # levels by weight so a single forward mixes all scales at ~constant cost.
    from leanmap.sampling.edges import basin_balanced_edge_weights

    mix = float(getattr(config, "landmark_sample_mix", 0.0) or 0.0)
    edge_samps = []
    for li, g in enumerate(graphs):
        w_override = None
        if mix > 0.0:
            # Primary landmark of each cell via the cell representative's row.
            rep_rows = g.reps.rep_idx.detach().cpu().long()
            # assign_top1 is aligned with the ambient rows used to build the graph.
            a1 = assign_top1.detach().cpu().long()
            if int(a1.shape[0]) <= int(rep_rows.max().item()):
                raise RuntimeError(
                    "assign_top1 shorter than max rep_idx; cannot basin-weight edges"
                )
            cell_lm = a1[rep_rows].numpy()
            w_override = basin_balanced_edge_weights(
                g.edges.cpu().numpy(),
                g.weights.cpu().numpy(),
                cell_lm,
                mix=mix,
            )
        edge_samps.append(
            EdgeSampler(
                X_train,
                g,
                seed=config.seed + li,
                weights=w_override,
            )
        )
    if mix > 0.0:
        log.info(
            "landmark basin edge mix=%.3g (epoch_unit=%s)",
            mix,
            getattr(config, "epoch_unit", "edges"),
        )
    # Per-level base weights (before epoch active-set masking).
    edge_base_weights = [
        np.asarray(s._base_weights, dtype=np.float64).copy() for s in edge_samps
    ]
    n_levels = len(graphs)
    if config.pyramid_level_weights is not None:
        lvl_w = [float(x) for x in config.pyramid_level_weights]
        if len(lvl_w) > n_levels:
            # Weights are finest-first and the LAST entry is the coarse/global
            # one. Plain truncation would drop exactly the long-range attraction
            # that anchors global geodesic structure, so keep the coarsest weight
            # and drop from the middle instead.
            dropped = lvl_w[n_levels - 1 : -1]
            lvl_w = lvl_w[: n_levels - 1] + [lvl_w[-1]]
            log.warning(
                "pyramid_level_weights has %d entries but only %d level(s) were "
                "built (coarsening stops at pyramid_min_reps=%d); dropped middle "
                "weights %s and kept the coarsest weight %.3g. Pass a %d-tuple to "
                "set this explicitly.",
                len(config.pyramid_level_weights),
                n_levels,
                config.pyramid_min_reps,
                [round(w, 3) for w in dropped],
                lvl_w[-1],
                n_levels,
            )
        elif len(lvl_w) < n_levels:
            lvl_w += [lvl_w[-1] if lvl_w else 1.0] * (n_levels - len(lvl_w))
    else:
        lvl_w = [1.0] * n_levels
    lvl_counts = _split_budget(config.batch_edges, lvl_w, list(range(n_levels)))
    if n_levels > 1:
        log.info(
            "pyramid training: %d levels, per-step edge split=%s (weights=%s)",
            n_levels,
            lvl_counts,
            [round(w, 3) for w in lvl_w],
        )
    neg_samp = NegativeSampler(X_train, graph.reps, seed=config.seed)
    # Local-rigidity term: sample fine-graph neighbourhoods ("stars") from the
    # finest level so ambient distance is a good local-geodesic proxy.
    # The density term reads the same stars: a node's neighbourhood radius is
    # what both terms are about, one constraining its shape and the other its
    # size.
    star_samp = (
        StarSampler(X_train, graphs[0], m=config.frame_neighbors, seed=config.seed)
        if config.lambda_frame > 0
        else None
    )
    frame_centers = FRAME_CENTERS
    # The density term gets its own stars: a wider neighbourhood (``n_neighbors``
    # rather than the frame term's 6) for a steadier radius, and a fixed
    # neighbour set so the ambient target and the layout estimate are measured
    # over identical stars.
    density_on = budget is not None
    dens_samp = dens_target = None
    if density_on:
        dens_samp = StarSampler(
            X_train,
            graphs[0],
            m=config.n_neighbors,
            seed=config.seed,
            deterministic=True,
        )
        nbr_idx, nbr_mask = dens_samp.padded_neighbours()
        star_budget = budget.on_stars(
            star_log_radius(
                X_train, graph.reps.rep_idx, nbr_idx, nbr_mask, metric
            )
        )
        dens_target = star_budget.target.to(device)
        density_info["star_ambient_log_r_sd"] = star_budget.ambient_sd
        log.info(
            "density term active: correlating log radii against an ambient "
            "spread of %.3f on %d-neighbour stars, weight %.3g, gate %s, "
            "var_shift %.3g",
            star_budget.ambient_sd,
            config.n_neighbors,
            config.lambda_density,
            config.density_ramp,
            config.density_var_shift,
        )
    # Coarse geodesic backbone: classical MDS of landmark geodesics on a
    # selectable pyramid level (metric edge lengths) + Procrustes pull
    # (absolute gauge) plus optional pairwise stress.
    geo_pack = None
    if config.lambda_geo > 0:
        n_levels = len(graphs)
        n_reps0 = int(graphs[0].reps.rep_idx.shape[0])
        if n_levels <= 1:
            resolved_gauge = 0
        elif config.gauge_level is not None:
            resolved_gauge = int(config.gauge_level)
            resolved_gauge = max(0, min(resolved_gauge, n_levels - 1))
        else:
            resolved_gauge = select_gauge_level(n_reps0)
            resolved_gauge = max(0, min(resolved_gauge, n_levels - 1))

        g_gauge = graphs[resolved_gauge]
        X_rep_g = X_train[g_gauge.reps.rep_idx]
        lengths = metric_edge_lengths(X_rep_g, g_gauge.edges, metric)
        # Map each landmark to its nearest training row, then to the level node.
        _, nn_idx = chunked_cdist(metric, M, X_train, topk=1, out_device=X_train.device)
        lm_raw = nn_idx[:, 0].detach().cpu().to(torch.int64)
        X_lm = X_train[lm_raw].contiguous()
        lm_level = g_gauge.reps.member_of[lm_raw].to(torch.int64)

        G_geo = landmark_geodesics_on_level(g_gauge, lengths, lm_level)
        finite_geo = torch.isfinite(G_geo)
        finite_geo = finite_geo.clone()
        finite_geo.fill_diagonal_(False)
        ii, jj = torch.where(torch.triu(finite_geo, diagonal=1))
        if ii.numel() == 0:
            log.warning(
                "geodesic backbone: no finite landmark pairs — disabling lambda_geo"
            )
        else:
            Z_mds, mds_diag = classical_mds(
                G_geo, d=config.d_out, finite=finite_geo, return_diagnostics=True
            )
            nu = gauge_nu_diagnostic(Z_mds, G_geo, finite=finite_geo)
            gauge_info = {
                "gauge_level": int(resolved_gauge),
                "nu": float(nu),
                **mds_diag,
            }
            graph.stats.extra.update(gauge_info)
            neg_ratio = float(nu)
            if neg_ratio > MDS_NEG_EIGEN_WARN:
                # Reported, not acted on: making a loss weight a hidden function
                # of a diagnostic is the buried coupling this config avoids.
                log.warning(
                    "classical MDS negative-eigenvalue mass ν=%.3f (> %.2f) on "
                    "gauge level %d: the landmark geodesics are not well "
                    "embeddable in %d-D, so the Procrustes target is a lossy "
                    "projection. Consider lambda_anchor=0 with lambda_frame > 0, "
                    "which keeps metric fidelity without pinning a gauge that "
                    "does not exist.",
                    neg_ratio,
                    MDS_NEG_EIGEN_WARN,
                    resolved_gauge,
                    config.d_out,
                )
            else:
                log.info(
                    "classical MDS ν=%.3f on gauge level %d; top-%d "
                    "eigenvalues carry %.3f of the positive spectrum",
                    neg_ratio,
                    resolved_gauge,
                    config.d_out,
                    float(mds_diag["mds_top_eigen_frac"]),
                )
            g_vals = G_geo[ii, jj]
            geo_pack = {
                "X_lm": X_lm,
                "Z_mds": Z_mds,
                "ii": ii,
                "jj": jj,
                "g": g_vals,
            }
            n_pairs = int(ii.numel())
            log.info(
                "geodesic backbone: L=%d level=%d MDS+Procrustes + stress "
                "pairs=%d (%.1f%%) geo median=%.4g lambda_geo=%.3g ramp=%s",
                X_lm.shape[0],
                resolved_gauge,
                n_pairs,
                100.0 * n_pairs / max(X_lm.shape[0] * (X_lm.shape[0] - 1) / 2, 1),
                float(g_vals.median().item()),
                config.lambda_geo,
                tuple(config.geo_ramp),
            )
    primary_name = stack.primary_factor.name
    ord_by_factor: Dict[str, OrdinalTripletSampler] = {}
    for f, aff in zip(factor_list, affs):
        top1_f, _ = assign_buckets(
            f.view(X_train), aff.M.detach().cpu(), f.metric, c=min(C_BUCKETS, aff.M.shape[0])
        )
        ord_by_factor[f.name] = OrdinalTripletSampler(
            X_train, top1_f, f.metric, seed=config.seed, view=f.view
        )
    # Ordinal loss / retention always uses PRIMARY factor anchors + view metric
    ord_samp = ord_by_factor[primary_name]

    # Chance level for retention_f, measured on this data rather than asserted.
    try:
        retention_chance = estimate_retention_null(
            ord_samp, edge_samps[0], stack.primary_affinity, device=device
        )
        log.info(
            "empirical retention null = %.3f (shuffled landmark ranks; "
            "the old asserted constant was %.3f)",
            retention_chance,
            RETENTION_CHANCE,
        )
    except Exception as exc:  # never block a fit on a diagnostic
        retention_chance = RETENTION_CHANCE
        log.warning(
            "retention null estimation failed (%s); falling back to %.3f",
            exc,
            RETENTION_CHANCE,
        )
    retention_warn = retention_chance + (RETENTION_WARN - RETENTION_CHANCE)

    class_on = bool(axes_list) and float(config.lambda_class) > 0.0
    class_samplers: List[Tuple[ClassAxis, ClassOrderSampler]] = []
    class_spread: List[Dict[str, Any]] = []
    class_pinned: Tuple[int, ...] = tuple(
        int(a.axis) for a in axes_list if a.is_pinned
    )
    if axes_list and not class_on:
        log.warning(
            "class_axes were supplied but lambda_class=%.3g, so the ordering is "
            "not applied; the axes are only used for the order_* diagnostics",
            config.lambda_class,
        )
    if class_on:
        assert labels_train is not None
        for ax in axes_list:
            samp_c = ClassOrderSampler(
                X_train, labels_train, ax.rank, seed=config.seed
            )
            class_samplers.append((ax, samp_c))
            class_spread.append({})
            log.info(
                "class axis %r on %s: %d ordered class pairs, weight=%.3g x "
                "lambda_class=%.3g ramp=%s margin=%.3g; %d of %d coordinates "
                "unnamed",
                ax.name,
                f"coordinate {ax.axis}" if ax.is_pinned else "a free direction",
                samp_c.n_pairs,
                ax.weight,
                config.lambda_class,
                tuple(config.class_ramp),
                config.class_margin,
                config.d_out - sum(a.is_pinned for a in axes_list),
                config.d_out,
            )

    path_on = bool(path_constraints) and float(config.lambda_path) > 0.0
    path_samplers: List[Tuple[PathConstraint, PathTripletSampler]] = []
    path_states: List[Dict[str, float]] = []
    if path_constraints and not path_on:
        log.warning(
            "path_constraints were supplied but lambda_path=%.3g, so the path "
            "term is not applied",
            config.lambda_path,
        )
    if path_on:
        assert path_constraints is not None
        for pc in path_constraints:
            if pc.n_rows_required() > N:
                raise ValueError(
                    f"PathConstraint {pc.name!r} indexes row {pc.n_rows_required()-1} "
                    f"but X has {N} rows"
                )
            kept = (
                pc.restrict(train_idx.cpu().numpy(), N)
                if train_idx is not None
                else pc
            )
            if kept is None:
                log.warning(
                    "path %r: no triplets survived the train split — skipping",
                    pc.name,
                )
                continue
            samp_p = PathTripletSampler(X_train, kept, seed=config.seed + 17)
            path_samplers.append((kept, samp_p))
            path_states.append({})
            n_anch = int(np.unique(kept.triplets[:, 0]).size)
            log.info(
                "path %r: %d triplets (%d anchors of %d train), weight=%.3g x "
                "lambda_path=%.3g ramp=%s c=%.3g C=%.3g",
                kept.name,
                int(kept.triplets.shape[0]),
                n_anch,
                int(X_train.shape[0]),
                kept.weight,
                config.lambda_path,
                tuple(config.path_ramp),
                kept.c,
                kept.C,
            )
        if not path_samplers:
            path_on = False
            log.warning("lambda_path>0 but no usable triplets after the split")

    # Fixed once so the order diagnostic tracks the same points every epoch and
    # its trajectory is a statement about the layout, not about resampling.
    class_diag_idx: Optional[torch.Tensor] = None
    if axes_list and labels_train is not None:
        take = min(4096, X_train.shape[0])
        g_diag = torch.Generator().manual_seed(config.seed + 11)
        class_diag_idx = torch.randperm(X_train.shape[0], generator=g_diag)[:take]

    # Warm start: fit the encoder to the coarse landmark layout before the main
    # objective sees a step, so training refines a topologically sensible map
    # instead of building one from PCA or from zero.
    warm_info: Dict[str, Any] = {}
    if config.warm_start_steps > 0:
        layout = str(config.warm_start_layout)
        if layout not in WARM_START_LAYOUTS:
            raise ValueError(
                f"warm_start_layout={layout!r} is not one of {WARM_START_LAYOUTS}"
            )
        if layout == "isomap" and geo_pack is None:
            raise ValueError(
                "warm_start_layout='isomap' needs the classical MDS of the landmark "
                "geodesics, which is only built when lambda_geo > 0 (got "
                f"{config.lambda_geo}). Use 'spectral', or raise lambda_geo."
            )
        if layout == "auto" and graph.knn_idx.numel() == 0:
            # Ranking needs the representatives' ambient neighbours. Rather than
            # silently skipping the warm start, fall back to naming a layout.
            layout = "isomap" if geo_pack is not None else "spectral"
            log.warning(
                "warm_start_layout='auto' needs the representative kNN to rank "
                "candidates and it is empty; using %r instead",
                layout,
            )
        # Only build what will be used: an explicit choice should not pay for the
        # eigensolve or the geodesics behind the candidates it did not ask for.
        wanted = set(WARM_START_LAYOUTS[1:]) if layout == "auto" else {layout}
        X_rep = X_train[graph.reps.rep_idx.to(X_train.device)]
        candidates: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if "isomap" in wanted and geo_pack is not None:
            candidates["isomap"] = (geo_pack["X_lm"], geo_pack["Z_mds"])
        if "spectral" in wanted:
            candidates["spectral"] = (
                X_rep,
                spectral_layout(
                    graph.edges,
                    graph.weights,
                    int(X_rep.shape[0]),
                    config.d_out,
                    seed=config.seed,
                ),
            )
        if "pca" in wanted:
            # PCA is the control: it is what the encoder starts from anyway, so it
            # scoring best is the answer "these coarse layouts add nothing here".
            W_pca = (
                pca_weight
                if pca_weight is not None
                else fit_pca_weight(
                    (X_enc_train - x_mean) / x_std, config.d_out, center=PCA_CENTER
                )
            )
            rep_sel = graph.reps.rep_idx.to(X_enc_train.device)
            X_enc_rep_n = (X_enc_train[rep_sel] - x_mean) / x_std
            candidates["pca"] = (X_rep, (X_enc_rep_n @ W_pca.T).contiguous())
        warm_info = warm_start(
            model,
            X_train,
            candidates,
            metric,
            layout=layout,
            X_ref=X_rep,
            reference_knn=graph.knn_idx,
            steps=int(config.warm_start_steps),
            batch=config.batch_edges,
            lr=float(
                config.warm_start_lr
                if config.warm_start_lr is not None
                else config.lr
            ),
            min_dist=float(config.min_dist),
            seed=config.seed,
        )

    E = graph.edges.shape[0]
    steps_per_epoch = max(1, math.ceil(E / config.batch_edges))
    n_lm_plan = int(M.shape[0]) if M is not None else int(config.n_landmarks)
    plan = coarse_to_fine_plan(
        config.epochs,
        [int(g.edges.shape[0]) for g in graphs],
        config.batch_edges,
        lvl_w,
        config.coarse_first_frac,
        epoch_unit=str(getattr(config, "epoch_unit", "edges")),
        n_landmarks=n_lm_plan,
        landmark_epoch_samples=float(
            getattr(config, "landmark_epoch_samples", 128.0)
        ),
    )
    total_steps = sum(steps for _, steps in plan)
    if str(getattr(config, "epoch_unit", "edges")).lower() in (
        "landmarks",
        "landmark",
        "basin",
    ):
        log.info(
            "epoch_unit=landmarks: %d steps/epoch (L=%d × %.4g samples / batch=%d); "
            "edge-unit would be %d steps/epoch on finest (E=%d)",
            plan[-1][1],
            n_lm_plan,
            float(getattr(config, "landmark_epoch_samples", 128.0)),
            int(config.batch_edges),
            steps_per_epoch,
            int(E),
        )
    if config.coarse_first_frac > 0 and n_levels > 1:
        log.info(
            "coarse-to-fine: %d total steps vs %d flat (%.0f%% of the work), "
            "first epoch %d step(s) on levels %s",
            total_steps,
            plan[-1][1] * config.epochs,
            100.0 * total_steps / max(plan[-1][1] * config.epochs, 1),
            plan[0][1],
            [i for i, c in enumerate(plan[0][0]) if c > 0],
        )
    opt = AdamW(
        _param_groups(model, config), lr=config.lr, weight_decay=WEIGHT_DECAY
    )
    if config.lr_after is not None and config.lr_switch_epochs > 0:
        switch_steps = sum(steps for _, steps in plan[: config.lr_switch_epochs])
        lr0 = float(config.lr)
        lr1 = float(config.lr_after)
        ratio = lr1 / max(lr0, 1e-12)

        def _lr_lambda(step: int) -> float:
            return 1.0 if step < switch_steps else ratio

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        log.info(
            "LR schedule: %.4g for %d epoch(s) (%d steps), then %.4g",
            lr0,
            config.lr_switch_epochs,
            switch_steps,
            lr1,
        )
    else:
        warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
        sched = SequentialLR(
            opt,
            [
                LinearLR(opt, start_factor=0.01, total_iters=warmup_steps),
                CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps)),
            ],
            milestones=[warmup_steps],
        )

    scale_state = {"mean_af": 1.0}
    ckpt_every = max(1, config.epochs // 10)
    global_step = 0

    # Overlapping epoch active-set passes (optional; large-N default via for_scale).
    from leanmap.sampling.epoch_pass import (
        active_member_csr,
        cell_intersects_active,
        edge_weights_for_active_cells,
        estimate_cover_passes,
        format_cover_passes,
        next_epoch_active_set,
    )

    epoch_active_rows = getattr(config, "epoch_active_rows", None)
    epoch_overlap = float(getattr(config, "epoch_overlap", 0.2) or 0.0)
    epoch_cover_visits = int(getattr(config, "epoch_cover_visits", 1) or 1)
    use_epoch_pass = epoch_active_rows is not None and int(epoch_active_rows) > 0
    prev_active = None
    pass_rng = np.random.default_rng(int(config.seed) + 17)
    if use_epoch_pass:
        B_active = min(int(epoch_active_rows), int(X_train.shape[0]))
        cover_rep = estimate_cover_passes(
            int(X_train.shape[0]),
            B_active,
            epoch_overlap,
            n_visits=epoch_cover_visits,
        )
        log.info(
            "epoch active-set: B=%d overlap=%.2f — %s",
            B_active,
            epoch_overlap,
            format_cover_passes(cover_rep),
        )
        if int(config.epochs) < int(cover_rep["epochs"]):
            log.warning(
                "epochs=%d < cover estimate %d for %d× visits; "
                "raise epochs or epoch_active_rows / lower epoch_overlap",
                int(config.epochs),
                int(cover_rep["epochs"]),
                epoch_cover_visits,
            )

    for epoch in range(config.epochs):
        model.train()
        if use_epoch_pass:
            B_active = min(int(epoch_active_rows), int(X_train.shape[0]))
            active = next_epoch_active_set(
                int(X_train.shape[0]),
                B_active,
                prev_active,
                epoch_overlap,
                pass_rng,
            )
            if prev_active is not None and active.size and prev_active.size:
                inter = float(np.intersect1d(active, prev_active).size)
                ov = inter / float(active.size)
            else:
                ov = 0.0
            log.info(
                "epoch %d active-set: |A|=%d overlap_with_prev=%.3f (target=%.2f)",
                epoch + 1,
                int(active.size),
                ov,
                epoch_overlap,
            )
            for li, (samp, g) in enumerate(zip(edge_samps, graphs)):
                cell_hit = cell_intersects_active(g.reps, active)
                w_mask = edge_weights_for_active_cells(
                    samp.edges, edge_base_weights[li], cell_hit, require_both=True
                )
                samp.set_weights(w_mask)
                off_a, val_a = active_member_csr(g.reps, active)
                samp.set_member_csr(off_a, val_a)
            cell_hit0 = cell_intersects_active(graphs[0].reps, active)
            neg_samp.set_active_cells(cell_hit0)
            prev_active = active
        totals = {
            "geom": 0.0,
            "ord": 0.0,
            "lm": 0.0,
            "frame": 0.0,
            "geo": 0.0,
            "dens": 0.0,
            "class": 0.0,
            "path": 0.0,
        }
        retentions = []
        gammas = []
        min_dm = []
        usage = []
        class_active: Dict[str, List[float]] = {}
        path_ords: List[float] = []

        t_frac = epoch / max(config.epochs - 1, 1)
        frame_ramp = alignment_ramp(t_frac, *config.frame_ramp)
        geo_ramp = alignment_ramp(t_frac, *config.geo_ramp)
        dens_ramp = alignment_ramp(t_frac, *config.density_ramp) if density_on else 0.0
        class_ramp = alignment_ramp(t_frac, *config.class_ramp) if class_on else 0.0
        path_ramp = alignment_ramp(t_frac, *config.path_ramp) if path_on else 0.0
        lvl_counts, epoch_steps = plan[epoch]

        pbar = tqdm(
            range(epoch_steps),
            desc=f"epoch {epoch + 1}/{config.epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        for _ in pbar:
            # Sample edges from every scale; concat into one batch. Positive
            # weights carry the per-level attraction weight (default equal).
            parts_i, parts_j, parts_w = [], [], []
            for li, cnt in enumerate(lvl_counts):
                if cnt <= 0:
                    continue
                xi_l, xj_l, w_l, _ = edge_samps[li].sample(cnt)
                parts_i.append(xi_l)
                parts_j.append(xj_l)
                parts_w.append(w_l * lvl_w[li])
            x_i = torch.cat(parts_i, dim=0)
            x_j = torch.cat(parts_j, dim=0)
            w = torch.cat(parts_w, dim=0)
            x_neg = neg_samp.sample(x_i.shape[0], config.n_negatives)
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            w = w.to(device)
            x_neg = x_neg.to(device)

            z_i, a_map, dm_map, gamma, beta, g_by, hit = model.forward_detailed(x_i)
            z_j, _, _ = model(x_j)
            B, n_neg, _ = x_neg.shape
            z_neg, _, _ = model(x_neg.reshape(B * n_neg, -1))
            z_neg = z_neg.view(B, n_neg, -1)

            L_geom = fuzzy_cross_entropy(z_i, z_j, w, z_neg, a_param, b_param)

            primary_name = stack.primary_factor.name
            a_i = a_map[primary_name]
            Dm_i = dm_map[primary_name]
            x_mid, x_far, mask, retention = ord_samp.sample(
                x_i, x_j, stack.primary_affinity
            )
            x_mid, x_far, mask = x_mid.to(device), x_far.to(device), mask.to(device)
            z_mid, _, _ = model(x_mid)
            z_far, _, _ = model(x_far)
            # DDP: allreduce ordinal path-scale batch mean (ā of ||z_a-z_f||)
            # via allreduce_path_scale / sync_train_stats before the EMA update.
            L_ord, scale_state = ordinal_triplet_loss(
                z_i, z_j, z_mid, z_far, mask, scale_state
            )
            retentions.append(retention)

            # DDP: allreduce_mean_affinity(a_i.mean(0)) before entropy in
            # landmark_regularisation (per-rank ā biases H(ā)).
            L_lm = landmark_regularisation(a_i, Dm_i, eta=ETA_BALANCE)

            # Dedicated pairs rather than reusing the edge batch: the edge
            # sampler draws neighbours, which are mostly same-class, so the
            # cross-class pairs a global ordering needs would be both rare and
            # biased toward whichever classes happen to touch.
            L_class = z_i.sum() * 0.0
            if class_samplers and class_ramp > 0:
                class_parts = []
                for (ax_c, samp_c), spread_c in zip(class_samplers, class_spread):
                    x_lo, x_hi = samp_c.sample(CLASS_PAIRS_PER_STEP)
                    z_lo, _, _ = model(x_lo.to(device))
                    z_hi, _, _ = model(x_hi.to(device))
                    if ax_c.is_pinned:
                        l_ax, _, active = class_order_loss(
                            z_lo,
                            z_hi,
                            ax_c.axis,
                            spread_c,
                            margin=float(config.class_margin),
                        )
                    else:
                        l_ax, _, active, _ = class_direction_loss(
                            z_lo,
                            z_hi,
                            class_pinned,
                            spread_c,
                            margin=float(config.class_margin),
                            orthogonal=ax_c.orthogonal,
                            whiten=ax_c.whiten,
                        )
                    class_parts.append(ax_c.weight * l_ax)
                    class_active.setdefault(ax_c.name, []).append(active)
                L_class = torch.stack(class_parts).sum()

            L_path = z_i.sum() * 0.0
            if path_samplers and path_ramp > 0:
                path_parts = []
                for pi, ((pc_p, samp_p), st_p) in enumerate(zip(path_samplers, path_states)):
                    xa, xn, xm, xf, dtn, dtm = samp_p.sample(PATH_PAIRS_PER_STEP)
                    za, _, _ = model(xa.to(device))
                    zn, _, _ = model(xn.to(device))
                    zm, _, _ = model(xm.to(device))
                    zf, _, _ = model(xf.to(device))
                    # DDP: allreduce_path_scale(batch_s) before scale_state EMA.
                    l_p, st_p, ofrac = path_constraint_loss(
                        za,
                        zn,
                        zm,
                        zf,
                        dtn.to(device),
                        dtm.to(device),
                        c=float(pc_p.c),
                        C=float(pc_p.C),
                        margin=float(config.path_margin),
                        scale_state=st_p,
                    )
                    path_states[pi] = st_p
                    path_parts.append(pc_p.weight * l_p)
                    path_ords.append(ofrac)
                L_path = torch.stack(path_parts).sum()

            L_frame = z_i.sum() * 0.0
            if star_samp is not None and frame_ramp > 0:
                xc, xnbr, fmask = star_samp.sample(frame_centers)
                xc = xc.to(device)
                B_f, m_f, _ = xnbr.shape
                xnbr = xnbr.to(device)
                fmask = fmask.to(device)
                zc, _, _ = model(xc)
                znbr, _, _ = model(xnbr.reshape(B_f * m_f, -1))
                znbr = znbr.view(B_f, m_f, -1)
                L_frame = local_rigidity_loss(
                    zc,
                    znbr,
                    xc,
                    xnbr,
                    fmask,
                    tangent=config.frame_tangent,
                    tangent_dim=FRAME_TANGENT_DIM,
                    normal_thresh=FRAME_NORMAL_THRESH,
                )

            L_dens = z_i.sum() * 0.0
            if dens_target is not None and dens_samp is not None and dens_ramp > 0:
                xdc, xdn, dmask, dcells = dens_samp.sample_indexed(
                    config.density_centers
                )
                B_d, m_d, _ = xdn.shape
                zdc, _, _ = model(xdc.to(device))
                zdn, _, _ = model(xdn.reshape(B_d * m_d, -1).to(device))
                # DDP: allreduce_density_moments(mean, sq_mean, count) before
                # correlation centering (local means ≠ global moments).
                L_dens = density_correlation_loss(
                    zdc,
                    zdn.view(B_d, m_d, -1),
                    dmask.to(device),
                    dens_target[dcells.to(device)],
                    var_shift=config.density_var_shift,
                )

            L_geo = z_i.sum() * 0.0
            if geo_pack is not None and geo_ramp > 0:
                # DDP: geo is replicated on every rank — do NOT allreduce geo
                # pair tensors / geodesic distances (identical inputs → grads
                # average cleanly under DDP).
                # Embed ALL landmarks; Procrustes-pull toward classical MDS
                # layout (untwists the global gauge). Mild pairwise stress on
                # a subsample keeps metric fidelity without trapping a twist.
                x_all = geo_pack["X_lm"].to(device)
                z_all, _, _ = model(x_all)
                L_anchor = (
                    procrustes_anchor_loss(z_all, geo_pack["Z_mds"])
                    if config.lambda_anchor != 0.0
                    else z_all.sum() * 0.0
                )
                n_avail = int(geo_pack["ii"].shape[0])
                n_take = (
                    GEO_PAIRS if GEO_PAIRS is not None else min(n_avail, 2048)
                )
                n_take = max(1, min(n_take, n_avail))
                pick = torch.randint(0, n_avail, (n_take,))
                ia = geo_pack["ii"][pick]
                ib = geo_pack["jj"][pick]
                L_stress = geodesic_stress_loss(
                    z_all[ia], z_all[ib], geo_pack["g"][pick].to(device)
                )
                L_geo = config.lambda_anchor * L_anchor + 0.25 * L_stress

            loss = (
                L_geom
                + LAMBDA_ORD * L_ord
                + config.lambda_lm * L_lm
                + config.lambda_frame * frame_ramp * L_frame
                + config.lambda_geo * geo_ramp * L_geo
                + config.lambda_density * dens_ramp * L_dens
                + config.lambda_class * class_ramp * L_class
                + config.lambda_path * path_ramp * L_path
            )
            opt.zero_grad()
            if not torch.isfinite(loss):
                log.warning("non-finite loss at step %d — skipping", global_step)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            global_step += 1
            if callbacks:
                for cb in callbacks:
                    on_step = getattr(cb, "on_step", None)
                    if on_step is not None:
                        on_step(
                            epoch + 1,
                            global_step,
                            model,
                            {"batch_n": int(x_i.shape[0])},
                        )

            totals["geom"] += float(L_geom.item())
            totals["ord"] += float(L_ord.item())
            totals["lm"] += float(L_lm.item())
            totals["frame"] += float(L_frame.item()) if torch.is_tensor(L_frame) else 0.0
            totals["geo"] += float(L_geo.item()) if torch.is_tensor(L_geo) else 0.0
            totals["dens"] += float(L_dens.item()) if torch.is_tensor(L_dens) else 0.0
            totals["class"] += float(L_class.item()) if torch.is_tensor(L_class) else 0.0
            totals["path"] += float(L_path.item()) if torch.is_tensor(L_path) else 0.0
            with torch.no_grad():
                gammas.append(gamma.mean().item())
                gammas.append(gamma.std().item())
                min_dm.append(float(Dm_i.min(dim=1).values.mean().item()))
                usage.append(a_i.mean(dim=0).cpu())
                for fname, samp in ord_by_factor.items():
                    if fname == primary_name:
                        continue
                    idx = stack.names.index(fname)
                    _, _, _, ret_f = samp.sample(x_i, x_j, stack.affinities[idx])
                    # store on metrics via side channel
                    if not hasattr(model, "_ret_extra"):
                        model._ret_extra = {}  # type: ignore[attr-defined]
                    model._ret_extra.setdefault(fname, []).append(ret_f)  # type: ignore[attr-defined]
                for fname, g_f in g_by.items():
                    if not hasattr(model, "_g_extra"):
                        model._g_extra = {}  # type: ignore[attr-defined]
                    model._g_extra.setdefault(fname, []).append(  # type: ignore[attr-defined]
                        (float(g_f.mean().item()), float(g_f.std().item()))
                    )

            steps_done = pbar.n + 1
            pbar.set_postfix(
                {
                    "loss": f"{float(loss.item()):.3f}",
                    "geom": f"{totals['geom'] / steps_done:.3f}",
                    "ord": f"{totals['ord'] / steps_done:.3f}",
                    "lm": f"{totals['lm'] / steps_done:.3f}",
                    "frame": f"{totals['frame'] / steps_done:.3f}",
                    "geo": f"{totals['geo'] / steps_done:.3f}",
                    "dens": f"{totals['dens'] / steps_done:.3f}",
                    "cls": f"{totals['class'] / steps_done:.3f}" if class_on else "—",
                    "path": f"{totals['path'] / steps_done:.3f}" if path_on else "—",
                    "ret": f"{float(np.mean(retentions)):.2f}" if retentions else "—",
                    "lr": f"{opt.param_groups[0]['lr']:.4g}",
                },
                refresh=False,
            )

        nstep = epoch_steps
        if not usage:
            log.warning("epoch %d: every step skipped (non-finite loss) — aborting", epoch + 1)
            raise RuntimeError(
                f"all {epoch_steps} steps in epoch {epoch + 1} had non-finite loss"
            )
        mean_a = torch.stack(usage).mean(dim=0).clamp_min(1e-12)
        usage_ent = float((-(mean_a * mean_a.log()).sum()).item())
        g_mean = float(np.mean(gammas[0::2])) if gammas else 0.0
        g_std = float(np.mean(gammas[1::2])) if gammas else 0.0
        ret = float(np.mean(retentions)) if retentions else 0.0
        metrics = {
            "geom": totals["geom"] / nstep,
            "ord": totals["ord"] / nstep,
            "lm": totals["lm"] / nstep,
            "frame": totals["frame"] / nstep,
            "geo": totals["geo"] / nstep,
            "dens": totals["dens"] / nstep,
            "class": totals["class"] / nstep,
            "path": totals["path"] / nstep,
            "path_ord": float(np.mean(path_ords)) if path_ords else 0.0,
            "retention": ret,
            "mean_gamma": g_mean,
            "std_gamma": g_std,
            "usage_ent": usage_ent,
            "min_dm": float(np.mean(min_dm)) if min_dm else 0.0,
        }
        log.info(
            "epoch %d: geom=%.4f ord=%.4f "
            "lm=%.4f frame=%.4f geo=%.4f dens=%.4f class=%.4f path=%.4f "
            "path_ord=%.3f retention=%.3f (chance≈%.3f) mean(gamma)=%.3f std(gamma)=%.3f "
            "usage_ent=%.3f minDm=%.4f",
            epoch + 1,
            metrics["geom"],
            metrics["ord"],
            metrics["lm"],
            metrics["frame"],
            metrics["geo"],
            metrics["dens"],
            metrics["class"],
            metrics["path"],
            metrics["path_ord"],
            metrics["retention"],
            retention_chance,
            metrics["mean_gamma"],
            metrics["std_gamma"],
            metrics["usage_ent"],
            metrics["min_dm"],
        )
        extra = getattr(model, "_ret_extra", {})
        for fname, vals in extra.items():
            rf = float(np.mean(vals)) if vals else 0.0
            metrics[f"retention_{fname}"] = rf
            if rf < retention_warn:
                log.warning(
                    "factor %r retention_f=%.3f < %.2f (chance≈%.3f) — conditioning "
                    "on noise; other metrics in this run are unreliable",
                    fname,
                    rf,
                    retention_warn,
                    retention_chance,
                )
        model._ret_extra = {}  # type: ignore[attr-defined]
        g_extra = getattr(model, "_g_extra", {})
        for fname, vals in g_extra.items():
            if vals:
                metrics[f"mean_gamma_{fname}"] = float(np.mean([v[0] for v in vals]))
                metrics[f"std_gamma_{fname}"] = float(np.mean([v[1] for v in vals]))
        model._g_extra = {}  # type: ignore[attr-defined]
        for name_c, vals in class_active.items():
            metrics[f"class_active_{name_c}"] = float(np.mean(vals)) if vals else 0.0
        if class_diag_idx is not None and (
            (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs
        ):
            assert labels_train is not None
            with torch.no_grad():
                xb_c = X_train[class_diag_idx]
                z_c = []
                for s in range(0, xb_c.shape[0], 4096):
                    zb_c, _, _ = model(xb_c[s : s + 4096].to(device))
                    z_c.append(zb_c.cpu())
                report = class_axis_report(
                    torch.cat(z_c, dim=0), labels_train[class_diag_idx], axes_list
                )
            metrics.update(report)
            for ax_c in axes_list:
                adj = report.get(f"order_adjacent_{ax_c.name}", float("nan"))
                where = "coordinate %d" % ax_c.axis if ax_c.is_pinned else (
                    "a direction tilted %.1f deg into the pinned axes%s"
                    % (
                        report.get(f"tilt_{ax_c.name}", float("nan")),
                        " (forced square)" if ax_c.orthogonal else "",
                    )
                )
                log.info(
                    "class axis %r on %s: ordering accuracy %.3f (adjacent "
                    "classes %.3f, chance %.2f%s)",
                    ax_c.name,
                    where,
                    report.get(f"order_{ax_c.name}", float("nan")),
                    adj,
                    ORDER_CHANCE,
                    "" if ax_c.is_pinned else " nominal, higher in practice",
                )
                if np.isfinite(adj) and adj < ORDER_WARN:
                    log.warning(
                        "class axis %r: adjacent-class ordering accuracy %.3f < "
                        "%.2f (chance %.2f) — the features do not separate "
                        "consecutive classes in the requested order, so this "
                        "ordering is not present in the layout. Raising "
                        "lambda_class will distort the layout rather than fix it",
                        ax_c.name,
                        adj,
                        ORDER_WARN,
                        ORDER_CHANCE,
                    )
        if ret < retention_warn:
            log.warning(
                "factor %r retention_f=%.3f < %.2f (chance≈%.3f) — conditioning "
                "on noise; other metrics in this run are unreliable",
                stack.primary_factor.name,
                ret,
                retention_warn,
                retention_chance,
            )
        if g_std < 1e-3:
            log.info("std(gamma) near zero — FiLM conditioning may be unused")
        if ret < 0.3:
            log.info("triplet retention < 0.3 — landmark ranking is a poor proxy")

        if (epoch + 1) % 10 == 0 and len(factor_list) > 1:
            # Affinity correlation on a train batch
            with torch.no_grad():
                xb = X_train[: min(512, X_train.shape[0])].to(device)
                _, a_map, _, _, _, _, _ = model.forward_detailed(xb)
                names = list(a_map.keys())
                vecs = [a_map[n].mean(dim=0).cpu() for n in names]
                C = torch.zeros(len(names), len(names))
                for i in range(len(names)):
                    for j in range(len(names)):
                        # Factors may have different anchor counts (e.g. a
                        # conditioning pyramid); correlation is only defined for
                        # equal-length affinity vectors.
                        if vecs[i].shape != vecs[j].shape:
                            C[i, j] = 0.0
                            continue
                        vi, vj = vecs[i] - vecs[i].mean(), vecs[j] - vecs[j].mean()
                        den = float(vi.norm() * vj.norm())
                        C[i, j] = (vi @ vj) / den if den > 0 else 0.0
                log.info("affinity correlation C=\n%s", C)
                off = C.clone()
                off.fill_diagonal_(0.0)
                if float(off.abs().max()) > 0.7:
                    log.warning(
                        "high off-diagonal affinity correlation (max=%.3f)",
                        float(off.abs().max()),
                    )

        if callbacks:
            for cb in callbacks:
                cb(epoch + 1, model, metrics)

        if (epoch + 1) % ckpt_every == 0:
            pass  # checkpoints are caller-driven via callbacks; hook reserved

    # 11. conformal on raw calib
    model.eval()
    calibrator = ConformalCalibrator()
    calibrator.fit(model, X_cal.to(device))

    # 11b. geodesic-fidelity diagnostic on the finest graph (global structure)
    geo_stats: Dict[str, float] = {}
    try:
        with torch.no_grad():
            rep_pts = X_train[graph.reps.rep_idx]
            z_reps = []
            for s in range(0, rep_pts.shape[0], 8192):
                zb, _, _ = model(rep_pts[s : s + 8192].to(device))
                z_reps.append(zb.cpu())
            Z_reps = torch.cat(z_reps, dim=0)
        geo_stats = geodesic_fidelity(
            graph.edges, graph.weights, Z_reps, seed=config.seed
        )
        log.info(
            "geodesic fidelity: spearman=%.3f stress=%.3f (pairs=%d)",
            geo_stats.get("geodesic_spearman", float("nan")),
            geo_stats.get("geodesic_stress", float("nan")),
            int(geo_stats.get("geodesic_pairs", 0)),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("geodesic fidelity diagnostic failed: %s", exc)

    result = PLANEResult(
        model=model,
        config=config,
        calibrator=calibrator,
        a=a_param,
        b=b_param,
        graph_stats=asdict(graph.stats),
        metric_name=_metric_name_from_dist_fn(dist_fn),
    )
    result.natural_scale = getattr(metric, "natural_scale", None)
    result.density = density_info
    if warm_info:
        result.graph_stats = {**result.graph_stats, **warm_info}
    if factor_metric is not None:
        result.factor_scales = dict(factor_metric.scales)
    if geo_stats:
        result.graph_stats = {**result.graph_stats, **geo_stats}
    if axes_list:
        # The split is internal, so the caller cannot otherwise know which rows
        # trained and which calibrated — and building the readout on training
        # points while calibrating on held-out ones is exactly the discipline
        # LandmarkSupport already requires.
        result.class_axes = list(axes_list)
        result.class_labels_train = labels_train
        result.class_labels_calib = labels_calib
        result.X_train = X_train
        result.X_calib = X_cal

    return result

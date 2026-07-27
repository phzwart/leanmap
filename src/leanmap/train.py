"""Training loop, checkpointing, and artefact I/O."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .conditioning import (
    RETENTION_CHANCE,
    RETENTION_WARN,
    ConditioningFactor,
    FactorStack,
    Role,
    build_factor_stack,
    default_primary_factor,
    metric_from_factors,
)
from .config import AlignmentSpec, PLANEConfig
from .conformal import ConformalCalibrator, geometry_consistency_score, model_weight_hash
from .distance import DistanceFn, EuclideanDistance
from .evaluate import geodesic_fidelity
from .graph import Graph, build_graph, build_graph_pyramid
from .landmarks import AnchorAffinity, LandmarkAffinity, classical_mds, landmark_geodesic_matrix
from .losses import (
    alignment_ramp,
    axial_alignment_loss,
    find_ab_params,
    fuzzy_cross_entropy,
    geodesic_stress_loss,
    landmark_regularisation,
    lipschitz_penalty,
    local_rigidity_loss,
    ordinal_triplet_loss,
    prepare_alignment_targets,
    procrustes_anchor_loss,
    reconstruction_loss,
    regional_alignment_loss,
    sigreg_loss,
)
from .metrics import MetricSpec, wrap_metric
from .model import Decoder, FiLMEncoder, PLANE, fit_pca_weight
from .negative_space import (
    ALL_FEATURES,
    DistanceQuantileHead,
    PerturbationConfig,
    _median_nn_scale,
    calibrate_head,
    distance_to_support,
    features_with_grad,
    pinball_loss,
)
from .sampler import EdgeSampler, NegativeSampler, OrdinalTripletSampler, StarSampler
from .utils import ensure_2d_float32, get_logger, resolve_device, seed_everything


@dataclass
class PLANEResult:
    """Fitted artefact: model + calibration + provenance (no graph / no N-arrays)."""

    model: PLANE
    config: PLANEConfig
    calibrator: ConformalCalibrator
    a: float
    b: float
    graph_stats: dict
    alignment_meta: list

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
            "alignment_metadata": self.alignment_meta,
            "graph_stats": self.graph_stats,
            "D": enc.D,
            "L": aff.M.shape[0],
            "metric_name": self.config.metric,
            "natural_scale": getattr(self, "natural_scale", None),
            "factor_scales": getattr(self, "factor_scales", None),
        }
        torch.save(payload, str(path))

    def embed(self, X, **kwargs):
        return self.model.embed(torch.as_tensor(ensure_2d_float32(X)), **kwargs)


def load_plane(path: Union[str, Path], device: Optional[str] = None) -> PLANE:
    """Load a saved artefact into a ``PLANE`` ready for ``embed()``."""
    from .metrics import get_metric

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
        from .conditioning import identity_view

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
            hyper_width=cfg.hyper_width,
            d_out=cfg.d_out,
        ).to(device_t)
        affinity_dim = sum(a.M.shape[0] for a in affs)
        enc = FiLMEncoder(
            D,
            cfg.d_out,
            width=cfg.width,
            depth=cfg.depth,
            L=L,
            affinity_dim=affinity_dim,
            hyper_width=cfg.hyper_width,
            spectral_norm_flag=cfg.spectral_norm,
            concat_affinity=cfg.concat_affinity,
            pca_skip=cfg.pca_skip,
        )
        enc.set_normalization(payload["x_mean"], payload["x_std"])
        dec = Decoder(cfg.d_out, D, cfg.width) if cfg.use_decoder else None
        model = PLANE(stack, enc, dec).to(device_t)
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
            hyper_width=cfg.hyper_width,
            spectral_norm_flag=cfg.spectral_norm,
            concat_affinity=cfg.concat_affinity,
            pca_skip=cfg.pca_skip,
        )
        enc.set_normalization(payload["x_mean"], payload["x_std"])
        dec = Decoder(cfg.d_out, D, cfg.width) if cfg.use_decoder else None
        model = PLANE(aff, enc, dec).to(device_t)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model


def fit(
    X: np.ndarray | torch.Tensor,
    dist_fn: Union[str, MetricSpec, DistanceFn] = "l2",
    config: Optional[PLANEConfig] = None,
    alignments: Optional[Sequence[AlignmentSpec]] = None,
    X_calib: Optional[np.ndarray | torch.Tensor] = None,
    callbacks: Optional[List[Callable]] = None,
    factors: Optional[Sequence[ConditioningFactor]] = None,
    encoder_view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    init_state_dict: Optional[dict] = None,
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
    """
    from .conditioning import identity_view
    from .landmarks import assign_buckets, init_anchors

    log = get_logger()
    if config is None:
        X_tmp = ensure_2d_float32(X)
        config = PLANEConfig.for_scale(X_tmp.shape[0])
    seed_everything(config.seed)
    device = resolve_device(config.device)
    # Spectral-norm power iteration uses aten::vdot, which MPS does not implement
    # even with PYTORCH_ENABLE_MPS_FALLBACK in some torch builds.
    if device.type == "mps" and config.spectral_norm:
        log = get_logger()
        log.info("MPS: disabling spectral_norm (aten::vdot unsupported)")
        from dataclasses import replace

        config = replace(config, spectral_norm=False)
    X_all = torch.as_tensor(ensure_2d_float32(X), dtype=torch.float32)
    enc_view = encoder_view if encoder_view is not None else (lambda t: t)

    # 2. Split calibration first
    N = X_all.shape[0]
    if X_calib is not None:
        X_cal = torch.as_tensor(ensure_2d_float32(X_calib), dtype=torch.float32)
        X_train = X_all
        calib_idx = None
    else:
        n_cal = min(int(config.calib_frac * N), config.calib_max)
        n_cal = max(n_cal, 1)
        g = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(N, generator=g)
        calib_idx = perm[:n_cal]
        train_idx = perm[n_cal:]
        X_cal = X_all[calib_idx]
        X_train = X_all[train_idx]

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
        from .metrics import get_metric

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
        # Conditioning pyramid: add coarse MODULATOR levels (multi-resolution
        # FiLM). Each gets its own auto-scaled tau via AnchorAffinity._default_tau.
        for j, na in enumerate(config.conditioning_pyramid_levels or []):
            factor_list.append(
                ConditioningFactor(
                    name=f"coarse{j}",
                    view=identity_view,
                    metric=metric,
                    n_anchors=int(na),
                    role=Role.MODULATOR,
                    learn_anchors=config.learn_landmarks,
                    learn_temperature=config.learn_tau,
                )
            )
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
    graphs, M, assign_top1, assign_topc = build_graph_pyramid(
        X_train,
        metric,
        pyramid_scales=config.pyramid_scales,
        pyramid_rep_ratio=config.pyramid_rep_ratio,
        pyramid_min_reps=config.pyramid_min_reps,
        pyramid_coarse_backbone=config.pyramid_coarse_backbone,
        n_neighbors=config.n_neighbors,
        n_landmarks=n_graph_landmarks,
        c_buckets=config.c_buckets,
        epsilon=config.epsilon,
        dedup=config.dedup,
        local_connectivity=config.local_connectivity,
        beta_multiplicity=config.beta_multiplicity,
        hub_correction=config.hub_correction,
        lambda_backbone=config.lambda_backbone,
        knn_mode=config.knn_mode,
        c_search=config.c_search,
        seed=config.seed,
        extra_ivf_anchors=extra_ivf or None,
        fps_view=fps_view,
        fps_view_metric=fps_view_metric,
        fps_geodesic=config.landmark_geodesic,
        fps_geodesic_k=config.landmark_geodesic_k,
        fps_poisson=config.landmark_poisson,
    )
    graph = graphs[0]  # finest graph: reps/negatives/knn_idx/stats live here

    if calib_idx is not None:
        assert X_cal.shape[0] > 0

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
        hyper_width=config.hyper_width,
        d_out=config.d_out,
    ).to(device)
    L = stack.primary_affinity.M.shape[0]
    affinity_dim = sum(a.M.shape[0] for a in affs)
    pca_weight = None
    if config.pca_skip:
        X_n = (X_enc_train - x_mean) / x_std
        pca_weight = fit_pca_weight(X_n, config.d_out, center=config.pca_center)
        log.info(
            "PCA skip: d_out=%d pca_center=%s (fit on encoder-normalized train)",
            config.d_out,
            config.pca_center,
        )
    encoder = FiLMEncoder(
        D,
        config.d_out,
        width=config.width,
        depth=config.depth,
        L=L,
        affinity_dim=affinity_dim,
        hyper_width=config.hyper_width,
        spectral_norm_flag=config.spectral_norm,
        concat_affinity=config.concat_affinity,
        pca_skip=config.pca_skip,
        pca_weight=pca_weight,
    )
    encoder.set_normalization(x_mean, x_std)
    decoder = Decoder(config.d_out, D, config.width) if config.use_decoder else None
    model = PLANE(
        stack, encoder, decoder, encoder_view=encoder_view
    ).to(device)
    if init_state_dict is not None:
        missing, unexpected = model.load_state_dict(init_state_dict, strict=False)
        log.info(
            "warm-start: loaded init_state_dict (missing=%d unexpected=%d)",
            len(missing),
            len(unexpected),
        )

    a_param, b_param = find_ab_params(config.spread, config.min_dist)
    # One EdgeSampler per pyramid level; per-step batch budget is split across
    # levels by weight so a single forward mixes all scales at ~constant cost.
    edge_samps = [
        EdgeSampler(X_train, g, seed=config.seed + li) for li, g in enumerate(graphs)
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
    _wsum = sum(lvl_w) or 1.0
    lvl_counts = [int(round(config.batch_edges * (w / _wsum))) for w in lvl_w]
    lvl_counts[0] += config.batch_edges - sum(lvl_counts)  # fix rounding on finest
    lvl_counts = [max(0, c) for c in lvl_counts]
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
    star_samp = (
        StarSampler(X_train, graphs[0], m=config.frame_neighbors, seed=config.seed)
        if config.lambda_frame > 0
        else None
    )
    frame_centers = (
        config.frame_centers if config.frame_centers is not None else 128
    )
    # Coarse geodesic backbone: classical MDS of landmark geodesics +
    # Procrustes pull (absolute gauge) plus optional pairwise stress.
    geo_pack = None
    if config.lambda_geo > 0:
        gk = (
            config.landmark_geodesic_k
            if config.landmark_geodesic_k is not None
            else config.n_neighbors
        )
        X_lm, G_geo, finite_geo = landmark_geodesic_matrix(
            X_train, M, metric, n_neighbors=gk
        )
        ii, jj = torch.where(torch.triu(finite_geo, diagonal=1))
        if ii.numel() == 0:
            log.warning(
                "geodesic backbone: no finite landmark pairs — disabling lambda_geo"
            )
        else:
            Z_mds = classical_mds(G_geo, d=config.d_out, finite=finite_geo)
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
                "geodesic backbone: L=%d MDS+Procrustes + stress pairs=%d "
                "(%.1f%%) geo median=%.4g lambda_geo=%.3g ramp=%s down=%s",
                X_lm.shape[0],
                n_pairs,
                100.0 * n_pairs / max(X_lm.shape[0] * (X_lm.shape[0] - 1) / 2, 1),
                float(g_vals.median().item()),
                config.lambda_geo,
                tuple(config.geo_ramp),
                bool(config.geo_ramp_down),
            )
    primary_name = stack.primary_factor.name
    ord_by_factor: Dict[str, OrdinalTripletSampler] = {}
    for f, aff in zip(factor_list, affs):
        top1_f, _ = assign_buckets(
            f.view(X_train), aff.M.detach().cpu(), f.metric, c=min(8, aff.M.shape[0])
        )
        ord_by_factor[f.name] = OrdinalTripletSampler(
            X_train, top1_f, f.metric, seed=config.seed, view=f.view
        )
    # Ordinal loss / retention always uses PRIMARY factor anchors + view metric
    ord_samp = ord_by_factor[primary_name]

    prepared_align = prepare_alignment_targets(
        list(alignments or []), whiten_multi_axis=config.whiten_multi_axis
    )
    alignment_meta = [
        {"axis": s.axis, "kind": s.kind, "weight": s.weight, "sign": s.sign}
        for s in prepared_align
    ]

    # Optional negative-space co-training: an auxiliary distance-to-support
    # quantile head trained jointly with the encoder (grad flows into backbone).
    dist_head: Optional[DistanceQuantileHead] = None
    dist_feature_groups = tuple(config.dist_features) if config.dist_features else ALL_FEATURES
    dist_nn_scale = 1.0
    dist_support_ref: Optional[torch.Tensor] = None
    if config.lambda_dist > 0:
        from .negative_space import extract_features as _extract_features

        with torch.no_grad():
            feat_dim = _extract_features(
                model, X_train[:2], dist_feature_groups
            ).shape[1]
        dist_head = DistanceQuantileHead(
            feat_dim,
            width=config.dist_head_width,
            depth=config.dist_head_depth,
            input_norm=True,
        ).to(device)
        dist_nn_scale = _median_nn_scale(X_train, EuclideanDistance())
        ref_n = min(X_train.shape[0], 20000)
        dist_support_ref = X_train[:ref_n].to(device)
        log.info(
            "negative-space co-training: lambda_dist=%.3g ramp=%s features=%d "
            "perturb/step=%d nn_scale=%.4g",
            config.lambda_dist, tuple(config.dist_ramp), feat_dim,
            config.dist_perturb_per_step, dist_nn_scale,
        )

    E = graph.edges.shape[0]
    steps_per_epoch = max(1, math.ceil(E / config.batch_edges))
    total_steps = steps_per_epoch * config.epochs
    opt_params = list(model.parameters())
    if dist_head is not None:
        opt_params += list(dist_head.parameters())
    opt = AdamW(opt_params, lr=config.lr, weight_decay=config.weight_decay)
    if config.lr_after is not None and config.lr_switch_epochs > 0:
        switch_steps = int(config.lr_switch_epochs) * steps_per_epoch
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
        warmup_steps = max(1, int(config.warmup_frac * total_steps))
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

    for epoch in range(config.epochs):
        model.train()
        totals = {
            "geom": 0.0,
            "ord": 0.0,
            "align": 0.0,
            "rec": 0.0,
            "lip": 0.0,
            "sigreg": 0.0,
            "lm": 0.0,
            "frame": 0.0,
            "geo": 0.0,
            "dist": 0.0,
        }
        retentions = []
        gammas = []
        min_dm = []
        usage = []

        t_frac = epoch / max(config.epochs - 1, 1)
        ramp = alignment_ramp(t_frac, *config.align_ramp)
        frame_ramp = alignment_ramp(t_frac, *config.frame_ramp)
        geo_ramp = alignment_ramp(
            t_frac, *config.geo_ramp, down=bool(config.geo_ramp_down)
        )
        dist_ramp_val = alignment_ramp(t_frac, *config.dist_ramp)

        pbar = tqdm(
            range(steps_per_epoch),
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
            L_ord, scale_state = ordinal_triplet_loss(
                z_i, z_j, z_mid, z_far, mask, scale_state
            )
            retentions.append(retention)

            L_align = z_i.sum() * 0.0
            if prepared_align and ramp > 0:
                # Map batch points back to training indices — use values by matching
                # We don't have indices; for axial, sample property from random train rows
                # SPEC: alignments have values (N,) on the original training array.
                # Edge sampler returns raw members; we need their indices.
                # Re-sample: store last indices in EdgeSampler
                pass
            # Axial: use a random train batch for alignment (property from prepared values)
            if prepared_align and ramp > 0:
                g = torch.randint(0, X_train.shape[0], (config.batch_edges,))
                xb = X_train[g].to(device)
                zb, _, _ = model(xb)
                for spec in prepared_align:
                    if spec.kind == "axial":
                        r = torch.as_tensor(spec.values, dtype=torch.float32)[g].to(device)
                        L_align = L_align + spec.weight * axial_alignment_loss(
                            zb, r, spec.axis, spec.sign
                        )
                    elif spec.kind == "regional":
                        labels = torch.as_tensor(spec.labels)[g].to(device)
                        L_align = L_align + spec.weight * regional_alignment_loss(
                            zb, labels, spec.targets or {}
                        )

            L_rec = z_i.sum() * 0.0
            if decoder is not None:
                x_hat = decoder(z_i)
                x_enc_i = model._x_enc(x_i)
                x_n = (x_enc_i - encoder.x_mean) / encoder.x_std
                L_rec = reconstruction_loss(x_hat, x_n)

            L_lip = z_i.sum() * 0.0
            if config.lambda_lip > 0:
                L_lip = lipschitz_penalty(model, x_i)

            L_sigreg = z_i.sum() * 0.0
            if config.lambda_sigreg > 0:
                L_sigreg = sigreg_loss(
                    z_i,
                    n_slices=config.sigreg_slices,
                    n_points=config.sigreg_points,
                    t_max=config.sigreg_domain,
                    target_std=config.sigreg_target_std,
                )

            L_lm = landmark_regularisation(a_i, Dm_i, eta=config.eta_balance)

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
                    tangent_dim=config.frame_tangent_dim,
                    normal_thresh=config.frame_normal_thresh,
                )

            L_geo = z_i.sum() * 0.0
            if geo_pack is not None and geo_ramp > 0:
                # Embed ALL landmarks; Procrustes-pull toward classical MDS
                # layout (untwists the global gauge). Mild pairwise stress on
                # a subsample keeps metric fidelity without trapping a twist.
                x_all = geo_pack["X_lm"].to(device)
                z_all, _, _ = model(x_all)
                L_anchor = procrustes_anchor_loss(z_all, geo_pack["Z_mds"])
                n_avail = int(geo_pack["ii"].shape[0])
                n_take = (
                    config.geo_pairs
                    if config.geo_pairs is not None
                    else min(n_avail, 2048)
                )
                n_take = max(1, min(n_take, n_avail))
                pick = torch.randint(0, n_avail, (n_take,))
                ia = geo_pack["ii"][pick]
                ib = geo_pack["jj"][pick]
                L_stress = geodesic_stress_loss(
                    z_all[ia], z_all[ib], geo_pack["g"][pick].to(device)
                )
                L_geo = L_anchor + 0.25 * L_stress

            # Negative-space co-training: score perturbations of the batch points
            # (half on-manifold, half at a random shell radius) with the aux
            # quantile head; pinball loss back-props into the encoder. The label
            # is the fixed ambient distance to the train support.
            L_dist = z_i.sum() * 0.0
            if dist_head is not None and dist_ramp_val > 0 and dist_support_ref is not None:
                k = min(config.dist_perturb_per_step, x_i.shape[0])
                base = x_i[:k]
                Ddim = base.shape[1]
                log_lo = math.log(config.dist_r_min_mult * dist_nn_scale)
                log_hi = math.log(config.dist_r_max_mult * dist_nn_scale)
                r = torch.empty(k, 1, device=device).uniform_(log_lo, log_hi).exp()
                dirs = torch.randn(k, Ddim, device=device)
                dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
                x_pert = torch.cat([base, base + r * dirs], dim=0)
                with torch.no_grad():
                    y_lbl = distance_to_support(
                        x_pert, dist_support_ref, dist_fn=EuclideanDistance()
                    ).to(device)
                phi_d = features_with_grad(model, x_pert, dist_feature_groups)
                lo_d, med_d, hi_d = dist_head(phi_d, detach_median=True)
                a_lo, a_hi = config.dist_alpha / 2.0, 1.0 - config.dist_alpha / 2.0
                L_dist = (
                    pinball_loss(y_lbl, med_d, 0.5)
                    + pinball_loss(y_lbl, lo_d, a_lo)
                    + pinball_loss(y_lbl, hi_d, a_hi)
                )

            loss = (
                L_geom
                + config.lambda_ord * L_ord
                + ramp * L_align
                + config.lambda_rec * L_rec
                + config.lambda_lip * L_lip
                + config.lambda_sigreg * L_sigreg
                + config.lambda_lm * L_lm
                + config.lambda_frame * frame_ramp * L_frame
                + config.lambda_geo * geo_ramp * L_geo
                + config.lambda_dist * dist_ramp_val * L_dist
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

            totals["geom"] += float(L_geom.item())
            totals["ord"] += float(L_ord.item())
            totals["align"] += float(L_align.item()) if torch.is_tensor(L_align) else 0.0
            totals["rec"] += float(L_rec.item()) if torch.is_tensor(L_rec) else 0.0
            totals["lip"] += float(L_lip.item()) if torch.is_tensor(L_lip) else 0.0
            totals["sigreg"] += (
                float(L_sigreg.item()) if torch.is_tensor(L_sigreg) else 0.0
            )
            totals["lm"] += float(L_lm.item())
            totals["frame"] += float(L_frame.item()) if torch.is_tensor(L_frame) else 0.0
            totals["geo"] += float(L_geo.item()) if torch.is_tensor(L_geo) else 0.0
            totals["dist"] += float(L_dist.item()) if torch.is_tensor(L_dist) else 0.0
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
                    "dist": f"{totals['dist'] / steps_done:.3f}",
                    "ret": f"{float(np.mean(retentions)):.2f}" if retentions else "—",
                    "lr": f"{opt.param_groups[0]['lr']:.4g}",
                },
                refresh=False,
            )

        nstep = steps_per_epoch
        mean_a = torch.stack(usage).mean(dim=0).clamp_min(1e-12)
        usage_ent = float((-(mean_a * mean_a.log()).sum()).item())
        g_mean = float(np.mean(gammas[0::2])) if gammas else 0.0
        g_std = float(np.mean(gammas[1::2])) if gammas else 0.0
        ret = float(np.mean(retentions)) if retentions else 0.0
        metrics = {
            "geom": totals["geom"] / nstep,
            "ord": totals["ord"] / nstep,
            "align": totals["align"] / nstep,
            "rec": totals["rec"] / nstep,
            "lip": totals["lip"] / nstep,
            "sigreg": totals["sigreg"] / nstep,
            "lm": totals["lm"] / nstep,
            "frame": totals["frame"] / nstep,
            "geo": totals["geo"] / nstep,
            "dist": totals["dist"] / nstep,
            "retention": ret,
            "mean_gamma": g_mean,
            "std_gamma": g_std,
            "usage_ent": usage_ent,
            "min_dm": float(np.mean(min_dm)) if min_dm else 0.0,
        }
        log.info(
            "epoch %d: geom=%.4f ord=%.4f align=%.4f rec=%.4f lip=%.4f "
            "sigreg=%.4f lm=%.4f frame=%.4f geo=%.4f "
            "retention=%.3f (chance≈%.3f) mean(gamma)=%.3f std(gamma)=%.3f "
            "usage_ent=%.3f minDm=%.4f",
            epoch + 1,
            metrics["geom"],
            metrics["ord"],
            metrics["align"],
            metrics["rec"],
            metrics["lip"],
            metrics["sigreg"],
            metrics["lm"],
            metrics["frame"],
            metrics["geo"],
            metrics["retention"],
            RETENTION_CHANCE,
            metrics["mean_gamma"],
            metrics["std_gamma"],
            metrics["usage_ent"],
            metrics["min_dm"],
        )
        extra = getattr(model, "_ret_extra", {})
        for fname, vals in extra.items():
            rf = float(np.mean(vals)) if vals else 0.0
            metrics[f"retention_{fname}"] = rf
            if rf < RETENTION_WARN:
                log.warning(
                    "factor %r retention_f=%.3f < %.2f (chance≈%.3f) — conditioning "
                    "on noise; other metrics in this run are unreliable",
                    fname,
                    rf,
                    RETENTION_WARN,
                    RETENTION_CHANCE,
                )
        model._ret_extra = {}  # type: ignore[attr-defined]
        g_extra = getattr(model, "_g_extra", {})
        for fname, vals in g_extra.items():
            if vals:
                metrics[f"mean_gamma_{fname}"] = float(np.mean([v[0] for v in vals]))
                metrics[f"std_gamma_{fname}"] = float(np.mean([v[1] for v in vals]))
        model._g_extra = {}  # type: ignore[attr-defined]
        if ret < RETENTION_WARN:
            log.warning(
                "factor %r retention_f=%.3f < %.2f (chance≈%.3f) — conditioning "
                "on noise; other metrics in this run are unreliable",
                stack.primary_factor.name,
                ret,
                RETENTION_WARN,
                RETENTION_CHANCE,
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
        alignment_meta=alignment_meta,
    )
    result.natural_scale = getattr(metric, "natural_scale", None)
    if factor_metric is not None:
        result.factor_scales = dict(factor_metric.scales)
    if geo_stats:
        result.graph_stats = {**result.graph_stats, **geo_stats}

    # Finalize negative-space co-training: freeze + recalibrate (CQR) the aux
    # head on a fresh perturbation set, and attach it to the result.
    result.negative_space = None
    result.negative_space_stats = None
    if dist_head is not None:
        model.eval()
        ns_model, ns_stats = calibrate_head(
            model,
            dist_head,
            X_train,
            feature_groups=dist_feature_groups,
            alpha=config.dist_alpha,
            perturb=PerturbationConfig(
                r_min_mult=config.dist_r_min_mult,
                r_max_mult=config.dist_r_max_mult,
                seed=config.seed + 1,
            ),
        )
        result.negative_space = ns_model
        result.negative_space_stats = ns_stats

    return result

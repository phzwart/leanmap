"""Command-line interface: ``leanmap fit | transform | info | mondrian``.

Canonical CLI module (``_cli`` remains a shim).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32, copy=False)
    if p.suffix == ".npz":
        data = np.load(p)
        key = "X" if "X" in data else data.files[0]
        return data[key].astype(np.float32, copy=False)
    raise SystemExit(f"unsupported input format: {p.suffix} (use .npy or .npz)")


def _parse_alphas(s: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(s).split(",") if x.strip())
    if not vals:
        raise SystemExit("--alphas must list at least one value, e.g. 0.01,0.05,0.1")
    for a in vals:
        if not (0.0 < a < 1.0):
            raise SystemExit(f"alpha must be in (0, 1), got {a}")
    return vals


def _print_levels(levels: dict, n_by_group: dict[str, int]) -> None:
    alphas = sorted({a for d in levels.values() for a in d})
    hdr = f"{'group':12}" + "".join(f"  α={a:<8.3g}" for a in alphas) + f"  {'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for g, d in levels.items():
        cells = "".join(
            f"  {d[a]:>10.4f}" if np.isfinite(d[a]) else f"  {'+inf':>10}" for a in alphas
        )
        print(f"{g:12}{cells}  {n_by_group.get(g, 0):6d}")


def cmd_fit(args: argparse.Namespace) -> int:
    from leanmap import PLANEConfig, fit

    X = _load_array(args.input)
    cfg = PLANEConfig.for_scale(len(X))
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.n_components is not None:
        cfg.d_out = int(args.n_components)
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.no_pyramid:
        cfg.pyramid_scales = 0
        cfg.pyramid_level_weights = None
        cfg.pyramid_coarse_backbone = 0.0
    if getattr(args, "exemplar_policy", None) is not None:
        cfg.exemplar_policy = str(args.exemplar_policy)

    result = fit(X, dist_fn=args.metric, config=cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    if args.embedding is not None:
        with torch.no_grad():
            Z, _ = result.embed(X)
        np.save(args.embedding, Z.detach().cpu().numpy())
    print(f"saved {out}")
    return 0


def cmd_transform(args: argparse.Namespace) -> int:
    from leanmap import load_plane

    model = load_plane(args.model, device=args.device)
    X = _load_array(args.input)
    with torch.no_grad():
        Z, S = model.embed(torch.as_tensor(X, dtype=torch.float32))
    Z = Z.detach().cpu().numpy()
    np.save(args.output, Z)
    print(f"wrote {args.output}  shape={tuple(Z.shape)}")
    if args.scores is not None:
        np.save(args.scores, S.detach().cpu().numpy())
        print(f"wrote {args.scores}  (landmark cover scores)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    print(f"model: {args.model}")
    print(
        f"d_out={cfg.get('d_out')} width={cfg.get('width')} depth={cfg.get('depth')}"
    )
    print(
        f"n_landmarks={cfg.get('n_landmarks')} n_neighbors={cfg.get('n_neighbors')}"
    )
    print(
        f"pyramid_scales={cfg.get('pyramid_scales')} "
        f"pyramid_level_weights={cfg.get('pyramid_level_weights')} "
        f"pyramid_coarse_backbone={cfg.get('pyramid_coarse_backbone')}"
    )
    s_calib = payload.get("s_calib")
    if s_calib is not None:
        n = int(getattr(s_calib, "numel", lambda: len(s_calib))())
        print(f"pooled conformal cover calib: n={n}  tau_embed={payload.get('tau_embed')}")
    gs = payload.get("graph_stats") or {}
    if gs:
        print(f"graph_stats keys: {sorted(gs.keys())[:12]}...")
    return 0


def cmd_mondrian(args: argparse.Namespace) -> int:
    """Fit Mondrian levels (digit / gauss / shuffle) and optionally score data."""
    from leanmap import MondrianCalibrator, list_nonconformity_scores, load_plane

    if args.list_scores:
        print("\n".join(list_nonconformity_scores()))
        return 0

    if args.load is not None:
        state = torch.load(args.load, map_location="cpu", weights_only=False)
        cal = MondrianCalibrator.from_state_dict(state)
        print(f"loaded Mondrian calibrator from {args.load}")
        print(f"score={cal.score_name}  groups={list(cal.group_names())}")
    else:
        if args.model is None or args.calib is None:
            raise SystemExit(
                "mondrian fit needs MODEL and CALIB.npy "
                "(or pass --load mondrian.pt, or --list-scores)"
            )
        model = load_plane(args.model, device=args.device)
        X_cal = torch.as_tensor(_load_array(args.calib), dtype=torch.float32)
        cal = MondrianCalibrator(score=args.score)
        cal.fit_from_digits(
            model,
            X_cal,
            n_gauss=args.n_gauss,
            n_shuffle=args.n_shuffle,
            seed=int(args.seed),
        )
        print(f"score={cal.score_name}  model={args.model}")

    alphas = _parse_alphas(args.alphas)
    levels = cal.levels(alphas=alphas)
    n_by = {g: int(s.numel()) for g, s in cal.s_calib.items()}
    print()
    _print_levels(levels, n_by)

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cal.state_dict(), str(out))
        print(f"\nsaved {out}")

    if args.levels_json is not None:
        serializable = {
            g: {str(a): (None if not np.isfinite(t) else float(t)) for a, t in d.items()}
            for g, d in levels.items()
        }
        Path(args.levels_json).write_text(json.dumps(serializable, indent=2) + "\n")
        print(f"wrote {args.levels_json}")

    if args.eval is not None:
        if args.model is None and args.load is None:
            raise SystemExit("--eval needs --model (to score points)")
        # Prefer explicit model; if only --load was used for cal, model is still required.
        if args.model is None:
            raise SystemExit("--eval requires MODEL so points can be scored")
        model = load_plane(args.model, device=args.device)
        X_ev = torch.as_tensor(_load_array(args.eval), dtype=torch.float32)
        s = cal.score_points(model, X_ev)
        p_upper = cal.p_values(s, model=model, sided="upper")
        p_two = cal.p_values(s, model=model, sided="two")
        sets = cal.prediction_set(s, alpha=float(args.alpha), model=model, sided="two")
        thr = {g: cal.threshold(g, float(args.alpha)) for g in cal.group_names()}
        print(f"\neval n={len(s)}  α={args.alpha}  (two-sided prediction sets)")
        from collections import Counter

        top = Counter(sets).most_common(6)
        print("prediction_set counts:", top)
        for g in cal.group_names():
            frac = float((p_two[g] > float(args.alpha)).float().mean())
            print(f"  fraction with two-sided p_{g} > α: {frac:.3f}")

        if args.eval_out is not None:
            payload = {
                "score_name": cal.score_name,
                "scores": s.numpy(),
                "alpha": float(args.alpha),
                "thresholds": np.array([thr[g] for g in cal.group_names()], dtype=np.float64),
                "groups": np.array(list(cal.group_names())),
                "prediction_set": np.array(["|".join(t) for t in sets], dtype=object),
            }
            for g in cal.group_names():
                payload[f"p_upper_{g}"] = p_upper[g].numpy()
                payload[f"p_two_{g}"] = p_two[g].numpy()
            np.savez(args.eval_out, **payload)
            print(f"wrote {args.eval_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leanmap",
        description=(
            "leanmap — parametric landmark-conditioned neighbor embedding "
            "(fit / transform / Mondrian conformal levels)"
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="fit a model and save it")
    fit_p.add_argument("input", help="training features (.npy or .npz with X)")
    fit_p.add_argument("-o", "--output", required=True, help="output path (.pt)")
    fit_p.add_argument("--embedding", default=None, help="optional .npy for train embedding")
    fit_p.add_argument("--epochs", type=int, default=None)
    fit_p.add_argument("--n-components", type=int, default=None, dest="n_components")
    fit_p.add_argument("--metric", default="l2", help="distance (default: l2)")
    fit_p.add_argument("--device", default=None)
    fit_p.add_argument("--seed", type=int, default=None)
    fit_p.add_argument(
        "--no-pyramid",
        action="store_true",
        help="disable cohesive multi-scale pyramid (single-scale graph)",
    )
    fit_p.add_argument(
        "--exemplar-policy",
        choices=("uniform", "sufficient_v1"),
        default=None,
        dest="exemplar_policy",
        help=(
            "within-epoch exemplar measure p_t (default: uniform = prior "
            "edge-mass sampling)"
        ),
    )
    fit_p.set_defaults(func=cmd_fit)

    tr_p = sub.add_parser("transform", help="embed new points with a saved model")
    tr_p.add_argument("model", help="saved .pt model")
    tr_p.add_argument("input", help="features (.npy or .npz with X)")
    tr_p.add_argument("-o", "--output", required=True, help="output embedding .npy")
    tr_p.add_argument(
        "--scores",
        default=None,
        help="optional .npy for landmark-cover scores from embed()",
    )
    tr_p.add_argument("--device", default=None)
    tr_p.set_defaults(func=cmd_transform)

    info_p = sub.add_parser("info", help="print summary of a saved model")
    info_p.add_argument("model", help="saved .pt model")
    info_p.set_defaults(func=cmd_info)

    md_p = sub.add_parser(
        "mondrian",
        help=(
            "Mondrian conformal levels for digit / gauss / shuffle "
            "(default score: affinity_entropy)"
        ),
        description=(
            "Fit category-conditional conformal thresholds on three groups "
            "(real points as 'digit', μ/σ-matched Gaussian noise, pixel-shuffled "
            "copies). Default nonconformity is affinity entropy; choose another "
            "with --score. Prints levels and optionally scores an eval set."
        ),
    )
    md_p.add_argument(
        "model",
        nargs="?",
        default=None,
        help="saved leanmap .pt (required to fit or --eval)",
    )
    md_p.add_argument(
        "calib",
        nargs="?",
        default=None,
        help="calibration features (.npy/.npz) treated as the digit group",
    )
    md_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="save MondrianCalibrator state_dict (.pt)",
    )
    md_p.add_argument(
        "--load",
        default=None,
        help="load a previously saved Mondrian state_dict instead of fitting",
    )
    md_p.add_argument(
        "--score",
        default="affinity_entropy",
        help=(
            "nonconformity score name (default: affinity_entropy). "
            "Use --list-scores for options."
        ),
    )
    md_p.add_argument(
        "--list-scores",
        action="store_true",
        help="print registered nonconformity score names and exit",
    )
    md_p.add_argument(
        "--alphas",
        default="0.01,0.05,0.1",
        help="comma-separated miscoverage levels for thresholds (default: 0.01,0.05,0.1)",
    )
    md_p.add_argument(
        "--n-gauss",
        type=int,
        default=None,
        help="Gaussian noise pool size (default: len(calib))",
    )
    md_p.add_argument(
        "--n-shuffle",
        type=int,
        default=None,
        help="pixel-shuffle pool size (default: len(calib))",
    )
    md_p.add_argument("--seed", type=int, default=0, help="RNG seed for noise pools")
    md_p.add_argument("--device", default=None)
    md_p.add_argument(
        "--levels-json",
        default=None,
        dest="levels_json",
        help="write levels table as JSON",
    )
    md_p.add_argument(
        "--eval",
        default=None,
        help="optional features to score against the Mondrian levels (.npy/.npz)",
    )
    md_p.add_argument(
        "--eval-out",
        default=None,
        dest="eval_out",
        help="savez path for eval scores / p-values / prediction sets",
    )
    md_p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="α for prediction sets / eval summary (default: 0.05)",
    )
    md_p.set_defaults(func=cmd_mondrian)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def main_graph_build(argv: list[str] | None = None) -> int:
    """``leanmap-graph-build`` — build/freeze a graph store from ``X``."""
    ap = argparse.ArgumentParser(prog="leanmap-graph-build")
    ap.add_argument("--X", required=True, help="features .npy / .npz")
    ap.add_argument("--out", required=True, help="output graph.pt or graph_store/ directory")
    ap.add_argument("--knn-mode", default="auto")
    ap.add_argument("--pyramid-scales", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--delta", default=None, help="None|eps|auto|<float>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stages",
        default=None,
        help="shared stages / bunch work directory (required for --bunch-partition fs)",
    )
    ap.add_argument(
        "--bunch-partition",
        choices=("local", "fs", "ddp", "mpi"),
        default="local",
        help=(
            "local single-process (default); fs=shared --stages FileStore; "
            "ddp=torch.distributed; mpi=mpi4py (leanmap[hpc])"
        ),
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=None,
        help="worker rank (default: env RANK or 0); used with --bunch-partition fs",
    )
    ap.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="worker world size (default: env WORLD_SIZE or 1); used with fs",
    )
    args = ap.parse_args(argv)
    from leanmap import PLANEConfig
    from leanmap.graph import (
        build_graph_pyramid,
        save_graph_pyramid,
        tensor_fingerprint,
    )
    from leanmap.metrics import wrap_metric
    from leanmap.store import open_graph_store, select_backend
    from leanmap.utils import seed_everything

    X = _load_array(args.X)
    seed_everything(args.seed)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.seed = args.seed
    cfg.knn_mode = args.knn_mode
    cfg.pyramid_scales = args.pyramid_scales
    cfg.epsilon = args.epsilon
    if args.delta is not None:
        try:
            cfg.delta = float(args.delta)
        except ValueError:
            cfg.delta = args.delta
    if args.stages:
        cfg.graph_stages_dir = args.stages
    Xt = torch.as_tensor(X, dtype=torch.float32)
    metric = wrap_metric("l2", X=Xt, n_neighbors=cfg.n_neighbors, seed=args.seed)
    build_kw = dict(
        pyramid_scales=cfg.pyramid_scales,
        n_neighbors=cfg.n_neighbors,
        n_landmarks=cfg.n_landmarks,
        seed=args.seed,
        knn_mode=cfg.knn_mode,
        epsilon=cfg.epsilon,
        delta=cfg.delta,
        stages_dir=cfg.graph_stages_dir,
    )
    if args.bunch_partition == "local":
        graphs, M, a1, ac = build_graph_pyramid(Xt, metric, **build_kw)
    else:
        from leanmap.build.bunches import build_graph_pyramid_bunches
        from leanmap.build.transport import make_transport

        transport = make_transport(
            args.bunch_partition,
            stages_dir=args.stages or cfg.graph_stages_dir,
            rank=args.rank,
            world_size=args.world_size,
        )
        result = build_graph_pyramid_bunches(
            Xt,
            metric,
            transport=transport,
            transport_kind=args.bunch_partition,
            stages_dir=args.stages or cfg.graph_stages_dir,
            **build_kw,
        )
        if result is None:
            # Non-root worker: freeze is root-only.
            return 0
        graphs, M, a1, ac = result
    out = Path(args.out)
    n = int(Xt.shape[0])
    train_idx = torch.arange(n, dtype=torch.long)
    calib_idx = torch.zeros(0, dtype=torch.long)
    fp = tensor_fingerprint(Xt)
    eps_val = float(graphs[0].stats.epsilon) if graphs[0].stats.epsilon else 0.0
    backend = select_backend(out, n_reps=int(graphs[0].reps.rep_idx.shape[0]))
    if str(backend) == "dirstore" or out.suffix != ".pt":
        store = open_graph_store(out)
        state = {
            "version": 1,
            "graphs": graphs,
            "M": M,
            "assign_top1": a1,
            "assign_topc": ac,
            "fingerprint": fp,
            "seed": args.seed,
            "epsilon": eps_val,
            "delta": getattr(cfg, "delta", None),
            "n_neighbors": cfg.n_neighbors,
            "n_landmarks": int(M.shape[0]),
            "metric_name": "l2",
            "train_idx": train_idx,
            "calib_idx": calib_idx,
            "n_all": n,
            "dedup": cfg.dedup,
        }
        if hasattr(store, "save_from_state"):
            store.save_from_state(state, X=Xt)  # type: ignore[call-arg]
        elif hasattr(store, "save"):
            store.save(  # type: ignore[call-arg]
                graphs=graphs,
                M=M,
                assign_top1=a1,
                assign_topc=ac,
                train_idx=train_idx,
                calib_idx=calib_idx,
                fingerprint=fp,
                metric_name="l2",
                n_all=n,
                n_neighbors=cfg.n_neighbors,
                epsilon=eps_val,
                seed=args.seed,
                dedup=cfg.dedup,
            )
    else:
        save_graph_pyramid(
            out,
            graphs=graphs,
            M=M,
            assign_top1=a1,
            assign_topc=ac,
            train_idx=train_idx,
            calib_idx=calib_idx,
            fingerprint=fp,
            metric_name="l2",
            n_all=n,
            n_neighbors=cfg.n_neighbors,
            epsilon=eps_val,
            seed=args.seed,
            dedup=cfg.dedup,
        )
    print(f"wrote {out}")
    return 0


def main_train(argv: list[str] | None = None) -> int:
    """``leanmap-train`` — train against a frozen graph store."""
    ap = argparse.ArgumentParser(prog="leanmap-train")
    ap.add_argument("--X", required=True)
    ap.add_argument("--graph-path", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-path", type=float, default=None)
    ap.add_argument(
        "--epoch-unit",
        choices=("edges", "landmarks"),
        default=None,
        help="edges=pass over graph edges; landmarks=basin cover (δ-independent)",
    )
    ap.add_argument(
        "--landmark-epoch-samples",
        type=float,
        default=None,
        help="edge draws per landmark per epoch when --epoch-unit landmarks",
    )
    ap.add_argument(
        "--landmark-sample-mix",
        type=float,
        default=None,
        help="0..1 blend toward equal landmark-basin edge sampling",
    )
    ap.add_argument(
        "--exemplar-policy",
        choices=("uniform", "sufficient_v1"),
        default="uniform",
    )
    ap.add_argument("--out", default="plane.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    from leanmap import PLANEConfig, fit

    X = _load_array(args.X)
    cfg = PLANEConfig.for_scale(len(X))
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.exemplar_policy = args.exemplar_policy
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lambda_path is not None:
        cfg.lambda_path = args.lambda_path
    if args.epoch_unit is not None:
        cfg.epoch_unit = args.epoch_unit
    if args.landmark_epoch_samples is not None:
        cfg.landmark_epoch_samples = float(args.landmark_epoch_samples)
    if args.landmark_sample_mix is not None:
        cfg.landmark_sample_mix = float(args.landmark_sample_mix)
    result = fit(X, config=cfg, graph_path=args.graph_path)
    result.save(args.out)
    print(f"wrote {args.out}")
    return 0


def main_eps_crawl(argv: list[str] | None = None) -> int:
    """``leanmap-eps-crawl`` — browse ε → R on a random subsample."""
    ap = argparse.ArgumentParser(prog="leanmap-eps-crawl")
    ap.add_argument("--X", required=True, help="features .npy / .npz")
    ap.add_argument(
        "--n-sample",
        type=int,
        default=10_000,
        help="random subsample size for the crawl (default: 10000)",
    )
    ap.add_argument(
        "--eps",
        default=None,
        help="comma-separated ε grid (default: auto from sample 1-NN quantiles)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--json-out",
        default=None,
        help="optional path to write the full crawl report as JSON",
    )
    args = ap.parse_args(argv)
    import torch

    from leanmap.build.resolution import crawl_epsilon, format_epsilon_crawl
    from leanmap.metrics import wrap_metric

    X = _load_array(args.X)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    metric = wrap_metric("l2", X=Xt, n_neighbors=15, seed=args.seed)
    epsilons = None
    if args.eps:
        epsilons = [float(x.strip()) for x in str(args.eps).split(",") if x.strip()]
    report = crawl_epsilon(
        Xt,
        metric,
        n_sample=args.n_sample,
        epsilons=epsilons,
        seed=args.seed,
        n_rows=int(Xt.shape[0]),
    )
    print(format_epsilon_crawl(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

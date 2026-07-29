"""Command-line interface: ``leanmap fit | transform | info | mondrian``."""

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


if __name__ == "__main__":
    sys.exit(main())

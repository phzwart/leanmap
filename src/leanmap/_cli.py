"""Command-line interface: ``leanmap fit | transform | info`` (PLANE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32, copy=False)
    if p.suffix == ".npz":
        return np.load(p)["X"].astype(np.float32, copy=False)
    raise SystemExit(f"unsupported input format: {p.suffix} (use .npy or .npz)")


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

    model = load_plane(args.model)
    X = _load_array(args.input)
    with torch.no_grad():
        Z, _ = model.embed(torch.as_tensor(X, dtype=torch.float32))
    Z = Z.detach().cpu().numpy()
    np.save(args.output, Z)
    print(f"wrote {args.output}  shape={tuple(Z.shape)}")
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
    gs = payload.get("graph_stats") or {}
    if gs:
        print(f"graph_stats keys: {sorted(gs.keys())[:12]}...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leanmap",
        description="PLANE: parametric landmark-conditioned neighbor embedding",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="fit a PLANE model and save it")
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
    tr_p.set_defaults(func=cmd_transform)

    info_p = sub.add_parser("info", help="print summary of a saved model")
    info_p.add_argument("model", help="saved .pt model")
    info_p.set_defaults(func=cmd_info)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

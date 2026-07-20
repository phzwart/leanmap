"""Command-line interface: ``leanmap fit | transform | info``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from ._api import LeanMap


# --------------------------------------------------------------------------- IO
def _load_matrix(path: str, *, delimiter: str = ",", skip_header: int = 0) -> np.ndarray:
    """Load a 2D float matrix from .npy, .npz (key 'X' or first array), or text."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".npz":
        npz = np.load(p)
        key = "X" if "X" in npz else npz.files[0]
        arr = npz[key]
    else:  # csv / tsv / txt
        arr = np.loadtxt(p, delimiter=delimiter, skiprows=skip_header)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _save_embedding(path: str, emb: np.ndarray, *, delimiter: str = ",") -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        np.save(p, emb)
    else:
        np.savetxt(p, emb, delimiter=delimiter, header="x,y", comments="")


# ------------------------------------------------------------------- subcommands
def _cmd_fit(args: argparse.Namespace) -> int:
    X = _load_matrix(args.input, delimiter=args.delimiter, skip_header=args.skip_header)
    print(f"[fit] loaded {X.shape[0]} x {X.shape[1]} from {args.input}", file=sys.stderr)

    mapper = LeanMap(
        device=args.device,
        verbose=not args.quiet,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        index_kind=args.index_kind,
        local_connectivity=args.local_connectivity,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_sample_rate=args.negative_sample_rate,
        repulsion_strength=args.repulsion_strength,
        learning_rate=args.learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
    )
    emb = mapper.fit_transform(X)
    mapper.save(args.model_out)
    print(f"[fit] saved model -> {args.model_out}", file=sys.stderr)

    if args.embedding_out:
        _save_embedding(args.embedding_out, emb, delimiter=args.delimiter)
        print(f"[fit] saved embedding -> {args.embedding_out}", file=sys.stderr)
    return 0


def _cmd_transform(args: argparse.Namespace) -> int:
    mapper = LeanMap.load(args.model, device=args.device)
    X = _load_matrix(args.input, delimiter=args.delimiter, skip_header=args.skip_header)
    if mapper.n_features_in_ is not None and X.shape[1] != mapper.n_features_in_:
        raise ValueError(
            f"Model expects {mapper.n_features_in_} features, got {X.shape[1]}"
        )
    print(f"[transform] embedding {X.shape[0]} rows", file=sys.stderr)
    emb = mapper.transform(X, batch_size=args.batch_size)
    _save_embedding(args.output, emb, delimiter=args.delimiter)
    print(f"[transform] saved embedding -> {args.output}", file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    mapper = LeanMap.load(args.model, device="cpu")
    enc = mapper.torch_module.encoder
    n_params = sum(p.numel() for p in mapper.torch_module.parameters())
    print(f"leanmap model: {args.model}")
    print(f"  input features : {mapper.n_features_in_}")
    print(f"  encoder input  : {enc.input_dim}")
    print(f"  hidden dims    : {enc.hidden_dims}")
    print(f"  parameters     : {n_params}")
    print("  config:")
    for k, v in vars(mapper.config).items():
        print(f"    {k:22s}= {v}")
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leanmap",
        description="Small, deployable parametric UMAP (FAISS k-NN + PCA-anchored net).",
    )
    parser.add_argument("--version", action="version", version=f"leanmap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # shared data-IO args
    def add_io(p: argparse.ArgumentParser) -> None:
        p.add_argument("--delimiter", default=",", help="text-file delimiter (default ',')")
        p.add_argument("--skip-header", type=int, default=0, help="header rows to skip")
        p.add_argument("--device", default=None, help="torch device (cpu, cuda, ...)")
        p.add_argument("--batch-size", type=int, default=4096)

    # fit
    pf = sub.add_parser("fit", help="fit a mapper on a data matrix and save it")
    pf.add_argument("input", help="training matrix (.npy/.npz/.csv/.tsv)")
    pf.add_argument("-o", "--model-out", required=True, help="output model path (.mmap)")
    pf.add_argument("-e", "--embedding-out", default=None, help="optional 2D embedding out")
    pf.add_argument("--n-neighbors", type=int, default=50)
    pf.add_argument("--min-dist", type=float, default=0.1)
    pf.add_argument("--spread", type=float, default=1.0)
    pf.add_argument("--index-kind", choices=["flat", "hnsw"], default="hnsw")
    pf.add_argument("--local-connectivity", type=float, default=1.0)
    pf.add_argument("--epochs", type=int, default=25)
    pf.add_argument("--negative-sample-rate", type=int, default=5)
    pf.add_argument("--repulsion-strength", type=float, default=1.0)
    pf.add_argument("--learning-rate", type=float, default=1e-3)
    pf.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--quiet", action="store_true", help="suppress per-epoch logs")
    add_io(pf)
    pf.set_defaults(func=_cmd_fit)

    # transform
    pt = sub.add_parser("transform", help="embed new data with a saved model")
    pt.add_argument("input", help="matrix to embed (.npy/.npz/.csv/.tsv)")
    pt.add_argument("-m", "--model", required=True, help="saved model path (.mmap)")
    pt.add_argument("-o", "--output", required=True, help="embedding output (.npy/.csv)")
    add_io(pt)
    pt.set_defaults(func=_cmd_transform)

    # info
    pi = sub.add_parser("info", help="print a saved model's config and shape")
    pi.add_argument("model", help="saved model path (.mmap)")
    pi.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

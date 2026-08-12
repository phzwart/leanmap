#!/usr/bin/env python
"""Frozen Cell-DINO (Cell Painting ViT-S/8) features on CellCycle → leanmap.

Loads Ch3/Ch4/Ch6 as (3, H, W), zero-pads to 5 channels (Cell Painting input),
resizes to 128×128, extracts 384-d CLS embeddings, writes zarr, optionally
fits leanmap with per-epoch frames.

    python examples/cellcycle_celldino.py
    python examples/cellcycle_celldino.py --fit-only

Requires: torch, zarr, Pillow, scikit-learn; dinov2 checkout + Cell-DINO ckpt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from _demo import OUT_DIR, default_config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cellcycle_lejepa import (  # noqa: E402
    CHANNELS,
    PHASES,
    list_cells,
    load_stack,
)

DEFAULT_SOURCE = Path.home() / "Projects" / "cells" / "CellCycle"
DEFAULT_CKPT = Path.home() / "Projects" / "cells" / "dino" / "cell_dino_vits8_pretrain_cp-37d20e9c.pth"
DEFAULT_REPO = ROOT / "models" / "dinov2"
DEFAULT_OUT = ROOT / "examples" / "out" / "cellcycle_celldino.zarr"
DEFAULT_RUN = OUT_DIR / "cellcycle_celldino"


def pad_to_5(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) → (B, 5, H, W) with two trailing zero channels."""
    if x.shape[1] == 5:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"expected 3 channels, got {x.shape[1]}")
    z = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
    return torch.cat([x, z], dim=1)


class StackDataset(Dataset):
    def __init__(self, images: np.ndarray) -> None:
        self.images = images  # (N, 3, H, W) float32 [0,1]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.images[i]), i


def load_cell_dino(repo: Path, ckpt: Path, device: str):
    model = torch.hub.load(
        str(repo),
        "cell_dino_cp_vits8",
        source="local",
        pretrained_path=str(ckpt),
    )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(
    model,
    images: np.ndarray,
    *,
    device: str,
    batch_size: int,
    img_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(
        StackDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )
    outs = []
    for x, _idx in loader:
        x = x.to(device, non_blocking=True)
        x = pad_to_5(x)
        if x.shape[-1] != img_size or x.shape[-2] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        outs.append(model(x).float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def featurize(args) -> Path:
    import zarr

    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    ckpt = args.ckpt.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not repo.is_dir():
        raise FileNotFoundError(repo)

    triples, labels, cell_ids = list_cells(
        source, max_per_phase=int(args.max_per_phase), seed=int(args.seed)
    )
    print(f"loading {len(triples)} stacks…")
    images = np.stack([load_stack(t) for t in triples], axis=0)
    print(f"images {images.shape} device={device}")

    model = load_cell_dino(repo, ckpt, device)
    print(
        f"Cell-DINO ViT-S/8  patch={tuple(model.patch_embed.proj.weight.shape)}  "
        f"emb={model.embed_dim}  pad 3→5 zeros"
    )

    feats = extract_features(
        model,
        images,
        device=device,
        batch_size=int(args.batch_size),
        img_size=int(args.img_size),
    )
    print(f"features {feats.shape} mean={feats.mean():.4f} std={feats.std():.4f}")

    if out.exists():
        shutil.rmtree(out)
    root = zarr.open_group(str(out), mode="w")
    root.create_array("features", data=feats)
    root.create_array("images", data=images.astype(np.float32))
    root.create_array("labels", data=labels)
    root.create_array("cell_ids", data=cell_ids)
    attrs = {
        "phases": list(PHASES),
        "channels": list(CHANNELS),
        "source": str(source),
        "seed": int(args.seed),
        "max_per_phase": int(args.max_per_phase),
        "method": "Cell-DINO cell_dino_cp_vits8 (frozen), Ch3/4/6 + 2 zero channels",
        "ckpt": str(ckpt),
        "repo": str(repo),
        "img_size": int(args.img_size),
        "in_channels_model": 5,
        "channel_pad": "zeros trailing",
        "emb_dim": int(feats.shape[1]),
        "counts": {
            p: int((labels == PHASES.index(p)).sum())
            for p in PHASES
            if (labels == PHASES.index(p)).any()
        },
    }
    for k, v in attrs.items():
        root.attrs[k] = v
    meta = out.with_suffix(out.suffix + ".meta.json")
    meta.write_text(json.dumps(attrs, indent=2) + "\n")
    print(f"wrote {out}")
    print(root.tree())
    return out


def fit_leanmap(args, zarr_path: Path) -> None:
    import zarr
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    from leanmap import fit

    try:
        from cellcycle_emd import EmbeddingRecorder, save_phase_scatter
        from cellcycle_emd import PHASES as P
    except ImportError:
        from cellcycle_emd import EmbeddingRecorder, save_phase_scatter  # type: ignore

        P = PHASES

    root = zarr.open_group(str(zarr_path), mode="r")
    X = np.asarray(root["features"], dtype=np.float32)
    labels = np.asarray(root["labels"], dtype=np.int64)
    phases = tuple(root.attrs.get("phases", P))
    feat_dim0 = int(X.shape[1])

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("epoch_*"):
        p.unlink(missing_ok=True)
    progress = run_dir / "progress.csv"
    if progress.exists():
        progress.unlink()

    pca_dim = int(args.pca_dim)
    if pca_dim > 0 and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=int(args.seed))
        X = pca.fit_transform(X).astype(np.float32)
        var = float(pca.explained_variance_ratio_.sum())
        print(f"PCA {feat_dim0} → {pca_dim}  explained_var={var:.3f}")
        np.save(
            run_dir / "pca_explained_variance.npy",
            pca.explained_variance_ratio_.astype(np.float64),
        )
        np.save(run_dir / "features_pca.npy", X)
    # unit-normalize so cosine ≡ angular geometry (also stabilizes L2 fallback)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)

    rng = np.random.default_rng(int(args.seed))
    n = len(X)
    n_cal = max(1, int(round(float(args.holdout) * n)))
    perm = rng.permutation(n)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    X_train, X_cal = X[train_idx], X[cal_idx]
    y_train = labels[train_idx]

    k = min(int(args.k), len(X_train) - 1)
    metric = str(args.metric)
    print(f"{metric} knn k={k} on features {X_train.shape}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X_train)
    dist, idx = nn.kneighbors(X_train)
    knn_idx = idx[:, 1:].astype(np.int64)
    knn_dist = dist[:, 1:].astype(np.float32)

    cfg = default_config(len(X_train), epochs=int(args.leanmap_epochs))
    cfg.seed = int(args.seed)
    cfg.dedup = False
    cfg.n_neighbors = k
    cfg.min_dist = float(args.min_dist)
    if args.device:
        cfg.device = args.device

    recorder = EmbeddingRecorder(
        X_train,
        y_train,
        out_dir=run_dir,
        every=int(args.frame_every) if int(args.frame_every) > 0 else 10**9,
        total_epochs=int(args.leanmap_epochs),
        phase_names=phases,
    )
    callbacks = [recorder] if int(args.frame_every) > 0 else None

    # leanmap registry: "cosine" / "l2" / …
    dist_fn = "cosine" if metric == "cosine" else metric
    result = fit(
        X_train,
        dist_fn=dist_fn,
        config=cfg,
        X_calib=X_cal,
        precomputed_knn=(
            torch.as_tensor(knn_idx, dtype=torch.int64),
            torch.as_tensor(knn_dist, dtype=torch.float32),
        ),
        callbacks=callbacks,
    )
    with torch.no_grad():
        Z, _ = result.embed(X_train)
    Z = Z.detach().cpu().numpy()

    result.save(str(run_dir / "model.pt"))
    save_phase_scatter(
        Z,
        y_train,
        title=(
            f"leanmap — Cell-DINO PCA{pca_dim} {metric} (min_dist={args.min_dist})"
            if pca_dim > 0 and pca_dim < feat_dim0
            else f"leanmap — Cell-DINO {metric} (min_dist={args.min_dist})"
        ),
        path=run_dir / "final.png",
        phase_names=phases,
    )
    np.save(run_dir / "Z_final.npy", Z.astype(np.float32))
    np.save(run_dir / "y_train.npy", y_train)
    np.save(run_dir / "train_idx.npy", train_idx.astype(np.int64))
    if recorder is not None and int(args.frame_every) > 0:
        recorder.finalize()
    print(f"saved leanmap run → {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--max-per-phase", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--skip-leanmap", action="store_true")
    ap.add_argument("--leanmap-epochs", type=int, default=80)
    ap.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--frame-every", type=int, default=1)
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=32,
        help="PCA dims before leanmap (0 = use raw 384-d features)",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=("cosine", "euclidean"),
        help="ambient knn / leanmap metric on features",
    )
    args = ap.parse_args()

    zarr_path = args.out.expanduser().resolve()
    if not args.fit_only:
        featurize(args)
    elif not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    if not args.skip_leanmap:
        fit_leanmap(args, zarr_path)


if __name__ == "__main__":
    main()

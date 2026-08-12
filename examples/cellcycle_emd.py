#!/usr/bin/env python
"""Sample CellCycle merged cells → zarr + kNN + leanmap fit.

Reads ``*_merged.jpg`` under a CellCycle root (default
``~/Projects/cells/CellCycle``), takes a random subset of at most ``--max-per-phase``
images per phase, writes a zarr store, builds a kNN graph, and fits leanmap
with ``precomputed_knn``.

``--knn l1`` (default): unit-mass flatten → L1 kNN on full-res luminance.
``--knn emd``: L1 candidate filter → torchemd EMD rescore (slow at 66×66).

Zarr layout (under ``--out``):

* ``images``       (N, H, W, 3) uint8   — raw RGB merged crops
* ``images_proc``  (N, H, W, 3) uint8   — CLAHE + Otsu-masked RGB (features/display)
* ``mask``         (N, H, W) uint8      — Otsu foreground (0/255)
* ``rgb``          (N, H, W, 3) float32 — processed bands /255 → [0, 1]
* ``labels``       (N,) int64           — phase index
* ``cell_ids``     (N,) int64
* ``knn_idx``      (N, k) int64         — full-set graph
* ``knn_dist``     (N, k) float32
* attrs: ``phases``, ``source``, ``seed``, ``k``, ``knn``, ``clahe``, ``otsu``, …

Leanmap run dir (``--run-dir``): model, final scatter, per-epoch frames + live.png.

Requires: ``scikit-learn``, ``zarr``, ``Pillow``; ``torchemd`` only for ``--knn emd``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from _demo import OUT_DIR, default_config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Interphase → mitosis order
PHASES = ("G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase")

DEFAULT_SOURCE = Path.home() / "Projects" / "cells" / "CellCycle"
DEFAULT_OUT = ROOT / "examples" / "out" / "cellcycle_l1.zarr"
DEFAULT_RUN = OUT_DIR / "cellcycle_l1"

# One discrete color per phase (not tab10's 10-class palette).
_PHASE_COLORS = (
    "#4C78A8",  # G1
    "#F58518",  # S
    "#54A24B",  # G2
    "#E45756",  # Prophase
    "#72B7B2",  # Metaphase
    "#B279A2",  # Anaphase
    "#FF9DA6",  # Telophase
)


def save_phase_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    path: Path,
    phase_names: tuple[str, ...] | list[str] = PHASES,
) -> Path:
    """Scatter colored by phase index with exactly len(phase_names) colors.

    Uses a 3-D axes when ``Z`` has ≥3 columns (plots the first three).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(phase_names)
    n_phases = len(names)
    colors = list(_PHASE_COLORS[:n_phases])
    if len(colors) < n_phases:
        # fallback if PHASES ever grows past the palette
        import matplotlib as mpl

        colors = list(mpl.colormaps["tab10"](np.linspace(0, 1, n_phases)))
    cmap = ListedColormap(colors)
    bounds = np.arange(n_phases + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    Z = np.asarray(Z)
    y = np.asarray(y)

    if Z.shape[1] >= 3:
        fig = plt.figure(figsize=(6.0, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            Z[:, 2],
            c=y,
            s=6,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            depthshade=True,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y,
            s=6,
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(n_phases))
    cb.ax.set_yticklabels(names)
    cb.set_label("phase")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _rgb01(images_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB (N,H,W,3) → float32 in [0, 1] per band."""
    return (np.asarray(images_u8, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def _clahe_rgb(
    images_u8: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Per-channel CLAHE on a stack of uint8 RGB images.

    Applied independently to R, G, B so dim marker channels are not swamped
    by bright background / DNA signal before L1.
    """
    import cv2

    imgs = np.asarray(images_u8)
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {imgs.shape}")
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    out = np.empty_like(imgs)
    for i in range(len(imgs)):
        for c in range(3):
            out[i, :, :, c] = clahe.apply(imgs[i, :, :, c])
    return out


def _otsu_mask_rgb(
    images_u8: np.ndarray,
    *,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Otsu foreground mask from luminance; zero background in RGB.

    Returns ``(masked_rgb uint8, mask uint8 0/255)``. Mask is computed on
    grayscale of each image (OpenCV Otsu). Default: bright = foreground
    (dark fluorescence background). Background pixels are set to 0 so L1
    ignores the exterior.
    """
    import cv2

    imgs = np.asarray(images_u8)
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {imgs.shape}")
    masked = np.zeros_like(imgs)
    masks = np.zeros(imgs.shape[:3], dtype=np.uint8)
    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    for i in range(len(imgs)):
        gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        _, m = cv2.threshold(gray, 0, 255, thr_type | cv2.THRESH_OTSU)
        masks[i] = m
        masked[i] = np.where(m[:, :, None] > 0, imgs[i], 0)
    return masked, masks


def _preprocess_rgb(
    images_u8: np.ndarray,
    *,
    clahe_clip: float,
    clahe_tile: int,
    otsu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE (+ optional Otsu mask). Returns ``(proc_u8, mask_u8)``."""
    proc = _clahe_rgb(
        images_u8,
        clip_limit=clahe_clip,
        tile_grid_size=(clahe_tile, clahe_tile),
    )
    if not otsu:
        mask = np.full(proc.shape[:3], 255, dtype=np.uint8)
        return proc, mask
    return _otsu_mask_rgb(proc)


def _flat(imgs: np.ndarray) -> np.ndarray:
    """(N, ...) → (N, D) float32."""
    return np.asarray(imgs, dtype=np.float32).reshape(len(imgs), -1)


def _unit_mass_flat(imgs: np.ndarray) -> np.ndarray:
    flat = _flat(imgs).astype(np.float64)
    total = flat.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (flat / total).astype(np.float32)


def _luminance01(rgb01: np.ndarray) -> np.ndarray:
    """RGB float [0,1] (N,H,W,3) → luminance (N,H,W)."""
    return (
        0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]
    ).astype(np.float32)


def sample_merged(
    source: Path,
    *,
    max_per_phase: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return images (N,H,W,3), labels, cell_ids, phase names present."""
    rng = np.random.default_rng(seed)
    imgs: list[np.ndarray] = []
    labels: list[int] = []
    cell_ids: list[int] = []
    present: list[str] = []

    for li, phase in enumerate(PHASES):
        d = source / phase
        if not d.is_dir():
            print(f"skip missing phase dir: {d}")
            continue
        paths = sorted(d.glob("*_merged.jpg"))
        if not paths:
            print(f"skip empty phase: {phase}")
            continue
        present.append(phase)
        order = rng.permutation(len(paths))[:max_per_phase]
        for j in order:
            p = paths[int(j)]
            cid = int(p.name.split("_", 1)[0])
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            imgs.append(arr)
            labels.append(li)
            cell_ids.append(cid)
        print(f"{phase}: kept {len(order)} / {len(paths)}")

    if not imgs:
        raise FileNotFoundError(f"no merged images under {source}")
    return (
        np.stack(imgs, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(cell_ids, dtype=np.int64),
        present,
    )


def build_l1_knn(
    imgs: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten full RGB [0,1] → exact L1 (minkowski p=1) kNN, drop self."""
    n = len(imgs)
    if k >= n:
        k = n - 1
    flat = _flat(imgs)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="minkowski", p=1).fit(flat)
    dist, idx = nn.kneighbors(flat)
    return idx[:, 1 : k + 1].astype(np.int64), dist[:, 1 : k + 1].astype(np.float32)


def build_l1_emd_knn(
    imgs: np.ndarray,
    *,
    k: int,
    M: int,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """L1 candidate tree → NNS-EMD rescore → prune to top-k."""
    from torchemd import image_to_histogram
    from torchemd.graph import rescore_neighbors

    if M < k:
        raise ValueError(f"candidate width M={M} must be >= k={k}")
    n = len(imgs)
    if M >= n:
        M = n - 1
    if k > M:
        k = M
    flat = _unit_mass_flat(imgs)
    nn = NearestNeighbors(n_neighbors=M + 1, metric="minkowski", p=1).fit(flat)
    _, cand = nn.kneighbors(flat)
    cand = cand[:, 1 : M + 1]

    hs = [image_to_histogram(im) for im in imgs]
    W = [h[0] for h in hs]
    coords = [h[1] for h in hs]
    idx, dist = rescore_neighbors(
        cand,
        W,
        coords,
        metric="euclidean",
        batch_size=batch_size,
        device=device,
    )
    return idx[:, :k].astype(np.int64), dist[:, :k].astype(np.float32)


def build_knn(imgs: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, int, int | None]:
    """Dispatch ``--knn``; returns idx, dist, k, M (M is None for pure L1)."""
    n = len(imgs)
    k = int(args.k)
    k = min(k, n - 1)
    mode = str(args.knn).lower()
    if mode == "l1":
        print(f"building L1 kNN k={k} on full RGB {tuple(imgs.shape)} (bands in [0,1])")
        idx, dist = build_l1_knn(imgs, k=k)
        return idx, dist, k, None
    if mode == "emd":
        # torchemd histograms are 2-D; use luminance of the [0,1] RGB stack
        gray = imgs if imgs.ndim == 3 else _luminance01(imgs)
        M = int(args.M) if args.M is not None else max(30, 3 * k)
        M = min(M, n - 1)
        k = min(k, M)
        print(
            f"building L1 tree (M={M}) → EMD rescore on luminance {gray.shape} → "
            f"prune k={k} (device={args.emd_device}, batch={args.batch_size})"
        )
        idx, dist = build_l1_emd_knn(
            gray,
            k=k,
            M=M,
            batch_size=int(args.batch_size),
            device=args.emd_device,
        )
        return idx, dist, k, M
    raise ValueError(f"unknown --knn {args.knn!r} (expected l1|emd)")


def write_zarr(
    path: Path,
    *,
    images: np.ndarray,
    images_proc: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    cell_ids: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    attrs: dict,
) -> Path:
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    root = zarr.open_group(str(path), mode="w")
    root.create_array("images", data=images)
    root.create_array("images_proc", data=images_proc)
    root.create_array("mask", data=mask)
    root.create_array("rgb", data=rgb)
    root.create_array("labels", data=labels)
    root.create_array("cell_ids", data=cell_ids)
    root.create_array("knn_idx", data=knn_idx)
    root.create_array("knn_dist", data=knn_dist)
    for key, val in attrs.items():
        root.attrs[key] = val
    return path


def load_zarr(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    root = zarr.open_group(str(path), mode="r")
    if "images_proc" in root:
        images_proc = np.asarray(root["images_proc"])
    elif "images_clahe" in root:
        images_proc = np.asarray(root["images_clahe"])
    else:
        images_proc = np.asarray(root["images"])
    arrays = {
        "images": np.asarray(root["images"]),
        "images_proc": images_proc,
        "mask": np.asarray(root["mask"])
        if "mask" in root
        else np.full(images_proc.shape[:3], 255, dtype=np.uint8),
        "rgb": np.asarray(root["rgb"]),
        "labels": np.asarray(root["labels"]),
        "cell_ids": np.asarray(root["cell_ids"]),
        "knn_idx": np.asarray(root["knn_idx"]),
        "knn_dist": np.asarray(root["knn_dist"]),
    }
    attrs = dict(root.attrs)
    return arrays, attrs


class EmbeddingRecorder:
    """Write per-epoch embeddings (``.npy``) + scatter frames during fit."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        out_dir: Path,
        every: int,
        total_epochs: int,
        phase_names: tuple[str, ...] | list[str],
    ) -> None:
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = np.asarray(y)
        self.every = max(1, int(every))
        self.total = int(total_epochs)
        self.out_dir = Path(out_dir)
        self.frame_dir = self.out_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.width = max(4, len(str(self.total)))
        self.epochs: list[int] = []
        self.Zs: list[np.ndarray] = []
        self.phase_names = list(phase_names)
        self.csv_path = self.out_dir / "progress.csv"
        self._csv_cols: list[str] | None = None

    def _log(self, epoch: int, metrics) -> None:
        row = {"epoch": epoch}
        row.update(
            {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
        )
        new = self._csv_cols is None
        if new:
            self._csv_cols = list(row)
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self._csv_cols, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def __call__(self, epoch: int, model, metrics) -> None:
        self._log(epoch, metrics)
        if epoch % self.every and epoch != self.total:
            return
        was_training = model.training
        with torch.no_grad():
            Z, _ = model.embed(self.X, return_score=False)
        if was_training:
            model.train()
        Z = Z.detach().cpu().numpy().astype(np.float32)
        self.epochs.append(int(epoch))
        self.Zs.append(Z)
        tag = f"epoch_{epoch:0{self.width}d}"
        np.save(self.frame_dir / f"{tag}.npy", Z)
        title = f"leanmap cellcycle — epoch {epoch}/{self.total}"
        save_phase_scatter(
            Z,
            self.y,
            title=title,
            path=self.frame_dir / f"{tag}.png",
            phase_names=self.phase_names,
        )
        save_phase_scatter(
            Z,
            self.y,
            title=title,
            path=self.out_dir / "live.png",
            phase_names=self.phase_names,
        )

    def finalize(self) -> Path | None:
        if not self.Zs:
            return None
        path = self.out_dir / "frames.npz"
        np.savez_compressed(
            path,
            epochs=np.asarray(self.epochs, dtype=np.int32),
            Z=np.stack(self.Zs),
            y=self.y,
            phases=np.asarray(self.phase_names),
        )
        return path


def build_dataset(args) -> Path:
    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    print(f"source={source}")
    images, labels, cell_ids, present = sample_merged(
        source, max_per_phase=int(args.max_per_phase), seed=int(args.seed)
    )
    clip = float(args.clahe_clip)
    tile = int(args.clahe_tile)
    use_otsu = not bool(args.no_otsu)
    print(
        f"preprocess: CLAHE clipLimit={clip} tile={tile}x{tile} "
        f"otsu={'on' if use_otsu else 'off'} invert={bool(args.otsu_invert)}"
    )
    if use_otsu:
        images_clahe = _clahe_rgb(
            images, clip_limit=clip, tile_grid_size=(tile, tile)
        )
        images_proc, mask = _otsu_mask_rgb(
            images_clahe, invert=bool(args.otsu_invert)
        )
    else:
        images_proc, mask = _preprocess_rgb(
            images, clahe_clip=clip, clahe_tile=tile, otsu=False
        )
    rgb = _rgb01(images_proc)
    n = len(images)
    fg_frac = float((mask > 0).mean())
    print(
        f"N={n} proc={images_proc.shape} rgb01 range=[{rgb.min():.3f},{rgb.max():.3f}] "
        f"fg_frac={fg_frac:.3f} phases={present}"
    )

    knn_idx, knn_dist, k, M = build_knn(rgb, args)
    assert knn_idx.shape == (n, k)
    assert np.isfinite(knn_dist).all()
    assert not (knn_idx == np.arange(n)[:, None]).any()
    print(
        f"kNN ready ({args.knn}): median nn={float(np.median(knn_dist[:, 0])):.4f} "
        f"mean={float(knn_dist.mean()):.4f}"
    )

    attrs = {
        "phases": list(PHASES),
        "phases_present": present,
        "source": str(source),
        "seed": int(args.seed),
        "max_per_phase": int(args.max_per_phase),
        "k": k,
        "knn": str(args.knn),
        "clahe": {
            "clip_limit": clip,
            "tile_grid_size": [tile, tile],
            "per_channel": True,
        },
        "otsu": {
            "enabled": use_otsu,
            "invert": bool(args.otsu_invert),
            "fg_frac": fg_frac,
        },
        "features": "CLAHE + Otsu-masked RGB /255 → [0,1], flattened HxWx3",
        "counts": {p: int((labels == PHASES.index(p)).sum()) for p in present},
    }
    if M is not None:
        attrs["M"] = M
    write_zarr(
        out,
        images=images,
        images_proc=images_proc,
        mask=mask,
        rgb=rgb,
        labels=labels,
        cell_ids=cell_ids,
        knn_idx=knn_idx,
        knn_dist=knn_dist,
        attrs=attrs,
    )
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(attrs, indent=2) + "\n")
    print(f"wrote {out}")
    print(zarr.open_group(str(out), mode="r").tree())
    print(f"meta {meta_path}")
    return out


def fit_leanmap(args, zarr_path: Path) -> None:
    from leanmap import fit

    arrays, attrs = load_zarr(zarr_path)
    rgb = arrays["rgb"].astype(np.float32)
    labels = arrays["labels"].astype(np.int64)
    phases = tuple(attrs.get("phases", PHASES))
    X = _flat(rgb)  # ambient + knn: full RGB in [0,1]
    n = len(X)

    rng = np.random.default_rng(int(args.seed))
    n_cal = max(1, int(round(float(args.holdout) * n)))
    perm = rng.permutation(n)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    X_train, X_cal = X[train_idx], X[cal_idx]
    y_train = labels[train_idx]
    imgs_train = rgb[train_idx]

    print(
        f"leanmap fit: N_train={len(X_train)} N_cal={len(X_cal)} "
        f"knn={args.knn} min_dist={args.min_dist} "
        f"rgb={rgb.shape[1]}x{rgb.shape[2]}x{rgb.shape[3]} d={X.shape[1]}"
    )
    knn_idx, knn_dist, k, M = build_knn(imgs_train, args)
    assert knn_idx.shape == (len(X_train), k)
    assert np.isfinite(knn_dist).all()
    print(
        f"kNN ready ({args.knn}): median nn={float(np.median(knn_dist[:, 0])):.4f} "
        f"mean={float(knn_dist.mean()):.4f}"
    )

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("epoch_*"):
        p.unlink(missing_ok=True)
    progress = run_dir / "progress.csv"
    if progress.exists():
        progress.unlink()

    cfg = default_config(len(X_train), epochs=int(args.epochs))
    cfg.seed = int(args.seed)
    cfg.dedup = False
    cfg.n_neighbors = k
    cfg.min_dist = float(args.min_dist)
    if args.device is not None:
        cfg.device = args.device

    recorder = None
    callbacks = None
    if int(args.frame_every) > 0:
        recorder = EmbeddingRecorder(
            X_train,
            y_train,
            out_dir=run_dir,
            every=int(args.frame_every),
            total_epochs=int(args.epochs),
            phase_names=phases,
        )
        callbacks = [recorder]

    result = fit(
        X_train,
        dist_fn="l1",
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

    model_path = run_dir / "model.pt"
    result.save(str(model_path))
    knn_label = str(args.knn).upper()
    final_png = save_phase_scatter(
        Z,
        y_train,
        title=f"leanmap — cellcycle {knn_label} (min_dist={args.min_dist})",
        path=run_dir / "final.png",
        phase_names=phases,
    )
    np.save(run_dir / "Z_final.npy", Z.astype(np.float32))
    np.save(run_dir / "y_train.npy", y_train)
    np.save(run_dir / "train_idx.npy", train_idx.astype(np.int64))
    (run_dir / "phases.json").write_text(json.dumps(list(phases), indent=2) + "\n")

    frames_npz = recorder.finalize() if recorder else None
    print(f"N_train={len(X_train)} N_cal={len(X_cal)} d={X_train.shape[1]} -> {Z.shape}")
    extra = f" M={M}" if M is not None else ""
    print(f"min_dist={result.config.min_dist} knn=precomputed {args.knn} k={k}{extra}")
    print(f"saved {model_path}")
    print(f"saved {final_png}")
    if frames_npz is not None:
        print(f"saved {len(recorder.Zs)} intermediate embeddings -> {frames_npz}")
        print(f"frames dir: {recorder.frame_dir}")
        print(f"live: {run_dir / 'live.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="CellCycle root with phase subdirs",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output zarr path",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help="leanmap run output directory",
    )
    ap.add_argument("--max-per-phase", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--knn",
        choices=("l1", "emd"),
        default="l1",
        help="graph metric: L1 on images (default) or L1-filter→EMD",
    )
    ap.add_argument("--k", type=int, default=15, help="neighbors kept")
    ap.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="OpenCV CLAHE clipLimit (per channel)",
    )
    ap.add_argument(
        "--clahe-tile",
        type=int,
        default=8,
        help="OpenCV CLAHE tileGridSize (tile x tile)",
    )
    ap.add_argument(
        "--no-otsu",
        action="store_true",
        help="skip Otsu foreground mask after CLAHE",
    )
    ap.add_argument(
        "--otsu-invert",
        action="store_true",
        help="treat dark pixels as foreground (THRESH_BINARY_INV)",
    )
    ap.add_argument(
        "--M",
        type=int,
        default=None,
        help="EMD mode only: L1 candidate width (default: max(30, 3*k))",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="EMD edge batch (full-res 66x66 supports need small batches)",
    )
    ap.add_argument(
        "--emd-device",
        default="cuda",
        help="torchemd device for EMD rescore (default cuda)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="leanmap training device (default: auto)",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    ap.add_argument("--holdout", type=float, default=0.1, help="calib fraction")
    ap.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="write embedding snapshot every N epochs (0 disables)",
    )
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help="skip sampling/zarr build; load --out and fit",
    )
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="build zarr + full-set knn, skip leanmap fit",
    )
    args = ap.parse_args()

    zarr_path = args.out.expanduser().resolve()
    if not args.fit_only:
        build_dataset(args)
    elif not zarr_path.exists():
        raise FileNotFoundError(f"--fit-only but zarr missing: {zarr_path}")

    if not args.build_only:
        fit_leanmap(args, zarr_path)


if __name__ == "__main__":
    main()

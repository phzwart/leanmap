#!/usr/bin/env python
"""Cheap LeJEPA on CellCycle Ch3/Ch4/Ch6 → features → leanmap.

LeJEPA = multi-view embedding agreement + SIGReg (isotropic Gaussian),
no teacher / stop-grad / negatives (Balestriero & LeCun, arXiv:2511.08544).

Pipeline:
  1. load ``{id}_Ch{3,4,6}.ome.jpg`` stacks as (3, 66, 66) float in [0, 1]
  2. train a tiny ConvNet encoder + projector with LeJEPA
  3. freeze encoder → write ``features`` to zarr
  4. optionally fit leanmap on those features (L2 knn) with epoch frames

    python examples/cellcycle_lejepa.py --max-per-phase 100 --epochs 100
    python examples/cellcycle_lejepa.py --fit-only   # leanmap on existing features

Requires: torch, lejepa, zarr, Pillow, scikit-learn (for leanmap path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from _demo import OUT_DIR, default_config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PHASES = ("G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase")
DEFAULT_SOURCE = Path.home() / "Projects" / "cells" / "CellCycle"
DEFAULT_OUT = ROOT / "examples" / "out" / "cellcycle_lejepa.zarr"
DEFAULT_RUN = OUT_DIR / "cellcycle_lejepa"
CHANNELS = (3, 4, 6)
# Soft disk: effective radius = this × half the box width (⇒ diameter = 75% of box).
SOFT_MASK_RADIUS_FRAC = 0.75
SOFT_MASK_EDGE_FRAC = 0.12  # falloff width as a fraction of the effective radius


# --------------------------------------------------------------------------- data
def soft_circular_mask(
    h: int,
    w: int,
    *,
    radius_frac: float = SOFT_MASK_RADIUS_FRAC,
    edge_frac: float = SOFT_MASK_EDGE_FRAC,
) -> np.ndarray:
    """Soft-edged circular aperture centered in the frame.

    ``radius_frac`` is relative to half the shorter side, so 0.75 → effective
    diameter = 75% of the box width.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = float(radius_frac) * 0.5 * float(min(h, w))
    soft = max(1.0, float(edge_frac) * r0)
    # cosine falloff from 1 inside r0-soft to 0 at r0+soft
    t = np.clip((r0 + soft - r) / (2.0 * soft), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def estimated_focus_quality(images: np.ndarray) -> np.ndarray:
    """Laplacian-variance focus score per image.

    ``images``: (N, C, H, W) or (C, H, W) float. Higher ⇒ sharper.
    """
    x = np.asarray(images, dtype=np.float32)
    single = x.ndim == 3
    if single:
        x = x[None]
    # luminance = mean over channels
    gray = x.mean(axis=1)
    # 3×3 Laplacian kernel
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=-1)
        + np.roll(gray, -1, axis=-1)
        + np.roll(gray, 1, axis=-2)
        + np.roll(gray, -1, axis=-2)
    )
    # ignore 1-px border (roll wrap)
    scores = lap[:, 1:-1, 1:-1].var(axis=(1, 2)).astype(np.float32)
    return scores[0] if single else scores


def preprocess_stack(
    x: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    radius_frac: float = SOFT_MASK_RADIUS_FRAC,
) -> np.ndarray:
    """Soft circular mask + equal per-channel signal contrast.

    - Soft disk keeps the center (effective radius = 75% of half-box by default).
    - Per channel: subtract low-percentile background (absence → ~0), keep
      positive signal under the mask, then scale so each channel's peak is 1
      (equal contribution even when a channel is weak).
    """
    x = np.asarray(x, dtype=np.float32)
    c, h, w = x.shape
    if mask is None:
        mask = soft_circular_mask(h, w, radius_frac=radius_frac)
    out = np.zeros_like(x)
    # background from the soft exterior (mask < 0.2), fallback to global low %ile
    exterior = mask < 0.2
    for ci in range(c):
        ch = x[ci]
        if exterior.any():
            bg = float(np.percentile(ch[exterior], 50))
        else:
            bg = float(np.percentile(ch, 10))
        sig = np.clip(ch - bg, 0.0, None) * mask
        peak = float(sig.max())
        if peak > 1e-8:
            out[ci] = sig / peak
        else:
            out[ci] = sig
    return out.astype(np.float32)


def annotate_focus_quality(zarr_path: Path) -> np.ndarray:
    """Write ``estimated_focus_quality`` into an existing features zarr; return scores."""
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r+")
    images = np.asarray(root["images"], dtype=np.float32)
    scores = estimated_focus_quality(images)
    if "estimated_focus_quality" in root:
        del root["estimated_focus_quality"]
    root.create_array("estimated_focus_quality", data=scores)
    root.attrs["focus_metric"] = "laplacian_variance_luminance"
    print(
        f"wrote estimated_focus_quality → {zarr_path}  "
        f"N={len(scores)}  med={float(np.median(scores)):.6g}  "
        f"p25={float(np.percentile(scores, 25)):.6g}  "
        f"p75={float(np.percentile(scores, 75)):.6g}"
    )
    return scores


def list_cells(
    source: Path, *, max_per_phase: int, seed: int
) -> tuple[list[tuple[Path, Path, Path]], np.ndarray, np.ndarray]:
    """Return (ch paths triples), labels, cell_ids."""
    rng = np.random.default_rng(seed)
    paths: list[tuple[Path, Path, Path]] = []
    labels: list[int] = []
    cell_ids: list[int] = []
    for li, phase in enumerate(PHASES):
        d = source / phase
        if not d.is_dir():
            continue
        ch3 = sorted(d.glob("*_Ch3.ome.jpg"))
        if not ch3:
            continue
        n_keep = len(ch3) if max_per_phase <= 0 else min(max_per_phase, len(ch3))
        order = rng.permutation(len(ch3))[:n_keep]
        for j in order:
            p3 = ch3[int(j)]
            cid = p3.name.split("_", 1)[0]
            p4 = d / f"{cid}_Ch4.ome.jpg"
            p6 = d / f"{cid}_Ch6.ome.jpg"
            if not (p4.is_file() and p6.is_file()):
                continue
            paths.append((p3, p4, p6))
            labels.append(li)
            cell_ids.append(int(cid))
        print(f"{phase}: kept {len(order)} / {len(ch3)}")
    if not paths:
        raise FileNotFoundError(f"no Ch3/4/6 triples under {source}")
    return (
        paths,
        np.asarray(labels, dtype=np.int64),
        np.asarray(cell_ids, dtype=np.int64),
    )


def load_stack(triple: tuple[Path, Path, Path]) -> np.ndarray:
    """Load Ch3/4/6 → float32 (3, H, W) in [0, 1]."""
    chans = []
    for p in triple:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        chans.append(a)
    return np.stack(chans, axis=0)


class CellCycleTripleDataset(Dataset):
    def __init__(
        self,
        triples: list[tuple[Path, Path, Path]],
        labels: np.ndarray,
        *,
        n_global: int = 2,
        n_local: int = 2,
        train: bool = True,
        preprocess: bool = False,
        mask_radius_frac: float = SOFT_MASK_RADIUS_FRAC,
        focus_quality: np.ndarray | None = None,
    ) -> None:
        self.triples = triples
        self.labels = np.asarray(labels)
        self.n_global = int(n_global)
        self.n_local = int(n_local)
        self.train = bool(train)
        self.preprocess = bool(preprocess)
        self.mask_radius_frac = float(mask_radius_frac)
        # cache arrays in RAM (N≈500–32k × 3 × 66 × 66 is fine)
        raw = np.stack([load_stack(t) for t in triples], axis=0)
        if focus_quality is None:
            self.focus_quality = estimated_focus_quality(raw)
        else:
            self.focus_quality = np.asarray(focus_quality, dtype=np.float32)
            if len(self.focus_quality) != len(raw):
                raise ValueError(
                    f"focus_quality length {len(self.focus_quality)} != N={len(raw)}"
                )
        if self.preprocess:
            _, h, w = raw[0].shape
            mask = soft_circular_mask(h, w, radius_frac=self.mask_radius_frac)
            self.images = np.stack(
                [preprocess_stack(im, mask=mask) for im in raw], axis=0
            )
            print(
                f"preprocess: soft mask r={self.mask_radius_frac:.2f}×half-box, "
                f"equal-channel signal contrast (N={len(self.images)})"
            )
        else:
            self.images = raw

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.images[i])  # (3,H,W)
        if not self.train:
            return x, int(self.labels[i]), i
        views = [_augment(x, global_view=True) for _ in range(self.n_global)]
        views += [_augment(x, global_view=False) for _ in range(self.n_local)]
        return views, int(self.labels[i]), i


def _augment(x: torch.Tensor, *, global_view: bool) -> torch.Tensor:
    """Cheap multi-crop augs for 66×66 fluorescence stacks (no ColorJitter RGB)."""
    c, h, w = x.shape
    # geometric
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])

    # scale crop
    if global_view:
        scale = float(torch.empty(1).uniform_(0.5, 1.0).item())
    else:
        scale = float(torch.empty(1).uniform_(0.3, 0.6).item())
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    top = int(torch.randint(0, h - nh + 1, (1,)).item()) if nh < h else 0
    left = int(torch.randint(0, w - nw + 1, (1,)).item()) if nw < w else 0
    x = x[:, top : top + nh, left : left + nw]
    x = F.interpolate(
        x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)

    # per-channel brightness / contrast (independent — not photo ColorJitter)
    for ci in range(c):
        if torch.rand(1).item() < 0.8:
            b = float(torch.empty(1).uniform_(0.7, 1.3).item())
            contrast = float(torch.empty(1).uniform_(0.7, 1.3).item())
            mean = x[ci].mean()
            x[ci] = (x[ci] - mean) * contrast + mean
            x[ci] = x[ci] * b
    # mild gaussian noise
    if torch.rand(1).item() < 0.3:
        x = x + 0.02 * torch.randn_like(x)
    return x.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- model
class TinyEncoder(nn.Module):
    """~200k-param ConvNet for 66×66 × 3 → emb_dim."""

    def __init__(self, in_ch: int = 3, emb_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 33
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)


class Projector(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 256, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class LeJEPA(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int = 128,
        proj_dim: int = 64,
        num_slices: int = 256,
        lambda_sigreg: float = 0.05,
        n_classes: int = len(PHASES),
        use_class_token: bool = False,
        lambda_cls: float = 0.5,
    ) -> None:
        super().__init__()
        import lejepa

        self.encoder = TinyEncoder(in_ch=3, emb_dim=emb_dim)
        self.projector = Projector(in_dim=emb_dim, out_dim=proj_dim)
        self.lambda_sigreg = float(lambda_sigreg)
        self.use_class_token = bool(use_class_token)
        self.lambda_cls = float(lambda_cls)
        self.n_classes = int(n_classes)
        if self.use_class_token:
            # per-phase token added to encoder features before the projector;
            # CE head sits on raw encoder features (exported for leanmap).
            self.class_token = nn.Embedding(self.n_classes, emb_dim)
            self.classifier = nn.Linear(emb_dim, self.n_classes)
            nn.init.zeros_(self.class_token.weight)
        else:
            self.class_token = None
            self.classifier = None
        uni = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=uni, num_slices=int(num_slices)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Image features only (no class token) — what leanmap consumes."""
        return self.encoder(x)

    def project(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        if self.use_class_token and y is not None:
            h = h + self.class_token(y)
        return self.projector(h)

    def forward_views(
        self,
        views: list[torch.Tensor],
        y: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        hs = [self.encoder(v) for v in views]
        if self.use_class_token and y is not None:
            tok = self.class_token(y)
            zs = [self.projector(h + tok) for h in hs]
            logits = self.classifier(hs[0])
            cls = F.cross_entropy(logits, y, weight=class_weight)
        else:
            zs = [self.projector(h) for h in hs]
            cls = hs[0].new_zeros(())
        # predictive: mean pairwise MSE (identity predictor)
        pred = 0.0
        n = 0
        for i in range(len(zs)):
            for j in range(len(zs)):
                if i == j:
                    continue
                pred = pred + F.mse_loss(zs[i], zs[j])
                n += 1
        pred = pred / max(n, 1)
        z_cat = torch.cat(zs, dim=0)
        sig = self.sigreg(z_cat)
        loss = pred + self.lambda_sigreg * sig + self.lambda_cls * cls
        return loss, {
            "loss": float(loss.detach()),
            "pred": float(pred.detach()),
            "sigreg": float(sig.detach()),
            "cls": float(cls.detach()),
        }


# --------------------------------------------------------------------------- train / export
def collate_views(batch):
    # batch: list of (views_list, label, idx)
    n_views = len(batch[0][0])
    views = [
        torch.stack([b[0][v] for b in batch], dim=0) for v in range(n_views)
    ]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return views, labels, idxs


@torch.no_grad()
def extract_features(
    model: LeJEPA, images: np.ndarray, *, device: str, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(images[start : start + batch_size]).to(device)
        outs.append(model.embed(x).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def _save_ckpt(
    path: Path,
    model: LeJEPA,
    *,
    opt: torch.optim.Optimizer | None,
    epoch: int,
    attrs: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": model.encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "epoch": int(epoch),
        "attrs": attrs,
    }
    if model.use_class_token:
        payload["class_token"] = model.class_token.state_dict()
        payload["classifier"] = model.classifier.state_dict()
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, path)
    print(f"ckpt → {path} (epoch {epoch})")


def _load_ckpt(model: LeJEPA, path: Path, opt: torch.optim.Optimizer | None = None) -> int:
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.projector.load_state_dict(ckpt["projector"])
    if model.use_class_token:
        if "class_token" not in ckpt or "classifier" not in ckpt:
            raise KeyError(f"{path} missing class_token/classifier (train with --class-token)")
        model.class_token.load_state_dict(ckpt["class_token"])
        model.classifier.load_state_dict(ckpt["classifier"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("epoch", 0) or 0)
    if start <= 0 and isinstance(ckpt.get("attrs"), dict):
        start = int(ckpt["attrs"].get("epochs", 0) or 0)
    print(f"resumed {path} @ epoch {start}")
    return start


def train_lejepa(args) -> Path:
    import zarr
    from torch.utils.data import Subset

    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    triples, labels, cell_ids = list_cells(
        source, max_per_phase=int(args.max_per_phase), seed=int(args.seed)
    )
    n_raw = len(triples)
    use_preprocess = bool(getattr(args, "soft_mask", False))
    mask_radius = float(getattr(args, "mask_radius_frac", SOFT_MASK_RADIUS_FRAC))
    focus_keep_frac = float(getattr(args, "focus_keep_frac", 0.0) or 0.0)

    # Load + optional soft-mask / equal-channel prep; always score focus on raw.
    ds_all = CellCycleTripleDataset(
        triples,
        labels,
        n_global=int(args.n_global),
        n_local=int(args.n_local),
        train=True,
        preprocess=use_preprocess,
        mask_radius_frac=mask_radius,
    )
    focus_all = ds_all.focus_quality.copy()
    print(
        f"focus quality (laplacian var): med={float(np.median(focus_all)):.6g}  "
        f"p75={float(np.percentile(focus_all, 75)):.6g}  "
        f"p90={float(np.percentile(focus_all, 90)):.6g}"
    )

    # Prefilter: keep the sharpest focus_keep_frac of images (global).
    keep_idx = np.arange(n_raw, dtype=np.int64)
    if focus_keep_frac > 0.0 and focus_keep_frac < 1.0:
        n_keep = max(1, int(round(focus_keep_frac * n_raw)))
        order = np.argsort(-focus_all)  # sharpest first
        keep_idx = np.sort(order[:n_keep].astype(np.int64))
        thr = float(focus_all[order[n_keep - 1]])
        print(
            f"focus prefilter: keep top {focus_keep_frac:.0%} → {n_keep}/{n_raw} "
            f"(threshold={thr:.6g})"
        )
        for li, phase in enumerate(PHASES):
            n_p = int((labels == li).sum())
            n_k = int((labels[keep_idx] == li).sum())
            if n_p:
                print(f"  {phase}: {n_k}/{n_p}")
        # Restrict the in-memory dataset to the kept rows.
        triples = [triples[int(i)] for i in keep_idx]
        labels = labels[keep_idx]
        cell_ids = cell_ids[keep_idx]
        focus_kept = focus_all[keep_idx]
        ds_all.triples = triples
        ds_all.labels = labels
        ds_all.images = ds_all.images[keep_idx]
        ds_all.focus_quality = focus_kept
    else:
        focus_kept = focus_all

    n_all = len(triples)

    # Optional: restrict the LeJEPA optimisation set (features still exported for all).
    train_sel: np.ndarray | None = None
    train_idx_path = getattr(args, "train_idx_path", None)
    if train_idx_path is not None:
        train_sel = np.load(Path(train_idx_path).expanduser().resolve()).astype(np.int64)
        if train_sel.ndim != 1:
            raise ValueError(f"--train-idx must be 1-D, got {train_sel.shape}")
        if train_sel.min() < 0 or train_sel.max() >= n_all:
            raise ValueError(
                f"--train-idx out of range for N={n_all} "
                f"(min={train_sel.min()}, max={train_sel.max()})"
            )
        print(
            f"LeJEPA train subset from {train_idx_path}: "
            f"{len(train_sel)} / {n_all} cells"
        )
    exclude_train = tuple(getattr(args, "exclude_phases_train", ()) or ())
    if exclude_train and train_sel is None:
        excl_ids = set(_phase_indices(PHASES, exclude_train))
        train_sel = np.array(
            [i for i, y in enumerate(labels) if int(y) not in excl_ids],
            dtype=np.int64,
        )
        print(
            f"LeJEPA train exclude {list(exclude_train)} → "
            f"{len(train_sel)} / {n_all} cells"
        )

    if train_sel is not None:
        ds_fit = Subset(ds_all, train_sel.tolist())
        labels_fit = labels[train_sel]
    else:
        ds_fit = ds_all
        labels_fit = labels
        train_sel = np.arange(n_all, dtype=np.int64)

    bs = min(int(args.batch_size), len(ds_fit))
    loader = DataLoader(
        ds_fit,
        batch_size=bs,
        shuffle=True,
        num_workers=int(args.workers),
        collate_fn=collate_views,
        drop_last=len(ds_fit) > bs,  # BatchNorm in projector needs >1
        pin_memory=str(device).startswith("cuda"),
    )
    print(
        f"N_fit={len(ds_fit)} N_export={n_all} views={args.n_global}g+{args.n_local}l "
        f"batch={args.batch_size} device={device}"
    )

    use_cls = bool(args.class_token)
    model = LeJEPA(
        emb_dim=int(args.emb_dim),
        proj_dim=int(args.proj_dim),
        num_slices=int(args.num_slices),
        lambda_sigreg=float(args.lambda_sigreg),
        n_classes=len(PHASES),
        use_class_token=use_cls,
        lambda_cls=float(args.lambda_cls),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model params={n_params:,}  class_token={use_cls}  "
        f"lambda_cls={args.lambda_cls}"
    )

    class_weight = None
    if use_cls:
        # inverse-frequency over the *fit* set only
        bc = np.bincount(labels_fit, minlength=len(PHASES)).astype(np.float64)
        present = bc > 0
        w = np.zeros(len(PHASES), dtype=np.float64)
        w[present] = 1.0 / bc[present]
        w[present] = w[present] / w[present].sum() * int(present.sum())
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(
            "class_weight",
            {PHASES[i]: float(w[i]) for i in range(len(PHASES)) if present[i]},
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    extra_epochs = int(args.epochs)
    start_epoch = 0
    resume = args.resume
    if resume is None and args.continue_from_out:
        cand = out.with_suffix(".pt")
        if cand.is_file():
            resume = cand
    if resume is not None:
        resume = Path(resume).expanduser().resolve()
        start_epoch = _load_ckpt(model, resume, opt=opt)
        model.to(device)

    method = "LeJEPA (cheap ConvNet + SIGReg)"
    if use_cls:
        method += " + class-token inject + CE"
    if len(train_sel) < n_all:
        method += f" [fit on {len(train_sel)}/{n_all} cells; export all]"
    attrs = {
        "phases": list(PHASES),
        "channels": list(CHANNELS),
        "source": str(source),
        "seed": int(args.seed),
        "max_per_phase": int(args.max_per_phase),
        "emb_dim": int(args.emb_dim),
        "proj_dim": int(args.proj_dim),
        "lambda_sigreg": float(args.lambda_sigreg),
        "class_token": use_cls,
        "lambda_cls": float(args.lambda_cls) if use_cls else 0.0,
        "method": method,
        "n_fit": int(len(train_sel)),
        "n_export": int(n_all),
        "n_raw_before_focus_filter": int(n_raw),
        "focus_keep_frac": float(focus_keep_frac),
        "focus_metric": "laplacian_variance_luminance",
        "soft_mask": bool(use_preprocess),
        "mask_radius_frac": float(mask_radius) if use_preprocess else 0.0,
        "train_idx_source": (
            str(Path(train_idx_path).expanduser().resolve())
            if train_idx_path is not None
            else ""
        ),
        "exclude_phases_train": list(exclude_train),
        "counts_fit": {
            p: int((labels_fit == PHASES.index(p)).sum())
            for p in PHASES
            if (labels_fit == PHASES.index(p)).any()
        },
        "counts": {
            p: int((labels == PHASES.index(p)).sum())
            for p in PHASES
            if (labels == PHASES.index(p)).any()
        },
    }
    ckpt_every = int(args.ckpt_every)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else out.parent / f"{out.stem}_ckpts"
    end_epoch = start_epoch + extra_epochs
    print(
        f"train epochs {start_epoch + 1}…{end_epoch}  "
        f"(+{extra_epochs})  ckpt_every={ckpt_every}  ckpt_dir={ckpt_dir}"
    )

    model.train()
    for epoch in range(start_epoch + 1, end_epoch + 1):
        totals = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0, "cls": 0.0}
        steps = 0
        for views, y, _idx in loader:
            views = [v.to(device, non_blocking=True) for v in views]
            y = y.to(device, non_blocking=True)
            loss, stats = model.forward_views(
                views, y=y if use_cls else None, class_weight=class_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in totals:
                totals[k] += stats.get(k, 0.0)
            steps += 1
        if steps:
            for k in totals:
                totals[k] /= steps
        msg = (
            f"epoch {epoch:04d}/{end_epoch}  loss={totals['loss']:.4f}  "
            f"pred={totals['pred']:.4f}  sigreg={totals['sigreg']:.4f}"
        )
        if use_cls:
            msg += f"  cls={totals['cls']:.4f}"
        print(msg)

        attrs_ep = {**attrs, "epochs": epoch}
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            _save_ckpt(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                model,
                opt=opt,
                epoch=epoch,
                attrs=attrs_ep,
            )
            # also refresh the main sidecar checkpoint
            _save_ckpt(out.with_suffix(".pt"), model, opt=opt, epoch=epoch, attrs=attrs_ep)

    # final checkpoint + features for *all* cells (even if fit used a subset)
    attrs = {**attrs, "epochs": end_epoch}
    _save_ckpt(out.with_suffix(".pt"), model, opt=opt, epoch=end_epoch, attrs=attrs)
    # persist the fit-row indices alongside the model for provenance
    np.save(out.with_name(out.stem + "_fit_idx.npy"), train_sel.astype(np.int64))

    feats = extract_features(model, ds_all.images, device=device)
    print(
        f"features {feats.shape} (exported all cells) "
        f"mean={feats.mean():.4f} std={feats.std():.4f}"
    )

    if out.exists():
        import shutil

        shutil.rmtree(out)
    root = zarr.open_group(str(out), mode="w")
    root.create_array("features", data=feats)
    root.create_array("images", data=ds_all.images.astype(np.float32))  # (N,3,H,W)
    root.create_array("labels", data=labels)
    root.create_array("cell_ids", data=cell_ids)
    root.create_array("fit_idx", data=train_sel.astype(np.int64))
    root.create_array(
        "estimated_focus_quality", data=np.asarray(ds_all.focus_quality, dtype=np.float32)
    )
    if focus_keep_frac > 0.0 and focus_keep_frac < 1.0:
        root.create_array("raw_keep_idx", data=keep_idx.astype(np.int64))
    for k, v in attrs.items():
        root.attrs[k] = v
    meta = out.with_suffix(out.suffix + ".meta.json")
    meta.write_text(json.dumps(attrs, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"cached model → {out.with_suffix('.pt')}")
    print(root.tree())
    return out


_MITOTIC = ("Prophase", "Metaphase", "Anaphase", "Telophase")


def _phase_indices(phases: tuple[str, ...] | list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    lookup = {p: i for i, p in enumerate(phases)}
    out: list[int] = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"unknown phase {name!r}; have {list(phases)}")
        out.append(lookup[name])
    return out


def _score_split(
    result,
    X: np.ndarray,
    *,
    batch: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z, cover, p_value) for ``X`` via the fitted conformal calibrator."""
    if len(X) == 0:
        d = int(getattr(result.config, "d_out", 2))
        empty = np.zeros((0,), dtype=np.float32)
        return np.zeros((0, d), dtype=np.float32), empty, empty
    model = result.model
    cal = result.calibrator
    device = next(model.parameters()).device
    zs: list[np.ndarray] = []
    covers: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i : i + batch], dtype=torch.float32, device=device)
            z, _ = model.embed(xb, return_score=False)
            s = cal.cover_score(model, xb)
            p = cal.p_value(s, model=model)
            zs.append(z.detach().cpu().numpy())
            covers.append(s.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())
    return (
        np.concatenate(zs).astype(np.float32),
        np.concatenate(covers).astype(np.float32),
        np.concatenate(ps).astype(np.float32),
    )


def _novelty_table(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    alphas: tuple[float, ...] = (0.05, 0.1),
) -> str:
    lines = [
        f"{'set':16} {'n':>6} {'cover50':>9} {'p50':>8}"
        + "".join(f" {'p<'+str(a):>8}" for a in alphas)
    ]
    for name, (cover, p) in groups.items():
        parts = [
            f"{name:16}",
            f"{len(p):6d}",
            f"{float(np.median(cover)):9.3f}",
            f"{float(np.median(p)):8.3f}",
        ]
        for a in alphas:
            parts.append(f"{float((p < a).mean()):8.3f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _save_novelty_scatter(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_ood: np.ndarray,
    y_ood: np.ndarray,
    p_ood: np.ndarray,
    *,
    path: Path,
    phase_names: tuple[str, ...] | list[str],
    title: str,
    alpha: float = 0.05,
) -> None:
    """Train phases + mitotic OOD overlaid (rejected at ``alpha`` marked ×)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from cellcycle_emd import _PHASE_COLORS

    names = list(phase_names)
    n_phases = len(names)
    colors = list(_PHASE_COLORS[:n_phases])
    cmap = ListedColormap(colors)
    bounds = np.arange(n_phases + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(
        Z_train[:, 0],
        Z_train[:, 1],
        c=y_train,
        s=4,
        cmap=cmap,
        norm=norm,
        linewidths=0,
        alpha=0.55,
    )
    rej = p_ood < alpha
    if (~rej).any():
        ax.scatter(
            Z_ood[~rej, 0],
            Z_ood[~rej, 1],
            c="0.25",
            s=18,
            marker="o",
            linewidths=0.4,
            edgecolors="k",
            label=f"mitotic p≥{alpha} (n={(~rej).sum()})",
            zorder=3,
        )
    if rej.any():
        ax.scatter(
            Z_ood[rej, 0],
            Z_ood[rej, 1],
            c="k",
            s=28,
            marker="x",
            linewidths=1.0,
            label=f"mitotic p<{alpha} (n={rej.sum()})",
            zorder=4,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_leanmap(args, zarr_path: Path) -> None:
    import zarr
    from sklearn.neighbors import NearestNeighbors

    from leanmap import fit

    # local import of phase scatter from cellcycle_emd if available
    try:
        from cellcycle_emd import save_phase_scatter, EmbeddingRecorder, PHASES as P
    except ImportError:
        from cellcycle_emd import save_phase_scatter, EmbeddingRecorder  # type: ignore
        P = PHASES

    root = zarr.open_group(str(zarr_path), mode="r")
    X = np.asarray(root["features"], dtype=np.float32)
    labels = np.asarray(root["labels"], dtype=np.int64)
    phases = tuple(root.attrs.get("phases", P))

    rng = np.random.default_rng(int(args.seed))
    n = len(X)
    max_points = int(getattr(args, "max_points", 0) or 0)
    if max_points > 0 and n > max_points:
        keep = rng.choice(n, size=max_points, replace=False)
        keep.sort()
        X, labels = X[keep], labels[keep]
        subsample_idx = keep.astype(np.int64)
        print(f"subsample {n} → {max_points} for leanmap")
        n = max_points
    else:
        subsample_idx = np.arange(n, dtype=np.int64)

    exclude_names = tuple(getattr(args, "exclude_phases", ()) or ())
    test_frac = float(getattr(args, "test_frac", 0.0) or 0.0)
    calib_frac = float(args.holdout)

    if exclude_names:
        excl_ids = set(_phase_indices(phases, exclude_names))
        in_mask = np.array([int(y) not in excl_ids for y in labels], dtype=bool)
        ood_mask = ~in_mask
        print(
            f"embedding phases: {[p for i, p in enumerate(phases) if i not in excl_ids]} "
            f"(n={int(in_mask.sum())}); held-out novelty: {list(exclude_names)} "
            f"(n={int(ood_mask.sum())})"
        )
    else:
        in_mask = np.ones(n, dtype=bool)
        ood_mask = np.zeros(n, dtype=bool)

    in_local = np.flatnonzero(in_mask)
    ood_local = np.flatnonzero(ood_mask)
    X_in, y_in = X[in_local], labels[in_local]
    X_ood = X[ood_local] if len(ood_local) else np.zeros((0, X.shape[1]), np.float32)
    y_ood = labels[ood_local] if len(ood_local) else np.zeros((0,), np.int64)
    idx_in = subsample_idx[in_local]
    idx_ood = subsample_idx[ood_local]

    n_in = len(X_in)
    n_test = int(round(test_frac * n_in)) if test_frac > 0 else 0
    n_cal = max(1, int(round(calib_frac * n_in))) if n_in > 1 else 1
    if n_test + n_cal >= n_in:
        raise ValueError(
            f"test_frac+holdout leave no train points "
            f"(n_in={n_in}, n_test={n_test}, n_cal={n_cal})"
        )
    perm = rng.permutation(n_in)
    test_local = perm[:n_test]
    cal_local = perm[n_test : n_test + n_cal]
    train_local = perm[n_test + n_cal :]

    X_train, X_cal = X_in[train_local], X_in[cal_local]
    y_train = y_in[train_local]
    y_cal = y_in[cal_local]
    X_test = X_in[test_local] if n_test else np.zeros((0, X.shape[1]), np.float32)
    y_test = y_in[test_local] if n_test else np.zeros((0,), np.int64)
    train_idx = idx_in[train_local]
    cal_idx = idx_in[cal_local]
    test_idx = idx_in[test_local] if n_test else np.zeros((0,), np.int64)

    # Optional: fold selected global indices (e.g. conformal novel probes)
    # into the embedding train set, removing them from the OOD pool.
    include_path = getattr(args, "include_idx_path", None)
    if include_path is not None:
        want = np.unique(
            np.load(Path(include_path).expanduser().resolve()).astype(np.int64)
        )
        want_set = set(int(i) for i in want)
        # Prefer pulling from current OOD; also accept any other held-out rows.
        ood_take = np.array(
            [j for j, gi in enumerate(idx_ood) if int(gi) in want_set], dtype=np.int64
        )
        if len(ood_take):
            X_train = np.concatenate([X_train, X_ood[ood_take]], axis=0)
            y_train = np.concatenate([y_train, y_ood[ood_take]], axis=0)
            train_idx = np.concatenate([train_idx, idx_ood[ood_take]], axis=0)
            keep_ood = np.ones(len(idx_ood), dtype=bool)
            keep_ood[ood_take] = False
            X_ood, y_ood, idx_ood = X_ood[keep_ood], y_ood[keep_ood], idx_ood[keep_ood]
        n_added = int(len(ood_take))
        still = (
            want_set
            - set(int(i) for i in train_idx)
            - set(int(i) for i in cal_idx)
            - set(int(i) for i in test_idx)
            - set(int(i) for i in idx_ood)
        )
        print(
            f"include-idx {include_path}: added {n_added} → train "
            f"(missing/unmatched={len(still)})"
        )

    print(
        f"split (in-manifold): train={len(X_train)} calib={len(X_cal)} "
        f"test={len(X_test)} | novelty={len(X_ood)}"
    )

    # Mean-center with the *train* mean, then L2-normalize (cosine-like geometry).
    mu = X_train.mean(axis=0, keepdims=True).astype(np.float32)

    def _center_l2(A: np.ndarray) -> np.ndarray:
        if len(A) == 0:
            return A
        A = A - mu
        return (A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)).astype(
            np.float32
        )

    X_train = _center_l2(X_train)
    X_cal = _center_l2(X_cal)
    X_test = _center_l2(X_test)
    X_ood = _center_l2(X_ood)
    print(
        f"features: mean-centered on train (||μ||₂={float(np.linalg.norm(mu)):.4f}), "
        f"then L2-normalized"
    )

    k = min(int(args.k), len(X_train) - 1)
    print(f"L2 knn k={k} on features {X_train.shape}")
    # +2: exact feature duplicates can push self out of slot 0
    nn = NearestNeighbors(n_neighbors=min(k + 2, len(X_train)), metric="euclidean").fit(
        X_train
    )
    dist, idx = nn.kneighbors(X_train)
    rows = np.arange(len(X_train))[:, None]
    not_self = idx != rows
    knn_idx = np.zeros((len(X_train), k), dtype=np.int64)
    knn_dist = np.zeros((len(X_train), k), dtype=np.float32)
    for i in range(len(X_train)):
        m = not_self[i]
        knn_idx[i] = idx[i, m][:k]
        knn_dist[i] = dist[i, m][:k]

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("epoch_*"):
        p.unlink(missing_ok=True)
    progress = run_dir / "progress.csv"
    if progress.exists():
        progress.unlink()

    cfg = default_config(len(X_train), epochs=int(args.leanmap_epochs))
    cfg.seed = int(args.seed)
    cfg.dedup = False
    cfg.n_neighbors = k
    cfg.min_dist = float(args.min_dist)
    cfg.d_out = int(args.d_out)
    n_lm = int(getattr(args, "n_landmarks", 0) or 0)
    if n_lm > 0:
        cfg.n_landmarks = n_lm
    if getattr(args, "lambda_geo", None) is not None:
        cfg.lambda_geo = float(args.lambda_geo)
    pw = getattr(args, "pyramid_level_weights", None)
    if pw is not None:
        cfg.pyramid_level_weights = tuple(float(x) for x in pw)
    if args.device:
        cfg.device = args.device
    print(
        f"leanmap config: d_out={cfg.d_out} min_dist={cfg.min_dist} "
        f"lambda_geo={cfg.lambda_geo} n_landmarks={cfg.n_landmarks} "
        f"pyramid_level_weights={cfg.pyramid_level_weights} "
        f"epochs={cfg.epochs}"
    )

    recorder = EmbeddingRecorder(
        X_train,
        y_train,
        out_dir=run_dir,
        every=int(args.frame_every) if int(args.frame_every) > 0 else 10**9,
        total_epochs=int(args.leanmap_epochs),
        phase_names=phases,
    )
    callbacks = [recorder] if int(args.frame_every) > 0 else None

    result = fit(
        X_train,
        dist_fn="l2",
        config=cfg,
        X_calib=X_cal,
        precomputed_knn=(
            torch.as_tensor(knn_idx, dtype=torch.int64),
            torch.as_tensor(knn_dist, dtype=torch.float32),
        ),
        callbacks=callbacks,
    )

    Z_train, cover_train, p_train = _score_split(result, X_train)
    Z_cal, cover_cal, p_cal = _score_split(result, X_cal)
    packs: dict[str, object] = {
        "Z_train": Z_train,
        "cover_train": cover_train,
        "p_train": p_train,
        "y_train": y_train,
        "train_idx": train_idx.astype(np.int64),
        "Z_cal": Z_cal,
        "cover_cal": cover_cal,
        "p_cal": p_cal,
        "y_cal": y_cal,
        "cal_idx": cal_idx.astype(np.int64),
        "phases": np.asarray(phases),
        "exclude_phases": np.asarray(exclude_names),
    }
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (cover_train, p_train),
        "calib": (cover_cal, p_cal),
    }

    if len(X_test):
        Z_test, cover_test, p_test = _score_split(result, X_test)
        packs.update(
            Z_test=Z_test,
            cover_test=cover_test,
            p_test=p_test,
            y_test=y_test,
            test_idx=test_idx.astype(np.int64),
        )
        groups["test"] = (cover_test, p_test)
        for i, name in enumerate(phases):
            m = y_test == i
            if m.any():
                groups[f"test/{name}"] = (cover_test[m], p_test[m])

    if len(X_ood):
        Z_ood, cover_ood, p_ood = _score_split(result, X_ood)
        packs.update(
            Z_ood=Z_ood,
            cover_ood=cover_ood,
            p_ood=p_ood,
            y_ood=y_ood,
            ood_idx=idx_ood.astype(np.int64),
        )
        groups["mitotic"] = (cover_ood, p_ood)
        for i, name in enumerate(phases):
            m = y_ood == i
            if m.any():
                groups[f"ood/{name}"] = (cover_ood[m], p_ood[m])

    table = _novelty_table(groups)
    print("\nconformal novelty (landmark cover)\n" + table)
    (run_dir / "novelty.txt").write_text(table + "\n")

    result.save(str(run_dir / "model.pt"))
    save_phase_scatter(
        Z_train,
        y_train,
        title=(
            f"leanmap train — LeJEPA "
            f"(min_dist={args.min_dist}, λ_geo={cfg.lambda_geo}, L={cfg.n_landmarks})"
        ),
        path=run_dir / "final.png",
        phase_names=phases,
    )
    if len(X_ood):
        _save_novelty_scatter(
            Z_train,
            y_train,
            packs["Z_ood"],  # type: ignore[arg-type]
            y_ood,
            packs["p_ood"],  # type: ignore[arg-type]
            path=run_dir / "novelty.png",
            phase_names=phases,
            title=(
                f"train + mitotic novelty (× = p<0.05) — "
                f"min_dist={args.min_dist}, L={cfg.n_landmarks}"
            ),
            alpha=0.05,
        )
    np.savez_compressed(run_dir / "scores.npz", **packs)  # type: ignore[arg-type]
    np.save(run_dir / "Z_final.npy", Z_train.astype(np.float32))
    np.save(run_dir / "y_train.npy", y_train)
    np.save(run_dir / "train_idx.npy", train_idx.astype(np.int64))
    np.save(run_dir / "cal_idx.npy", cal_idx.astype(np.int64))
    if n_test:
        np.save(run_dir / "test_idx.npy", test_idx.astype(np.int64))
    if recorder is not None and int(args.frame_every) > 0:
        recorder.finalize()
    print(f"saved leanmap run → {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument(
        "--max-per-phase",
        type=int,
        default=0,
        help="cap samples per phase (0 = use all)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100, help="LeJEPA pretrain epochs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--lambda-sigreg", type=float, default=0.05)
    ap.add_argument("--num-slices", type=int, default=256)
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fit-only", action="store_true", help="skip LeJEPA; leanmap only")
    ap.add_argument("--skip-leanmap", action="store_true")
    ap.add_argument(
        "--annotate-focus",
        action="store_true",
        help="only write estimated_focus_quality into --out zarr and exit",
    )
    ap.add_argument(
        "--soft-mask",
        action="store_true",
        help="soft circular mask + equal-channel signal contrast before training",
    )
    ap.add_argument(
        "--mask-radius-frac",
        type=float,
        default=SOFT_MASK_RADIUS_FRAC,
        dest="mask_radius_frac",
        help="effective radius as a fraction of half-box width (0.75 → 75%% diameter)",
    )
    ap.add_argument(
        "--focus-keep-frac",
        type=float,
        default=0.0,
        dest="focus_keep_frac",
        help="keep only this fraction of sharpest images (e.g. 0.25); 0 = keep all",
    )
    ap.add_argument("--leanmap-epochs", type=int, default=80)
    ap.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    ap.add_argument(
        "--lambda-geo",
        type=float,
        default=None,
        help="override leanmap lambda_geo (geodesic stress weight)",
    )
    ap.add_argument(
        "--d-out",
        type=int,
        default=2,
        dest="d_out",
        help="embedding dimensionality (2 or 3)",
    )
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument(
        "--holdout",
        type=float,
        default=0.1,
        help="calibration fraction of the in-manifold (embedding) set",
    )
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        dest="test_frac",
        help="downstream test fraction of the in-manifold set (held out of fit)",
    )
    ap.add_argument(
        "--n-landmarks",
        type=int,
        default=0,
        dest="n_landmarks",
        help="override leanmap n_landmarks (0 = default_config)",
    )
    ap.add_argument(
        "--pyramid-level-weights",
        type=float,
        nargs="+",
        default=None,
        dest="pyramid_level_weights",
        help="per-level attraction weights, finest first (e.g. 1 1 1)",
    )
    ap.add_argument(
        "--exclude-phases",
        nargs="*",
        default=None,
        help=(
            "phase names excluded from embedding / used as novelty probes "
            f"(default: none; pass {' '.join(_MITOTIC)} to hold out mitosis)"
        ),
    )
    ap.add_argument(
        "--include-idx",
        type=Path,
        default=None,
        dest="include_idx_path",
        help=(
            ".npy of zarr indices to fold into the embedding train set "
            "(e.g. conformal novel cells pulled out of --exclude-phases)"
        ),
    )
    ap.add_argument("--frame-every", type=int, default=1)
    ap.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="subsample this many cells for leanmap (0 = all features)",
    )
    ap.add_argument(
        "--class-token",
        action="store_true",
        help="inject learnable per-phase token into projector path + CE on encoder",
    )
    ap.add_argument(
        "--lambda-cls",
        type=float,
        default=0.5,
        help="weight for classification CE when --class-token is set",
    )
    ap.add_argument(
        "--train-idx",
        type=Path,
        default=None,
        dest="train_idx_path",
        help=(
            ".npy of zarr/row indices to use for LeJEPA optimisation; "
            "features are still exported for every cell"
        ),
    )
    ap.add_argument(
        "--exclude-phases-train",
        nargs="*",
        default=None,
        dest="exclude_phases_train",
        help=(
            "phase names excluded from LeJEPA fit when --train-idx is not set "
            f"(e.g. {' '.join(_MITOTIC)})"
        ),
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint .pt to resume weights/optimizer from",
    )
    ap.add_argument(
        "--continue-from-out",
        action="store_true",
        help="resume from <out>.pt if present",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=0,
        help="write checkpoints every N epochs (0 = only final)",
    )
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="directory for periodic checkpoints (default: <out>_ckpts)",
    )
    args = ap.parse_args()

    # default separate outputs when using class tokens (don't clobber SSL run)
    if args.class_token and args.out == DEFAULT_OUT:
        args.out = ROOT / "examples" / "out" / "cellcycle_lejepa_cls.zarr"
    if args.class_token and args.run_dir == DEFAULT_RUN:
        args.run_dir = OUT_DIR / "cellcycle_lejepa_cls"

    zarr_path = args.out.expanduser().resolve()
    if args.annotate_focus:
        if not zarr_path.exists():
            raise FileNotFoundError(zarr_path)
        annotate_focus_quality(zarr_path)
        return
    if not args.fit_only:
        train_lejepa(args)
    elif not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    if not args.skip_leanmap:
        fit_leanmap(args, zarr_path)


if __name__ == "__main__":
    main()

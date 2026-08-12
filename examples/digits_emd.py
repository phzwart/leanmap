#!/usr/bin/env python
"""leanmap on 8x8 digits with an L1-tree → torchemd EMD-rescored kNN.

Mirrors the production pattern in torchemd's ``emd_knn_graph`` /
``rescore_graph_umap`` examples:

1. unit-mass flatten → L1 kNN with ``M ≫ k`` candidates (cheap tree)
2. rescore those edges with NNS-EMD on spatial histograms
3. prune to top-``k`` by EMD
4. fit leanmap with ``precomputed_knn=(idx, dist)``; ambient metric stays L1
   for landmarks / assignment

During training, intermediate embeddings are written under
``out/digits_emd/frames/`` (``.npy`` + scatter PNG) and ``live.png`` is
refreshed each snapshot. At the end, ``frames.npz`` stacks all snapshots.

Requires: ``torchemd``, ``scikit-learn``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors

from _demo import OUT_DIR, default_config, save_scatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _unit_mass_flat(imgs: np.ndarray) -> np.ndarray:
    flat = imgs.reshape(len(imgs), -1).astype(np.float64)
    total = flat.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (flat / total).astype(np.float32)


def _cap_per_class(
    X: np.ndarray, y: np.ndarray, imgs: np.ndarray, n_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        keep.append(idx[:n_class])
    keep = np.concatenate(keep)
    return X[keep], y[keep], imgs[keep]


def build_l1_emd_knn(
    imgs: np.ndarray,
    *,
    k: int,
    M: int,
    batch_size: int = 8192,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """L1 candidate tree → NNS-EMD rescore → prune to top-k."""
    from torchemd import image_to_histogram
    from torchemd.graph import rescore_neighbors

    if M < k:
        raise ValueError(f"candidate width M={M} must be >= k={k}")
    flat = _unit_mass_flat(imgs)
    nn = NearestNeighbors(n_neighbors=M + 1, metric="minkowski", p=1).fit(flat)
    _, cand = nn.kneighbors(flat)
    cand = cand[:, 1 : M + 1]  # drop self

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
        save_scatter(
            Z,
            self.y,
            title=f"leanmap digits EMD — epoch {epoch}/{self.total}",
            path=self.frame_dir / f"{tag}.png",
            cmap="tab10",
            colorbar_label="digit",
        )
        save_scatter(
            Z,
            self.y,
            title=f"leanmap digits EMD — epoch {epoch}/{self.total}",
            path=self.out_dir / "live.png",
            cmap="tab10",
            colorbar_label="digit",
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
        )
        return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-class",
        type=int,
        default=None,
        help="optional cap on samples per digit class",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=15, help="final EMD neighbors kept")
    ap.add_argument(
        "--M",
        type=int,
        default=None,
        help="L1 candidate width (default: max(30, 3*k))",
    )
    ap.add_argument("--holdout", type=float, default=0.1, help="calib fraction")
    ap.add_argument("--min-dist", type=float, default=None, dest="min_dist")
    ap.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="write embedding snapshot every N epochs (0 disables)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="digits_emd",
        help="output subdirectory under examples/out/",
    )
    args = ap.parse_args()

    data = load_digits()
    imgs = data.images.astype(np.float32)
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    if args.n_class is not None:
        X, y, imgs = _cap_per_class(X, y, imgs, args.n_class, args.seed)

    rng = np.random.default_rng(args.seed)
    n = len(X)
    n_cal = max(1, int(round(args.holdout * n)))
    perm = rng.permutation(n)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    X_train, X_cal = X[train_idx], X[cal_idx]
    y_train = y[train_idx]
    imgs_train = imgs[train_idx]

    k = int(args.k)
    M = int(args.M) if args.M is not None else max(30, 3 * k)
    print(f"building L1 tree (M={M}) → EMD rescore → prune k={k} on N={len(X_train)}")
    knn_idx, knn_dist = build_l1_emd_knn(
        imgs_train, k=k, M=M, device=args.device
    )
    assert knn_idx.shape == (len(X_train), k)
    assert np.isfinite(knn_dist).all()
    assert not (knn_idx == np.arange(len(X_train))[:, None]).any()
    print(
        f"EMD kNN ready: median nn dist={float(np.median(knn_dist[:, 0])):.4f} "
        f"mean={float(knn_dist.mean()):.4f}"
    )

    from leanmap import fit

    run_dir = OUT_DIR / str(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    # clear prior frames from an interrupted run
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("epoch_*"):
        p.unlink(missing_ok=True)

    cfg = default_config(len(X_train), epochs=args.epochs)
    cfg.seed = int(args.seed)
    cfg.dedup = False
    cfg.n_neighbors = k
    if args.device is not None:
        cfg.device = args.device
    if args.min_dist is not None:
        cfg.min_dist = float(args.min_dist)

    recorder = None
    callbacks = None
    if int(args.frame_every) > 0:
        recorder = EmbeddingRecorder(
            X_train,
            y_train,
            out_dir=run_dir,
            every=int(args.frame_every),
            total_epochs=int(args.epochs),
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
    title = f"leanmap — digits EMD ({args.out})"
    final_png = save_scatter(
        Z,
        y_train,
        title=title,
        path=run_dir / "final.png",
        cmap="tab10",
        colorbar_label="digit",
    )
    # legacy top-level aliases only for the default run name
    if str(args.out) == "digits_emd":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result.save(str(OUT_DIR / "digits_emd.pt"))
        save_scatter(
            Z,
            y_train,
            title="leanmap — digits (L1→EMD kNN)",
            path=OUT_DIR / "digits_emd.png",
            cmap="tab10",
            colorbar_label="digit",
        )
    np.save(run_dir / "Z_final.npy", Z.astype(np.float32))
    np.save(run_dir / "y_train.npy", y_train)

    frames_npz = recorder.finalize() if recorder else None
    print(f"N_train={len(X_train)} N_cal={len(X_cal)} d={X_train.shape[1]} -> {Z.shape}")
    print(f"min_dist={result.config.min_dist} knn=precomputed EMD k={k} M={M}")
    print(f"saved {model_path}")
    print(f"saved {final_png}")
    if frames_npz is not None:
        print(f"saved {len(recorder.Zs)} intermediate embeddings -> {frames_npz}")
        print(f"frames dir: {recorder.frame_dir}")
        print(f"live: {run_dir / 'live.png'}")


if __name__ == "__main__":
    main()

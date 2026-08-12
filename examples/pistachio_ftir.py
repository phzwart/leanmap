#!/usr/bin/env python
"""leanmap on pistachio-stem FTIR spectra (cosine similarity).

Loads ``GD_Pistachio_Stem_Ctr_whole_10um__ftir.zarr``, optionally converts
% transmittance → absorbance, fits leanmap with ``dist_fn="cosine"``, and
writes per-epoch embedding snapshots under ``out/<run>/frames/`` plus a
refreshed ``live.png`` every epoch (default ``--frame-every 1``).

Requires: ``zarr``, ``scikit-learn`` (not used for knn — leanmap/faiss builds
the cosine graph after L2 normalisation).
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

from _demo import OUT_DIR, default_config, save_scatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ZARR = ROOT / "examples" / "GD_Pistachio_Stem_Ctr_whole_10um__ftir.zarr"
DEFAULT_OUT = "pistachio_ftir_cosine"


def load_spectra(
    zarr_path: Path,
    *,
    absorbance: bool,
    max_n: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return spectra, wavenumbers, x, y, color (band index for scatter)."""
    root = zarr.open_group(str(zarr_path), mode="r")
    spectra = np.asarray(root["spectra"][:], dtype=np.float32)
    wn = np.asarray(root["wavenumbers"][:], dtype=np.float64)
    x = np.asarray(root["x"][:], dtype=np.float64)
    y = np.asarray(root["y"][:], dtype=np.float64)

    if max_n is not None and max_n < len(spectra):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(spectra), size=int(max_n), replace=False))
        spectra, x, y = spectra[keep], x[keep], y[keep]

    if absorbance:
        # %T → absorbance; floor avoids log(0)
        t = np.clip(spectra, 1e-3, None) / 100.0
        spectra = (-np.log10(t)).astype(np.float32)

    # Colour by CH₂ asymmetric stretch (~2920 cm⁻¹) — strong plant-tissue band.
    i2920 = int(np.argmin(np.abs(wn - 2920.0)))
    color = spectra[:, i2920].astype(np.float32)
    return spectra, wn, x, y, color


class EmbeddingRecorder:
    """Write per-epoch embeddings + scatter frames; refresh ``live.png``."""

    def __init__(
        self,
        X: np.ndarray,
        color: np.ndarray,
        *,
        out_dir: Path,
        every: int,
        total_epochs: int,
        colorbar_label: str,
    ) -> None:
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.color = np.asarray(color, dtype=np.float32)
        self.every = max(1, int(every))
        self.total = int(total_epochs)
        self.out_dir = Path(out_dir)
        self.frame_dir = self.out_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.width = max(4, len(str(self.total)))
        self.colorbar_label = colorbar_label
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
        title = f"leanmap FTIR cosine — epoch {epoch}/{self.total}"
        save_scatter(
            Z,
            self.color,
            title=title,
            path=self.frame_dir / f"{tag}.png",
            cmap="viridis",
            colorbar_label=self.colorbar_label,
        )
        save_scatter(
            Z,
            self.color,
            title=title,
            path=self.out_dir / "live.png",
            cmap="viridis",
            colorbar_label=self.colorbar_label,
        )

    def finalize(self) -> Path | None:
        if not self.Zs:
            return None
        path = self.out_dir / "frames.npz"
        np.savez_compressed(
            path,
            epochs=np.asarray(self.epochs, dtype=np.int32),
            Z=np.stack(self.Zs),
            color=self.color,
        )
        return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=15, help="kNN neighbors")
    ap.add_argument("--holdout", type=float, default=0.05, help="calib fraction")
    ap.add_argument("--min-dist", type=float, default=None, dest="min_dist")
    ap.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="write embedding snapshot every N epochs (0 disables)",
    )
    ap.add_argument(
        "--no-absorbance",
        action="store_true",
        help="keep %% transmittance instead of converting to absorbance",
    )
    ap.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="optional random subset for smoke tests",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="output subdirectory under examples/out/",
    )
    args = ap.parse_args()

    zarr_path = args.zarr.expanduser().resolve()
    if not zarr_path.exists():
        raise SystemExit(f"missing zarr: {zarr_path}")

    absorbance = not args.no_absorbance
    print(f"loading {zarr_path.name} (absorbance={absorbance})")
    X, wn, x_coord, y_coord, color = load_spectra(
        zarr_path,
        absorbance=absorbance,
        max_n=args.max_n,
        seed=args.seed,
    )
    color_label = "A(2920 cm⁻¹)" if absorbance else "%T(2920 cm⁻¹)"

    rng = np.random.default_rng(args.seed)
    n = len(X)
    n_cal = max(1, int(round(args.holdout * n)))
    perm = rng.permutation(n)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    X_train, X_cal = X[train_idx], X[cal_idx]
    color_train = color[train_idx]
    x_train, y_train = x_coord[train_idx], y_coord[train_idx]

    from leanmap import fit

    run_dir = OUT_DIR / str(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("epoch_*"):
        p.unlink(missing_ok=True)
    progress = run_dir / "progress.csv"
    if progress.exists():
        progress.unlink()

    k = int(args.k)
    cfg = default_config(len(X_train), epochs=int(args.epochs))
    cfg.seed = int(args.seed)
    cfg.dedup = False  # spectra are continuous; ε-net is wasted cost at this N
    # Full pyramid coarsens via FPS over all N reps — ~minutes on N≈70k×3319.
    # Single-scale cosine kNN is enough for a first spectral embedding.
    cfg.pyramid_scales = 0
    cfg.pyramid_level_weights = None
    cfg.pyramid_coarse_backbone = 0.0
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
            color_train,
            out_dir=run_dir,
            every=int(args.frame_every),
            total_epochs=int(args.epochs),
            colorbar_label=color_label,
        )
        callbacks = [recorder]

    print(
        f"leanmap fit: N_train={len(X_train)} N_cal={len(X_cal)} "
        f"d={X_train.shape[1]} metric=cosine k={k} epochs={args.epochs} "
        f"frame_every={args.frame_every}"
    )
    result = fit(
        X_train,
        dist_fn="cosine",
        config=cfg,
        X_calib=X_cal,
        callbacks=callbacks,
    )
    with torch.no_grad():
        Z, _ = result.embed(X_train)
    Z = Z.detach().cpu().numpy()

    model_path = run_dir / "model.pt"
    result.save(str(model_path))
    title = f"leanmap — pistachio FTIR cosine ({args.out})"
    final_png = save_scatter(
        Z,
        color_train,
        title=title,
        path=run_dir / "final.png",
        cmap="viridis",
        colorbar_label=color_label,
    )
    # spatial colouring of the same embedding
    save_scatter(
        Z,
        x_train,
        title=f"{title} — coloured by x",
        path=run_dir / "final_by_x.png",
        cmap="coolwarm",
        colorbar_label="x (OMNIC)",
    )
    save_scatter(
        Z,
        y_train,
        title=f"{title} — coloured by y",
        path=run_dir / "final_by_y.png",
        cmap="coolwarm",
        colorbar_label="y (OMNIC)",
    )

    np.save(run_dir / "Z_final.npy", Z.astype(np.float32))
    np.save(run_dir / "train_idx.npy", train_idx.astype(np.int64))
    np.save(run_dir / "color_train.npy", color_train)
    np.save(run_dir / "xy_train.npy", np.stack([x_train, y_train], axis=1))
    meta = {
        "zarr": str(zarr_path),
        "metric": "cosine",
        "absorbance": absorbance,
        "k": k,
        "epochs": int(args.epochs),
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "n_bands": int(X_train.shape[1]),
        "wavenumber_range": [float(wn[0]), float(wn[-1])],
        "frame_every": int(args.frame_every),
        "color_band_cm": 2920.0,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    frames_npz = recorder.finalize() if recorder else None
    print(f"N_train={len(X_train)} N_cal={len(X_cal)} d={X_train.shape[1]} -> {Z.shape}")
    print(f"min_dist={result.config.min_dist} metric=cosine k={k}")
    print(f"saved {model_path}")
    print(f"saved {final_png}")
    if frames_npz is not None:
        print(f"saved {len(recorder.Zs)} intermediate embeddings -> {frames_npz}")
        print(f"frames dir: {recorder.frame_dir}")
        print(f"live: {run_dir / 'live.png'}")


if __name__ == "__main__":
    main()

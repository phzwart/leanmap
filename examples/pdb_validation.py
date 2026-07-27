#!/usr/bin/env python
"""leanmap on PDB X-ray validation metrics (resolution-colored epochs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _demo import OUT_DIR, fit_embed, save_scatter

DATA = Path(__file__).resolve().parent / "data" / "pdb_xray_reslt2_5k.csv"

# Features for the embedding (resolution is color only, not a coordinate).
FEATURE_COLS = [
    "r_work",
    "r_free",
    "clashscore",
    "angles_rmsz",
    "bonds_rmsz",
    "percent_ramachandran_outliers",
    "percent_rotamer_outliers",
    "percent_rsrz_outliers",
    "data_completeness",
]


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    for col in ["pdb_id", "resolution", *FEATURE_COLS]:
        if col not in df.columns:
            raise SystemExit(f"missing column {col!r} in {path}")
    X_raw = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    # Fill rare blanks (e.g. RSRZ) with column median so N stays 5k.
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        missing = ~np.isfinite(col)
        if missing.any():
            med = float(np.nanmedian(col))
            col[missing] = med
            X_raw[:, j] = col
    # Per-channel min–max → [0, 1] so R-factors / clashscore / % outliers
    # contribute on a common scale under L2.
    lo = X_raw.min(axis=0)
    hi = X_raw.max(axis=0)
    span = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
    X = ((X_raw - lo) / span).astype(np.float32)
    resolution = df["resolution"].to_numpy(dtype=np.float32)
    pdb_ids = df["pdb_id"].to_numpy()
    return X, resolution, pdb_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3, help="learning rate (default 5× base 1e-3)")
    ap.add_argument("--n-landmarks", type=int, default=500)
    ap.add_argument("--tau-init", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR / "pdb_validation_epochs",
        help="directory for per-epoch PNGs",
    )
    args = ap.parse_args()

    X, resolution, pdb_ids = load_table(args.csv)
    print(f"loaded {args.csv}: N={len(X)} d={X.shape[1]} features={FEATURE_COLS}")
    print(
        f"resolution: {resolution.min():.3f}–{resolution.max():.3f} Å "
        f"(mean {resolution.mean():.3f})"
    )

    epoch_dir = Path(args.out_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for p in epoch_dir.glob("epoch_*.png"):
        p.unlink()

    X_t = torch.as_tensor(X)

    def on_epoch(epoch: int, model, metrics: dict) -> None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            Z_ep, _ = model.embed(X_t, return_score=False)
        if was_training:
            model.train()
        ret = metrics.get("retention", float("nan"))
        path = save_scatter(
            Z_ep.detach().cpu().numpy(),
            resolution,
            title=f"PDB X-ray validation  epoch {epoch:03d}  ret={ret:.3f}",
            path=epoch_dir / f"epoch_{epoch:03d}.png",
            cmap="viridis_r",  # high-res (good) → yellow/green; low-res → purple
            colorbar_label="resolution (Å)",
        )
        print(f"epoch {epoch:03d}: saved {path.name}  retention={ret:.3f}")

    result, Z, _ = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        callbacks=[on_epoch],
        n_landmarks=args.n_landmarks,
        tau_init=args.tau_init,
        learn_tau=False,
        pyramid_level_weights=(1.0, 1.0, 2.0, 4.0),
        lr=args.lr,
    )
    final = save_scatter(
        Z,
        resolution,
        title=f"PDB X-ray validation  final ({args.epochs} ep)",
        path=OUT_DIR / "pdb_validation.png",
        cmap="viridis_r",
        colorbar_label="resolution (Å)",
    )
    meta = OUT_DIR / "pdb_validation_meta.npz"
    np.savez_compressed(meta, Z=Z, resolution=resolution, pdb_id=pdb_ids, X=X)

    print(f"N={len(X)} d={X.shape[1]} -> embedding {Z.shape}")
    print(
        f"pyramid_scales={result.config.pyramid_scales} "
        f"level_weights={result.config.pyramid_level_weights}"
    )
    print(f"epoch PNGs → {epoch_dir}/epoch_XXX.png")
    print(f"final → {final}")
    print(f"meta → {meta}")


if __name__ == "__main__":
    main()

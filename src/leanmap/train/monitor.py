"""Training callbacks: latent plots and epoch checkpoints."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from ..utils import get_logger


def _save_scatter(
    Z: np.ndarray,
    color: np.ndarray,
    *,
    title: str,
    path: Path,
    cmap: str = "tab10",
    colorbar_label: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Z = np.asarray(Z, dtype=np.float32)
    d = int(Z.shape[1]) if Z.ndim == 2 else 1
    if d >= 3:
        fig = plt.figure(figsize=(6.0, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            Z[:, 2],
            c=color,
            s=3,
            cmap=cmap,
            linewidths=0,
            alpha=0.85,
            depthshade=True,
        )
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")
        ax.set_zlabel("z2")
        ax.set_title(title)
        # Drop tick clutter; keep axis labels for 3D orientation.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        x = Z[:, 0] if d >= 1 else np.zeros(len(Z))
        y = Z[:, 1] if d >= 2 else np.zeros(len(Z))
        sc = ax.scatter(x, y, c=color, s=4, cmap=cmap, linewidths=0, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


class TrainMonitor:
    """Latent scatter every ``plot_every`` steps; checkpoint every ``ckpt_every`` epochs.

    Callables:
    - ``on_step(epoch, global_step, model, info)`` — optional, used by ``fit``
    - ``__call__(epoch, model, metrics)`` — end-of-epoch hook used by ``fit``
    """

    def __init__(
        self,
        X: np.ndarray,
        *,
        out_dir: Union[str, Path],
        plot_every: int = 10,
        ckpt_every: int = 1,
        max_plot_n: int = 5000,
        color: Optional[np.ndarray] = None,
        colorbar_label: str = "",
        seed: int = 0,
        config: Any = None,
        X_calib: Optional[np.ndarray] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.frame_dir = self.out_dir / "frames"
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every = max(0, int(plot_every))
        self.ckpt_every = max(0, int(ckpt_every))
        self.config = config
        self.log = get_logger()
        self.csv_path = self.out_dir / "progress.csv"
        self._csv_cols: Optional[list[str]] = None
        self.X_calib: Optional[torch.Tensor] = None
        if X_calib is not None:
            self.set_calib(X_calib)

        X = np.asarray(X, dtype=np.float32)
        n = int(X.shape[0])
        rng = np.random.default_rng(int(seed))
        if n > int(max_plot_n):
            idx = np.sort(rng.choice(n, size=int(max_plot_n), replace=False))
        else:
            idx = np.arange(n, dtype=np.int64)
        self.plot_idx = idx.astype(np.int64)
        self.X_plot = torch.as_tensor(X[self.plot_idx], dtype=torch.float32)
        if color is None:
            self.color = np.zeros(len(self.plot_idx), dtype=np.float32)
            self.colorbar_label = colorbar_label or "const"
            self.cmap = "viridis"
        else:
            c = np.asarray(color).reshape(-1)
            if c.shape[0] != n:
                raise ValueError(f"color length {c.shape[0]} != X rows {n}")
            self.color = c[self.plot_idx].astype(np.float32)
            self.colorbar_label = colorbar_label or "color"
            nuniq = int(np.unique(self.color).size)
            self.cmap = "tab10" if nuniq <= 10 else "viridis"

    def set_calib(self, X_calib: Union[np.ndarray, torch.Tensor]) -> None:
        """Attach the train-time conformal calibration matrix (raw ambient rows)."""
        self.X_calib = torch.as_tensor(
            np.asarray(X_calib, dtype=np.float32), dtype=torch.float32
        )

    def _embed(self, model) -> np.ndarray:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        chunks = []
        with torch.no_grad():
            xb = self.X_plot
            for s in range(0, xb.shape[0], 8192):
                z, _ = model.embed(xb[s : s + 8192].to(device), return_score=False)
                chunks.append(z.detach().cpu())
        if was_training:
            model.train()
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)

    def _plot(self, Z: np.ndarray, *, title: str, stem: str) -> None:
        _save_scatter(
            Z,
            self.color,
            title=title,
            path=self.frame_dir / f"{stem}.png",
            cmap=self.cmap,
            colorbar_label=self.colorbar_label,
        )
        _save_scatter(
            Z,
            self.color,
            title=title,
            path=self.out_dir / "live.png",
            cmap=self.cmap,
            colorbar_label=self.colorbar_label,
        )
        np.save(self.frame_dir / f"{stem}.npy", Z)

    def on_step(self, epoch: int, global_step: int, model, info: dict) -> None:
        if self.plot_every <= 0:
            return
        if global_step % self.plot_every != 0:
            return
        Z = self._embed(model)
        stem = f"step_{global_step:06d}_ep{epoch:03d}"
        title = f"leanmap latent — step {global_step} (epoch {epoch})"
        self._plot(Z, title=title, stem=stem)
        self.log.info(
            "wrote latent plot %s (%d pts)",
            self.frame_dir / f"{stem}.png",
            int(Z.shape[0]),
        )

    def _log_metrics(self, epoch: int, metrics: Optional[dict]) -> None:
        row = {"epoch": int(epoch)}
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

    def _save_ckpt(self, path: Path, model, *, epoch: int, metrics: Optional[dict]) -> None:
        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": dict(metrics or {}),
        }
        if self.config is not None:
            try:
                payload["config"] = asdict(self.config)
            except Exception:  # noqa: BLE001
                payload["config"] = None
        # Match PLANEResult.save conformal fields when calib rows are available.
        enc = getattr(model, "encoder", None)
        aff = getattr(model, "affinity", None)
        if enc is not None:
            payload["D"] = int(enc.D)
            if hasattr(enc, "x_mean"):
                payload["x_mean"] = enc.x_mean.detach().cpu()
            if hasattr(enc, "x_std"):
                payload["x_std"] = enc.x_std.detach().cpu()
        if aff is not None and hasattr(aff, "M"):
            payload["L"] = int(aff.M.shape[0])
            payload["landmark_coordinates"] = aff.M.detach().cpu()
            if hasattr(aff, "log_tau"):
                payload["log_tau"] = aff.log_tau.detach().cpu()
            # Cover scores live in affinity.dist_fn units (often L2 / natural_scale).
            # Without these fields, loaders that rebuild with bare L2 mis-calibrate.
            dist_fn = getattr(aff, "dist_fn", None)
            if dist_fn is not None:
                payload["metric_name"] = getattr(dist_fn, "name", None) or "l2"
                payload["natural_scale"] = getattr(dist_fn, "natural_scale", None)
        if self.X_calib is not None and int(self.X_calib.shape[0]) > 0:
            from ..conformal import ConformalCalibrator

            was_training = model.training
            model.eval()
            cal = ConformalCalibrator()
            cal.fit(model, self.X_calib.to(next(model.parameters()).device))
            if was_training:
                model.train()
            payload["tau_embed"] = cal.tau_embed
            payload["s_calib"] = (
                None if cal.s_calib is None else cal.s_calib.detach().cpu()
            )
            payload["weight_hash"] = cal.weight_hash
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    def __call__(self, epoch: int, model, metrics) -> None:
        self._log_metrics(epoch, metrics)
        # Always refresh live latent at epoch end (even if plot_every skips).
        Z = self._embed(model)
        stem = f"epoch_{epoch:03d}"
        self._plot(
            Z,
            title=f"leanmap latent — epoch {epoch}",
            stem=stem,
        )
        if self.ckpt_every > 0 and (epoch % self.ckpt_every == 0):
            ckpt = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
            self._save_ckpt(ckpt, model, epoch=epoch, metrics=metrics)
            latest = self.out_dir / "checkpoint_latest.pt"
            self._save_ckpt(latest, model, epoch=epoch, metrics=metrics)
            self.log.info("wrote checkpoint %s", ckpt)

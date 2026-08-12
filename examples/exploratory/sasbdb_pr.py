#!/usr/bin/env python
"""leanmap on SASBDB P(r) profiles (pr_000..pr_099 from pr_profiles.parquet).

Each row stores a pair distance distribution resampled onto 100 bins spanning
r in [0, dmax], so bin index is already a relative coordinate r/dmax. The
stored ``pr_norm`` column integrates to 1 in physical r, which makes its
amplitude scale like 1/dmax; ``--normalize unit-sum`` (default) instead makes
each profile a distribution over the 100 relative-r bins, giving a scale-free
shape descriptor.

Frames are written *during* training: ``live.png`` is overwritten every epoch
(keep it open in the editor to watch the map form) alongside a numbered frame
and a row in ``progress.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent
_ROOT = _EXAMPLES.parent
for _p in (_EXAMPLES, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics_run import compute_metrics, write_json  # noqa: E402

from _demo import fit_embed, save_density, save_scatter, save_shepard  # noqa: E402

DEFAULT_PARQUET = Path.home() / "Desktop" / "pr_profiles.parquet"
DEFAULT_OUT = _ROOT / "runs" / "sasbdb_pr"
META_COLS = ("sasbdb_code", "dmax", "rg_pr", "rg_guinier", "length_unit")


def daemonize(log_path: Path, pid_path: Path) -> None:
    """Double-fork into a new session so the run outlives the launching shell.

    A long fit started from an agent/CI shell is otherwise killed when that
    shell's process group is torn down. Must be called before torch or
    matplotlib are imported, since forking a threaded runtime is unsafe.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    pid_path.write_text(f"{os.getpid()}\n")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)


def load_profiles(parquet: Path, column: str):
    import pyarrow.parquet as pq

    df = pq.read_table(parquet).to_pandas()
    P = np.stack(df[column].to_numpy()).astype(np.float64)
    return P, df


def normalize(P: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return P
    if mode == "unit-sum":
        return P / P.sum(axis=1, keepdims=True)
    if mode == "unit-max":
        return P / P.max(axis=1, keepdims=True)
    if mode == "unit-l2":
        return P / np.linalg.norm(P, axis=1, keepdims=True)
    raise ValueError(f"unknown normalize mode {mode!r}")


def quality_mask(P: np.ndarray, df) -> np.ndarray:
    """Drop profiles that cannot be a physical P(r).

    Beyond non-finite / all-zero rows this rejects negative densities and
    Rg/Dmax ratios outside the geometric range (a sphere sits at ~0.39, a thin
    rod approaches ~0.29 of its length; anything under 0.1 or over 0.6 means
    the stored Rg and Dmax disagree).
    """
    ok = np.isfinite(P).all(axis=1) & (P.sum(axis=1) > 0)
    ok &= P.min(axis=1) >= -1e-12
    ratio = df["rg_pr"].to_numpy(dtype=np.float64) / df["dmax"].to_numpy(dtype=np.float64)
    ok &= np.isfinite(ratio) & (ratio > 0.1) & (ratio < 0.6)
    return ok


def pca2(X: np.ndarray) -> np.ndarray:
    """2-D PCA scores, used as a fixed rotational gauge for the frames."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def procrustes(Z: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Center ``Z`` and rotate/reflect it onto ``ref`` (scale untouched)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Zc.T @ (ref - ref.mean(axis=0, keepdims=True)))
    return Zc @ (U @ Vt)


def square_limits(lo: np.ndarray, hi: np.ndarray, pad: float = 0.05):
    center = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) / 2.0, 1e-6) * (1.0 + pad)
    return (
        (center[0] - half, center[0] + half),
        (center[1] - half, center[1] + half),
    )


class FrameFigure:
    """One reusable matplotlib figure; redrawing is ~10x cheaper than rebuilding."""

    def __init__(self, color, label: str, cmap: str, n: int):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        # Full range, matching _demo.save_scatter, so the frames share a colour
        # scale with the final panels and with the UMAP baseline. Percentile
        # clipping here would make the same data look like a different variable.
        vmin, vmax = float(np.min(color)), float(np.max(color))
        self.fig, self.ax = plt.subplots(figsize=(5.5, 5.0))
        self.sc = self.ax.scatter(
            np.zeros(n),
            np.zeros(n),
            c=color,
            s=4 if n > 2000 else 6,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        cb = self.fig.colorbar(self.sc, ax=self.ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        self.fig.tight_layout()

    def draw(self, Z: np.ndarray, title: str, xlim, ylim, paths, dpi: int = 100) -> None:
        self.sc.set_offsets(Z)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_title(title)
        for p in paths:
            self.fig.savefig(p, dpi=dpi)

    def close(self) -> None:
        self.plt.close(self.fig)


class LiveRecorder:
    """Snapshot + render the training-set embedding as training runs."""

    def __init__(self, X, color, *, out_dir: Path, label: str, cmap: str,
                 every: int, total_epochs: int, dpi: int):
        import torch

        self._torch = torch
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.ref = pca2(np.asarray(X, dtype=np.float64))
        self.every = max(1, int(every))
        self.total = int(total_epochs)
        self.dpi = int(dpi)
        self.out_dir = out_dir
        self.frame_dir = out_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.width = max(4, len(str(self.total)))
        self.fig = FrameFigure(color, label, cmap, len(X))
        self.epochs: list[int] = []
        self.frames: list[np.ndarray] = []
        # Limits only ever grow, so the live view zooms out smoothly instead of
        # jumping around as the embedding expands.
        self.lo = None
        self.hi = None
        self.csv_path = out_dir / "progress.csv"
        self._csv_cols: list[str] | None = None

    def _limits(self, Z):
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        self.lo = lo if self.lo is None else np.minimum(self.lo, lo)
        self.hi = hi if self.hi is None else np.maximum(self.hi, hi)
        return square_limits(self.lo, self.hi)

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
        with self._torch.no_grad():
            Z, _ = model.embed(self.X, return_score=False)
        if was_training:
            model.train()
        Z = procrustes(Z.detach().cpu().numpy().astype(np.float64), self.ref)
        self.epochs.append(int(epoch))
        self.frames.append(Z.astype(np.float32))
        xlim, ylim = self._limits(Z)
        self.fig.draw(
            Z,
            f"epoch {epoch}/{self.total}",
            xlim,
            ylim,
            [
                self.frame_dir / f"epoch_{epoch:0{self.width}d}.png",
                self.out_dir / "live.png",
            ],
            dpi=self.dpi,
        )

    def rerender(self) -> None:
        """Second pass with the final gauge and one global set of limits."""
        if not self.frames:
            return
        final = self.frames[-1].astype(np.float64)
        allZ = np.concatenate(self.frames)
        xlim, ylim = square_limits(allZ.min(axis=0), allZ.max(axis=0))
        for i, (Z, ep) in enumerate(zip(self.frames, self.epochs)):
            Za = procrustes(Z.astype(np.float64), final)
            self.frames[i] = Za.astype(np.float32)
            self.fig.draw(
                Za,
                f"epoch {ep}/{self.total}",
                xlim,
                ylim,
                [self.frame_dir / f"epoch_{ep:0{self.width}d}.png"],
                dpi=self.dpi,
            )

    def write_gif(self, fps: int) -> None:
        try:
            from PIL import Image
        except ImportError:
            print("Pillow not available; skipping GIF")
            return
        paths = sorted(self.frame_dir.glob("epoch_*.png"))
        if not paths:
            return
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
        gif = self.out_dir / "evolution.gif"
        imgs[0].save(
            gif,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / max(1, fps)),
            loop=0,
        )
        print(f"wrote {gif}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--column", default="pr_norm", choices=("pr", "pr_norm"))
    ap.add_argument(
        "--normalize",
        default="unit-sum",
        choices=("unit-sum", "unit-max", "unit-l2", "raw"),
    )
    # pr_norm rows are discrete distributions on equal-width relative-r bins;
    # 1-D Wasserstein-1 (CDF L1) is the natural transport metric on that line.
    # L1 remains available as total-variation (up to a factor 2).
    ap.add_argument(
        "--metric",
        default="wasserstein1d",
        choices=(
            "wasserstein1d",
            "l1",
            "l2",
            "jensenshannon",
            "cosine",
            "correlation",
            "braycurtis",
        ),
    )
    # P(r) profiles sit in a narrow shell: the default temperature (mean distance
    # to the 32 nearest landmarks) leaves the softmax near-uniform, perplexity
    # ~110 of 128 landmarks, so conditioning carries no signal and retention sits
    # at chance. 0.12 brings it to the ~8 that calibrate.py targets.
    ap.add_argument("--tau-scale", type=float, default=None)
    ap.add_argument("--learn-tau", dest="learn_tau", action="store_true", default=None)
    ap.add_argument("--no-learn-tau", dest="learn_tau", action="store_false")
    # Learned landmarks drift off the data here: with tau frozen, mean distance
    # to the nearest landmark still climbs 1.3 -> 5.0 while usage collapses to
    # ~2 effective anchors. CONFIGURATION.md: "Freeze for stability at small N".
    ap.add_argument(
        "--learn-landmarks", dest="learn_landmarks", action="store_true", default=None
    )
    ap.add_argument("--no-learn-landmarks", dest="learn_landmarks", action="store_false")
    # Strength of the term correlating each neighbourhood's log radius with the
    # ambient graph's. 0 leaves density to the attraction/repulsion equilibrium,
    # whose min_dist default was measured on a uniform s-curve and has no claim on
    # this data. See leanmap.density.
    ap.add_argument("--lambda-density", type=float, default=1.0)
    # Reproducing a d-dimensional density field in the plane is not generally
    # possible; at d_out = 3 there is room for it. Frames and the 2-D scatter
    # panels are skipped above 2.
    ap.add_argument("--d-out", type=int, default=2)
    # Scaling handle for the mixture-of-experts tessellation. Unlicensed density
    # that comes from the tessellation must be organised at the scale of one cell,
    # i.e. N / n_landmarks points, so sweeping this moves the residual's
    # correlation length if the experts are the cause and leaves it alone if not.
    ap.add_argument("--n-landmarks", type=int, default=None)
    ap.add_argument("--min-dist", type=float, default=None, dest="min_dist")
    ap.add_argument("--lambda-geo", type=float, default=None, dest="lambda_geo")
    ap.add_argument("--n", type=int, default=0, help="random subsample size (0 = all)")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    # Per-epoch snapshots call model.embed() inside the training loop, which
    # stalls badly on the MPS backend; CPU is faster here at this problem size.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frame-every", type=int, default=1, help="0 disables frames")
    ap.add_argument(
        "--frame-color",
        default="mode_pos",
        choices=("dmax", "rg_over_dmax", "mode_pos", "skew"),
    )
    ap.add_argument("--frame-dpi", type=int, default=100)
    ap.add_argument("--no-rerender", action="store_true", help="keep the live frames as-is")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--no-filter", action="store_true", help="skip the P(r) sanity filter")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument(
        "--detach",
        action="store_true",
        help="daemonize so the run survives the launching shell",
    )
    args = ap.parse_args()

    if args.detach:
        out = Path(args.out)
        print(f"detaching; log -> {out.with_suffix('.log')}")
        daemonize(out.with_suffix(".log"), out.with_suffix(".pid"))

    P_all, df = load_profiles(args.parquet, args.column)
    n_total = len(P_all)
    if args.no_filter:
        keep = np.isfinite(P_all).all(axis=1) & (P_all.sum(axis=1) > 0)
    else:
        keep = quality_mask(P_all, df)
    P_all, df = P_all[keep], df.loc[keep].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    n = len(P_all) if args.n <= 0 else min(int(args.n), len(P_all))
    idx = rng.choice(len(P_all), size=n, replace=False) if n < len(P_all) else np.arange(n)
    idx.sort()
    X = normalize(P_all[idx], args.normalize).astype(np.float32)
    meta = df.loc[idx, [c for c in META_COLS if c in df.columns]].reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.csv").unlink(missing_ok=True)
    meta.to_csv(out / "meta.csv", index=False)

    dmax = meta["dmax"].to_numpy(dtype=np.float64)
    rg = meta["rg_pr"].to_numpy(dtype=np.float64)
    bins = np.linspace(0.0, 1.0, X.shape[1])
    w = X / X.sum(axis=1, keepdims=True)
    mode_pos = bins[np.argmax(X, axis=1)]
    mean_pos = w @ bins
    skew = w @ (bins**3) - 3 * mean_pos * (w @ (bins**2)) + 2 * mean_pos**3
    panels = [
        ("dmax", np.log10(dmax), "log10 Dmax", "viridis"),
        ("rg_over_dmax", rg / dmax, "Rg / Dmax", "coolwarm"),
        ("mode_pos", mode_pos, "peak position r/Dmax", "plasma"),
        ("skew", skew, "P(r) skewness (relative r)", "cividis"),
    ]

    print(
        f"{args.parquet.name}: {n_total} rows -> {len(P_all)} pass filter "
        f"(dropped {n_total - len(P_all)}) -> fitting {n} x {X.shape[1]} "
        f"[{args.column}, {args.normalize}, metric={args.metric}]\n"
        f"writing to {out}",
        flush=True,
    )

    recorder = None
    if args.frame_every and args.d_out != 2:
        print(f"d_out={args.d_out}: skipping per-epoch frames (2-D only)", flush=True)
    elif args.frame_every:
        _, color, label, cmap = next(p for p in panels if p[0] == args.frame_color)
        recorder = LiveRecorder(
            X,
            color,
            out_dir=out,
            label=label,
            cmap=cmap,
            every=args.frame_every,
            total_epochs=args.epochs,
            dpi=args.frame_dpi,
        )

    result, Z, score = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        metric=args.metric,
        tau_scale=args.tau_scale,
        learn_tau=args.learn_tau,
        learn_landmarks=args.learn_landmarks,
        callbacks=[recorder] if recorder else None,
        lambda_density=args.lambda_density,
        d_out=args.d_out,
        min_dist=args.min_dist,
        lambda_geo=args.lambda_geo,
        **({} if args.n_landmarks is None else {"n_landmarks": args.n_landmarks}),
    )

    np.save(out / "Z.npy", Z)
    np.save(out / "X.npy", X)
    for name, color, label, cmap in panels if args.d_out == 2 else []:
        save_scatter(
            Z,
            color,
            title=f"leanmap — SASBDB P(r), N={n} ({label})",
            path=out / f"scatter_{name}.png",
            cmap=cmap,
            colorbar_label=label,
        )
    if args.d_out == 2:
        save_density(
            Z, title=f"leanmap — SASBDB P(r) density, N={n}", path=out / "density.png"
        )

    if recorder and recorder.frames:
        if not args.no_rerender:
            recorder.rerender()
        np.savez_compressed(
            out / "frames.npz",
            epochs=np.asarray(recorder.epochs),
            Z=np.stack(recorder.frames),
        )
        if not args.no_movie:
            recorder.write_gif(args.fps)
        recorder.fig.close()
        print(f"{len(recorder.frames)} frames -> {out / 'frames'}", flush=True)

    from leanmap.evaluate import shepard_pairs_ambient
    from leanmap.metrics import get_metric

    d_orig, d_embed = shepard_pairs_ambient(
        X, Z, n_pairs=32768, seed=args.seed, dist_fn=get_metric(args.metric).fn
    )
    save_shepard(
        d_orig,
        d_embed,
        title=f"Shepard (ambient {args.metric}) — SASBDB P(r), N={n}",
        path=out / "shepard_ambient.png",
    )

    metrics = compute_metrics(X, Z, seed=args.seed, metric=args.metric)
    metrics.update(
        n=n,
        n_rows=n_total,
        n_filtered=int(n_total - len(P_all)),
        d_in=int(X.shape[1]),
        column=args.column,
        normalize=args.normalize,
        epochs=int(args.epochs),
        seed=int(args.seed),
        score_mean=float(np.mean(score)),
        score_p95=float(np.percentile(score, 95)),
        spearman_dim1_logdmax=float(
            np.corrcoef(
                np.argsort(np.argsort(Z[:, 0])), np.argsort(np.argsort(np.log10(dmax)))
            )[0, 1]
        ),
    )
    write_json(out / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Minimum-norm perturbation that lands a flagged point back inside the support.

The encoder is frozen; only the input moves. For leanmap this does not need an
optimiser, because the acceptance region has a shape that can be projected onto
in closed form. The score is ``cover(x) = min_l ||x - M_l|| / s`` -- an ambient
Euclidean distance to the nearest landmark, up to a fixed scale -- so

    {x : cover(x) <= tau}  =  union of balls B(M_l, tau*s)

and the Euclidean projection onto a union of balls is the projection onto the
nearest one: move straight at the nearest landmark until you touch its surface.
That is the exact global minimiser of ``||delta||``, not a local optimum:

    delta* = (1 - tau*s/d) (M_l* - x),      ||delta*|| = d - tau*s,  d = ||x - M_l*||

Written as a blend it is ``x_repaired = (tau*s/d) x + (1 - tau*s/d) M_l*`` -- a
convex combination of the flagged point and the nearest training exemplar, which
is worth staring at, because it means the repair cannot invent anything.

Passing the test the repair was built to satisfy is therefore guaranteed and
proves nothing. The question worth asking is whether the repaired image is
actually near real data, so it is scored against the EMD reference -- a metric
neither the model nor the repair has any access to.

Usage::

    python examples/exploratory/manifold_repair.py \\
      --run examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --X examples/exploratory/data/digits_X.npy \\
      --probes examples/exploratory/data/digits_probes_X.npy \\
      --probe-kind examples/exploratory/data/digits_probes_kind.npy \\
      --out examples/out/exploratory/digits_emd/repair.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from splits import load_split


def conformal_threshold(calib: np.ndarray, alpha: float) -> float:
    """Largest score that still passes ``p = (1+#{calib>=s})/(n+1) > alpha``.

    Solving for the count: ``p > alpha`` needs ``#{calib >= s} > alpha(n+1) - 1``,
    so the threshold is the ``k``-th largest calibration score for that count.
    """
    n = len(calib)
    k = int(np.floor(alpha * (n + 1) - 1.0)) + 1
    k = max(1, min(k, n))
    return float(np.sort(calib)[::-1][k - 1])


def pvalues(scores: np.ndarray, calib: np.ndarray) -> np.ndarray:
    s_calib = np.sort(calib)
    n_ge = len(s_calib) - np.searchsorted(s_calib, scores, side="left")
    return (1.0 + n_ge) / (len(s_calib) + 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="leanmap run dir with model.pt")
    ap.add_argument("--X", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--probe-kind", default=None)
    ap.add_argument("--image-shape", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--no-emd", action="store_true", help="skip the independent check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    from leanmap import load_plane

    X = np.load(args.X).astype(np.float32)
    P = np.load(args.probes).astype(np.float32)
    kinds = (
        np.asarray([str(v) for v in np.load(args.probe_kind, allow_pickle=True)])[: len(P)]
        if args.probe_kind
        else np.array(["probe"] * len(P))
    )
    train_idx, hold_idx = load_split(args.run, n=len(X))

    model = load_plane(args.run / "model.pt", device="cpu")
    model.eval()
    M = model.affinity.M.detach().cpu().numpy().astype(np.float64)
    scale = float(model.affinity.dist_fn.natural_scale)

    with torch.no_grad():
        _, cover_all = model.embed(torch.as_tensor(X))
        _, cover_p = model.embed(torch.as_tensor(P))
    cover_all = cover_all.numpy().astype(np.float64)
    cover_p = cover_p.numpy().astype(np.float64)

    rng = np.random.default_rng(args.seed)
    hold = np.asarray(hold_idx).copy()
    rng.shuffle(hold)
    cut = len(hold) // 2
    calib_idx, test_idx = hold[:cut], hold[cut:]
    calib = cover_all[calib_idx]
    tau = conformal_threshold(calib, args.alpha)

    # ---- the projection, in closed form -------------------------------------
    Pd = P.astype(np.float64)
    d_all = np.linalg.norm(Pd[:, None, :] - M[None, :, :], axis=2)
    nearest = d_all.argmin(axis=1)
    d = d_all[np.arange(len(Pd)), nearest]
    # A hair inside the surface rather than exactly on it: the model round-trips
    # in float32, and landing on the boundary leaves a third of the points a few
    # ulp outside it once re-scored.
    radius = tau * scale * (1.0 - 1e-3)
    keep = np.clip(radius / d, 0.0, 1.0)  # weight left on the original point
    blend = 1.0 - keep  # how far it is dragged toward the landmark
    R = keep[:, None] * Pd + blend[:, None] * M[nearest]
    delta = R - Pd

    with torch.no_grad():
        _, cover_r = model.embed(torch.as_tensor(R.astype(np.float32)))
    cover_r = cover_r.numpy().astype(np.float64)
    p_before, p_after = pvalues(cover_p, calib), pvalues(cover_r, calib)

    print(f"tau at alpha={args.alpha}: {tau:.4f} (cover units), radius {radius:.3f} in pixel units")
    print(f"calibration n={len(calib)}, holdout test n={len(test_idx)}\n")
    print(f"{'':<34}{'before':>12}{'after':>12}")
    print(f"{'cover score (median)':<34}{np.median(cover_p):>12.3f}{np.median(cover_r):>12.3f}")
    print(f"{'conformal p (median)':<34}{np.median(p_before):>12.4f}{np.median(p_after):>12.4f}")
    print(f"{'flagged at alpha':<34}{np.mean(p_before <= args.alpha):>11.0%}{np.mean(p_after <= args.alpha):>12.0%}")

    rel = np.linalg.norm(delta, axis=1) / np.linalg.norm(Pd, axis=1)
    print(f"\n{'perturbation ||delta|| / ||x||':<34}median {np.median(rel):.3f}  "
          f"[{np.percentile(rel, 10):.3f}, {np.percentile(rel, 90):.3f}]")
    print(f"{'blend weight on the landmark':<34}median {np.median(blend):.3f}  "
          f"[{np.percentile(blend, 10):.3f}, {np.percentile(blend, 90):.3f}]")

    # ---- the independent check ----------------------------------------------
    emd_stats = None
    if not args.no_emd:
        from leanmap.emd import pairwise_emd

        tr = np.asarray(train_idx)
        stack = np.concatenate([X[tr], X[test_idx], P, R.astype(np.float32)], axis=0)
        n_tr, n_te, n_p = len(tr), len(test_idx), len(P)
        q = np.arange(n_tr, len(stack))
        print(f"\ncomputing EMD for {len(q)} query images against {len(stack)} "
              f"({len(q) * len(stack) / 1e6:.2f}M pairs)...")
        D = pairwise_emd(stack, tuple(args.image_shape), query_idx=q,
                         n_jobs=args.n_jobs, progress=False)
        to_train = D[:, :n_tr].min(axis=1)
        emd_stats = {
            "holdout digit": to_train[:n_te],
            "probe": to_train[n_te : n_te + n_p],
            "repaired": to_train[n_te + n_p :],
        }
        print(f"\n{'EMD to the nearest training digit':<34}{'median':>10}{'p90':>10}{'x a real digit':>17}")
        base = float(np.median(emd_stats["holdout digit"]))
        for k, v in emd_stats.items():
            print(f"  {k:<32}{np.median(v):>10.3f}{np.percentile(v, 90):>10.3f}"
                  f"{np.median(v) / base:>16.2f}x")

        # How much blending would it actually take to look like real data? The
        # cover test asks for a specific amount; the reference metric has its own
        # opinion, and the two need not agree.
        ts = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        blends = np.concatenate(
            [(1 - t) * Pd + t * M[nearest] for t in ts], axis=0
        ).astype(np.float32)
        stack2 = np.concatenate([X[tr], blends], axis=0)
        q2 = np.arange(n_tr, len(stack2))
        print(f"computing the blend sweep ({len(q2) * len(stack2) / 1e6:.2f}M pairs)...")
        D2 = pairwise_emd(stack2, tuple(args.image_shape), query_idx=q2,
                          n_jobs=args.n_jobs, progress=False)
        curve = D2[:, :n_tr].min(axis=1).reshape(len(ts), len(Pd))
        sweep = {"t": ts, "median": np.median(curve, axis=1) / base}
        # Blend weight at which the median blended probe is as close to real data
        # as a genuine held-out digit is.
        t_need = float(np.interp(1.0, sweep["median"][::-1], ts[::-1]))
        emd_stats["_sweep"] = sweep
        emd_stats["_t_need"] = t_need
        print(f"\n{'blend t':<12}" + "".join(f"{t:>9.1f}" for t in ts))
        print(f"{'EMD x digit':<12}" + "".join(f"{v:>9.2f}" for v in sweep["median"]))
        # Projecting to the boundary is the weakest repair that passes, so the
        # obvious objection is that a stricter target would do better. Aiming at
        # a *typical* digit's cover rather than the acceptance threshold:
        blend_typ = float(np.median(1.0 - np.median(calib) * scale / d))
        emd_typ = float(np.interp(blend_typ, ts, sweep["median"]))
        print(f"\ncover test is satisfied at blend {np.median(blend):.2f}; "
              f"EMD parity with a real digit needs {t_need:.2f}")
        print(f"aiming instead at a typical digit's cover needs blend "
              f"{blend_typ:.2f}, still only {emd_typ:.2f}x")

    # ---- figure --------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = sorted(set(kinds.tolist()))[:8]
    ncol = len(fams)
    fig = plt.figure(figsize=(2.0 * ncol + 2.0, 9.4))
    gs = fig.add_gridspec(4, ncol, height_ratios=[1, 1, 1, 1.5], hspace=0.35)
    sh = tuple(args.image_shape)
    for j, fam in enumerate(fams):
        i = int(np.flatnonzero(kinds == fam)[0])
        for row, (img, lab) in enumerate((
            (Pd[i], f"{fam}\nflagged p={p_before[i]:.3f}"),
            (R[i], f"repaired p={p_after[i]:.3f}\n||d||/||x||={rel[i]:.2f}"),
            (M[nearest[i]], f"nearest landmark\nblend {blend[i]:.0%}"),
        )):
            ax = fig.add_subplot(gs[row, j])
            ax.imshow(img.reshape(sh), cmap="gray_r", vmin=0, vmax=16)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab, fontsize=7)

    if emd_stats is not None:
        ax = fig.add_subplot(gs[3, : max(1, ncol // 2)])
        bins = np.linspace(0, max(np.percentile(emd_stats["probe"], 99), 1e-6), 55)
        for k, c in (("holdout digit", "tab:green"), ("probe", "tab:red"),
                     ("repaired", "tab:blue")):
            ax.hist(emd_stats[k], bins=bins, density=True, alpha=0.55, color=c,
                    label=f"{k} (median {np.median(emd_stats[k]):.2f})")
        ax.set_xlabel("EMD to nearest training digit -- a metric the repair never saw")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.set_title("the repair passes the test but stops short", fontsize=10)

        ax = fig.add_subplot(gs[3, max(1, ncol // 2) :])
        sw = emd_stats["_sweep"]
        ax.plot(sw["t"], sw["median"], "o-", color="tab:blue", lw=2)
        ax.axhline(1.0, color="tab:green", ls="--", lw=1.2, label="a real held-out digit")
        ax.axvline(float(np.median(blend)), color="tab:purple", ls="--", lw=1.2,
                   label=f"cover test satisfied at {np.median(blend):.2f}")
        ax.axvline(emd_stats["_t_need"], color="tab:orange", ls=":", lw=1.6,
                   label=f"EMD parity needs {emd_stats['_t_need']:.2f}")
        ax.set_xlabel("blend weight on the nearest training exemplar")
        ax.set_ylabel("EMD to nearest digit (x a real digit's)")
        ax.set_title("how far the repair would have to go", fontsize=10)
        ax.legend(fontsize=7.5)
    else:
        fig.add_subplot(gs[3, :]).axis("off")

    fig.suptitle(
        f"Minimum-norm repair onto the cover sublevel set (alpha={args.alpha}, "
        f"exact projection, encoder frozen)",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

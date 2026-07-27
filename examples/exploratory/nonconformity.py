#!/usr/bin/env python
"""Direction-aware nonconformity scores, tested against cover's own blind spot.

`manifold_repair.py` shows that ``cover = min_l ||x - M_l||`` accepts points that
are demonstrably off-manifold, because its acceptance region is a union of
**isotropic** balls around 179 landmarks and the data occupies a thin
~9-dimensional sheet inside each one. The fix has to know about direction.

The candidates here all keep the property that makes leanmap worth using: they are
fixed-size additions fitted post-hoc to a frozen encoder, and inference stays a
handful of matrix-vector products with no training data retained. Around each
landmark a local chart is fitted from its ``k`` nearest *training* points --
Voronoi cells hold a median of 6 points here, too few for a 9-dimensional tangent,
so the neighbourhoods deliberately overlap:

* ``cover``        baseline, isotropic distance to the nearest landmark.
* ``residual``     distance off the local tangent sheet -- the direction cover
                   cannot see.
* ``residual_z``   the same, divided by how far real training points sit off that
                   landmark's sheet, so one threshold means the same everywhere.
* ``mahalanobis``  anisotropic distance in the local chart: off-sheet and
                   along-sheet displacements scaled by their own spreads. This is
                   cover with the balls replaced by ellipsoids.
* ``knn_train``    L2 to the nearest training point. Not fixed-size and so not a
                   candidate -- included as the reference for what retaining the
                   whole training set would buy.

The comparison is deliberately not AUROC-only. The repaired points sit *at* cover's
threshold by construction, so cover still ranks them highly while flagging none of
them. The honest question is power at a fixed, separately calibrated alpha: each
score gets its own threshold from the same calibration half, is checked for
validity on a disjoint test half, and only then compared on the probes and on the
repaired points.

Usage::

    python examples/exploratory/nonconformity.py \\
      --run examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --X examples/exploratory/data/digits_X.npy \\
      --probes examples/exploratory/data/digits_probes_X.npy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree

from manifold_repair import conformal_threshold, pvalues
from splits import load_split

# Spurious Accelerate BLAS flags on large float matmuls; values stay finite.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")


class LocalCharts:
    """Per-landmark tangent frames, fitted from training data only.

    Artefact is ``L x (D*q + D + 2)`` floats: a basis, a centre and two scales per
    landmark. Nothing that grows with the size of the training set.
    """

    def __init__(self, M: np.ndarray, X_train: np.ndarray, q: int = 10, k: int = 48):
        self.q, self.k = int(q), int(k)
        L, D = M.shape
        k = min(int(k), len(X_train))
        _, idx = cKDTree(X_train).query(M, k=k)
        self.M = M
        self.centre = np.empty((L, D))
        self.basis = np.empty((L, D, self.q))
        self.s_perp = np.empty(L)
        self.s_par = np.empty(L)
        for li in range(L):
            A = X_train[idx[li]]
            c = A.mean(axis=0)
            Vt = np.linalg.svd(A - c, full_matrices=False)[2]
            V = Vt[: self.q].T
            self.centre[li] = c
            self.basis[li] = V
            Rr = (A - c) - (A - c) @ V @ V.T
            self.s_perp[li] = max(np.sqrt(np.mean((Rr**2).sum(axis=1))), 1e-6)
            self.s_par[li] = max(np.sqrt(np.mean((((A - c) @ V) ** 2).sum(axis=1))), 1e-6)

    def _decompose(self, Xq: np.ndarray, li: np.ndarray):
        """Off-sheet and along-sheet displacement magnitudes for each query."""
        d = Xq - self.centre[li]
        par = np.einsum("nd,ndq->nq", d, self.basis[li])
        perp = np.linalg.norm(d - np.einsum("nq,ndq->nd", par, self.basis[li]), axis=1)
        return perp, np.linalg.norm(par, axis=1)

    def residual(self, Xq: np.ndarray, nearest: np.ndarray, standardise: bool):
        perp, _ = self._decompose(Xq, nearest)
        return perp / self.s_perp[nearest] if standardise else perp

    def mahalanobis(self, Xq: np.ndarray, block: int = 256) -> np.ndarray:
        """``min_l`` anisotropic distance -- the union of ellipsoids."""
        out = np.empty(len(Xq))
        L = len(self.M)
        for s in range(0, len(Xq), block):
            Q = Xq[s : s + block]
            best = np.full(len(Q), np.inf)
            for li in range(L):
                d = Q - self.centre[li]
                par = d @ self.basis[li]
                perp = np.linalg.norm(d - par @ self.basis[li].T, axis=1)
                val = np.sqrt(
                    (perp / self.s_perp[li]) ** 2
                    + (np.linalg.norm(par, axis=1) / self.s_par[li]) ** 2
                )
                np.minimum(best, val, out=best)
            out[s : s + block] = best
        return out

    def project_ellipsoid(self, Xq: np.ndarray, tau: float, n_theta: int = 4001):
        """Minimum-norm move into ``{x : mahalanobis(x) <= tau}``.

        Within the tangent block and within the normal block the ellipsoid is
        rotationally symmetric, so the projection collapses to a 2-D problem:
        push the point ``(along, off)`` onto the ellipse with semi-axes
        ``tau*s_par`` and ``tau*s_perp``. That is a 1-D search over the ellipse
        parameter, solved on a grid fine enough that the residual error is well
        below the float32 noise the model round-trip introduces anyway.
        """
        th = np.linspace(0.0, np.pi / 2, n_theta)
        ct, st = np.cos(th), np.sin(th)
        out = Xq.copy()
        best_cost = np.full(len(Xq), np.inf)
        for li in range(len(self.M)):
            d = Xq - self.centre[li]
            par = d @ self.basis[li]
            a = np.linalg.norm(par, axis=1)
            perp_vec = d - par @ self.basis[li].T
            b = np.linalg.norm(perp_vec, axis=1)
            A, B = tau * self.s_par[li], tau * self.s_perp[li]
            # cost of landing at each candidate point on the ellipse
            cost = (a[:, None] - A * ct[None, :]) ** 2 + (b[:, None] - B * st[None, :]) ** 2
            j = cost.argmin(axis=1)
            c_best = cost[np.arange(len(Xq)), j]
            inside = (a / A) ** 2 + (b / B) ** 2 <= 1.0
            c_best = np.where(inside, 0.0, c_best)
            take = c_best < best_cost
            if not take.any():
                continue
            a_t, b_t = A * ct[j], B * st[j]
            # rescale each block to its target magnitude, directions unchanged
            fa = np.where(a > 1e-12, a_t / np.maximum(a, 1e-12), 0.0)
            fb = np.where(b > 1e-12, b_t / np.maximum(b, 1e-12), 0.0)
            cand = (
                self.centre[li]
                + (par * fa[:, None]) @ self.basis[li].T
                + perp_vec * fb[:, None]
            )
            cand = np.where(inside[:, None], Xq, cand)
            out[take] = cand[take]
            best_cost[take] = c_best[take]
        return out


def build_scores(M, Xtr, charts, cover_fn) -> Dict[str, callable]:
    tree_lm = cKDTree(M)
    tree_tr = cKDTree(Xtr)

    def nearest(Xq):
        return tree_lm.query(Xq, k=1)[1]

    return {
        "cover": lambda Xq: cover_fn(Xq),
        "residual": lambda Xq: charts.residual(Xq, nearest(Xq), False),
        "residual_z": lambda Xq: charts.residual(Xq, nearest(Xq), True),
        "mahalanobis": lambda Xq: charts.mahalanobis(Xq),
        "knn_train": lambda Xq: tree_tr.query(Xq, k=1)[0],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--X", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=int, default=10, help="local tangent dimension")
    ap.add_argument("--k", type=int, default=48, help="training points per chart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attack", action="store_true", help="min-norm repair vs the new score")
    ap.add_argument("--image-shape", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args(argv)

    import torch

    from leanmap import load_plane

    X = np.load(args.X).astype(np.float64)
    P = np.load(args.probes).astype(np.float64)
    train_idx, hold_idx = load_split(args.run, n=len(X))
    Xtr = X[np.asarray(train_idx)]

    model = load_plane(args.run / "model.pt", device="cpu")
    model.eval()
    M = model.affinity.M.detach().cpu().numpy().astype(np.float64)
    scale = float(model.affinity.dist_fn.natural_scale)

    def cover_fn(Xq):
        with torch.no_grad():
            _, c = model.embed(torch.as_tensor(np.asarray(Xq, dtype=np.float32)))
        return c.numpy().astype(np.float64)

    rng = np.random.default_rng(args.seed)
    hold = np.asarray(hold_idx).copy()
    rng.shuffle(hold)
    cut = len(hold) // 2
    calib_idx, test_idx = hold[:cut], hold[cut:]

    # The adversarial set: cover's own minimum-norm repair of every probe.
    cover_calib = cover_fn(X[calib_idx])
    tau_cover = conformal_threshold(cover_calib, args.alpha)
    d_all = np.linalg.norm(P[:, None, :] - M[None, :, :], axis=2)
    nn = d_all.argmin(axis=1)
    d = d_all[np.arange(len(P)), nn]
    keep = np.clip(tau_cover * scale * (1 - 1e-3) / d, 0.0, 1.0)
    R = keep[:, None] * P + (1.0 - keep)[:, None] * M[nn]

    charts = LocalCharts(M, Xtr, q=args.q, k=args.k)
    scores = build_scores(M, Xtr, charts, cover_fn)

    sets = {
        "calib": X[calib_idx],
        "test": X[test_idx],
        "probe": P,
        "repaired": R,
    }

    lm_bytes = M.size * 4
    chart_bytes = (charts.basis.size + charts.centre.size + 2 * len(M)) * 4
    print(f"charts: q={args.q} tangent dims, k={args.k} training points each")
    print(f"artefact added: {chart_bytes / 1024:.0f} KB on top of a 944 KB model "
          f"(landmarks alone are {lm_bytes / 1024:.0f} KB); nothing scales with n_train\n")

    print(f"{'score':<14}{'valid: real digits':>20}{'probes':>10}{'REPAIRED':>11}"
          f"{'AUROC probe':>13}{'AUROC repaired':>16}")
    print("-" * 84)

    def auroc(pos, neg):
        p = np.concatenate([pos, neg])
        o = np.argsort(p, kind="mergesort")
        r = np.empty(len(p))
        r[o] = np.arange(1, len(p) + 1)
        i = 0
        while i < len(p):
            j = i
            s = p[o]
            while j + 1 < len(p) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[o[i : j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    for name, fn in scores.items():
        s = {k: fn(v) for k, v in sets.items()}
        tau = conformal_threshold(s["calib"], args.alpha)
        p = {k: pvalues(v, s["calib"]) for k, v in s.items()}
        flag = {k: float(np.mean(v <= args.alpha)) for k, v in p.items()}
        tag = "" if abs(flag["test"] - args.alpha) < 0.035 else "  <-- MISCALIBRATED"
        print(
            f"{name:<14}{flag['test']:>19.3f}{flag['probe']:>10.0%}{flag['repaired']:>11.0%}"
            f"{auroc(s['probe'], s['test']):>13.3f}{auroc(s['repaired'], s['test']):>16.3f}{tag}"
        )
        _ = tau

    print(f"\nnominal alpha = {args.alpha}; 'valid' column must sit near it or the "
          f"score's power is meaningless.")
    print(f"n_calib={len(calib_idx)}, n_test={len(test_idx)}, n_probe={len(P)}, "
          f"n_repaired={len(R)}")

    if not args.attack:
        return 0

    # ---- attack the new score with its own minimum-norm repair ---------------
    # Catching points built to defeat *cover* is close to circular. The test that
    # counts is whether the new score's own boundary is somewhere real.
    print("\n" + "=" * 84)
    print("minimum-norm repair against `residual` itself, then judged by EMD")
    print("=" * 84)
    res_calib = scores["residual"](X[calib_idx])
    tau_res = conformal_threshold(res_calib, args.alpha)
    nn_p = cKDTree(M).query(P, k=1)[1]
    dvec = P - charts.centre[nn_p]
    par = np.einsum("nd,ndq->nq", dvec, charts.basis[nn_p])
    perp_vec = dvec - np.einsum("nq,ndq->nd", par, charts.basis[nn_p])
    perp = np.linalg.norm(perp_vec, axis=1)
    shrink = np.clip(tau_res * (1 - 1e-3) / np.maximum(perp, 1e-12), 0.0, 1.0)
    R2 = P - (1.0 - shrink)[:, None] * perp_vec

    s_after = scores["residual"](R2)
    p_after = pvalues(s_after, res_calib)
    rel = np.linalg.norm(R2 - P, axis=1) / np.linalg.norm(P, axis=1)
    print(f"residual   : median {np.median(perp):>7.2f} -> {np.median(s_after):>6.2f} "
          f"(tau={tau_res:.2f}); flagged {np.mean(p_after <= args.alpha):>4.0%}; "
          f"||delta||/||x|| {np.median(rel):.3f}")

    # The one score that bounds displacement in *both* directions. Its acceptance
    # region is a union of ellipsoids, so the projection is a 2-D problem.
    mah_calib = scores["mahalanobis"](X[calib_idx])
    tau_mah = conformal_threshold(mah_calib, args.alpha)
    R3 = charts.project_ellipsoid(P, tau_mah * (1 - 1e-3))
    s3 = scores["mahalanobis"](R3)
    p3 = pvalues(s3, mah_calib)
    rel3 = np.linalg.norm(R3 - P, axis=1) / np.linalg.norm(P, axis=1)
    print(f"mahalanobis: median {np.median(scores['mahalanobis'](P)):>7.2f} -> "
          f"{np.median(s3):>6.2f} (tau={tau_mah:.2f}); flagged "
          f"{np.mean(p3 <= args.alpha):>4.0%}; ||delta||/||x|| {np.median(rel3):.3f}")

    # The control that separates two explanations. If the trouble is that 179
    # charts are too coarse a summary of 1438 points, then a score built on the
    # actual data should have a tight boundary. If its attack also lands at ~2x,
    # the trouble is the metric rather than the summary.
    knn_calib = scores["knn_train"](X[calib_idx])
    tau_knn = conformal_threshold(knn_calib, args.alpha)
    tree_tr = cKDTree(Xtr)
    dn, jn = tree_tr.query(P, k=1)
    step = np.clip(tau_knn * (1 - 1e-3) / np.maximum(dn, 1e-12), 0.0, 1.0)
    R4 = Xtr[jn] + (P - Xtr[jn]) * step[:, None]
    s4 = scores["knn_train"](R4)
    p4 = pvalues(s4, knn_calib)
    rel4 = np.linalg.norm(R4 - P, axis=1) / np.linalg.norm(P, axis=1)
    print(f"knn_train  : median {np.median(dn):>7.2f} -> {np.median(s4):>6.2f} "
          f"(tau={tau_knn:.2f}); flagged {np.mean(p4 <= args.alpha):>4.0%}; "
          f"||delta||/||x|| {np.median(rel4):.3f}   [reference, keeps all data]")

    # Confound to rule out: tau is a 95th percentile, so part of any remaining EMD
    # gap could just be "the 95th percentile is genuinely far". Repair the probes
    # to the *median* L2-to-training distance instead and compare like with like.
    d_te = tree_tr.query(X[test_idx], k=1)[0]
    t_med = float(np.median(d_te))
    step_m = np.clip(t_med / np.maximum(dn, 1e-12), 0.0, 1.0)
    R5 = Xtr[jn] + (P - Xtr[jn]) * step_m[:, None]

    from leanmap.emd import pairwise_emd

    tr = np.asarray(train_idx)
    stack = np.concatenate(
        [Xtr, X[test_idx], P, R, R2, R3, R4, R5], axis=0
    ).astype(np.float64)
    n_tr, n_te, n_p = len(tr), len(test_idx), len(P)
    q = np.arange(n_tr, len(stack))
    print(f"\ncomputing EMD ({len(q) * len(stack) / 1e6:.2f}M pairs)...")
    D = pairwise_emd(stack, tuple(args.image_shape), query_idx=q,
                     n_jobs=args.n_jobs, progress=False)
    tt = D[:, :n_tr].min(axis=1)
    groups = {
        "real held-out digit": tt[:n_te],
        "probe (unrepaired)": tt[n_te : n_te + n_p],
        "min-norm repair vs cover": tt[n_te + n_p : n_te + 2 * n_p],
        "min-norm repair vs residual": tt[n_te + 2 * n_p : n_te + 3 * n_p],
        "min-norm repair vs mahalanobis": tt[n_te + 3 * n_p : n_te + 4 * n_p],
        "min-norm repair vs knn_train": tt[n_te + 4 * n_p : n_te + 5 * n_p],
        "probe pulled to MEDIAN train L2": tt[n_te + 5 * n_p :],
    }
    base = float(np.median(groups["real held-out digit"]))
    l2 = {
        "real held-out digit": float(np.median(d_te)),
        "min-norm repair vs knn_train": tau_knn,
        "probe pulled to MEDIAN train L2": t_med,
    }
    print(f"\n{'':<34}{'L2 to train':>13}{'EMD':>9}{'x a real digit':>16}")
    for k, v in groups.items():
        col = f"{l2[k]:>13.1f}" if k in l2 else f"{'':>13}"
        print(f"  {k:<32}{col}{np.median(v):>9.3f}{np.median(v) / base:>15.2f}x")
    print(
        "\nThe last row is the like-for-like test: a probe dragged to exactly the L2\n"
        "distance a typical real digit sits at. If its EMD stays well above 1.00x,\n"
        "then L2 proximity to training data does not imply manifold membership, and\n"
        "no sublevel set of an L2-based score can be tight -- independent of whether\n"
        "its region is a ball, a slab, an ellipsoid, or a union over every data point."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

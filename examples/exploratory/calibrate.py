#!/usr/bin/env python
"""Derive leanmap's landmark/tau/pyramid settings from the data, training-free.

Three settings should never be typed in by hand:

``tau_scale``
    leanmap's default tau is *anchor spacing* (``AnchorAffinity._default_tau``:
    mean distance from each anchor to its 32 nearest anchors), so an absolute tau
    means nothing without knowing that scale, and the same value implies a
    completely different regime at different ``n_landmarks``. The interpretable
    quantity is the affinity **perplexity** ``exp(H(a_i))``: the effective number
    of anchors a point attends to. Perplexity near 1 is one-hot assignment, which
    turns the conditioning code into a Voronoi indicator and lets the decoder emit
    only piecewise-constant offsets; perplexity near ``L`` is uniform and carries
    no information at all. Calibrating perplexity (as t-SNE and UMAP do for sigma)
    also decouples tau from ``n_landmarks``, without which no landmark sweep is
    interpretable.

``n_landmarks``
    Bracketed from both sides: anchors coarser than the graph's neighbourhood
    radius cannot modulate local geometry, and anchors with too few points each
    cannot be estimated.

``pyramid_level_weights``
    Its length must match the number of levels actually built, which
    ``pyramid_min_reps`` caps well below ``pyramid_scales + 1`` at small N.

``min_dist``
    Enters the loss only through the ``find_ab_params`` curve fit, whose exponent
    ``b`` decides whether attraction is self-limiting. The near-contact force
    goes as ``d^(2b-1)``, so at ``b < 1`` a close pair is pulled proportionally
    harder and the layout knots up. Unlike the other three this needs no
    measurement -- ``b`` is a closed-form function of ``(min_dist, spread)``
    alone, independent of the data, the graph and the layout scale -- but it is
    reported here so the whole configuration is derived in one place.

Usage::

    python examples/exploratory/calibrate.py \\
      --X examples/exploratory/data/digits_X.npy --target-perp 8
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# Exponent at which the s-curve ladder stops clumping further as it trains;
# min_dist_for_b(B_STABLE) is 0.478 at spread 1, i.e. the 0.5 default.
B_STABLE = 1.3


def euclid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b)


def intrinsic_dim(X: torch.Tensor, k: int = 10) -> float:
    """Levina-Bickel maximum-likelihood intrinsic dimension."""
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    vals, _ = torch.sort(d, dim=1)
    r = vals[:, :k].clamp_min(1e-12).double()
    ratio = torch.log(r[:, k - 1 : k] / r[:, : k - 1]).mean(dim=1)
    return float(1.0 / ratio.clamp_min(1e-12).mean())


def knn_radius(X: torch.Tensor, k: int) -> np.ndarray:
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    vals, _ = torch.topk(d, k, dim=1, largest=False)
    return vals[:, -1].cpu().numpy()


def default_tau(M: torch.Tensor) -> torch.Tensor:
    """Replicate ``AnchorAffinity._default_tau``: mean dist to 32 nearest anchors."""
    L = M.shape[0]
    if L == 1:
        return torch.ones(1)
    k = min(32, L - 1)
    d = torch.cdist(M, M)
    vals, _ = torch.topk(d, k + 1, dim=1, largest=False)
    return vals[:, 1 : k + 1].mean(dim=1).clamp_min(1e-3)


def affinity_stats(Dm: torch.Tensor, tau: torch.Tensor) -> Dict[str, float]:
    """Perplexity of each point's anchor distribution, and global anchor usage."""
    a = torch.softmax(-Dm / tau.unsqueeze(0), dim=1)
    ent = -(a * a.clamp_min(1e-12).log()).sum(dim=1)
    perp = ent.exp()
    mass = a.sum(dim=0)
    p = mass / mass.sum()
    usage_ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    L = Dm.shape[1]
    return {
        "perp_median": float(perp.median()),
        "perp_p10": float(perp.quantile(0.10)),
        "perp_p90": float(perp.quantile(0.90)),
        "usage_ent": usage_ent,
        "usage_frac_of_max": usage_ent / float(np.log(L)) if L > 1 else float("nan"),
    }


def solve_tau_scale(
    Dm: torch.Tensor,
    tau0: torch.Tensor,
    target_perp: float,
    lo: float = 1e-4,
    hi: float = 1e4,
    iters: int = 48,
) -> float:
    """Bisect ``tau_scale`` for a target median perplexity (monotone in tau)."""
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if affinity_stats(Dm, tau0 * mid)["perp_median"] < target_perp:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def calibrate_tau_scale(
    X: np.ndarray | torch.Tensor,
    n_landmarks: int,
    target_perp: float = 8.0,
    seed: int = 0,
) -> float:
    """``tau_scale`` giving median perplexity ``target_perp`` at ``n_landmarks``."""
    from leanmap.landmarks import fps_init_indices

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    idx = fps_init_indices(Xt, euclid, int(n_landmarks), seed=seed)
    M = Xt[idx].contiguous()
    return solve_tau_scale(torch.cdist(Xt, M), default_tau(M), target_perp)


def predict_pyramid_levels(
    n: int,
    pyramid_scales: int = 3,
    rep_ratio: float = 4.0,
    min_reps: int = 256,
) -> int:
    """Number of levels ``build_graph_pyramid`` will actually build.

    Mirrors the loop in :func:`leanmap.graph.build_graph_pyramid`. Assumes
    ``R0 == n`` (exact when dedup is a no-op); verify against the
    ``pyramid built: N level(s)`` log line.
    """
    levels = 1
    prev = n
    for level in range(1, max(pyramid_scales, 0) + 1):
        target = max(int(round(n / (rep_ratio**level))), min_reps)
        if target >= prev:
            break
        levels += 1
        prev = target
        if target <= min_reps:
            break
    return levels


def curve_regime(min_dist: float, spread: float) -> Dict[str, float]:
    """Fitted attraction curve and whether it is self-limiting.

    ``d_half = a**(-1/(2b))`` is where the target similarity drops to 1/2, i.e.
    the distance the curve treats as "neighbours"; it is reported so the regime
    verdict can be read against the layout scale it implies.

    Three regimes. Below ``b = 1`` the near-contact force decays more slowly
    than the separation and neighbourhoods collapse. Between 1 and ``B_STABLE``
    the layout is merely marginal: measured on a uniform manifold its
    kNN-spacing CV still creeps up with every extra epoch. From ``B_STABLE`` the
    drift is flat, which is where the shipped ``min_dist`` default comes from.
    """
    from leanmap.losses import find_ab_params, min_dist_for_b

    a, b = find_ab_params(spread, min_dist)
    return {
        "a": a,
        "b": b,
        "d_half": float(a ** (-1.0 / (2.0 * b))) if a > 0 else float("inf"),
        "min_dist_b1": min_dist_for_b(1.0, spread),
        "min_dist_stable": min_dist_for_b(B_STABLE, spread),
        "regime": "collapse" if b < 1.0 else ("marginal" if b < B_STABLE else "stable"),
    }


def landmark_table(
    X: np.ndarray,
    candidates: Sequence[int],
    *,
    n_neighbors: int,
    target_perp: float,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Coverage / occupancy / tau_scale for each candidate ``n_landmarks``."""
    from leanmap.landmarks import fps_init_indices

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    n = len(Xt)
    r_med = float(np.median(knn_radius(Xt, n_neighbors)))
    rows = []
    for L in candidates:
        if L > n:
            continue
        idx = fps_init_indices(Xt, euclid, int(L), seed=seed)
        M = Xt[idx].contiguous()
        tau0 = default_tau(M)
        Dm = torch.cdist(Xt, M)
        spacing = float(tau0.median())
        ts = solve_tau_scale(Dm, tau0, target_perp)
        rows.append(
            {
                "L": int(L),
                "anchor_spacing": spacing,
                "coverage_ratio": spacing / max(r_med, 1e-12),
                "points_per_anchor": n / L,
                "tau_scale_for_target": ts,
                "perp_at_scale_1": affinity_stats(Dm, tau0)["perp_median"],
                **{f"tgt_{k}": v for k, v in affinity_stats(Dm, tau0 * ts).items()},
            }
        )
    return rows


def recommend(
    rows: Sequence[Dict[str, float]],
    *,
    coverage_max: float = 3.0,
    min_points_per_anchor: float = 10.0,
) -> Optional[Dict[str, float]]:
    """Finest L that still keeps every anchor populated.

    Coverage improves monotonically with L while occupancy degrades, so
    occupancy is the binding constraint and the answer is the largest L that
    satisfies it. Coverage is reported rather than used to select, because on
    sparse high-dimensional data (digits: kNN radius 22.9 against a diameter of
    77) it is already satisfied at every candidate and so cannot discriminate.
    """
    populated = [r for r in rows if r["points_per_anchor"] >= min_points_per_anchor]
    if not populated:
        return None
    best = max(populated, key=lambda r: r["L"])
    best = dict(best)
    best["coverage_ok"] = bool(best["coverage_ratio"] <= coverage_max)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True)
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--target-perp", type=float, default=8.0)
    ap.add_argument(
        "--landmarks",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 179, 256, 450, 900],
    )
    ap.add_argument("--min-dist", type=float, default=0.5)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--pyramid-scales", type=int, default=3)
    ap.add_argument("--min-reps", type=int, default=256)
    ap.add_argument("--coverage-max", type=float, default=3.0)
    ap.add_argument("--min-pts-per-anchor", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    X = np.load(args.X).astype(np.float32)
    Xt = torch.as_tensor(X)
    n = len(X)
    r = knn_radius(Xt, args.n_neighbors)
    print(f"data: N={n}  D={X.shape[1]}  diameter={float(torch.cdist(Xt, Xt).max()):.4f}")
    print(f"intrinsic dim (Levina-Bickel k=10) = {intrinsic_dim(Xt):.2f}")
    print(f"median kNN radius (k={args.n_neighbors}) = {float(np.median(r)):.4f}")
    n_lvl = predict_pyramid_levels(
        n, args.pyramid_scales, min_reps=args.min_reps
    )
    print(
        f"pyramid: {n_lvl} level(s) expected with scales={args.pyramid_scales} "
        f"min_reps={args.min_reps} -> pyramid_level_weights must have {n_lvl} entries"
    )
    reg = curve_regime(args.min_dist, args.spread)
    print(
        f"attraction curve: min_dist={args.min_dist:g} spread={args.spread:g} -> "
        f"a={reg['a']:.4f} b={reg['b']:.4f} d_half={reg['d_half']:.4f}  "
        f"[{reg['regime'].upper()}; b=1 at min_dist={reg['min_dist_b1']:.4f}, "
        f"b={B_STABLE:g} at {reg['min_dist_stable']:.4f}]"
    )
    print()

    rows = landmark_table(
        X,
        args.landmarks,
        n_neighbors=args.n_neighbors,
        target_perp=args.target_perp,
        seed=args.seed,
    )
    hdr = (
        f"{'L':>6} {'spacing':>9} {'cover':>7} {'pts/anc':>8} "
        f"{'tau_scale*':>11} {'perp@1':>8} {'usage/max':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r_ in rows:
        print(
            f"{r_['L']:>6} {r_['anchor_spacing']:>9.4f} {r_['coverage_ratio']:>7.2f} "
            f"{r_['points_per_anchor']:>8.1f} {r_['tau_scale_for_target']:>11.4f} "
            f"{r_['perp_at_scale_1']:>8.1f} {r_['tgt_usage_frac_of_max']:>10.2f}"
        )
    print()
    print(f"cover = anchor spacing / median kNN radius (want <= {args.coverage_max:g})")
    print(f"tau_scale* = multiplier on default tau for median perplexity {args.target_perp:g}")

    rec = recommend(
        rows,
        coverage_max=args.coverage_max,
        min_points_per_anchor=args.min_pts_per_anchor,
    )
    if rec is not None:
        print()
        flag = "" if rec.get("coverage_ok", True) else "  [coverage above target]"
        md = max(args.min_dist, reg["min_dist_stable"])
        print(
            f"RECOMMEND: n_landmarks={rec['L']} tau_scale={rec['tau_scale_for_target']:.4f} "
            f"min_dist={md:.4g} "
            f"(coverage={rec['coverage_ratio']:.2f}, {rec['points_per_anchor']:.1f} pts/anchor, "
            f"{n_lvl}-entry weight tuple){flag}"
        )
        if md > args.min_dist:
            print(
                f"  min_dist raised from {args.min_dist:g} to reach b = {B_STABLE:g}, "
                "past which the layout stops clumping further as it trains. Going "
                f"below {reg['min_dist_b1']:.3g} (b = 1) is never safe; between the "
                "two, expect uniformity to degrade with the epoch budget."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

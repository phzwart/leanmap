#!/usr/bin/env python
"""Training-free hyperparameter probe for leanmap (landmark count + tau).

Stage A of hyperparameter selection. Everything here is a property of the data
and the landmark/affinity geometry, so it runs in seconds and needs no fit.

Two questions get answered:

``tau``
    leanmap's default tau is *anchor spacing* (mean distance from each anchor to
    its 32 nearest anchors), so absolute tau values are meaningless without
    knowing that scale. The interpretable quantity is the **affinity
    perplexity** ``exp(H(a_i))`` -- the effective number of anchors each point
    attends to. Perplexity near 1 means one-hot assignment: the conditioning
    code becomes a Voronoi indicator and the decoder can only emit piecewise
    constant offsets, which manufactures crisp islands that are artifacts of
    tau, not structure in the data. This probe reports perplexity vs
    ``tau_scale`` and inverts it to hit a requested target.

``n_landmarks``
    Anchors must cover the manifold at the resolution the kNN graph operates at,
    and each anchor needs enough points to be estimated. The probe reports
    coverage (anchor spacing / kNN radius), points per anchor, and what fraction
    of anchors actually own points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _demo import OUT_DIR
from leanmap.config import PLANEConfig
from leanmap.landmarks import fps_init_indices
from pdb_validation import DATA, load_table


def euclid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b)


def intrinsic_dim(X: torch.Tensor, k: int = 10) -> float:
    """Levina-Bickel maximum-likelihood intrinsic dimension estimate."""
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    vals, _ = torch.sort(d, dim=1)
    r = vals[:, :k].clamp_min(1e-12).double()
    # m_hat(k)^-1 = mean_j log(r_k / r_j) over j < k
    ratio = torch.log(r[:, k - 1 : k] / r[:, : k - 1]).mean(dim=1)
    return float(1.0 / ratio.clamp_min(1e-12).mean())


def knn_radius(X: torch.Tensor, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour for every point."""
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    vals, _ = torch.topk(d, k, dim=1, largest=False)
    return vals[:, -1].cpu().numpy()


def default_tau(M: torch.Tensor) -> torch.Tensor:
    """Replicates AnchorAffinity._default_tau: mean dist to 32 nearest anchors."""
    L = M.shape[0]
    if L == 1:
        return torch.ones(1)
    k = min(32, L - 1)
    d = torch.cdist(M, M)
    vals, _ = torch.topk(d, k + 1, dim=1, largest=False)
    return vals[:, 1 : k + 1].mean(dim=1).clamp_min(1e-3)


def affinity_stats(Dm: torch.Tensor, tau: torch.Tensor) -> dict:
    """Perplexity of each point's anchor distribution + anchor usage."""
    a = torch.softmax(-Dm / tau.unsqueeze(0), dim=1)
    ent = -(a * a.clamp_min(1e-12).log()).sum(dim=1)
    perp = ent.exp()
    # Global anchor usage: how evenly mass spreads over anchors.
    mass = a.sum(dim=0)
    p = mass / mass.sum()
    usage_ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    L = Dm.shape[1]
    owners = torch.unique(Dm.argmin(dim=1)).numel()
    return {
        "perp_median": float(perp.median()),
        "perp_p10": float(perp.quantile(0.10)),
        "perp_p90": float(perp.quantile(0.90)),
        "usage_ent": usage_ent,
        "usage_frac_of_max": usage_ent / float(np.log(L)) if L > 1 else float("nan"),
        "anchors_owning_points": owners,
        "anchor_occupancy": owners / L,
    }


def solve_tau_scale(
    Dm: torch.Tensor,
    tau0: torch.Tensor,
    target_perp: float,
    lo: float = 1e-3,
    hi: float = 1e3,
    iters: int = 40,
) -> float:
    """Bisect tau_scale so median perplexity hits target (perplexity is
    monotone increasing in tau)."""
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        p = affinity_stats(Dm, tau0 * mid)["perp_median"]
        if p < target_perp:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DATA)
    ap.add_argument(
        "--landmarks",
        type=int,
        nargs="+",
        default=[5, 32, 64, 128, 250, 500, 1000],
        help="candidate n_landmarks values to probe",
    )
    ap.add_argument(
        "--target-perp",
        type=float,
        default=8.0,
        help="target median affinity perplexity (effective anchors per point)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=OUT_DIR / "hparam_probe.json")
    args = ap.parse_args()

    X_np, resolution, _ = load_table(args.csv)
    X = torch.as_tensor(X_np)
    N, D = X.shape
    cfg = PLANEConfig.for_scale(N)
    k_graph = cfg.n_neighbors

    d_int = intrinsic_dim(X)
    r_knn = knn_radius(X, k_graph)
    r_med = float(np.median(r_knn))
    diam = float(torch.cdist(X, X).max())

    print(f"=== data ===")
    print(f"N={N}  ambient d={D}  diameter={diam:.4f}")
    print(f"intrinsic dim (Levina-Bickel, k=10) = {d_int:.2f}")
    print(f"graph n_neighbors={k_graph}  median kNN radius={r_med:.4f}")
    print(f"config default n_landmarks for N={N}: {cfg.n_landmarks}")
    print()

    rows = []
    print("=== landmark count sweep (training-free) ===")
    header = (
        f"{'L':>6} {'spacing':>9} {'cover':>7} {'pts/anc':>8} "
        f"{'occup':>6} {'tau_scale*':>11} {'perp@1.0':>9} {'usage/max':>10}"
    )
    print(header)
    print("-" * len(header))
    for L in args.landmarks:
        if L > N:
            continue
        idx = fps_init_indices(X, euclid, L, seed=args.seed)
        M = X[idx].contiguous()
        tau0 = default_tau(M)
        Dm = torch.cdist(X, M)

        spacing = float(tau0.median())
        coverage = spacing / r_med
        at_default = affinity_stats(Dm, tau0)
        ts = solve_tau_scale(Dm, tau0, args.target_perp)
        at_target = affinity_stats(Dm, tau0 * ts)

        row = {
            "L": L,
            "anchor_spacing": spacing,
            "coverage_ratio": coverage,
            "points_per_anchor": N / L,
            "tau_default_median": spacing,
            "tau_scale_for_target": ts,
            "tau_abs_for_target": spacing * ts,
            "at_tau_scale_1": at_default,
            "at_target_perp": at_target,
        }
        rows.append(row)
        print(
            f"{L:>6} {spacing:>9.4f} {coverage:>7.2f} {N / L:>8.1f} "
            f"{at_default['anchor_occupancy']:>6.2f} {ts:>11.3f} "
            f"{at_default['perp_median']:>9.2f} "
            f"{at_target['usage_frac_of_max']:>10.2f}"
        )

    print()
    print("cover   = anchor spacing / median kNN radius (want O(1-3): anchors")
    print("          resolve the same scale the graph does)")
    print("tau_scale* = multiplier on leanmap's default tau that yields median")
    print(f"          perplexity {args.target_perp:g}")
    print("perp@1.0 = median effective anchors per point at tau_scale=1")
    print()

    # What did the absolute tau values we have been using actually mean?
    print("=== absolute tau_init we used historically -> perplexity ===")
    print(f"{'L':>6} {'tau_init':>9} {'perp_med':>9} {'perp_p10':>9} {'regime':>22}")
    print("-" * 60)
    history = []
    for L in [32, 500]:
        if L > N:
            continue
        idx = fps_init_indices(X, euclid, L, seed=args.seed)
        M = X[idx].contiguous()
        Dm = torch.cdist(X, M)
        for tau_abs in [0.01, 0.1, 1.0]:
            tau = torch.full((L,), float(tau_abs))
            st = affinity_stats(Dm, tau)
            pm = st["perp_median"]
            if pm < 1.5:
                regime = "ONE-HOT (Voronoi)"
            elif pm > 0.5 * L:
                regime = "near-uniform (no info)"
            else:
                regime = "soft (usable)"
            history.append({"L": L, "tau_init": tau_abs, **st, "regime": regime})
            print(f"{L:>6} {tau_abs:>9.3f} {pm:>9.2f} {st['perp_p10']:>9.2f} {regime:>22}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "N": N,
                "ambient_dim": D,
                "intrinsic_dim": d_int,
                "diameter": diam,
                "n_neighbors": k_graph,
                "median_knn_radius": r_med,
                "target_perplexity": args.target_perp,
                "landmark_sweep": rows,
                "absolute_tau_history": history,
            },
            indent=2,
        )
    )
    print()
    print(f"json -> {args.json_out}")


if __name__ == "__main__":
    main()

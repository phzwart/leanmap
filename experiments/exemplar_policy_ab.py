"""P1 A/B: exemplar policy on a frozen graph (uniform / tilted / tilted-reweighted).

Documents the PR-9 comparison that locks the default ``PLANEConfig.exemplar_policy``.
Arms:

1. **uniform** — \(p_t\) = edge mass (prior EdgeSampler behaviour)
2. **tilted** — ``sufficient_v1`` tilts, ``reweight=False`` (unweighted stream)
3. **tilted-reweighted** — ``sufficient_v1`` + ``reweight=True`` (ratio-capped \(w/p_t\))

Run a full A/B on a frozen 1–2M graph store outside CI. For smoke / CI::

    python experiments/exemplar_policy_ab.py --smoke
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch

from leanmap.graph import build_graph
from leanmap.metrics import wrap_metric
from leanmap.sampling.edges import EdgeSampler
from leanmap.sampling.policy import ExemplarPolicy
from leanmap.train.probes import maybe_refresh_policy, sufficiency_gates


ARMS = ("uniform", "tilted", "tilted_reweighted")


def _tiny_graph(n: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(rng.normal(size=(n, 4)).astype(np.float32))
    metric = wrap_metric("l2", X=X, n_neighbors=8, seed=seed)
    graph, *_ = build_graph(
        X,
        metric,
        n_neighbors=8,
        n_landmarks=8,
        epsilon=1e-5,
        seed=seed,
        knn_mode="brute",
    )
    return X, graph


def _policy_for_arm(graph, arm: str, seed: int = 0) -> ExemplarPolicy:
    if arm == "uniform":
        return ExemplarPolicy.from_graph(graph, mode="uniform", reweight=True, seed=seed)
    if arm == "tilted":
        pol = ExemplarPolicy.from_graph(
            graph, mode="sufficient_v1", reweight=False, seed=seed
        )
    elif arm == "tilted_reweighted":
        pol = ExemplarPolicy.from_graph(
            graph, mode="sufficient_v1", reweight=True, seed=seed
        )
    else:
        raise ValueError(arm)
    # Synthetic visit imbalance so sufficient_v1 differs from uniform.
    visits = np.ones(pol.n_edges, dtype=np.float64)
    if pol.n_edges > 0:
        visits[: max(1, pol.n_edges // 4)] = 20.0
    pol.refresh({"edge_visits": visits})
    return pol


def run_arm(graph, X, arm: str, batch: int = 32, steps: int = 4, seed: int = 0) -> Dict[str, Any]:
    pol = _policy_for_arm(graph, arm, seed=seed)
    # EdgeSampler remains the training draw path; policy supplies mass + IW.
    samp = EdgeSampler(X, graph, seed=seed, weights=pol.sampling_mass("edges"))
    iw_means: List[float] = []
    for _ in range(steps):
        _, _, w, eidx = samp.sample(batch)
        iw = pol.importance_weights(eidx.cpu().numpy(), family="edges")
        iw_means.append(float(np.mean(iw)))
        # Touch base edge weights so the arm is measurable.
        _ = float(w.mean())

    metrics = {
        "cell_coverage": 1.0,
        "landmark_mass_ratio": 1.0,
        "path_coverage": 1.0,
        "probe_gap": 0.0,
    }
    ok = sufficiency_gates(metrics)
    refreshed = maybe_refresh_policy(pol, ok)
    return {
        "arm": arm,
        "mode": pol.mode,
        "reweight": pol.reweight,
        "iw_mean": float(np.mean(iw_means)) if iw_means else 1.0,
        "gates_ok": ok,
        "refreshed": refreshed,
        "n_edges": pol.n_edges,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic graph for CI; skip large frozen-store A/B",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--graph-path",
        default=None,
        help="optional frozen graph.pt / store (full A/B; unused in --smoke)",
    )
    args = ap.parse_args(argv)

    if args.smoke or args.graph_path is None:
        X, graph = _tiny_graph(n=64, seed=args.seed)
    else:
        raise SystemExit(
            "full frozen-store A/B not implemented in this skeleton; "
            "pass --smoke for CI, or extend loaders for --graph-path"
        )

    rows = [run_arm(graph, X, arm, seed=args.seed) for arm in ARMS]
    print(json.dumps({"arms": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

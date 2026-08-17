"""Golden bit-compat harness for the 10M refactor series."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

GOLDEN_DIR = Path(__file__).resolve().parent
EXPECTED_PATH = GOLDEN_DIR / "expected.json"
SEED = 42


def _pin_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_digest(t: torch.Tensor) -> str:
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    return _sha256_bytes(arr.tobytes() + str(arr.dtype).encode() + str(arr.shape).encode())


def graph_artefact_digests(graphs_state: List[Dict[str, Any]], fingerprint: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, g in enumerate(graphs_state):
        for key in ("src", "dst", "weight", "rep_idx", "edges", "weights"):
            if key in g and torch.is_tensor(g[key]):
                out[f"g{i}.{key}"] = _tensor_digest(g[key])
    out["fingerprint"] = _sha256_bytes(json.dumps(fingerprint, sort_keys=True, default=str).encode())
    return out


def build_fixture_graph(name: str, X: np.ndarray) -> Dict[str, Any]:
    """Build pyramid under pinned protocol; return digests (no full train)."""
    from leanmap import PLANEConfig
    from leanmap.graph import build_graph_pyramid, graph_to_state, tensor_fingerprint
    from leanmap.metrics import wrap_metric
    from leanmap.utils import seed_everything

    _pin_threads()
    seed_everything(SEED)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    n = int(Xt.shape[0])
    cfg = PLANEConfig.for_scale(n)
    cfg.seed = SEED
    cfg.device = "cpu"
    cfg.knn_mode = "brute"
    cfg.dedup = True

    metric = wrap_metric("l2", X=Xt, n_neighbors=cfg.n_neighbors, seed=SEED)
    graphs, M, assign_top1, assign_topc = build_graph_pyramid(
        Xt,
        metric,
        n_neighbors=cfg.n_neighbors,
        n_landmarks=min(int(cfg.n_landmarks), max(8, n // 4)),
        seed=SEED,
        pyramid_scales=min(int(cfg.pyramid_scales), 2),
        knn_mode="brute",
    )
    graphs_state = [graph_to_state(g) for g in graphs]
    digests: Dict[str, Any] = {
        "fixture": name,
        "n": n,
        "seed": SEED,
        "n_graphs": len(graphs),
        "n_reps0": int(graphs[0].reps.rep_idx.shape[0]),
        "n_edges0": int(graphs[0].edges.shape[0]),
        "M_digest": _tensor_digest(M),
        "assign_top1_digest": _tensor_digest(assign_top1),
    }
    digests.update(graph_artefact_digests(graphs_state, tensor_fingerprint(Xt)))
    return digests


def load_expected() -> Dict[str, Any]:
    if not EXPECTED_PATH.exists():
        return {}
    return json.loads(EXPECTED_PATH.read_text())

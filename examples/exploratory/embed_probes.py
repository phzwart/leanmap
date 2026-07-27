#!/usr/bin/env python
"""Place a new probe set on already-trained runs, without refitting anything.

Both families of method here support out-of-sample placement on a trained model
-- ``PLANE.embed()`` for leanmap, ``transform()`` for UMAP/densMAP/PCA -- so
changing the probe set never requires a refit. It only requires that the fitted
model was kept: ``master.py`` writes ``model.pt`` and ``reference.py`` writes
``model.pkl``.

The one genuine exception is densMAP, whose ``transform()`` raises
``NotImplementedError`` in umap-learn.

Rewrites ``Z_probe.npy`` in each run directory, and ``probe_cover.npy`` where the
model exposes an OOD score.

Usage::

    python examples/exploratory/embed_probes.py \\
      --probes examples/exploratory/data/digits_probes_X.npy \\
      examples/out/exploratory/digits_emd_lm/matched__digits__seed* \\
      examples/out/exploratory/digits_holdout/reference/umap_default__none__seed*
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _place_leanmap(model_path: Path, P: np.ndarray, device):
    import torch

    from leanmap import load_plane

    model = load_plane(model_path, device=device)
    with torch.no_grad():
        Z, cover = model.embed(torch.as_tensor(P))
    return Z.detach().cpu().numpy(), cover.detach().cpu().numpy()


def _place_sklearn(model_path: Path, P: np.ndarray):
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    return np.asarray(model.transform(P), dtype=np.float32), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    P = np.load(args.probes).astype(np.float32)
    for run in args.runs:
        run = Path(run)
        pt, pkl = run / "model.pt", run / "model.pkl"
        try:
            if pt.is_file():
                Z, cover = _place_leanmap(pt, P, args.device)
                kind = "leanmap"
            elif pkl.is_file():
                Z, cover = _place_sklearn(pkl, P)
                kind = "sklearn"
            else:
                print(f"  skip {run.name}: no saved model")
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  {run.name}: transform unavailable -- {type(exc).__name__}: {exc}")
            continue
        np.save(run / "Z_probe.npy", np.asarray(Z, dtype=np.float32))
        if cover is not None:
            np.save(run / "probe_cover.npy", np.asarray(cover, dtype=np.float32))
        print(f"  {run.name}: placed {len(P)} probes ({kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

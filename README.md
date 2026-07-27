# leanmap

**leanmap** — *LEarn ANother MAPping*.

Parametric landmark-conditioned neighbour embedding with a **cohesive
multi-scale graph pyramid**. Fit once, then embed new points with a single
network forward pass. The neighbour graph is discarded after training; the
saved artefact holds weights, landmarks, and conformal calibration scores.

```python
from leanmap import PLANEConfig, fit, load_plane
import torch

X = ...  # (N, D) float32
cfg = PLANEConfig.for_scale(len(X))   # cohesive pyramid by default
result = fit(X, dist_fn="l2", config=cfg)
Z, score = result.embed(X)
result.save("model.pt")

model = load_plane("model.pt")
Z_new, score_new = model.embed(torch.as_tensor(X_new))
```

## Install

```bash
pip install -e ".[cpu]"      # + faiss-cpu (recommended for fit at scale)
pip install -e ".[dev,cpu]"  # + pytest, scikit-learn
```

Core deps: `numpy`, `scipy`, `torch`, `tqdm`. FAISS is optional for small
brute-force fits (`knn_mode="brute"`) but recommended via the `cpu` extra.

## Cohesive pyramid (default)

| knob | default |
|------|---------|
| `pyramid_scales` | `3` (fine + 3 coarsenings → up to 4 levels) |
| `pyramid_level_weights` | `(1, 1, 2, 4)` — coarse-heavy |
| `pyramid_coarse_backbone` | `1.0` — MST skeleton on the coarsest level |

Escape hatches: `pyramid_scales=0` for a single-scale graph, or set
`pyramid_coarse_backbone=0` / equal level weights to ablate cohesion.

## CLI

```bash
leanmap fit data.npy -o model.pt --epochs 50
leanmap transform model.pt data.npy -o embedding.npy
leanmap info model.pt
```

## Docs & tests

- Design notes (factors, roles, conformal, pyramid): [`src/leanmap/README.md`](src/leanmap/README.md)
- Toy demos (S-curve, Swiss roll, 8×8 digits): [`examples/README.md`](examples/README.md)
- Tests: `pytest` under `tests/`

Local research trees (`legacy/`, including old experiment scripts under
`legacy/examples/`) are gitignored and not part of the installable package.

## License

MIT

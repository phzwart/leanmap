# leanmap

**leanmap** — *LEarn ANother MAPping*.

Parametric landmark-conditioned neighbour embedding with a cohesive multi-scale
graph pyramid. Fit once, then embed new points with a single network forward
pass. The neighbour graph is discarded after training; the saved artefact holds
weights, landmarks, and conformal calibration scores.

```python
from leanmap import PLANEConfig, fit, load_plane
import torch

X = ...  # (N, D) float32
cfg = PLANEConfig.for_scale(len(X))
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

## Recommended configuration

`PLANEConfig.for_scale(N)` for `N ≤ 5k` ships the measured recipe:

| knob | value |
|------|-------|
| `pca_skip` | `False` |
| `lr` | `2e-2` |
| `lambda_geo` | `0.15` (raise to `0.5` on smooth manifolds, with flat weights) |
| `min_dist` | `0.5` |
| `epochs` | `240` |
| `pyramid_level_weights` | `(1, 2, 8)` |
| `width` / `depth` | `384` / `3` |

Two rules that interact: **`pca_skip` and `lr` are one decision** (either change
alone scores worse — or use `pca_lr_mult` to break the tie), and
**`lambda_geo` and `pyramid_level_weights` are one decision** (with a strong
anchor on a smooth manifold, flat weights beat the coarse-heavy ramp).
`min_dist=0.5` is the top of the measured ladder with no resolved loss. Full
guide: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

Derive landmarks and temperatures from the data:

```bash
python examples/exploratory/calibrate.py --X data.npy --target-perp 8
```

## Reproduce the paper battery

```bash
python examples/exploratory/prepare_feeds.py   # s-curve, swiss roll, digits, iris

python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name paper_digits --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8
```

Sweeps: `canonical` (all four ladders), `iris_canonical` (small-N),
`swiss_roll_frame` (fold-back rigidity). Results and interpretation:
[`docs/RESULTS.md`](docs/RESULTS.md). How to read metrics:
[`docs/METRICS.md`](docs/METRICS.md).

## CLI

```bash
leanmap fit data.npy -o model.pt --epochs 50
leanmap transform model.pt data.npy -o embedding.npy
leanmap info model.pt
```

### Mondrian conformal levels

Category-conditional thresholds for **digit / gauss / shuffle** (default
nonconformity: affinity entropy):

```bash
# list built-in scores
leanmap mondrian --list-scores

# fit levels from a held-out digit pool; print thresholds; save calibrator
leanmap mondrian model.pt calib.npy -o mondrian.pt \
  --score affinity_entropy --alphas 0.01,0.05,0.1

# reuse a saved calibrator and score new points
leanmap mondrian model.pt --load mondrian.pt \
  --eval test.npy --eval-out mondrian_eval.npz --alpha 0.05
```

`calib.npy` is treated as the digit group; Gaussian (μ/σ-matched) and
pixel-shuffled pools are synthesized automatically. Override the score with
`--score cover` (landmark cover), `soft_cover`, `dm_min+a_ent`, etc.

Python API: `MondrianCalibrator` in [`src/leanmap/README.md`](src/leanmap/README.md).

## Publication record (embeddings + OOD)

Start here for a complete, reproducible path (parameters, figures, tables,
frozen models):

**[`publication/GUIDE.md`](publication/GUIDE.md)** · index [`publication/README.md`](publication/README.md)

```bash
bash publication/reproduce.sh          # refresh tables / copy artefacts
bash publication/reproduce.sh --fit    # also retrain digits_clean.pt
```

## Docs & tests

| doc | content |
|-----|---------|
| [`publication/GUIDE.md`](publication/GUIDE.md) | **User guide + publication artefacts** |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Every public knob, defaults, measured ranges |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Paper evidence on four datasets |
| [`docs/METRICS.md`](docs/METRICS.md) | Battery, nulls, traps, Mondrian OOD notes |
| [`docs/math/leanmap.tex`](docs/math/leanmap.tex) | Mathematics (cover, Mondrian categories, LDA) |
| [`src/leanmap/README.md`](src/leanmap/README.md) | Design notes (conditioning, pyramid, conformal) |
| `pytest` under `tests/` | |

## License

MIT

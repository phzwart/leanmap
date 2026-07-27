# leanmap

**leanmap** — *LEarn ANother MAPping*.

Parametric landmark-conditioned neighbour embedding. After training, inference
is a single network forward pass. The neighbour graph is discarded; the saved
artefact holds only weights, landmarks, temperatures, normalisation stats, and
conformal calibration scores.

## Factored conditioning

The **metric** defines what neighbourhood structure the embedding must preserve
(`p_ij` via the graph). **Conditioning** defines which local chart the encoder
applies (`a_f(x)` → FiLM). They are independent objects — do not couple them in
code.

| | metric | conditioning |
|---|---|---|
| defines | `p_ij`, the target neighbourhood structure | `a_f(x)`, which local chart the network applies |
| changing it | changes what "correct" means | leaves the target unchanged; changes only reachability |
| requirement | **faithful** to the comparison you care about | **cheap, smooth, discriminative** |
| evaluated | training only (via the graph) | every forward pass, at inference |

### Roles

Each factor has a role that sets FiLM capacity:

| role | `gamma_f` | `beta_f` | can it reorganise the map? |
|---|---|---|---|
| `PRIMARY` | per-channel `(B, width)` | per-channel `(B, width)` | yes — by design |
| `MODULATOR` | per-layer scalar `(B, 1)` | per-channel `(B, width)` | no |
| `GAIN` | per-layer scalar `(B, 1)` | — | no |
| `AXIS` | — | — | no; dedicated monotone output axis |

Exactly one `PRIMARY` is recommended. `metric_weight` is independent of `role`:
a factor may condition only (`metric_weight=None`), score only, both, or neither.

Composition: `gamma = clamp(∏ gamma_f)`, `beta = Σ beta_f`.

### Domain declarations (same API)

| domain | `PRIMARY` | `MODULATOR` | `AXIS` / `GAIN` | notes |
|---|---|---|---|---|
| unit cells | shape `r/σ` (L=256) | — | `log σ` (L=16) | metric: shape 1.0, size ~0.3 |
| spectra | normalised spectrum | — | `log` total intensity | |
| spectral imaging | normalised spectrum | **spatial coords** (`metric_weight=None`) | `log` intensity | position conditions only |
| molecules | conformational invariants | composition / atom counts | `log` size | |
| images | content embedding | — | scale, illumination | |

Helper: `scale_quotient_factorization()` builds direction + log-magnitude factors.

### `retention_f`

Per-factor fraction of (near, mid, far) triplets — proposed by ranking that
factor's anchors — that satisfy the factor's **view metric** order. Chance is
≈ **0.475** when `near` comes from a graph edge. Below **0.55**, the factor is
conditioning on noise (WARNING); other metrics in the run are then unreliable.

---

```python
import numpy as np
import torch
from leanmap import PLANEConfig, fit

X = np.random.randn(3000, 16).astype("float32")
cfg = PLANEConfig.for_scale(len(X))
result = fit(X, dist_fn="l2", config=cfg)
Z, score = result.model.embed(torch.as_tensor(X))
result.save("plane.pt")
```

Use `"l1"`, `"cosine"`, a `CompositeMetric`, or any batched `DistanceFn`.

## Config presets (`PLANEConfig.for_scale`)

| | `N ≤ 5k` | `5k < N ≤ 2e5` | `N > 2e5` |
|---|---|---|---|
| `width` | 128 | 384 | 512 |
| `depth` | 2 | 3 | 3 |
| `n_landmarks` | 32 | 256 | 512 |
| `n_neighbors` | 10 | 15 | 15 |
| `lambda_lip` | 0.1 | 0.01 | 0.0 |
| `epochs` | 500 | 200 | 50 |
| `calib_max` | 200 | 2000 | 2000 |

At small `N` the map can memorise — the small preset shrinks capacity and raises
`lambda_lip`. The point at that scale is a reusable, differentiable OOS map.

## Graph pyramid (cohesive default)

By default PLANE trains against a **multi-scale graph pyramid**, not only the
finest fuzzy kNN graph. Coarser levels are Galerkin coarsenings of the fine
graph (connectivity-preserving; no Isomap short-circuits). They supply
medium/long-range attraction so distant regions do not drift under uniform
repulsion. The **coarse backbone** bridges any disconnected regions of the
coarsest level so the embedding stays one component.

| field | default | meaning |
|---|---|---|
| `pyramid_scales` | `3` | number of coarsenings (0 = single-scale) |
| `pyramid_level_weights` | `(1, 1, 2, 4)` | attraction weight per level, finest first |
| `pyramid_coarse_backbone` | `1.0` | weight of bridge edges on the coarsest level (`0` = off) |
| `pyramid_rep_ratio` | `4.0` | representative shrink factor per level |
| `pyramid_min_reps` | `256` | stop coarsening near this size |

**Depth is capped by `pyramid_min_reps`, so you often get fewer levels than
`pyramid_scales + 1`.** Coarsening stops once a level reaches `pyramid_min_reps`,
which with the defaults needs `N` of roughly 17k to reach 4 levels; below that
only 3 are built. `pyramid_level_weights` is then matched to the levels actually
built by keeping the first entries *and the last one* (the coarsest weight
carries the global attraction, so dropping it would be the worst choice) and
discarding from the middle, with a warning. Pass a tuple of the right length to
control this exactly — at `N = 5000`, `(1, 4, 16)` rather than `(1, 1, 2, 4)`.

Level weights are the strongest lever in the whole config for global structure.
On a 5k × 9 dataset, moving from an effective `(1, 1, 2)` to `(1, 4, 16)` took
held-out geodesic Spearman from 0.51 to 0.84 and density Spearman from −0.06 to
+0.73 (shuffled-null baselines 0.36 and 0.01). Always calibrate against a null:
refit on column-shuffled data, which destroys all joint structure while keeping
every marginal, and treat that as the chance level for each metric.

The encoder class is unchanged (`PLANE` / `FiLMEncoder`); only which positive
edges enter the batch changes. Inference never sees the pyramid.

## Graph dedup (ε-net)

By default (`dedup=True`, `epsilon=None`) PLANE collapses near-duplicates with an
ε-net before kNN. ε is estimated from a 1-NN subsample; if the low quantile is
≤0 (exact-duplicate mass), it falls back to the median of *positive* 1-NN
distances. Set `dedup=False` or `epsilon=0` to keep every point as its own node
(`R == N`).

## Graph diagnostics

| Warning / signal | Meaning | What to do |
|---|---|---|
| `frac_exact_zero > 0.5` | Majority exact duplicates | Fix upstream dedup / hashing |
| `compression_ratio < 1.05` | ε-net was a no-op | Usually fine; or raise `epsilon` |
| `n_degenerate / R > 0.05` | Neighbours collapsed near `rho` | Raise `local_connectivity` or `epsilon` |
| `n_no_bracket` / `n_hit_floor` | σ solve struggled | Inspect duplicates; check metric scale |
| Components before backbone ≫ 1 | Disconnected fuzzy graph | Often real structure; backbone softly links landmarks |
| IVF/ANN `recall@k < 0.9` | Approximate kNN miss | Auto-retries `c_search`/`nprobe`; switch to `brute` if needed |
| Triplet retention `< 0.3` | Landmark ranks ≠ true order | More landmarks / check metric |
| `std(gamma) ≈ 0` | FiLM unused | Train longer or check affinity collapse |
| Non-true metric + ε-net | Cells may exceed diameter `2ε` | Lower `epsilon` |

## Conformal / OOD caveats

Primary OOD score is **landmark cover** ``min_l ||x - M_l||`` (ambient distance
to the nearest landmark). Conformal `p_value` and `embed(..., return_score=True)`
use this score: large cover ⇒ small `p` ⇒ reject as OOD.

Affinity consistency (`0.5 ||a - a_embed||_1`) is retained as a secondary
*chart-quality* diagnostic via `geometry_consistency_score`'s second return; it
is **not** used for OOD gating (off-manifold points can look spuriously
consistent).

1. The test is on the **cover** distribution. A shift that leaves cover
   unchanged (e.g. sliding along the manifold) is invisible.
2. It answers "is this point near the landmark support", not "have I seen this
   exact point before". A novel point sitting on the manifold will correctly pass.

Retraining or updating landmarks invalidates calibration; `p_value` raises if
the model weight hash no longer matches.

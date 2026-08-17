# Configuration guide

Practical settings for leanmap. Prefer the **split configs** introduced in
0.3; `PLANEConfig` remains a compatibility facade for one deprecation cycle.

```python
from leanmap import (
    BuildConfig,
    TrainConfig,
    PolicyConfig,
    compose_plane_config,
    fit,
)

build = BuildConfig(n_landmarks=128, pyramid_scales=3)
train = TrainConfig(epochs=40, lr=2e-2, pca_skip=False)
policy = PolicyConfig(exemplar_policy="uniform")
cfg = compose_plane_config(build=build, train=train, policy=policy)
# or still: cfg = PLANEConfig.for_scale(len(X))
result = fit(X, dist_fn="l2", config=cfg)
Z, score = result.embed(X)
```

**Core capabilities** (path constraints, class-axis ordering, conformal /
Mondrian OOD, density ordering, conditioning) are first-class — not optional
extras. See `lambda_path`, `lambda_class`, `path_constraints=`, `class_axes=`.

Defaults match the measured recipe for `N ≤ 5k` when you call
`PLANEConfig.for_scale(N)` (or compose equivalent fields).

```python
from leanmap import PLANEConfig, fit

cfg = PLANEConfig.for_scale(len(X))
result = fit(X, dist_fn="l2", config=cfg)
Z, score = result.embed(X)
```

Derive `n_landmarks` and `tau_scale` from the data rather than guessing:

```bash
python examples/exploratory/calibrate.py --X data.npy --target-perp 8
```

### Scale knobs (v2)

| Knob | Config | Notes |
|------|--------|-------|
| `delta` | Build | `None`/`"eps"` = today; `"auto"` calibrates into R band |
| `gauge_level` | Build | `None` = auto (0 below \(R\approx3\times10^5\)) |
| `exemplar_policy` | Policy | `uniform` (default) or `sufficient_v1` |
| Store backend | Store | `auto` picks `ptfile` vs directory on R |

CLI (App. B):

```bash
leanmap-graph-build --X X.npy --out graph.pt
leanmap-train --X X.npy --graph-path graph.pt --exemplar-policy uniform
```

---

## Rules you cannot discover one axis at a time

### 1. `pca_skip` and `lr` are one decision

| `pca_skip` | `lr` | digits 5-NN acc (held out) |
|---|---|---|
| on (legacy default) | 1e-3 | ~0.65 (PCA-pinned) |
| off | 1e-3 | ~0.41 (worse) |
| on | 2e-2 | ~0.68 |
| **off** | **2e-2** | **~0.94** |

With the skip on, the output is `pca(x) + residual` and the residual head
starts near zero, so training begins at plain PCA. The unconstrained linear
path keeps supplying the layout. Turn the skip off and the head must build
the layout from a near-zero init, which it cannot do at `lr=1e-3`. Only both
changes together escape. `for_scale(N≤5k)` sets `pca_skip=False, lr=2e-2`.

### 2. `min_dist = 0.5` is the top of the ladder with no resolved loss

`min_dist` reaches the loss only through `find_ab_params`, which fits
`1/(1 + a d^{2b})`. Higher `min_dist` means higher `b`, which means less
relative pull between already-close points and so a more evenly packed layout.

Three seeds per rung, `±` is the sample sd, **bold** marks a difference
resolved against twice the pooled seed sd:

| `min_dist` | b | digits 5-NN | digits `trust_15` | digits `spacing_cv` | s-curve `knn_overlap` | s-curve `area_sd` |
|---|---|---|---|---|---|---|
| 0.1 | 0.895 | 0.936±0.029 | 0.948±0.009 | 0.612±0.110 | 0.813±0.006 | 0.231±0.022 |
| 0.2 | 1.003 | 0.929±0.028 | 0.944±0.009 | 0.504±0.077 | 0.824±0.019 | 0.193±0.019 |
| 0.5 | 1.334 | 0.921±0.034 | 0.938±0.010 | **0.398**±0.050 | **0.858**±0.016 | **0.152**±0.017 |
| 0.8 | 1.681 | 0.906±0.030 | **0.921**±0.011 | **0.334**±0.028 | 0.858±0.041 | **0.141**±0.032 |

Default **0.5**: the largest value with no resolved loss on any metric. At 0.8
digits trustworthiness is resolved-worse and the s-curve stops improving.

Two earlier claims are withdrawn. The digits 5-NN column does decline, but by
0.030 across the whole ladder against a seed sd of 0.028–0.034 — not one
pairwise difference is resolved, so the "degrades monotonically past 0.2"
reading came from a single seed. And "below 0.2 is never safe" is not
supported: 0.1 costs uniformity, not accuracy. On the s-curve, raising
`min_dist` to 0.5 *improves* neighbourhood preservation outright rather than
trading it away.

### 3. Fold-back manifolds need delayed frame rigidity

`lambda_frame` matches local neighbour Gram matrices (as-rigid-as-possible).
Local rigidity cannot tell a rolled manifold from its unrolled isometry, so
on swiss-roll-like data it must switch on **after** the neighbour loss has
unrolled:

```python
cfg.lambda_frame = 0.5
cfg.frame_ramp = (0.5, 0.75)   # delayed
cfg.frame_tangent = True        # drop across-sheet shortcuts
```

On non-folding sheets (s-curve) early or delayed both help; delayed is still
safer. On clustered data (digits) leave `lambda_frame=0` unless you care more
about local metric than label separation.

---

## Knobs

### Graph

| field | default | measured range | effect |
|---|---|---|---|
| `n_neighbors` | 15 (`for_scale` small) / 10 in some demos | 5–30 | Local graph connectivity. Flat on digits once training is long enough. |
| `local_connectivity` | 1 | 1–2 | Fuzzy-set floor. Raise if many degenerate σ solves. |
| `min_dist` | 0.5 | 0.2–0.8 | Packing / clumping. See rule 2. |
| `dedup` | True | on/off | ε-net collapse of near-duplicates before kNN. Keep on at large N. |
| `epsilon` | None | 0 to disable | Explicit ε; `None` estimates the 1% quantile of 1-NN distances over all N and **writes the value back**, so a saved artefact refits at the same radius. |
| `beta_multiplicity` | 0.5 | 0–1 | Exponent on the cell-multiplicity reweighting `(w_i w_j)^β` of edge memberships. See below. |

`beta_multiplicity` encodes what a repeated row *means*, so it is a property of
the dataset rather than of the solver. Set it near **0** when duplicates are a
deposition artefact and carry no density information — PDB resubmissions of the
same structure, redundant crystal forms — so an over-represented cell does not
acquire proportionally stronger edges. Set it near **1** when multiplicity is
genuine sampling density that the layout should reproduce. At 0.5 a cell of
weight `w` contributes as if it were `sqrt(w)` points.

### Pyramid

| field | default | measured range | effect |
|---|---|---|---|
| `pyramid_scales` | 3 | 0–3 | Coarsenings. 0 = single-scale. At N ≲ 300 the pyramid is inert (`pyramid_min_reps=256`). |
| `pyramid_min_reps` | 256 | — | Stop coarsening near this size. Caps real level count. |
| `pyramid_level_weights` | `(1,1,2,4)` base; `(1,2,8)` in `for_scale` | flat / ramp / steep / frontload | Coarse weight is the strongest global lever. Fine-emphasis `(8,1,1)` can produce *negative* density margins vs a matched null. |
| `pyramid_coarse_backbone` | 1.0 | 0 / 1 | MST bridges on the coarsest level. Keep on. |
| `pyramid_squash` | `"rational_q99"` | `"rational_q99"`, `"rational"`, `"quantile_clamp"` | How aggregated coarse weights become memberships. See below. |

Aggregated crossing weights are heavy-tailed, so they need squashing into
`(0,1)`. The three modes differ in both shape and where they anchor:

- `"quantile_clamp"` (the old behaviour) is `min(w/q99, 1)`, which flattens
  everything above the 0.99 quantile to a common weight of exactly 1 — the
  strongest long-range edges, precisely what a `(1,2,8)` pyramid exists to
  exploit, all tie.
- `"rational"` is `w/(w + median)`: strictly monotone, never saturating, but it
  sends the *median* to 0.5, lifting the bulk of the coarse graph.
- `"rational_q99"` (default) is the same rational shape anchored so that
  `q99 → 0.99`: monotone and non-saturating like `"rational"`, with the bulk
  magnitude of `"quantile_clamp"`.

The anchor matters more than the shape. On digits at three seeds, switching to
`"rational"` costs **0.21** of `density_spearman` (0.725 → 0.520) — five times
the seed spread — and re-tuning `pyramid_level_weights` across flat, matched,
ramp, and steep does *not* recover it (best 0.556). `"rational_q99"` recovers
most of it (0.678) while staying monotone, and comes with a resolved
improvement in `spacing_cv` (0.583 → 0.477). The residual −0.047 against the
clamp is the price of not flattening the top percentile.

Because magnitude and level weights interact, a squash change and a
`pyramid_level_weights` re-tune are **one experiment, not two**. `(1,2,8)`
transfers unchanged between `"quantile_clamp"` and `"rational_q99"` — that is
what matching the magnitude profile bought.

Pass a weight tuple matching the levels that will actually be built (check
`calibrate.py`). A holdout shrinks N and can drop a level.

### Landmarks

| field | default | measured range | effect |
|---|---|---|---|
| `n_landmarks` | 256 (128 in `for_scale` small) | 32–450 | Derive from coverage via `calibrate.py` (target cover ≲ 3). |
| `landmark_geodesic` | False | on/off | FPS on graph geodesics. Helps folded manifolds. |
| `landmark_poisson` | False | on/off | Geodesic Poisson-disk; more uniform interior than FPS. |
| `learn_landmarks` | True | on/off | Freeze for stability at small N / matched recipe. |
| `learn_tau` | True | on/off | Soft temperatures. Matched recipe freezes and sets `tau_scale`. |
| `tau_scale` | 1.0 | ~0.09–2.0 | Softness of landmark affinity. Derive with `--target-perp`. |
| `tau_init` | None | constant | Override all temperatures to a fixed value. |

### Model

| field | default | measured range | effect |
|---|---|---|---|
| `width` / `depth` | 384 / 3 | 128–768 / 2–4 | Capacity. Small-N iris uses 128/2. Wider did not help digits. |
| `d_out` | 2 | 2 | Embedding dimension. |
| `pca_skip` | True base; **False** in `for_scale` small | on/off | See rule 1. |
| `pca_lr_mult` | 1.0 | 1–20 | LR multiplier for the residual head and FiLM hypers over the PCA skip. See below. |
| `conditioning` | `"film"` | `"film"`, `"concat"` | How the landmark affinity `a(x)` enters the encoder. See below. |

With `pca_skip=True` a single flat `lr` couples two decisions: how fast the
skip drifts from the PCA solution it was seeded with, and how fast the head
builds a residual on top of it. Low `lr` keeps the skip near PCA but starves
the head; high `lr` frees the head but lets the skip wander. `pca_lr_mult`
is the escape — put the head and hypernetworks on a 10–20x higher AdamW rate
and the two stop trading off. `1.0` reproduces the flat-rate behaviour exactly
and is the default; the multiplier is ignored when `pca_skip=False`, where
there is no skip to hold back.

The landmark affinity `a(x)` is a deterministic function of `x`, so
conditioning adds no information the encoder could not compute itself — it is
an inductive bias (a soft partition-of-unity mixture of experts), not extra
capacity. `"film"` modulates each hidden layer, `gelu(γ ⊙ LN(h) + β)`;
`"concat"` appends `a(x)` to the normalised input and is otherwise identical in
width, depth, and PCA skip. `"concat"` exists as the honest baseline for
whether FiLM earns the roles, temperatures, and perplexity calibration it
carries.

Measured at three seeds, FiLM wins on exactly one axis. On digits it is
resolved-better on `geodesic_spearman` (0.724±0.060 vs 0.629±0.024, a +0.096
gap against a pooled spread of ~0.023) and unresolved on 5-NN accuracy,
trustworthiness, `knn_overlap`, `spacing_cv`, and `area_sd`. On swiss roll the
two are indistinguishable everywhere, matching to three decimals on
`geodesic_spearman`. So FiLM buys global structure on clustered data and
nothing else measurable; `"concat"` is a legitimate simpler choice when global
geodesic fidelity is not the goal.

### Losses

| field | default | measured range | effect |
|---|---|---|---|
| `n_negatives` | 5 | 5–20 | Repulsion samples. Raising alone does not fix clumping. |
| `lambda_geo` | 0.5 base; **0.15** in `for_scale` small | 0–1 | Global geodesic Procrustes. Weak on digits once trained long; essential for unrolling manifolds (use ≥ 0.5). |
| `geo_ramp` | (0.2, 0.45) | — | Delay before full geo weight. |
| `lambda_anchor` | 1.0 | 0–1 | Splits `lambda_geo` into `lambda_anchor * L_anchor + 0.25 * L_stress`. See below. |
| `lambda_frame` | 0.0 | 0–1 | Local rigidity. See rule 3. |
| `frame_neighbors` | 6 | — | Neighbours per star. |
| `frame_tangent` | True | on/off | Drop across-sheet neighbours. Required for swiss roll. |
| `frame_ramp` | (0, 0) | (0.5, 0.75) delayed | See rule 3. |
| `lambda_lm` | 0.1 | 0–0.1 | Landmark attraction. Not the clumping culprit. |

`lambda_anchor` weights the Procrustes pull toward the classical-MDS layout of
the landmark geodesics; the stress term keeps distances honest regardless. The
anchor pins a *gauge* (a global rotation and reflection), which is only
meaningful when the landmark geodesics actually embed in `d_out` dimensions.
The fit logs `mds_neg_eigen_ratio`, the fraction of eigenvalue mass classical
MDS has to clamp away; above 0.10 it warns. A high ratio means the Procrustes
target is a lossy projection of a non-Euclidean structure, and pinning to it
fights the neighbour loss. The weight is **never** adjusted automatically from
the diagnostic — a loss weight that is a hidden function of the data is exactly
the buried coupling this config avoids. Set `lambda_anchor=0` with
`lambda_frame > 0` instead: local rigidity keeps metric fidelity without
committing to a global gauge that does not exist.

Measured on swiss roll at three seeds with the frame term on, turning the
anchor off is unresolved on every metric mean — but it roughly triples the
*seed spread*: `area_sd` is 0.131±0.072 with the anchor off against
0.075±0.008 with it on. The default stays at 1.0 for that reason. If you turn
it off, check across seeds rather than one.

Note also that `lambda_geo` and `pyramid_level_weights` are not independent.
On swiss roll at `lambda_geo=0.5`, **flat** level weights beat the coarse-heavy
`(1,2,8)` ramp by a resolved 0.052 of `knn_overlap` and 0.043 of `area_sd`.
With a strong anchor already carrying global structure, coarse levels just
compete with fine ones for the edge budget. Raise `lambda_geo` on smooth
manifolds and flatten the weights at the same time.

### Optimisation

| field | default | measured range | effect |
|---|---|---|---|
| `epochs` | 200 base; **240** in `for_scale` small | 60–480 | Digits flat from ~60 on accuracy; 240 recovers geodesic when `lambda_geo=0`. |
| `lr` | 1e-3 base; **2e-2** in `for_scale` small | 1e-3–2e-2 | Paired with `pca_skip`. |
| `lr_after` / `lr_switch_epochs` | None / 0 | optional two-phase | Disables warmup+cosine when set. |
| `batch_edges` | 4096 | 512–4096 | Lower on tiny N. |

### Conformal / device

| field | default | notes |
|---|---|---|
| `calib_max` | 200 (small) / 2000 | Cap on conformal calibration split. |
| `knn_mode` | `"auto"` | `"brute"` / `"ivf"`. |
| `seed` / `device` | 0 / None | Device resolves to CUDA → MPS → CPU. |

---

## Small-N recipe (iris, N ≲ 300)

Below roughly 300 points:

1. **Pyramid is inert.** `pyramid_min_reps=256` means no coarsening is built.
   Set `pyramid_scales=0` explicitly and let `lambda_geo` carry global structure.
2. **Landmarks must be ≪ N.** Default 256 exceeds iris (N=150). Use
   `calibrate.py`; L=64 gives coverage ≈ 2.4 on iris.
3. **Shrink capacity.** `width=128, depth=2` avoids memorisation.
4. **Conformal resolution.** With a 20% holdout used for calibration, the
   smallest reachable p-value is `1/(n_calib+1)`. At n_calib=30 that floor is
   ≈ 0.032 — a property of the method at small N, not a defect.
5. **Artefact size vs speed.** At N=150 a pickled UMAP is only ~25 KB while a
   width-128 leanmap is ~664 KB — the fixed-size win appears as N grows. What
   still holds at tiny N is OOS throughput (~26× vs UMAP at batch 64 on iris).

```python
cfg = PLANEConfig.for_scale(150)
cfg.n_landmarks = 64
cfg.pyramid_scales = 0
cfg.pyramid_level_weights = None
cfg.pyramid_coarse_backbone = 0.0
cfg.lambda_geo = 0.5
cfg.width, cfg.depth = 128, 2
```

---

## Recommended starting points

**Clustered / image-like (digits):**

```python
cfg = PLANEConfig.for_scale(len(X))  # already pca_skip=False, lr=2e-2, λ_geo=0.15
# then override n_landmarks / tau_scale from calibrate.py
```

**Smooth manifold (s-curve):** same as above but `cfg.lambda_geo = 0.5`.

**Fold-back manifold (swiss roll):** `lambda_geo=0.5`, plus delayed frame:

```python
cfg.lambda_geo = 0.5
cfg.lambda_frame = 0.5
cfg.frame_ramp = (0.5, 0.75)
cfg.frame_tangent = True
```

**Tiny N (iris):** see small-N recipe above.

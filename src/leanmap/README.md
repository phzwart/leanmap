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

The encoder applies `gelu(gamma * LN(W h) + beta)` — normalise, *then* modulate.
The order matters for the scalar-`gamma` roles: `LN(c*h) == LN(h)` for `c > 0`,
so modulating before the norm would make `GAIN` a literal no-op and leave
`MODULATOR` acting only through `beta`'s relative size.

### Domain declarations (same API)

| domain | `PRIMARY` | `MODULATOR` | `AXIS` / `GAIN` | notes |
|---|---|---|---|---|
| unit cells | shape `r/σ` (L=256) | — | `log σ` (L=16) | metric: shape 1.0, size ~0.3 |
| spectra | normalised spectrum | — | `log` total intensity | |
| spectral imaging | normalised spectrum | **spatial coords** (`metric_weight=None`) | `log` intensity | position conditions only |
| molecules | conformational invariants | composition / atom counts | `log` size | |
| images | content embedding | — | scale, illumination | |

Helper: `scale_quotient_factorization()` builds direction + log-magnitude factors.

## Class order as a gauge fix (`d_out - 1` directions)

`AXIS` orders the map by a scalar *computable from `x`*. A **class label** is not
computable from `x` — that is what you want the map to help you infer — so it
cannot be a conditioning factor at all: every factor's `view` is evaluated on
every forward pass including inference (`affinities_forward` → `view_batch`), so a
label-reading view would demand the answer as input.

Labels instead enter as a **gauge fix**. The unsupervised objective determines
shape but not orientation — `fuzzy_cross_entropy` sees only distances,
`local_rigidity_loss` is rotation/reflection-invariant — so rotation and
reflection are free parameters the data never constrains, and refits spend them
arbitrarily. A user ordering of the classes is the natural thing to spend them on.

```python
from leanmap import PLANEConfig, fit, ordinal_class_axis

cfg = PLANEConfig.for_scale(len(X))
cfg.lambda_class = 1.0                      # 0 = off (default)
ax = ordinal_class_axis(n_classes=5, axis=0, order=[2, 0, 1, 4, 3])
res = fit(X, dist_fn="l2", config=cfg, class_labels=y, class_axes=[ax])
```

At most **`d_out - 1`** axes may *name a coordinate* (`validate_class_axes` raises,
it does not warn): the remainder must stay free, or the labels have stopped
choosing among equivalent layouts and started dictating the layout, and the map
can no longer disagree with them. Inference needs no label.

| property | how |
|---|---|
| order only | hinge on the *sign* of the gap; spacing and position stay the graph's business |
| zero force once satisfied | `relu`, not a pull — ordered pairs contribute exactly no gradient |
| scale free | margin is a fraction of the coordinate's own running spread |
| free axes untouched | `dL/dz` is exactly zero off the constrained coordinates |
| labels out of the graph | metric, kNN, ε-net and memberships are unchanged — no leakage into what "neighbour" means |

Only the *order* of `rank` is read, so equal ranks express a partial order
("don't order these two"). This differs deliberately from `ordinal_triplet_loss`,
whose `logsigmoid` never reaches zero: that is a ranking objective meant to keep
pressing, this is a gauge fix that should stop.

### Two orderings: pin the primary, let the secondary pick its own direction

Only one ordering is usually worth an axis. For a second, coarser factor — a
parity, a treatment arm, a broad stage — you generally have no basis for claiming
*which way* the map should lay it out, and claiming one anyway is friction you get
nothing for. `axis=None` asks only that the groups come apart along **some**
direction, recomputed each step and oriented low-to-high, so neither the direction
nor the sign is constrained:

```python
from leanmap import grouped_class_axis, ordinal_class_axis

digit  = ordinal_class_axis(10, axis=0, name="digit")            # pinned: z0 orders 0..9
parity = grouped_class_axis([[0,2,4,6,8], [1,3,5,7,9]],          # tied ranks within a group
                            axis=None, name="parity", weight=0.3)  # weight scales lambda_class
res = fit(X, dist_fn="l2", config=cfg, class_labels=y, class_axes=[digit, parity])
```

Because the chosen direction is zeroed on the pinned coordinates, a
free-direction term **provably cannot move a pinned axis** — `dL/dz` is exactly
zero there however hard it is weighted — so a secondary factor is safe to add to
a primary ordering you care about. Pinned axes are capped at `d_out - 1`;
free-direction axes are counted separately and the total is capped at `d_out`,
with a warning (not a refusal) when they fill it, since asking for a separation
with a free sign is strictly weaker than naming a coordinate.

On digits, `z0` ordering the value and parity on a free direction (20 epochs,
`lambda_class=16`, `margin=0.30`, early ramp; run-to-run spread ≈0.03):

| arm | digit adjacent | asked | parity | 5-NN |
|---|---|---|---|---|
| digit only | 0.95 | — | 0.62 | 0.95 |
| + parity free direction, `weight=0.3` | 0.95 | 1.00 | **1.00** | 0.97 |
| + parity free direction, `weight=0.3`, `d_out=3` | 0.91 | 1.00 | 1.00 | 0.97 |
| + parity pinned to `z1` | *refused by the ceiling* | | | |
| arbitrary 5/5 split | 0.96 | **1.00** | 0.82 | 0.98 |

Parity is *not* there incidentally (0.62 without the term), the digit ordering
does not suffer, and 5-NN accuracy holds — a second ordering can be close to free
when the features support it. In `d_out=3` the fit picks a direction inside the
free plane (`[0, −0.20, 0.98]`) rather than a coordinate.

The last row is the warning: an **arbitrary** 5/5 split of the digits also
separates essentially perfectly. A high score means the term worked, not that the
grouping means anything — so the claim worth making is always relative to a
baseline that never requested it. (Its 0.82 on *true* parity is mostly overlap:
that draw shares four of five members with the even digits.)
See `examples/digits_two_orderings.py`.

Two further caveats specific to free-direction axes:

- **Chance is above 0.5**, because the direction is fitted on the points it
  scores: ~0.53 at `n=600, d_out=2`, ~0.54 at `d_out=3`, ~0.515 by `n=3000`. A
  pinned axis lands on 0.500 exactly. Get the null from a shuffled refit.
- **More weight is not more ordering.** Unlike the pinned case, the direction
  co-adapts with the layout, and a heavy secondary term drags the map around while
  its direction estimate is still forming. Start at `weight ≤ 0.5`.

The direction is smoothed across steps (`DIRECTION_MOMENTUM`), which is load
bearing rather than cosmetic: before the groups separate the per-step estimate is
noise, its sign flips between steps, successive pushes cancel, and an unsmoothed
term sits at chance indefinitely (0.59 vs 0.90 on the same synthetic fit).

### Reading the result

`class_axis_report` returns `order_<name>` (pairwise ordering accuracy over
ordered class pairs, chance `0.5`) and `order_adjacent_<name>` (the same
restricted to consecutive classes). **Read the adjacent one.** Well-separated
classes order themselves incidentally once they separate at all, so the overall
mean is optimistic about whether the *sequence* was reproduced; `order=0.95` with
`order_adjacent=0.55` means grouped classes in essentially unresolved order.

An unsatisfiable ordering does **not** degrade gracefully. On digits with
shuffled labels at `lambda_class=1`, ordering stays at chance (0.541) while 5-NN
accuracy collapses 0.945 → 0.377 and the constrained axis shrinks from ~6 units
to ~0.5: contradictory pulls average toward the centre and flatten the coordinate.
Hence the warning below `order_adjacent ≈ 0.6` — raising `lambda_class` there
distorts the layout instead of fixing it. Always report the **shuffled-label
null** at the same `lambda_class`; if it also orders, the term is fitting any
labels it is handed. See `examples/digits_class_axis.py`.

### Inference readout

```python
from leanmap import ClassAxisReadout, ClassRegionConformal

readout = ClassAxisReadout.from_model(res.model, res.X_train, res.class_labels_train, ax)
cal = ClassRegionConformal(readout).fit(res.model, res.X_calib, y_calib)

pos = readout.position(Z)              # continuous place on the ordering
sets = cal.prediction_set(Z, 0.05)     # () = novel, (a,) = confident, (a, b) = ambiguous
```

`position` is the point of the exercise: it places a sample *on the user's
sequence*, interpolating between the class positions training settled into, so a
sample between two classes reads as between them instead of being forced to one
side. `ClassRegionConformal` is one conformal test per class on distance to that
class's region — it cannot go through `MondrianCalibrator`, which applies a single
score across all groups, and the shipped `cover` / `affinity_entropy` scores are
support-based and would separate nothing between class regions of one manifold.

Needs ~50+ calibration points **per class** (it warns below that) and, with an
explicit `X_calib`, the caller owns the calibration labels.

**These class regions do not detect ambient outliers.** The encoder maps every
input somewhere and LayerNorm discards most of an extreme input's scale, so an
absurd ambient point lands *inside* the occupied map: `(1e4, 1e4)` on a 2-D toy
embeds to `(0.52, -1.90)`. That is landmark cover's question, not this one — run
both.

### `retention_f`

Per-factor fraction of (near, mid, far) triplets — proposed by ranking that
factor's anchors — that satisfy the factor's **view metric** order.

Chance is **measured, not asserted**: at fit time the landmark ranks are
shuffled and the same sampler is re-run, which gives the rate at which a
triplet passes with no information in the ranking. That rate depends on the
sampler, the bucket-size distribution, and the data — measured values run
0.472 (iris) to 0.487 (digits) against the 0.475 that used to be hard-coded.
The warn threshold floats with the measured null, keeping the same margin.
Note how narrow that margin is: a `retention_f` of 0.55 is only ~0.07 above
chance, so treat values in the 0.5–0.6 band as barely-informative rather than
as a pass. Below the threshold the factor is
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

### Precomputed kNN (edge metric ≠ ambient metric)

Supply a caller-built neighbor graph when edge distances come from a metric that
is too expensive for all-pairs landmark work (e.g. EMD-rescored L1 candidates):

```python
result = fit(
    X_train,
    dist_fn="l1",                          # landmarks / ε-net / assignment
    config=cfg,                            # requires dedup=False
    X_calib=X_cal,                         # required — caller owns train rows
    precomputed_knn=(knn_idx, knn_dist),   # (N, k) int64 / float32
)
```

`knn_idx` / `knn_dist` index the same rows as `X_train`. Fuzzy-graph edges use
`knn_dist`; landmarks still use `dist_fn`. See `examples/digits_emd.py`.

## Config presets (`PLANEConfig.for_scale`)

| | `N ≤ 5k` | `5k < N ≤ 2e5` | `N > 2e5` |
|---|---|---|---|
| `width` | 384 | 384 | 512 |
| `depth` | 3 | 3 | 3 |
| `n_landmarks` | 128 | 256 | 512 |
| `n_neighbors` | 15 | 15 | 15 |
| `epochs` | 240 | 200 | 50 |
| `pca_skip` | False | True | True |
| `lr` | 2e-2 | 1e-3 | 1e-3 |
| `lambda_geo` | 0.15 | 0.5 | 0.5 |
| `pyramid_level_weights` | `(1, 2, 8)` | `(1, 1, 2, 4)` | `(1, 1, 2, 4)` |
| `calib_max` | 200 | 2000 | 2000 |

At small `N` the preset matches the measured digits/s-curve recipe (no PCA
skip, raised `lr`, mid capacity) so the shipped default reproduces the
documented result. Raise `lambda_geo` to 0.5 on smooth manifolds; on fold-back
manifolds add delayed frame rigidity. Full knob guide:
[`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md).

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

### Sharper support models

Plain cover uses one global scale, so it flags the sparse tail of the training
distribution as readily as anything off-manifold, and in `D ≫ m` a union of
balls is exponentially too generous for a thin sheet. `LandmarkSupport` fixes
both, fit from **training** points:

```python
from leanmap import ConformalCalibrator, LandmarkSupport

support = LandmarkSupport.from_model(model, X_train)   # mode="chart"
cal = ConformalCalibrator(support=support)
cal.fit(model, X_calib)
p = cal.p_value(cal.cover_score(model, X_test))
```

`mode="ball"` divides by a per-landmark radius `r_l` (median training
distance-to-landmark in bucket `l`), buying approximate conditional validity
from a single pooled calibration set. `mode="chart"` (the default) additionally
splits each neighbourhood into tangent and normal directions by local PCA and
scales them separately. On a 2-D sheet in 8-D, the score ratio between an
off-sheet and an equal-length along-sheet move is:

| sheet thickness | balls | charts |
|---|---|---|
| 0.05 | 1.34 | 2.3 |
| 0.005 | 1.34 | 14.9 |
| 0.001 | 1.34 | 74.2 |

Balls are indifferent to thickness; charts track it. Buckets with too few
training points fall back to isotropic balls automatically.

Always score through `cal.cover_score(...)`: calibrating with a support and
then scoring with the plain cover compares two different score functions and
voids the guarantee.

**Validity.** `s_nat` and any global `(mu, sigma)` may be estimated on all of
`X` — they are monotone rescalings and `p_value` depends only on ranks against
the calibration set. Per-landmark radii and charts are *not* rank-preserving,
which is why they are fit on the training split instead.

**Repair.** `support.repair(x, tau)` targets the landmark minimising
`d_l - tau * r_l`, not the nearest one: projection onto a union of balls equals
projection onto the nearest centre only when the radii are equal, and a farther
landmark with a generous radius can be the cheaper move.

### Mondrian levels (digit / gauss / shuffle)

For category-conditional thresholds, use `MondrianCalibrator` (or
`leanmap mondrian` on the CLI). By default it scores with **affinity entropy**
and calibrates three groups — real digits, μ/σ-matched Gaussian noise, and
pixel-shuffled digits — so you get a threshold (and p-value) per group:

```bash
leanmap mondrian --list-scores
leanmap mondrian model.pt calib.npy -o mondrian.pt \
  --score affinity_entropy --alphas 0.01,0.05,0.1
leanmap mondrian model.pt --load mondrian.pt \
  --eval test.npy --eval-out eval.npz --alpha 0.05
```

```python
from leanmap import MondrianCalibrator, list_nonconformity_scores

cal = MondrianCalibrator()                        # score="affinity_entropy"
# cal = MondrianCalibrator(score="cover")         # or soft_cover, dm_min+a_ent, …
cal.fit_from_digits(model, X_calib)
levels = cal.levels(alphas=(0.01, 0.05, 0.1))      # {group: {α: threshold}}
s = cal.score_points(model, X_test)
p = cal.p_values(s)                               # upper-tailed (OOD)
sets = cal.prediction_set(s, alpha=0.05)          # two-sided {g : p_g > α}
```

`list_nonconformity_scores()` lists built-in score names; pass a callable for a
custom nonconformity function. Persist with `cal.state_dict()` /
`MondrianCalibrator.from_state_dict`.

**LDA on (cover, entropy).** Fit a Fisher hyperplane on in-support vs OOD
features and use signed distance as the score (higher ⇒ more OOD):

```python
from leanmap import CoverEntropyLDA, MondrianCalibrator, make_mondrian_groups

g = make_mondrian_groups(X_train, seed=0)
lda = CoverEntropyLDA().fit(
    model, g["digit"], torch.cat([g["gauss"], g["shuffle"]], 0)
)
cal = MondrianCalibrator(score=lda)
cal.fit_from_digits(model, X_calib)
```

**Notes.** `levels` / `threshold` are upper-tailed (reject group *g* when
`score > q_g(α)`). `prediction_set` defaults to two-sided p-values so a typical
digit is not accepted as gauss/shuffle merely because its entropy is *below*
those pools. Retraining or updating landmarks invalidates calibration;
`p_value` raises if the model weight hash no longer matches.

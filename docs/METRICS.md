# Metrics and how to read them

The evaluation battery is shared across datasets. Numbers only mean something
when scored **held out**, against a **matched null**, over **seeds**.

## Habits

1. **Held out (`--holdout 0.2`).** leanmap is parametric — score points it never
   trained on via `result.embed()`. In-sample metrics flatter the embedding at
   small N.
2. **Matched null (`--null shuffle`).** Refit the *same* configuration on
   column-shuffled input. Chance depends on the config, not just the data. Only
   the margin above the null carries information.
3. **Seeds (`--seeds 0 1 2`).** Gives the run-to-run spread a difference has to
   clear. On digits that is ±0.016 on accuracy and ±0.075 on geodesic Spearman.

## Geometry columns

| metric | what it measures | higher better? |
|---|---|---|
| `trust_15` / `cont_15` | trustworthiness / continuity @15 | yes |
| `knn_overlap_15` | Jaccard overlap of ambient vs embed kNN | yes |
| `geodesic_spearman` | Spearman of graph-geodesic distances vs embed | yes |
| `ambient_spearman` | Spearman of ambient distances vs embed | yes |
| `density_spearman` | local density correspondence ambient ↔ embed | yes |
| `spacing_cv` | CV of embed kNN radii (uniformity) | **lower** |
| `area_sd` | local area distortion | **lower** |

## Label columns (when labels exist)

| metric | notes |
|---|---|
| `label_acc_Z` | 5-NN label accuracy in the embedding |
| `label_acc_X` | same in ambient — the ceiling the features support |
| `label_ari` | ARI of k-means vs truth |
| `label_sil_Z` | silhouette of true labels in Z |

## Four traps

### 1. Parameter-bin accuracy can reward *not* unrolling

On the s-curve, PCA-2D scores the highest `t`-bin accuracy (≈0.98) while managing
only ≈0.31 kNN overlap, because `t` is nearly linear in the ambient projection.
Use **kNN overlap** and **geodesic Spearman** to tell unrolling from projecting.

### 2. `spacing_cv` is only diagnostic where sampling density is known

On a uniformly sampled s-curve the true flattening has `spacing_cv ≈ 0.18`. On
digits the shuffled null scores 0.43 against the real data's 0.46 — so the metric
measures a habit of the algorithm, not a property of the data. Prefer **`area_sd`**
on clustered feeds.

### 3. Density Spearman is blind to knots and voids

leanmap can read 0.71 density Spearman against UMAP's 0.25 while being the
clumpier of the two. A rank correlation cannot see local packing failure.
**`area_sd`** and the per-epoch **`spacing_cv` drift** (`--monitor N`) are the
informative counterparts.

### 4. Unstructured map-AUROC is not “outside-manifold” detection

`UMAP.transform()` places each new point as a weighted mean of training
neighbours’ embedding coordinates — a convex combination — so it **cannot**
emit a coordinate outside the convex hull of the training embedding. High AUROC
on unstructured probes (`noise`, pixel-shuffled) therefore measures absorption
into filled interiors, not extrapolation. Prefer **local occupancy** (neighbours
within the train 15-NN radius) and **ambient cover** over raw map NN-distance
AUROC when claiming off-manifold behaviour. Leanmap’s structured-novelty lead
comes from voids between clusters (probe occupancy ≈0.16× a real digit’s vs
UMAP ≈0.39×), not from placing points outside the hull (~90% inside under both).

## Persistence: would you get the same map twice?

Every column above asks whether *a* map is good. None asks whether you would
get the same map again — which is the property a saved `.pt` actually sells.
`persistence_run.py` refits at ≥3 seeds and ≥3 training subsamples (80% draws)
and reports:

| metric | what it measures | higher better? |
|---|---|---|
| `rank_spearman_mean` | Spearman of neighbour ranks between two independently trained maps, on a shared probe set | yes |
| `rank_jaccard_15_mean` | 15-NN Jaccard overlap between two maps | yes |
| `coord_disagreement_median` | median Procrustes-aligned coordinate distance | **lower** |

**Protocol matters.** The Procrustes similarity is fitted on a shared *anchor*
set and scored on **held-out** points, so the alignment is not fit on the same
points it is graded on.

**Read rank agreement as primary.** Procrustes cannot repair genuine
topological ambiguity — which arm of a branching structure lands where — so
large coordinate disagreement *with* high rank agreement is a gauge artefact,
not instability. The s-curve subsample row is exactly this: coordinate
disagreement rises 8x (0.032 → 0.264) while rank Spearman moves 0.998 → 0.995.

Measured (120 epochs, shipping recipe):

| dataset | refit varies | rank Spearman | Jaccard@15 | coord. disagreement |
|---|---|---|---|---|
| digits | seed | 0.712 | 0.377 | 0.729 |
| digits | subsample | 0.633 | 0.377 | 0.726 |
| s-curve | seed | 0.998 | 0.727 | 0.032 |
| s-curve | subsample | 0.995 | 0.657 | 0.264 |
| swiss roll | seed | 0.998 | 0.716 | 0.030 |
| swiss roll | subsample | 0.862 | 0.609 | 0.724 |

Manifold maps are reproducible at fixed training data. **The digits map is
not** — two runs differing only in seed share under two neighbours in five.
Treat clustered-data coordinates as one sample from a distribution over
layouts, not as a stable artefact.

## Cross-metric diagnostics (EMD on image feeds)

An embedding is always relative to an **imposed metric** (here ambient L2 →
kNN graph → neighbour loss). L2-referenced columns score that geometry. They
are also partly circular: both leanmap and UMAP build their graph from L2, so
those columns reward reproducing the input metric.

Earth Mover’s Distance (exact W1 of ink mass) is a *different* metric — it
keeps ordering far pairs after L2 saturates (far-band L2↔EMD Spearman ≈ 0.37
on digits). Scoring an L2-trained map against EMD therefore measures
**transfer across metrics**, not fidelity under the training objective. Do not
treat it as a head-to-head “who preserved the image better” verdict unless both
methods were fit from an EMD graph.

```bash
python examples/exploratory/make_emd.py \
  --X examples/exploratory/data/digits_X.npy --image-shape 8 8 --n-jobs 8
```

Pass `--emd` to `master.py` or use `emd_bench.py` for the transfer numbers with
paired bootstrap CIs.

## OOD / conformal

Primary score is landmark **cover** `min_ℓ ||x − M_ℓ||`. Conformal p-values are
calibrated on a held-out split of real data. Scope limits:

1. The test is on the cover distribution. A shift that leaves cover unchanged
   (sliding along the manifold near a landmark) is invisible.
2. It answers "near the landmark support?", not "have I seen this exact point?".

Map distance is a weak detector in 2-D; ambient cover is the capability column.

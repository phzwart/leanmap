# Results

Uniform evidence for leanmap on four datasets: **s-curve**, **swiss roll**,
**digits**, and **iris**. Every claim reports a matched null where available.
Reproduction commands are at the end of each section and in
[CONFIGURATION.md](CONFIGURATION.md).

Scoring protocol: held-out 20%, seeds 0–2 (0–4 on iris), `--null shuffle`,
`--target-perp 8`. See [METRICS.md](METRICS.md) for how to read the battery.

## Scorecard (leanmap vs UMAP)

| axis | winner | evidence |
|---|---|---|
| kNN overlap@15 (digits, L2) | **leanmap** | 0.583 ± 0.012 vs UMAP ~0.538 (`paper_digits` recommended) |
| density Spearman (digits) | **leanmap** | 0.718 ± 0.030 vs ~0.248 |
| geodesic Spearman (s-curve) | **leanmap** | 0.985 raw; best null-corrected margin (+0.114 vs +0.097) |
| null-corrected trust / kNN (s-curve) | **leanmap** | +0.031 / +0.178 vs UMAP +0.004 / +0.111 |
| structured novelty (map AUROC) | **leanmap** | 0.809 vs 0.746; probes land in voids (occupancy 0.16× vs 0.39×) |
| ambient cover OOD | **leanmap only** | AUROC 1.000; conformal flags 100% of probes at α=0.05 |
| OOS inference cost | **leanmap** | 4.3 µs/pt vs 696 µs at batch 512 (~162×) |
| label accuracy / trust (digits) | UMAP | 0.987 / 0.988 vs leanmap 0.941 / 0.946 |
| unstructured map-AUROC | *deceptive* — see below | UMAP 0.893 vs 0.811; not a fair “outside-manifold” claim |

Fidelity under the **imposed metric** (ambient L2 → graph → neighbour
embedding) is the fair head-to-head. Cross-metric transfer to EMD is reported
separately below — it is not a scorecard row.

UMAP’s higher AUROC on unstructured controls (`noise`, pixel-`shuffled`) is
**not** evidence that it places points off the manifold. `UMAP.transform()`
initialises each new point as a weighted mean of training neighbours’
embedding coordinates — a **convex combination** — so every transform lands
inside the convex hull of the training embedding by construction. Measured on
digits: ~90% of probes sit inside the train hull under both methods; UMAP
additionally pulls the typical probe *onto* the data (local occupancy 0.39× a
real digit’s neighbourhood, vs leanmap 0.16×). Unstructured AUROC therefore
rewards how UMAP absorbs random ink into filled cluster interiors, not
extrapolation. Leanmap’s advantage is empty interior (voids between clusters)
plus an ambient cover score UMAP does not have.

---

## Digits (N=1797, D=64, 10 classes)

**Setup.** Sklearn 8×8 digits. Recommended config:
`pca_skip=False`, `lr=2e-2`, `lambda_geo=0.15`, `epochs=240`, `min_dist=0.5`,
`pyramid_level_weights=(1,2,8)`, `n_landmarks` / `tau_scale` from
`calibrate.py` (L≈179, τ≈0.089).

### Reference bar (held out, 3 seeds)

`paper_digits` recommended (`min_dist=0.5`, `--target-perp 8`), vs earlier
`min_dist=0.1` matched run and UMAP nn10 from the holdout bar:

| metric | leanmap (`paper_digits`) | leanmap (`min_dist=0.1`) | UMAP nn10 | PCA-2D | null (`paper_digits`) |
|---|---|---|---|---|---|
| 5-NN label accuracy | 0.931 ± 0.013 | 0.941 ± 0.016 | **0.987** | 0.603 | 0.093 |
| ARI vs truth | — | 0.825 ± 0.041 | **0.911** | 0.393 | — |
| trustworthiness@15 | 0.945 ± 0.008 | 0.946 ± 0.008 | **0.988** | 0.829 | 0.571 |
| **kNN overlap@15** | **0.583 ± 0.012** | **0.571 ± 0.019** | 0.538 | 0.151 | 0.070 |
| geodesic Spearman | **0.707 ± 0.053** | 0.644 ± 0.075 | 0.606 | 0.547 | 0.064 |
| **density Spearman** | **0.718 ± 0.030** | **0.709 ± 0.036** | 0.248 | ~0 | 0.148 |

Leanmap trails on label accuracy / trustworthiness, and **beats UMAP on kNN
overlap, geodesic Spearman, and density preservation** under L2-referenced
metrics. Null margins on the recommended run are large (overlap +0.51,
geodesic +0.64). At `min_dist=0.5` vs the older 0.1 matched layout, accuracy
is essentially unchanged here while packing stays in the winning band.

### Ablations (signal-carrying ladders)

All arms below are three seeds with the spread reported. A difference is called
**resolved** only when it exceeds twice the pooled seed sd; anything smaller is
inside run-to-run noise and is reported as unresolved rather than as a result.

#### Net effect of the review-response changes

A seeded snapshot taken *before* the FiLM reorder, ε quantile estimator, halo
pass, and squash change, compared against the same recipe after. Four of thirty
metric/dataset pairs are resolved:

| dataset | metric | before | after |
|---|---|---|---|
| digits | `density_spearman` | 0.717±0.020 | 0.670±0.007 |
| swiss roll | `knn_overlap_15` | 0.793±0.005 | 0.840±0.004 |
| swiss roll | `area_sd` | 0.201±0.015 | 0.149±0.009 |
| swiss roll | `trust_15` | 0.9953±0.0002 | 0.9972±0.0004 |

Three improvements on the manifold dataset, and one known cost — the monotone
squash, at exactly the magnitude the squash ablation predicts. Everything else
is unchanged within noise, including the whole s-curve column (where
`knn_overlap` rises 0.061 and `density_spearman` 0.105, but the spread is too
wide to resolve either). The FiLM reorder, ε estimator, and halo pass produce
no regression anywhere.

- **`min_dist`:** the only knob that moves uniformity, and it moves it a lot —
  digits `spacing_cv` falls from 0.612 to 0.334 across 0.1 → 0.8. The digits
  5-NN decline over the same range is 0.030 against a seed sd of ~0.030, so it
  is *not* resolved; the earlier "degrades monotonically past 0.2" reading was
  single-seed. Default 0.5 is the largest value with no resolved loss.
- **`lambda_geo`:** at 240 epochs the ladder 0 → 1.0 is flat within seed noise
  on accuracy; keep 0.15 for geodesic/ARI balance. On swiss roll, dropping the
  anchor at fixed `(1,2,8)` *is* resolved (−0.051 `knn_overlap`).
- **`pyramid_level_weights`:** coarse-heavy survives null calibration; fine
  emphasis can produce negative density margins. But with `lambda_geo=0.5` on a
  smooth manifold, **flat** weights are resolved-better than `(1,2,8)`
  (+0.052 `knn_overlap`) — the two knobs are one decision.
- **`pyramid_squash`:** the anchor point dominates. `"rational"` (median-
  anchored) costs a resolved 0.21 of `density_spearman` and no level-weight
  re-tune recovers it; `"rational_q99"` recovers most of it while staying
  monotone, and is the default.
- **`conditioning`:** FiLM beats concat on digits `geodesic_spearman` (+0.096,
  resolved) and on nothing else; on swiss roll the two are indistinguishable.
- **`beta_multiplicity`:** inert on digits (all deltas < 0.01, none resolved),
  which is expected — digits has no duplicate mass. It is a property of the
  corpus, not a solver knob.
- **`lambda_anchor`:** unresolved on the mean on swiss roll, but the seed spread
  of `area_sd` roughly triples with it off. Default stays 1.0.
- **`pca_lr_mult`:** with `pca_skip=True`, a 10–20x head/hypernetwork rate
  recovers the 0.098 `geodesic_spearman` loss the skip otherwise incurs
  (down to an unresolved 0.009). It does not make the skip path preferable —
  trustworthiness and `knn_overlap` stay resolved-worse than no-skip.
- **`lambda_frame`:** leave at 0 for clustered data unless local metric matters
  more than class separation.

### Persistence: would you get the same map twice?

Three refits, Procrustes fitted on a shared anchor set and scored on held-out
points. Rank agreement is the primary number: Procrustes cannot repair genuine
topological ambiguity, so high coordinate disagreement with high rank agreement
is a gauge artefact, not instability.

| dataset | refit varies | rank Spearman | Jaccard@15 | coord. disagreement |
|---|---|---|---|---|
| digits | seed | 0.712 | 0.377 | 0.729 |
| digits | subsample | 0.633 | 0.377 | 0.726 |
| s-curve | seed | 0.998 | 0.727 | 0.032 |
| s-curve | subsample | 0.995 | 0.657 | 0.264 |
| swiss roll | seed | 0.998 | 0.716 | 0.030 |
| swiss roll | subsample | 0.862 | 0.609 | 0.724 |

Manifold maps are reproducible at fixed training data; **the digits map is
not** — two runs differing only in seed share under two neighbours in five.
Treat digits coordinates as one sample from a distribution over layouts, not as
a stable artefact. The s-curve subsample row is the gauge artefact in pure
form: coordinate disagreement rises eightfold while rank agreement is
unchanged.

### Inference cost

Digits (N=1797, D=64), CPU, median of 15 repeats after warm-up:

| | model on disk | B=1 | B=512 | µs/point at B=512 |
|---|---|---|---|---|
| leanmap | 944 KB | 0.25 ms | 1.73 ms | 3.4 |
| UMAP `.transform` | 605 KB | 3.22 ms | 399.9 ms | 781.0 |
| PCA | 1 KB | 0.03 ms | 0.05 ms | 0.09 |

231x faster than `UMAP.transform` per point at B=512, 13x at B=1. PCA is
another 38x faster than leanmap and is the honest floor.

The neighbour search is not eliminated, it is **replaced**: an exact
variable-size search becomes a fixed L×D dense computation. The structural wins
are no index, no training data shipped, and a differentiable map. The artefact
difference is asymptotic — the UMAP pickle grows at a measured 431 bytes/point
(flat to 1% over N=250–1797), while leanmap's is independent of N. Crossover is
near N≈2200; by N=10⁶ that is ~400 MB against an unchanged 944 KB.

### Cross-metric note: EMD (not a fidelity verdict)

Both methods embed under an **imposed ambient metric** — here pixel L2, with
its own local/saturation properties — and the L2-referenced battery above is
the comparison under that geometry. Earth Mover’s Distance (exact W1 of ink
mass on the 8×8 grid) is a *different* metric with different structure: it
orders far pairs that L2 saturates on (far-band Spearman of L2 vs EMD is only
~0.37). Scoring an L2-trained embedding against EMD therefore asks how much of
an alien geometry happens to leak into the map — useful as a diagnostic, not as
a verdict on who preserved “image geometry better.”

Held out, 3 seeds, neither method fit on EMD:

| metric | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| EMD Shepard | 0.224 ± 0.066 | 0.337 ± 0.058 | 0.325 ± 0.013 |
| EMD far-band Spearman | −0.101 ± 0.114 | 0.166 ± 0.049 | −0.094 ± 0.016 |
| EMD kNN overlap@15 | 0.484 ± 0.008 | 0.545 ± 0.013 | 0.315 ± 0.008 |

UMAP’s L2 embedding happens to align more with EMD than leanmap’s does,
including in the far band where leanmap’s Spearman is negative. That is a
statement about **transfer across metrics**, not about failure under the
training objective. A fair EMD contest would fit both methods from an EMD
graph; this run never did that.

### Novelty: structured probes, voids, and why unstructured AUROC misleads

Fourteen probe families never in training (12 structured + `noise` +
pixel-`shuffled`), rescaled to median digit ink mass. AUROC via distance to the
nearest *training* point in the map:

| probe set | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| 12 structured families | **0.809** | 0.746 | 0.647 |
| 2 unstructured controls | 0.811 | 0.893 | 0.590 |
| NN-distance ratio (probe / held-out digit) | **3.28×** | 2.72× | 1.42× |

**Leanmap leads on structured novelty.** The unstructured row is the misleading
one: UMAP’s `transform()` is a convex combination of training embedding
coordinates, so it cannot emit a point outside the training hull no matter how
unlike the input is.

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| inside train hull: probe | 0.921 | 0.899 | 0.821 |
| probe local occupancy / digit’s | **0.16** | 0.39 | 0.70 |
| occupancy p75 / digit median | **0.35** | 0.90 | 1.05 |

Both methods keep ~90% of probes inside the global hull. They differ in the
*interior*: UMAP fills cluster cores (p75 probe has 90% of a real digit’s
neighbours; p90 has *more*), which is exactly what a convex NN blend produces.
Leanmap leaves voids — probes land central yet far from training mass — which
is why structured map-AUROC favours it. Detectability here is where a map
reserves empty space, not whether it can represent “far away.”

Ambient landmark-cover AUROC is **1.000** on all fourteen families; conformal
at α=0.05 flags 100% of probes while held-out digits stay at the nominal rate.
Map geometry catches only ~44% of probes at 5% FPR — use the ambient score for
OOD, the map to look at the result. Per-family novelty is stable under cover
(~1% across-seed spread) and seed noise under map distance (~48%).

### Reproduce

```bash
python examples/exploratory/prepare_feeds.py
python examples/exploratory/calibrate.py \
  --X examples/exploratory/data/digits_X.npy --target-perp 8
python examples/exploratory/reference.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy --name paper_digits \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --n-neighbors 10
python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name paper_digits --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8
```

---

## S-curve (N=2000, D=3, intrinsic dim ≈ 1.94)

**Setup.** Sklearn S-curve, uniform sampling. Same as digits recommended but
`lambda_geo=0.5` (smooth sheet wants the global pull). Labels = 8 quantile
bins of the intrinsic parameter `t`.

### Held out, 3 seeds (`paper_s_curve` recommended + earlier λ_geo=0.5 transfer)

`paper_s_curve` (DIGITS_MATCHED λ_geo=0.15):

| metric | leanmap | sd | null | margin |
|---|---|---|---|---|
| `t`-bin accuracy | 0.924 | 0.027 | 0.131 | +0.79 |
| trustworthiness@15 | 0.992 | 0.002 | 0.958 | **+0.034** |
| kNN overlap@15 | 0.754 | 0.015 | 0.562 | **+0.192** |
| geodesic Spearman | **0.984** | 0.006 | 0.869 | **+0.115** |

Earlier transfer at `lambda_geo=0.5` (3 seeds + null): accuracy 0.913 ± 0.022,
trust 0.994 / null 0.963 (margin +0.031), kNN 0.739 / 0.560 (+0.178), geodesic
**0.986** / 0.872 (+0.114). Same story: leanmap unrolls cleanly, best geodesic
fidelity, leads UMAP on null-corrected trust and kNN margins. Prefer
`lambda_geo=0.5` on this feed.

**Trap:** PCA-2D wins raw `t`-bin accuracy (~0.98) while scoring ~0.31 on kNN
overlap — that column rewards *not* unrolling. Density Spearman is uninformative
on this uniformly sampled feed (ambient density ≈ constant; null exposes it).

### Frame rigidity (in-sample ladder, λ_geo=0.5)

| `lambda_frame` | kNN overlap@15 | geodesic Spearman |
|---|---|---|
| 0.0 | 0.907 | 0.996 |
| 0.1 delayed | 0.928 | 0.996 |
| 0.25 delayed | 0.945 | 0.996 |
| 0.5 delayed | 0.957 | 0.996 |
| 0.5 early | 0.935 | 0.994 |
| 1.0 delayed | 0.966 | 0.996 |

Monotone gain; delayed ramp beats early. Promote to seeds+holdout on swiss roll
(next section) before treating 0.966 as a held-out claim.

### Reproduce

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --y examples/exploratory/data/s_curve_tbin.npy \
  --name paper_s_curve --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 --cmap Spectral
# for manifolds, raise geo:
#   edit overlay or use --only after setting lambda_geo=0.5 in axes.RECOMMENDED
```

---

## Swiss roll (N=2000, D=3, intrinsic dim ≈ 1.97)

**Setup.** Classic `sklearn.datasets.make_swiss_roll`. Fold-back manifold —
the case `frame_tangent=True` and delayed `frame_ramp` exist for.
`lambda_geo=0.5`, `n_landmarks` / `tau_scale` from calibrate (L≈179, τ≈0.18).

### Frame ladder (held out, 3 seeds, matched null)

Baseline without frame (`lambda_frame=0`, `lambda_geo=0.5`), held out, 3 seeds:

| metric | mean | sd |
|---|---|---|
| `t`-bin accuracy | 0.908 | 0.021 |
| trustworthiness@15 | 0.995 | 0.000 |
| kNN overlap@15 | 0.790 | 0.009 |
| geodesic Spearman | 0.747 | 0.030 |

Full ladder (`lambda_frame` ∈ {0, 0.25, 0.5, 1.0} delayed + early 0.5) is
produced by:

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/swiss_roll_X.npy \
  --y examples/exploratory/data/swiss_roll_tbin.npy \
  --name paper_swiss_roll --sweep swiss_roll_frame \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 --cmap Spectral
```

Results land in `examples/out/exploratory/paper_swiss_roll/summary.csv`.
Expected qualitative pattern (from s-curve / swiss-cone in-sample ladders):
kNN overlap rises with `lambda_frame`; delayed beats early. Front-loading the
geodesic gauge has been removed from the API (measured harmful).

Reference bar: `examples/out/exploratory/paper_swiss_roll/bar.json`.

### Reproduce (recommended)

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/swiss_roll_X.npy \
  --y examples/exploratory/data/swiss_roll_tbin.npy \
  --name paper_swiss_roll --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8
# then set lambda_frame=0.5, frame_ramp=(0.5,0.75) for the fold-back recipe
```

---

## Iris (N=150, D=4, 3 classes) — small-N showcase

Iris is not here to beat UMAP on fidelity. Three well-separated classes in 4-D
leave little headroom — PCA already does well. The showcase is the axes that
do **not** depend on winning there.

### Mechanical facts

| fact | consequence |
|---|---|
| `pyramid_min_reps=256` > N | Pyramid never builds; set `pyramid_scales=0` |
| Default `n_landmarks=256` > N | Use calibrate: L=64 → coverage ≈ 2.4 |
| Holdout 20% ≈ 30 points | Seeds 0–4; `trust@15` is undefined (k > n_eval); conformal p-floor ≈ `1/31` |
| densMAP `transform()` | Raises `NotImplementedError` — cannot OOS |

To show what `pyramid_level_weights` would do at this N, a separate didactic
sweep lowers `pyramid_min_reps` to 16 (`iris_pyramid_weights` →
`examples/out/exploratory/paper_iris_pyramid_weights/atlas.png`). That is not
the production small-N recipe.

### Held out, 5 seeds (`iris_canonical` recommended)

| metric | leanmap | null | margin | UMAP | UMAP null | PCA-2D |
|---|---|---|---|---|---|---|
| 5-NN label accuracy | 0.880 ± 0.065 | 0.380 | **+0.50** | 0.953 ± 0.051 | 0.387 | 0.920 ± 0.065 |
| kNN overlap@15 | 0.911 ± 0.016 | 0.864 | +0.05 | 0.912 ± 0.074 | 0.806 | 0.993 ± 0.010 |
| geodesic Spearman | 0.859 ± 0.049 | 0.882 | −0.02 | 0.891 ± 0.082 | 0.860 | 0.968 ± 0.011 |

Fidelity margins are thin: null kNN overlap is already ~0.86 at this N, and PCA
wins raw geodesic because the ambient metric is nearly the right geometry.
Report null-corrected numbers only. Chance accuracy for 3 classes is 1/3;
leanmap's null sits near that.

### Artefact size and inference (the actual showcase)

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| model on disk | 664 KB | **25 KB** | 1 KB |
| us / point at B=1 | 158 | 2286 | 39 |
| us / point at B=64 | **8.9** | 230 | 0.7 |
| speedup vs UMAP at B=64 | **~26×** | 1× | — |

At N=150 the fixed-size argument does **not** favour leanmap: UMAP's pickled
training set is tiny, while the encoder (width 128) still carries its full
parameter count. What still holds is **throughput** — ~26× cheaper OOS
placement at batch 64 — and densMAP's inability to transform at all. The
fixed-artefact win appears as N grows (digits: leanmap 944 KB fixed vs UMAP
605 KB already carrying 1438×64 data, and growing).

Wall time to fit iris recommended: ~46 s / seed on MPS.

### Reproduce

```bash
python examples/exploratory/calibrate.py \
  --X examples/exploratory/data/iris_X.npy --target-perp 8
python examples/exploratory/reference.py \
  --X examples/exploratory/data/iris_X.npy \
  --y examples/exploratory/data/iris_y.npy --name paper_iris \
  --holdout 0.2 --seeds 0 1 2 3 4 --null shuffle
python examples/exploratory/master.py \
  --X examples/exploratory/data/iris_X.npy \
  --y examples/exploratory/data/iris_y.npy \
  --name paper_iris --sweep iris_canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 3 4 --null shuffle --target-perp 8
python examples/exploratory/bench_inference.py \
  --X examples/exploratory/data/iris_X.npy \
  --leanmap leanmap=examples/out/exploratory/paper_iris/recommended__default__seed0 \
  --sklearn umap=examples/out/exploratory/paper_iris/reference/umap_default__none__seed0 \
  --sklearn pca=examples/out/exploratory/paper_iris/reference/pca2d__none__seed0
# Didactic pyramid-weights figure (not the production recipe):
python examples/exploratory/master.py \
  --X examples/exploratory/data/iris_X.npy \
  --y examples/exploratory/data/iris_y.npy \
  --name paper_iris_pyramid_weights --sweep iris_pyramid_weights \
  --holdout 0.2 --seeds 0 --target-perp 8 --atlas
```

---

## Cross-dataset: inference cost

Measured on digits (n_train=1438), CPU, warmed-up transforms verified against
saved embeddings:

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| model on disk | 944 KB | 605 KB | 1 KB |
| latency, 1 point | 0.20 ms | 2.37 ms | 0.02 ms |
| **per point at batch 512** | **4.3 µs** | **696 µs** | 0.09 µs |

**~162× cheaper out-of-sample placement at batch** while also returning the
cover score. The gap is structural: leanmap inference cost does not grow with
training-set size; UMAP's `transform()` searches the stored training data.

## Cross-dataset: design goal

Stated against amortised inference + a usable novelty flag, leanmap clears a
fidelity constraint under the imposed metric and delivers what UMAP cannot: fast
repeated placement, a layout that keeps empty interior between clusters, and a
calibrated ambient OOD score that catches probes buried inside a cluster. On L2
neighbourhood / density columns and on s-curve null-corrected margins it matches
or beats UMAP. Which trade is right depends on whether the map is the product
or the encoder is.

## Cover-certificate limit

Minimum-norm repair onto the cover acceptance region (union of balls around
landmarks) is closed form. Repaired probes pass the conformal test by
construction and remain ~2× farther from real digits under EMD than a real
held-out digit. The certificate is loose because landmarks are a coarse
isotropic support model in ambient space — tightening α cannot fix the shape.

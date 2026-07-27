# Matching UMAP on digits: result, and how it was found

## Summary

Leanmap matches UMAP on 8x8 digits. Held out on 3 seeds it reaches **0.941 +/- 0.016**
5-NN label accuracy against UMAP's 0.987, and it **beats** UMAP on kNN overlap
(0.571 vs 0.538) and on ambient/embedded density correlation (0.709 vs 0.248).
The matched null sits at 0.089, i.e. chance for 10 classes.

Those figures are at `min_dist=0.1`, which the [clumping](#the-layout-clumps-and-min_dist-is-the-only-knob-that-touches-it)
section later retires as the default: it makes layouts knot up, progressively
worse the longer they train. At the new default of 0.5 digits gives up about one
seed-sd of accuracy, 0.915 +/- 0.024, in exchange for halving the layout's area
distortion. Both configurations are reported side by side below; the digits
tables in the middle of this document are all at 0.1, since that is what the
UMAP-matching investigation ran under.

Before this work the same harness scored 0.705, and no amount of graph or loss
tuning moved it: 20+ configurations spanning every exposed axis all landed
between 0.588 and 0.705, against PCA-2D's 0.603. The embedding was a PCA in
disguise. The cause turned out to be a two-parameter interaction, `pca_skip` and
`lr`, which is invisible to any one-at-a-time sweep because each parameter alone
makes things *worse*.

## Why the old digits runs were discarded

The 20 runs previously under `examples/out/exploratory/digits/` were unusable and
have been deleted (only `bar.json` and `ingest.json` remain):

- `lambda_geo: 0.0` while the baseline specified `0.5` -- they predated the baseline.
- `pyramid_level_weights: [1,1,1,1]` truncated to `[1,1,1]` at the 3 levels
  actually built, so there was no coarse attraction at all.
- `pyramid_coarse_backbone: 1.0` under the pre-fix library, i.e. the MST that
  overwrote coarse edge weights.
- 60 epochs, one seed, no holdout, no null.

## How the comparison is made

Three habits, all wired into `master.py`, decide whether a number means anything.

**Matched nulls (`--null shuffle`).** The identical configuration is refit on
column-shuffled input. Chance depends on the configuration, not just the data, so
a null is only valid for the run it was produced with. This is not pedantry: on
PDB, shuffled input already reaches trustworthiness 0.926, so UMAP's raw 0.954 is
almost entirely chance. Only the margin above the null carries information.

**Held-out scoring (`--holdout 0.2`).** Leanmap is parametric, so it can be scored
on points it never trained on via `result.embed()`. At small N, in-sample metrics
flatter the embedding.

**Seeds (`--seeds 0 1 2`).** Gives the run-to-run spread that a difference has to
clear. On digits that spread is +/-0.016 on accuracy and +/-0.075 on geodesic
Spearman, which retires most of the small differences seen along the way.

The bar itself comes from `reference.py` (UMAP, densMAP, PCA-2D, and raw-X as the
ceiling, each with its own null), and the configuration comes from `calibrate.py`
rather than from guesswork: intrinsic dimension, median kNN radius, an
`n_landmarks` bracket by coverage and occupancy, `tau_scale` by perplexity
bisection, and the number of pyramid levels that will actually be built.
`--target-perp` re-derives `tau_scale` from the anchor geometry of whatever data
is passed, so it is never a carried-over literal.

## The result

Digits, held out, 3 seeds:

| metric | leanmap | sd | UMAP nn10 | UMAP default | PCA-2D | null |
|---|---|---|---|---|---|---|
| 5-NN label accuracy | 0.941 | 0.016 | 0.987 | 0.975 | 0.603 | 0.089 |
| ARI vs truth | 0.825 | 0.041 | 0.911 | 0.829 | 0.393 | -0.001 |
| silhouette of labels | 0.596 | 0.025 | 0.656 | 0.616 | 0.105 | -0.119 |
| trustworthiness@15 | 0.946 | 0.008 | 0.988 | 0.987 | 0.829 | 0.604 |
| **kNN overlap@15** | **0.571** | 0.019 | 0.538 | 0.535 | 0.151 | 0.085 |
| geodesic Spearman | 0.644 | 0.075 | 0.606 | 0.660 | 0.547 | 0.091 |
| **density Spearman** | **0.709** | 0.036 | 0.248 | 0.272 | 0.004 | 0.156 |

Leanmap matches UMAP's ARI, is a little short on accuracy, silhouette and
trustworthiness, and is clearly ahead on kNN overlap and density preservation.
`compare_digits.png` shows ten separated clusters where the old configuration
produced a PCA blob.

## How it was found

Working the axes in priority order produced a long run of null results. Each row
is 5-NN label accuracy, held out, against PCA-2D's 0.603:

| axis tried | range | verdict |
|---|---|---|
| corrected baseline, 3 seeds | 0.624 - 0.705 | starting point |
| pyramid weights, 5 tuples | 0.616 - 0.655 | flat |
| `min_dist` 0.0 / 0.5 | 0.657 - 0.663 | flat |
| `n_negatives` 15 | 0.632 | flat |
| `lambda_geo` 0.0 / 1.0 | 0.641 - 0.649 | flat |
| epochs 120 / 240 | 0.663 - 0.674 | converged, flat |
| objective stripped to local | 0.588 - 0.641 | *worse* |
| width 768 / depth 4 | 0.669 | flat |
| `pca_skip=False` | 0.412 | much worse |
| `lr` 5e-3 / 2e-2 alone | 0.683 - 0.685 | flat |

Everything within noise of PCA. Two hypotheses were killed without spending
compute on sweeps:

- *The encoder is too constrained to expand cluster gaps.* The backbone Linears
  are wrapped in `spectral_norm`, which would make the map roughly 1-Lipschitz.
  But `train.py` silently disables spectral norm on MPS (`aten::vdot` is
  unimplemented), and every run here was on MPS. The encoder was already an
  unconstrained 3x384 MLP, so the constraint theory was dead on arrival.
- *A long-range attraction term is holding the classes together.* The baseline
  carries three (pyramid coarse levels, `lambda_geo`, `lambda_lm`). Stripping all
  three to a purely local, UMAP-like objective made accuracy *worse* (0.588), so
  attraction was not the culprit.

The clue that mattered was the shape of the failure rather than any single
number: kNN overlap was well *above* PCA (0.38 vs 0.151) while label accuracy was
*equal* to PCA. Neighbourhoods were being preserved but classes were never pulled
apart, and the learned residual was contributing almost nothing -- with the skip
off it scored 0.412, worse than the linear path it replaced. That reads as an
undertrained head, not a bad objective, and `lr` was the one knob never varied on
digits.

### `pca_skip` and `lr` are one decision, not two

| `pca_skip` | `lr` | 5-NN acc |
|---|---|---|
| on (default) | 1e-3 (default) | 0.65 |
| off | 1e-3 | 0.41 |
| on | 2e-2 | 0.68 |
| **off** | **2e-2** | **0.93** |

With the skip on, the output is `pca(x_n) + residual` and the residual head is
initialized at std 1e-4, so training *starts at* plain PCA. The unconstrained
linear path keeps supplying the layout and pins the result near PCA no matter how
the graph or losses are tuned. Turn the skip off and the head must build the
layout from a near-zero init, which it cannot do at the default `lr` -- so it
scores worse than leaving the skip alone. Only both changes together escape. Any
one-at-a-time sweep reads both as dead ends.

This is documented on both fields in `src/leanmap/config.py`, because the shipped
defaults are a trap: `pca_skip=False` is effectively broken at the default `lr`.

### Refinement

From the no-skip base at 0.933, the two things that helped were lowering
`lambda_geo` and training to 240 epochs -- but only the second survives scrutiny.
At 60 epochs `lambda_geo=0` looked like the single best accuracy (0.964) while
collapsing geodesic Spearman to 0.503, below PCA's 0.547, suggesting a real
local/global trade. Re-run at 240 epochs that trade disappears: the whole ladder
from `lambda_geo=0` to `1.0` spans 0.944-0.955 accuracy, inside the +/-0.016 seed
spread, and geodesic at `lambda_geo=0` recovers to 0.663. The collapse was an
artifact of short training, not of the loss term.

| `lambda_geo` @240ep | acc | ARI | trust15 | ov15 | geodesic |
|---|---|---|---|---|---|
| 0.0 | 0.953 | 0.755 | 0.952 | 0.589 | 0.663 |
| 0.15 | 0.953 | 0.834 | 0.947 | 0.576 | **0.724** |
| 0.5 | 0.944 | 0.807 | 0.946 | 0.573 | 0.708 |
| 1.0 | 0.955 | 0.784 | 0.945 | 0.577 | 0.695 |

So `lambda_geo` is a weak knob on digits once training is long enough. The
shipped `matched` configuration keeps `lambda_geo=0.15` because it has the best
geodesic and ARI of the four, not because the others fail:

```
pca_skip=False, lr=2e-2, lambda_geo=0.15, epochs=240,
n_neighbors=15, pyramid_level_weights=(1,2,8),
n_landmarks and tau_scale derived by calibrate.py / --target-perp
```

## Does it transfer?

**s-curve** (N=2000, D=3, intrinsic dim measured at 1.94 -- correctly a 2-D
sheet). Straight transfer of the digits configuration, with `lambda_geo` raised
back to 0.5 since a smooth manifold wants the global pull. Held out, 3 seeds,
with matched nulls (chance accuracy 0.125):

| metric | leanmap | sd | null | margin | UMAP | UMAP null | margin |
|---|---|---|---|---|---|---|---|
| `t`-bin accuracy | 0.907 | 0.024 | 0.109 | +0.798 | 0.968 | 0.121 | +0.847 |
| trustworthiness@15 | 0.994 | 0.001 | 0.963 | **+0.030** | 0.999 | 0.995 | +0.004 |
| kNN overlap@15 | 0.737 | 0.010 | 0.560 | **+0.178** | 0.779 | 0.669 | +0.111 |
| geodesic Spearman | **0.985** | 0.002 | 0.872 | **+0.113** | 0.955 | 0.858 | +0.097 |
| density Spearman | 0.535 | 0.061 | 0.593 | *-0.057* | 0.485 | 0.448 | +0.037 |

Leanmap unrolls the sheet cleanly, has the best geodesic fidelity of any method
here (0.985 raw, and the best margin), and leads UMAP on the null-corrected
trustworthiness and kNN-overlap margins despite trailing slightly on the raw
values. For reference densMAP and PCA-2D score 0.677 and 0.306 on kNN overlap.

`lambda_geo` matters more here than on digits, but saturates: 0.0 / 0.15 / 0.5 /
1.0 give kNN overlap 0.709 / 0.710 / 0.746 / 0.752 at seed 0. The real step is
0.15 -> 0.5; across 3 seeds `lambda_geo=1.0` is within one standard deviation of
0.5 on every metric (kNN overlap 0.747 +/- 0.008 vs 0.737 +/- 0.010). Anything
from 0.5 up is equivalent on this feed.

Two cautions this feed makes obvious. PCA-2D "wins" `t`-bin accuracy at 0.984
while scoring 0.306 on kNN overlap -- `t` is close to linear, so that column
rewards *not* unrolling, and kNN overlap is what separates unrolling from
projecting. And the density margin is negative: the s-curve is sampled uniformly,
so ambient density is nearly constant and that correlation is mostly noise, which
the null exposes (UMAP's margin is only +0.037). Density preservation is not a
meaningful claim on this feed for any method.

**PDB** (N=5000, D=9 validation metrics). Little to find: the raw-feature ceiling
for resolution bins is 0.323 against 0.20 chance, and UMAP (0.300), densMAP
(0.298) and PCA-2D (0.291) are indistinguishable. Read only null-corrected
margins here. Leanmap wins the trustworthiness (+0.123 vs +0.029), kNN overlap
(+0.113 vs +0.060) and density (+0.110 vs +0.050) margins, and loses geodesic
(+0.195 vs +0.397).

On the weights question the null earned its keep: fine-emphasis `(8,1,1)` and
pyramid-off have *negative* density margins -- they score worse on real data than
the same configuration scores on shuffled data, so their apparent density
structure is not real. Coarse-weighted pyramids survive calibration; fine-weighted
ones do not. `flat` `(1,1,1)` is the balanced choice on PDB.

Most importantly, the discrete PDB clusters that motivated this investigation are
gone. Under the fixed pyramid and a calibrated configuration, leanmap agrees with
UMAP, densMAP and PCA that the data is one continuous cloud with a resolution
gradient. The old clusters were an artifact of the truncated weight tuple and the
coarse-backbone MST.

## Scoring against a metric neither method was given

Everything above has a hole in it. `trust_*`, `knn_overlap_*`, `ambient_spearman`
and `geodesic_spearman` all define the truth as pixel L2 on the 64-D vector, and
both leanmap and UMAP build their kNN graph from pixel L2. Those columns partly
reward reproducing the input each method was handed. The one column that escapes
that, `label_acc_Z`, rewards clumping: an embedding that shatters the data into
ten tight balls scores perfectly while destroying all geometry, and UMAP clumps
harder than leanmap.

Pixel L2 is also a bad image metric. Once two digits stop overlapping,
`||a-b||` saturates at `sqrt(||a||^2+||b||^2)` and carries no information about
how far apart they are.

So the comparison was rerun against **EMD** -- 2-D optimal transport of ink mass
across the pixel grid, exact `W1` via network simplex, each image normalised to
unit mass. Neither method ever sees it, which is the point.

### Is EMD its own geometry?

On the full 1797x1797 digit matrix, Spearman against EMD:

| band | pixel L2 | L2-graph geodesic |
|---|---|---|
| local | 0.691 | 0.531 |
| mid | 0.280 | 0.173 |
| global | 0.366 | 0.273 |
| overall | 0.762 | 0.588 |

L2 tracks EMD well locally and poorly at range, exactly as the saturation
argument predicts: among the third of pairs EMD calls most distant, L2's 99th
percentile is only 1.22x its median. At 0.37 in the far band, EMD is emphatically
not a relabelled L2, so it can arbitrate. Gate passed.

**But the geodesic claim failed.** Chaining short L2 hops was supposed to recover
EMD where the direct distance cannot. On a densely sampled synthetic manifold it
does -- a Gaussian blob swept along a curved path gives far-band Spearman 0.87
for the geodesic against 0.50 for L2, and that is what `tests/test_emd_geodesic.py`
pins down. On real digits it does the opposite: 0.59 overall against L2's 0.76.
1797 points in 64-D are too sparsely sampled for chaining to help. The geodesic
is a property of the sampling, not only of the metric.

### Held out, same rows, three seeds

Both methods now fit on the same training split and place the holdout through
their own out-of-sample path (`result.embed()` / `transform()`), scored on
identical rows. Every number below is EMD-referenced, held out, mean +/- sd over
seeds 0-2:

| metric | leanmap | UMAP | UMAP nn10 | PCA-2D |
|---|---|---|---|---|
| EMD Shepard | 0.224 +/- 0.066 | **0.337** +/- 0.058 | 0.358 +/- 0.018 | 0.325 +/- 0.013 |
| &nbsp;&nbsp;far band | -0.101 +/- 0.114 | **0.166** +/- 0.049 | 0.145 +/- 0.022 | -0.094 +/- 0.016 |
| EMD kNN overlap@15 | 0.484 +/- 0.008 | **0.545** +/- 0.013 | 0.546 +/- 0.014 | 0.315 +/- 0.008 |
| EMD trust@15 | 0.896 +/- 0.010 | **0.942** +/- 0.005 | 0.939 +/- 0.002 | 0.777 +/- 0.005 |
| EMD geodesic | 0.345 +/- 0.066 | **0.430** +/- 0.076 | 0.467 +/- 0.037 | 0.433 +/- 0.018 |
| EMD retrieval (new->train) | 0.288 +/- 0.003 | **0.433** +/- 0.007 | 0.440 +/- 0.004 | 0.152 +/- 0.004 |

Paired bootstrap on identical query points, leanmap minus UMAP: kNN overlap
-0.067 [-0.084, -0.052], retrieval -0.153 [-0.174, -0.133], Shepard
-0.090 [-0.103, -0.079]. Every interval excludes zero.

**UMAP wins on EMD fidelity, in every band, by more than the seed spread.** The
pre-registered "different, not better" outcome required leanmap to take the
global band while conceding the local one. It does not: leanmap's far-band
Spearman is *negative*, -0.10, meaning that among the pairs EMD calls most
distant, its embedding distances carry no usable ordering. PCA-2D is negative
there too, at -0.09. This is the sharpest correction to the table further up:
leanmap's apparent geodesic lead (0.644 vs 0.606) was measured against the
L2 graph, the same object it was fit from. Against a metric it was not given,
that lead reverses.

leanmap's run-to-run spread is also 2-4x UMAP's on these columns (+/-0.066 vs
+/-0.058 on Shepard, +/-0.114 vs +/-0.049 in the far band).

### The holdout penalty that was not there

UMAP was expected to degrade out of sample. It does not, on this feed:

| train - holdout, EMD kNN overlap@15 | leanmap | UMAP | UMAP nn10 | PCA-2D |
|---|---|---|---|---|
| gap | +0.010 +/- 0.026 | +0.006 +/- 0.029 | +0.002 +/- 0.033 | -0.001 +/- 0.021 |

All four are zero within noise. Two things had to be fixed before this number
meant anything. First, `transform()` was checked for silent degeneracy: UMAP's
holdout placement has spread ratio 0.967 and centroid shift 0.14 train-sd with
every point distinct, so it is working, and its holdout deficit is genuinely
absent rather than hidden behind a broken call. Second, the train regime is
subsampled to the holdout's size (359 rows), because overlap@15 among 359 points
is mechanically easier than among 1438 -- before that correction the "gap" read
-0.10 to -0.17 for every method, which was measuring set size.

**densMAP has no out-of-sample path at all**: `transform()` raises
`NotImplementedError: Transforming data into an existing embedding not supported
for densMAP`. It cannot take part in a holdout comparison, and its rows in the
older bar were in-sample only.

UMAP and PCA, by contrast, place new points on an already-trained model exactly
as leanmap does, and persisting the fitted estimator loses nothing: reloading a
pickled UMAP and calling `transform()` reproduces its fit-time placements
bit-for-bit (max abs difference 0.0 over 224 probes). Being parametric is
therefore *not* what distinguishes leanmap here -- all three can absorb new
points without a refit. What differs is the artifact: a pickled UMAP carries the
training data and its neighbour index and so grows with N (619 KB at N=1438),
while leanmap's `model.pt` is encoder weights plus landmarks, fixed by
architecture (967 KB here -- larger at this N, and unchanged if N grew).

### Probes: leanmap leads on structured novelty, and only on structured novelty

Fourteen families, 16 jittered variants each, **never in training**, all rescaled
to the median digit's ink mass so total intensity cannot give them away. Twelve
are structured -- smile, frown, neutral, surprised, cross, X, ring, checkerboard,
horizontal and vertical bars, dot, block -- and two are unstructured controls:

- `noise`: independent random pixels. The floor. A detector that misses this is
  broken, and it bounds how much of a structured-probe score is really about
  structure.
- `shuffled`: a real digit with its 64 pixels spatially permuted. The intensity
  histogram and total ink are preserved *exactly*, so nothing about brightness,
  contrast or sparsity can separate it from a digit. Only spatial layout can.

AUROC separating probes from held-out digits, using only distance to the nearest
training point in the map:

| probe set | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| 12 structured families | **0.809** | 0.746 | 0.647 |
| 2 random controls | 0.811 | **0.893** | 0.590 |

**The ranking reverses between the two rows.** leanmap is better at noticing that
a smiley is not a digit; UMAP is better at noticing that random pixels are not a
digit (`noise` 0.939 vs 0.837, `shuffled` 0.847 vs 0.784). Without the controls
the pooled average reads as a flat leanmap win, and that would have been the
wrong conclusion: the advantage is specific to *structured* novelty, not novelty
in general.

Per family, the spread matters as much as the mean:

| family | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| ring | **0.918** | 0.492 | 0.583 |
| checker | 0.913 | **0.941** | 0.601 |
| ex | **0.871** | 0.719 | 0.485 |
| cross | **0.819** | 0.622 | 0.755 |
| smile | 0.733 | 0.692 | **0.760** |
| frown | 0.761 | **0.804** | 0.607 |
| vbars | 0.761 | **0.925** | 0.410 |
| hbars | **0.802** | 0.756 | 0.391 |
| dot | 0.748 | 0.826 | **1.000** |
| *noise* | 0.837 | **0.939** | 0.512 |
| *shuffled* | 0.784 | **0.847** | 0.668 |

leanmap is the most uniform of the three, spanning 0.733-0.918. UMAP swings from
0.492 on `ring` -- chance, it places rings indistinguishably from real digits --
to 0.941 on `checker`. PCA-2D falls *below* chance on three families (`hbars`
0.391, `vbars` 0.410, `ex` 0.485), meaning those probes land closer to the
training data than genuine held-out digits do.

### How far away, in map units

AUROC says only which is farther, not by how much. The underlying quantity is the
distance from each probe to its nearest *training* point in the map. Raw values
are in each embedding's own units and are not comparable across methods -- the
three maps differ in scale by an order of magnitude -- so the comparable figure
is the ratio to what a genuine held-out digit does. A ratio of 1.0 means a probe
sits as close to the training data as a real new digit:

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| embedding scale (train sd) | 1.614 | 6.611 | 13.119 |
| NN distance, held-out digit -> train | 0.034 | 0.064 | 0.543 |
| NN distance, probe -> train | 0.113 | 0.174 | 0.770 |
| **ratio** | **3.28x** | 2.72x | 1.42x |
| probes caught at 5% FPR | **0.438** | 0.379 | 0.231 |

PCA barely moves probes at all -- 1.42x, which is why its AUROC is weakest. The
per-family ratios explain the outliers in the AUROC table exactly: UMAP places
rings at **1.02x**, indistinguishable from a real digit, hence chance AUROC; PCA
puts `hbars` at 0.86x and `vbars` at 0.87x, i.e. *closer* than a real digit,
hence below-chance AUROC; and PCA flings `dot` out to 40.47x, hence its lone
1.000. leanmap's extremes are `ring` 10.23x and `checker` 8.35x.

**The practical caveat is the last row.** At a threshold admitting 5% of real
held-out digits, even the best map catches only 44% of probes. Map distance is a
weak detector no matter which method draws the map, because 2-D is simply not
enough room to keep off-manifold points away from everything. leanmap's ambient
cover score flags 100% of the same probes at alpha=0.05 -- roughly 2.3x the
detection rate of its own 2-D geometry. If the goal is OOD detection rather than
visualisation, score in the ambient space and use the map to look at the result.

### Neither method puts an outlier outside its map

UMAP's `transform()` initialises a new point at a weighted mean of its nearest
training neighbours' embedding coordinates -- a convex combination -- so it
should be structurally unable to place anything beyond the existing embedding,
however unlike the training data the input is. leanmap's encoder is a learned
function with no such constraint, so it *could* extrapolate. Testing that against
the convex hull of the training embedding:

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| inside train hull: real held-out digit | 0.993 | 0.956 | 0.988 |
| inside train hull: probe | 0.921 | 0.899 | 0.821 |
| radius vs train p95: real digit | 0.658 | 0.573 | 0.652 |
| radius vs train p95: probe | 0.513 | 0.682 | 0.604 |

The prediction holds for UMAP -- and equally for leanmap, which was not expected.
About 90% of probes land inside the training hull under both, and leanmap keeps
*more* of them inside than UMAP does. Being parametric does not buy extrapolation
here; the encoder has only ever been asked to produce points in the region where
its landmarks live, and that is what it produces.

### But "inside the map" is not "on the data"

Both numbers above are global, and in a clustered embedding a global statistic
cannot see the difference between landing on a cluster and landing in the gap
between two. The hull swallows every void it encloses, so a probe dropped into
empty interior still counts as inside; the radius is measured from a centroid
that sits in a hole. Local occupancy is the measure that separates the two. Take
the training set's own 15-NN radius as the neighbourhood scale and count how many
training points a new point actually has around it:

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| neighbours within r: real held-out digit | 15.3 | 13.7 | 14.3 |
| neighbours within r: probe | 2.5 | 5.3 | 10.0 |
| probe occupancy as a fraction of a digit's | 0.16 | 0.39 | 0.70 |
| probes with **zero** neighbours | 0.286 | 0.272 | 0.143 |
| real held-out digits with zero neighbours | 0.017 | 0.030 | 0.020 |

The median probe under UMAP sits in a neighbourhood 2.4x as populated, relative
to what a real digit sees, as the median probe under leanmap. The distribution is
where it becomes unambiguous -- probe occupancy by percentile, as a fraction of
the holdout-digit median:

| | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| leanmap | 0.00 | 0.04 | 0.16 | 0.35 | 0.70 |
| UMAP | 0.00 | 0.04 | 0.39 | 0.90 | 1.34 |
| PCA-2D | 0.00 | 0.25 | 0.70 | 1.05 | 1.44 |

UMAP's upper half really is absorbed: at p75 a probe has 90% of the company a
real digit has, and at p90 it has *more* -- 1.34x -- meaning those probes are not
merely near a cluster but inside its core, which is exactly what a convex
combination of nearest-neighbour coordinates should produce. leanmap's upper half
reaches only 0.35 and 0.70; it has no equivalent population of probes buried in
the data.

What is *not* true is that UMAP absorbs all of them. 27.2% of UMAP's probes have
no training point within the neighbourhood radius at all, statistically the same
as leanmap's 28.6% (+/-0.136 across seeds), while real held-out digits are
isolated 2-3% of the time under either. The two maps differ in the bulk of the
distribution, not in the tail: UMAP pulls its typical probe onto the data and
leanmap does not, but both leave a comparable quarter of them stranded.

So leanmap's AUROC advantage is not distance from the data in any global sense --
its probes are more central than UMAP's by radius, and more often inside the
hull. It comes from empty interior: leanmap's clusters leave voids between them
and probes land in those voids, central yet 3.28x farther from the nearest
training point than a real digit is. UMAP's interior is filled, so a probe that
lands centrally lands *in* something. Detectability here is a question of where a
map reserves empty space, not whether it can represent "far away" at all. Neither
can.

Every family has an AUROC ceiling of 1.000 under both EMD and L2 to the nearest
training digit, so all three maps are discarding signal that was fully present in
the input. `emd_bench.png` shows the mechanism: UMAP scatters probes *into* its
digit clusters while leanmap gives them their own region. leanmap pays for that
with the worst pattern separation of the three (1.816 vs UMAP 1.977 and PCA
2.796) -- it flags probes as novel while squashing the differences between them.

leanmap's shipped OOD path is perfect on all fourteen families: landmark-cover
AUROC 1.000, and the conformal test calibrated on real held-out digits flags 100%
of probes at alpha=0.05. UMAP has no equivalent, so this is a capability column
rather than a win on a shared axis.

`conformal.png`, from `plot_conformal.py`, plots that path directly. The holdout
is split in half there, so the calibration set and the test set are disjoint and
no probe is ever involved in calibration. Four things come out of it:

- The cover distributions separate cleanly. Real held-out digits sit at 0.5-1.5x
  the calibration median, probes at 1.5-4x, with the alpha=0.05 threshold falling
  in the gap. The spike at exactly zero is the 179 landmarks, which are drawn from
  the training set and are their own nearest landmark.
- The test is valid, not just powerful: p-values on the disjoint test half track
  the uniform diagonal (0.056 / 0.107 / 0.217 / 0.502 against nominal 0.05 / 0.10
  / 0.20 / 0.50).
- **Every** probe is pinned at `p = 1/(n_calib+1) = 0.0056`. The 100% flag rate is
  therefore a floor effect and understates the margin -- the test has run out of
  resolution, not out of power. Resolving these probes further needs a bigger
  calibration set, not a better score.
- Cover and the picture disagree by a wide margin on the same points: AUROC 1.000
  against 0.814 for distance to the nearest training point in the 2-D map. The
  score is ambient and the map is a projection of it, which is the whole reason
  the section above finds map geometry a weak detector.

Colouring the probes by their own cover score, on the same scale as the digits,
makes the mechanism visible: probes saturate the scale *wherever they land*,
including the ones sitting inside a digit cluster. Position on the map carries
almost none of the information the score has. Scoring each probe both ways puts a
number on it -- **57.3%** of probes fall below the map's 5%-FPR threshold while
above the conformal threshold. For the majority of probes the picture and the
score do not merely differ in confidence, they disagree outright.

### Per-family novelty is stable under the score and is seed noise under the map

Plotting median novelty per pattern, both expressed as a multiple of what a real
held-out digit gets, produces the sharpest result in the figure. Averaged over the
fourteen families, the across-seed spread is **1% for cover and 48% for map
distance**. Cover reproduces to the third digit (`dot` scores 3.62 / 3.53 / 3.60
on the three seeds); the map does not come close (`checker` scores 19.00 / 2.53 /
4.56, `ring` 16.31 / 10.51 / 5.06).

The two also rank the families differently. Rank correlation is negative on every
seed (-0.19, -0.29, -0.49) though not individually significant at n=14, and the
stable disagreements are large: `dot` is the most anomalous pattern under cover on
all three seeds and only mid-table in the map, while `ring` and `checker` are the
map's two loudest alarms and among cover's quietest.

This retires the per-family reading of the AUROC table above. Those numbers come
from map geometry, so they inherit its instability -- `smile` ranges 0.63 to 0.83
across seeds under leanmap -- and the ordering of which pattern a map finds
strange is mostly an accident of that seed's layout. The aggregate structured-vs-
unstructured split survives because it averages over families; claims about
individual patterns should be made from the ambient score, which is reproducible,
and not from the picture.

### What the novelty score is actually established to do

**It flags genuine novelty, and the flag means something.** All 672 probes across
three seeds are caught at alpha=0.05. That is only informative because the same
test leaves real data alone: held-out digits it has never seen are flagged at
0.056 against a nominal 0.05. The score is not simply calling everything strange.

**A probe can sit inside a cluster and still be flagged.** This is the part the
map cannot do. Measuring local occupancy in the layout, 62.4% of probes land with
as much company as a real held-out digit, and 3.1% land in a cluster *core* --
denser than the median real digit's neighbourhood. Every one of them is flagged,
including all 21 core cases (mostly `frown` and `smile`, which the layout drops
straight into the digit mass). The reason is that "inside a cluster in 2-D" and
"on the manifold in 64-D" are different statements: the projection collapses an
ambient distance that the cover score still sees. This is the mechanism behind the
57.3% figure above, in its most extreme form.

**Two scope limits.** First, cover is `min_l ||x - M_l||`, an ambient L2 distance
to the nearest landmark, so what is established is detection of points off the
*landmark support*. Here that coincides with "outside the training data" because
every family is L2-far (the L2 AUROC ceiling is 1.000 across the board). The
documented blind spot -- a novel point lying on the manifold, near a landmark --
is untested, and `conformal.py` says as much.

Second, the separation is complete but not comfortable. The closest probe clears
the farthest real held-out digit by only 4%, 3% and 13% on the three seeds. AUROC
1.000 is an honest measurement of these probes and should not be read as
headroom; a probe designed to be L2-near would plausibly land inside that gap.
Constructing one requires optimisation rather than sampling, which is the separate
experiment noted below.

That perfection is also why one prediction remains untested. The cover score is
`Dm.min(dim=1)`, an ambient **L2** distance to the nearest landmark, and
`conformal.py` warns that shifts leaving cover unchanged are invisible to it. The
expectation was that L2 saturation would hide a probe sitting L2-near a real
digit. No probe here is L2-near: the L2 ceiling is 1.000 for all fourteen
families, including `shuffled`, which was the best candidate since it preserves
the intensity histogram exactly. Permuting pixels moves ink far enough to be
obvious under both metrics. An L2-near, EMD-far probe has to be constructed
deliberately -- by optimisation, not by sampling -- and that is a separate
experiment.

## The minimum-norm repair, and what it says about the certificate

If a point is flagged, the natural next question is what it would take to make it
acceptable. With the encoder frozen and only the input allowed to move, leanmap
does not need an optimiser for this, because the acceptance region has a shape
that can be projected onto exactly. Cover is `min_l ||x - M_l|| / s`, an ambient
Euclidean distance to the nearest landmark up to a fixed scale (`s = 25.83` here,
identity view, verified against the model rather than assumed), so

    {x : cover(x) <= tau}  =  union of balls B(M_l, tau*s)

and the Euclidean projection onto a union of balls is the projection onto the
nearest one. The global minimiser is closed form:

    delta* = (1 - tau*s/d) (M_l* - x),   ||delta*|| = d - tau*s,   d = ||x - M_l*||

which rearranges to `x_repaired = (tau*s/d) x + (1 - tau*s/d) M_l*` -- a convex
combination of the flagged point and the nearest training exemplar. The repair
cannot invent anything; it can only drag the input toward data the model already
has. `manifold_repair.py` runs it.

It does what it is supposed to. Median cover falls from 1.949 to the threshold
1.138, the conformal p-value goes from 0.0056 to 0.0556, and 0% of probes remain
flagged. That is guaranteed by construction and is not evidence of anything.

The perturbation is not small. `||delta|| / ||x||` has median 0.342, and the blend
weight on the landmark is 0.417 -- the minimum-norm fix drags a probe 42% of the
way to a real training digit. The middle row of `repair.png` shows the result:
still recognisably a block, a cross, a checkerboard, with a ghost digit over it.

**Scored against EMD -- which neither the model nor the repair has access to --
the repaired points are still off the manifold.** Distance to the nearest training
digit, as a multiple of what a real held-out digit gets:

| | EMD to nearest training digit | x a real digit |
|---|---|---|
| real held-out digit | 0.206 | 1.00x |
| probe | 0.792 | 3.85x |
| **repaired probe** | **0.442** | **2.15x** |

The repair closes a little over half the gap and then stops, because it stops the
instant the test is satisfied. Sweeping the blend weight shows where the test
would have to be set for the two to agree: EMD parity with a real digit needs a
blend of 0.79, nearly twice the 0.42 the cover test accepts.

The obvious objection is that projecting to the boundary is deliberately the
weakest repair that passes. It does not rescue the certificate: aiming instead at
a *typical* digit's cover score rather than the acceptance threshold needs a blend
of 0.54 and still leaves the point at 2.05x.

### Why the certificate is loose

The repaired points are not merely EMD-far. Measuring plain L2 to the nearest
training digit -- not to a landmark -- they sit at 29.36 against 16.28 for a real
held-out digit, 1.80x, and separate from real digits at AUROC 0.999. A repaired
point is therefore close to *a landmark* while being nowhere near *any actual
digit*, which is the whole failure in one sentence.

The cause is the shape of the acceptance region rather than the threshold on it.
Cover compares against 179 landmarks standing in for 1438 training points, and the
sublevel set is a union of **isotropic** balls around them. Real digits pass the
test while also being near other digits, so the calibration never notices that the
balls extend in directions the data does not. In 64 dimensions that
over-approximation is enormous, and the minimum-norm repair walks straight into
it: it moves toward the nearest landmark only far enough to cross the surface,
which puts it on a part of the sphere where no data lives.

Two consequences. First, tightening `alpha` cannot fix this -- it shrinks the
radius uniformly and the repair simply blends further, staying on the boundary
wherever that boundary is. Closing the gap needs a statistic that knows about
direction, or a support model finer than 179 isotropic balls -- a prediction the
next section tests directly, and largely refutes. Second, this is
*not* the L2-near, EMD-far probe the earlier section left open: these points are
L2-far too. That construction still has to be built deliberately, and the minimum
norm repair does not hand it over for free.

## Testing the prediction: direction-aware scores, and why they do not close it

The section above predicted that the fix was a statistic that knows about
direction, or a support model finer than 179 isotropic balls. That prediction is
half right, and the half that is wrong matters more.

`nonconformity.py` fits local charts post-hoc on the frozen encoder, changing
nothing about the model. The two-NN estimator puts the intrinsic dimension of
digits at 8.9, and Voronoi cells around the landmarks hold a median of 6 training
points -- far too few for a 9-dimensional tangent -- so each chart is fitted from
the landmark's `k=48` nearest training points, with deliberately overlapping
neighbourhoods, keeping `q=10` local principal directions. The artefact grows by
494 KB and nothing in it scales with the training set, so the fixed-size property
survives.

Four candidate scores, plus L2-to-nearest-training-point as a reference for what
retaining the entire training set would buy. Each is calibrated on one half of the
holdout, checked on the disjoint other half, and only then compared on power:

| score | flag rate, real digits | probes | cover-repaired |
|---|---|---|---|
| `cover` (isotropic balls) | 0.056 | 100% | **0%** |
| `residual` (off local sheet) | 0.050 | 100% | **100%** |
| `residual_z` (standardised) | 0.056 | 100% | **100%** |
| `mahalanobis` (local ellipsoids) | 0.056 | 100% | **100%** |
| `knn_train` (keeps all data; not fixed-size) | 0.050 | 100% | **100%** |

Every score holds its nominal 0.05 on real held-out digits, so the power column is
meaningful. And every direction-aware score catches all 224 points that defeat
cover by construction, at the same alpha, for 494 KB and no retained data.

That table is close to circular, though: those points were built to sit exactly on
*cover's* boundary, so a different statistic catching them is unsurprising. The
test that counts is giving each score its own minimum-norm attack -- for
`residual` the acceptance region is a slab and the projection is closed form, for
`mahalanobis` it is a union of ellipsoids and the projection collapses to a 2-D
problem because the region is rotationally symmetric within the tangent block and
within the normal block -- and then judging the results by EMD, which none of them
has seen.

| | L2 to train | EMD | x a real digit |
|---|---|---|---|
| real held-out digit | 16.3 | 0.206 | 1.00x |
| probe, unrepaired | | 0.792 | 3.85x |
| min-norm repair vs `cover` | | 0.442 | 2.15x |
| min-norm repair vs `residual` | | 0.434 | **2.11x** |
| min-norm repair vs `mahalanobis` | | 0.445 | **2.16x** |
| min-norm repair vs `knn_train` | 23.7 | 0.392 | **1.91x** |
| probe pulled to *median* train L2 | 16.3 | 0.278 | 1.35x |

Making the region direction-aware buys nothing at all: 2.15x becomes 2.11x with a
slab and 2.16x with ellipsoids. Making the support model as fine as it can possibly
be -- a union of balls around all 1438 actual training points, which abandons the
fixed-size property entirely -- buys 2.15x to 1.91x. The shape and the resolution
of the acceptance region were not the binding constraint.

The last row locates what is. Drag a probe to exactly the L2 distance from the
training set that a *typical* real digit sits at, and it is still 1.35x away in
EMD. So the 2.15x gap decomposes into roughly three parts: about 1.35x is
irreducible, because L2 proximity in pixel space simply does not imply manifold
membership; most of the remainder is that alpha=0.05 puts the threshold at a 95th
percentile, which is a genuinely permissive place to draw a boundary; and only
about 12% of it (1.91x against 2.15x) is the coarseness of the 179-ball summary
that the previous section blamed.

The conclusion is that a more complex nonconformity score is worth building, but
not for the reason assumed. Two goals were conflated:

* **Detection power on realistic novelty**, including inputs engineered or drifted
  to sit at the threshold. Here the direction-aware scores are a clear, cheap win:
  0% to 100% at fixed alpha and fixed artefact size. `mahalanobis` is the one to
  prefer, since it bounds displacement along the sheet as well as off it, closing
  the sliding-along-the-manifold blind spot `conformal.py` already documents.
* **A tight geometric certificate**, where "accepted" implies "genuinely on the
  data manifold". This is not reachable by refining an L2 sublevel set at any
  region complexity, and the experiments above bound how much is even available.
  It needs the score built in a metric that matches the data's own notion of
  similarity, or a density/energy model rather than a distance.

## Inference cost, which is what leanmap was built for

Everything above scores map quality, and by that measure UMAP wins. But the design
goal is amortised inference on a precomputed encoder plus a usable novelty flag,
which makes fidelity a constraint to clear rather than the objective. That axis had
not been measured. `bench_inference.py` measures it: each model is loaded from
disk, and new points are pushed through it on CPU, after a warm-up call that pays
numba's JIT and with every transform checked against the embedding the run already
committed to disk (UMAP and PCA bit-exact, leanmap to float32 rounding).

| | leanmap | UMAP | PCA-2D |
|---|---|---|---|
| model on disk (n_train=1438) | 944 KB | 605 KB | 1 KB |
| latency, single point | 0.20 ms | 2.37 ms | 0.02 ms |
| latency, batch of 512 | 2.20 ms | 356.6 ms | 0.05 ms |
| **per point at batch 512** | **4.3 us** | **696 us** | 0.09 us |

**leanmap places a point 162x more cheaply than UMAP at batch, and 12x more
cheaply one at a time** -- while returning the cover score in the same call, so it
is doing strictly more work than `transform()`, which returns coordinates only.

The gap is structural rather than an implementation detail, and inspecting the two
artefacts shows why. leanmap's `model.pt` is 226,920 parameters: an encoder whose
shape is fixed, plus `n_landmarks x D` landmark coordinates, and `n_landmarks` is
set by coverage rather than by N. Inference is a forward pass whose cost does not
depend on how much data the model was fitted to. UMAP's `model.pkl` stores
`_raw_data` (1438x64 -- the entire training set), `embedding_`, and the 1438x1438
graph, because `transform()` has to find each new point's neighbours among the
training data and then run optimisation epochs to place it. Both the artefact and
the per-point cost grow with N, so the 162x measured at n=1438 is a lower bound on
what it would be at a larger fit. Confirming that shape needs a scaling sweep,
which this run does not do.

PCA is the honest floor here: 0.09 us per point and a 1 KB model, but 0.639 probe
AUROC and 0.151 kNN overlap. It is cheap because it is barely modelling anything,
which is the trade the other two are being measured against.

### Verdict

The pre-registered rule scored map fidelity, and on that axis the answer is
unchanged. But stated against the design goal -- fast repeated inference on a
frozen encoder, with problematic points flagged -- leanmap does the job UMAP
cannot: **162x cheaper out-of-sample placement, a fixed-size artefact, and a
calibrated novelty flag that catches probes buried inside a cluster**, at a
fidelity cost that is real but bounded (kNN overlap 0.571 vs 0.535 in its favour,
Shepard correlation 0.090 against it). Whether that trade is worth taking depends
on whether the map is the product or the encoder is.

On the narrower question the rule actually asked: **UMAP is better, not merely
different, at preserving image geometry** -- every EMD-referenced fidelity column,
every band, every CI excluding zero. **leanmap is better at noticing structured
novelty**
specifically: +0.06 AUROC on the twelve structured families, but -0.08 on the
unstructured controls, where UMAP wins. It also has a calibrated OOD score UMAP
does not. The expected out-of-sample penalty for UMAP does not exist on this
feed.

One caveat bounds all of it: the `matched` configuration was tuned against
L2-referenced metrics and label accuracy, and was never tuned for EMD. This
measures the shipped configuration, not the best leanmap could do if EMD
fidelity were the objective.

Reproducing:

```bash
python examples/exploratory/make_emd.py \
  --X examples/exploratory/data/digits_X.npy --image-shape 8 8 --n-jobs 8

python examples/exploratory/reference.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy --name digits_holdout \
  --probes examples/exploratory/data/digits_probes_X.npy \
  --seeds 0 1 2 --holdout 0.2 --null shuffle --n-neighbors 10

python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name digits_emd_lm --sweep matched --holdout 0.2 --seeds 0 1 2 \
  --target-perp 8 --probes examples/exploratory/data/digits_probes_X.npy \
  --emd examples/exploratory/data/digits_emd.npy

python examples/exploratory/emd_bench.py \
  --X examples/exploratory/data/digits_X.npy \
  --emd examples/exploratory/data/digits_emd.npy \
  --probes examples/exploratory/data/digits_probes_X.npy \
  --probe-kind examples/exploratory/data/digits_probes_kind.npy \
  --Z leanmap=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \
  --Z umap=examples/out/exploratory/digits_holdout/reference/umap_default__none__seed0 \
  --out examples/out/exploratory/digits_emd
```

## The layout clumps, and `min_dist` is the only knob that touches it

Good scores on the battery hid a plain visual defect: leanmap layouts are knots
and voids rather than an even spread. The s-curve makes it measurable, because a
uniformly sampled sheet has a known target -- the coefficient of variation of the
kNN radius in the layout (`spacing_cv`) is 0.181 for the true flattening. Leanmap
scored 0.561. UMAP scores 0.371, PCA-2D 0.299.

Isolating the cause took one sweep: of `lambda_lm` (the landmark attraction),
`n_negatives` (repulsion) and `min_dist`, only the last moved the number at all.
Dropping `lambda_lm` to 0 or raising `n_negatives` to 20 changed nothing
(0.561 -> 0.565, 0.567); `min_dist=0.5` took it to 0.345, and stacking the other
two on top added nothing. So landmark attraction is not the culprit and more
repulsion is not the cure.

### Why `min_dist` and nothing else

`min_dist` never enters the loss directly. It is consumed once, by
`find_ab_params`, which fits `1/(1 + a d^2b)` to UMAP's piecewise target curve
and hands the training loop the pair `(a, b)`. The attractive force near contact
then goes as `d^(2b-1)`, which makes `b = 1` a real boundary: below it the force
decays *more slowly* than the separation, so a pair that is already close is
pulled proportionally harder, and neighbourhoods run away into knots. `b` is a
closed-form function of `(min_dist, spread)` alone -- independent of the data,
the graph and the layout scale -- and crosses 1 at `min_dist = 0.197 * spread`.
The old default of 0.1 gives `b = 0.895`, inside the collapse regime.

UMAP ships the same 0.1 and is fine, because its SGD kernel clips gradients to
+/-4. Leanmap differentiates the analytic loss with no equivalent safeguard, so
what UMAP absorbs by clipping, leanmap has to avoid by staying out of the regime.

### The ladder, and where the argument fails

A 0.1-0.8 ladder on both feeds, bracketing the `b = 1` crossing at 0.2. `drift`
is the slope of `spacing_cv` against epoch over the last 100 epochs, from the new
per-epoch monitor:

| `min_dist` | b | s-curve `spacing_cv` | drift /100ep | digits 5-NN acc | digits `area_sd` |
|---|---|---|---|---|---|
| 0.1 | 0.895 | 0.561 | **+0.064** | 0.958 | 0.561 |
| 0.15 | 0.949 | 0.528 | **+0.075** | 0.950 | 0.531 |
| 0.2 | 1.003 | 0.486 | **+0.069** | **0.969** | 0.539 |
| 0.3 | 1.112 | 0.424 | **+0.046** | 0.967 | 0.470 |
| 0.5 | 1.334 | 0.345 | -0.005 | 0.947 | 0.294 |
| 0.8 | 1.681 | 0.291 | -0.006 | 0.939 | 0.289 |

For scale, on the same measurement: true flattening 0.181, a Poisson sample
0.188, PCA-2D 0.299, UMAP 0.371, densMAP 0.603.

**There is no knee at `b = 1`.** `spacing_cv` falls smoothly across the whole
ladder and keeps falling well past the boundary, so the exponent argument does
not by itself pick a value, and the default cannot be justified as "`b >= 1`".

What does change discretely is the *drift*, and only the per-epoch monitor could
see it. Up to `min_dist=0.3` the layout keeps clumping the longer it trains; from
0.5 it is flat. That is the runaway the mechanism predicts -- it simply arrives
at `b ~ 1.3` rather than at 1. It also means anything below 0.5 makes the result
a function of the epoch budget, which is the practical reason to avoid it.

### What was chosen

`min_dist` now defaults to **0.5**, the smallest value that stops the clumping
growing during training. It is not free. Digits, held out, 3 seeds, old default
against new:

| metric | `min_dist=0.1` | `min_dist=0.5` | UMAP nn10 |
|---|---|---|---|
| 5-NN label accuracy | 0.941 +/- 0.016 | 0.915 +/- 0.024 | 0.987 |
| ARI vs truth | 0.825 +/- 0.041 | 0.786 +/- 0.055 | 0.911 |
| trustworthiness@15 | 0.946 +/- 0.008 | 0.925 +/- 0.008 | 0.988 |
| kNN overlap@15 | 0.571 +/- 0.019 | 0.549 +/- 0.017 | 0.538 |
| geodesic Spearman | 0.644 +/- 0.075 | 0.562 +/- 0.083 | 0.606 |
| density Spearman | 0.709 +/- 0.036 | 0.667 +/- 0.048 | 0.248 |
| `area_sd` (lower = less distortion) | 0.561 | **0.340 +/- 0.026** | -- |

Accuracy costs about one seed-sd and the geodesic difference is inside two;
what is bought is a 40% cut in area distortion and a layout that no longer
degrades with training. On a clustered feed that is a judgement call, so the
guidance is explicit in `config.py`: lower `min_dist` toward 0.2 when class
separation matters more than an undistorted layout, and never go below 0.2.

One caveat on reading these columns. On digits the matched null reaches
`spacing_cv` 0.434 against the real data's 0.457, and `area_sd` 0.355 against
0.340 -- i.e. neither uniformity metric separates real structure from shuffled
input there. They measure a habit of the algorithm, not a property of the data,
and are only diagnostic on a feed with a known sampling density like the s-curve.

### Tooling added

- `find_ab_params` warns when the fitted `b < 1`, and `min_dist_for_b(target_b,
  spread)` bisects for the value that fixes it, so the warning names a number.
- `calibrate.py` reports `a`, `b`, `d_half`, the regime (collapse / marginal /
  stable) and both thresholds, and folds a `min_dist` suggestion into
  `RECOMMEND`. No measurement needed -- `b` depends on nothing but the config.
- `master.py --monitor N` logs `spacing_cv`, `area_sd` and the density Spearman
  every N epochs to `uniformity_trace.csv`, plus a plot against the Poisson floor
  and the UMAP mark. This is what exposed the drift; end-of-run numbers cannot.
  Note the density Spearman is logged for context but is *not* the alarm: it
  reads 0.709 for leanmap against UMAP's 0.248 while leanmap is the clumpier of
  the two, because a rank correlation is blind to knots and voids. `area_sd` is
  its informative counterpart.

Gradient clipping is the untried alternative -- it is precisely what lets UMAP
run at 0.1 -- and was left out of scope here.

## Open items

- **Spectral norm is silently device-dependent.** `config.spectral_norm` defaults
  to `True` but is disabled at runtime on MPS, so the same configuration trains a
  different model on Apple silicon than on CPU or CUDA. Every result in this
  report is therefore an unnormalized backbone. Whether the constraint helps is
  untested; settling it needs a CPU run of the `matched` configuration.
- `calibrate.py` reports the pyramid level count for the full N, but a holdout
  fits fewer points and a level can disappear (PDB: 4 at N=5000, 3 at N=4000).
  Weight tuples are currently sized to the training split by hand; the function
  should take the holdout fraction.
- PDB was run at one seed. The geodesic/density ordering across weight tuples is
  large and systematic, but the accuracy differences are inside the seed spread.
- **PDB has not been re-run at `min_dist=0.5`.** Every PDB number in this report
  is at 0.1, i.e. in the collapse regime, so the layouts behind them are clumpier
  than the current default would produce. Since the PDB weight conclusions rest
  on null-corrected margins rather than on layout geometry, they are unlikely to
  reverse, but that is an assumption until the `pdbw__flat` run is repeated.
- **leanmap loses global structure under a metric it was not fit on.** Far-band
  EMD Spearman is -0.10 on digits, i.e. no usable ordering among the most
  distant pairs, where UMAP holds +0.17. The `matched` configuration was tuned
  against L2-referenced metrics, so the obvious next step is a sweep with
  `emd_spearman_global` as the objective rather than `label_acc_Z`, to find out
  whether this is the configuration or the method.
- **The cover-score failure mode is still untested.** Landmark cover is an
  ambient L2 distance, so it should be blind to anything that is L2-near and
  EMD-far. All fourteen probe families are far under both metrics (AUROC ceiling
  1.000 either way), including the pixel-shuffled control that preserves the
  intensity histogram exactly. Sampling does not produce an L2-near, EMD-far
  image on this grid; it would have to be constructed by optimisation --
  minimise L2 to a target digit subject to a large EMD -- which is the
  experiment that would settle it.
- **UMAP places rings at chance (AUROC 0.492).** Rings are the one structured
  family it cannot tell from a digit at all, while it handles checkerboards at
  0.941. Whether that is a `min_dist` artifact or something about closed curves
  under its local-connectivity assumption is unexplained.
- The EMD-referenced numbers in the section above are only for the `matched`
  config. The other sweeps in `axes.py` have not been rescored, and
  `master.py --emd` now makes that cheap: the reference matrix is built once and
  reused.
- **`mahalanobis` is not wired into the library.** It dominates `cover` on every
  measured axis at fixed alpha and adds 494 KB, but it lives in
  `nonconformity.py` as an evaluation script. Promoting it means fitting the
  charts inside `fit_embed` and storing them in the checkpoint, plus deciding
  whether `embed` returns it alongside cover or in place of it.
- **The chart hyperparameters were not tuned.** `q=10` follows the two-NN
  intrinsic dimension of 8.9 and `k=48` is the smallest neighbourhood comfortably
  above it; neither was swept. Since power is already saturated at 100% on
  everything tested, a harder probe set is needed before tuning means anything.
- **The 1.35x floor is a digits-and-EMD number, not a law.** It says L2 proximity
  does not imply manifold membership *on this data under this ground metric*. How
  much of it is the 8x8 grid, and how much would survive on larger images or on
  the PDB features where the ground metric is not spatial at all, is untested.
- **No score was tested against an attack it had not been shown.** Each score was
  attacked with its own minimum-norm repair. Whether `mahalanobis` survives an
  attack designed against `knn_train`, or against a combination, is open, and is
  the natural next step if the flag is ever load-bearing against an adversary
  rather than against drift.

## Reproducing

```bash
# bar, configuration, then the matched run
python examples/exploratory/reference.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy --name digits

python examples/exploratory/calibrate.py \
  --X examples/exploratory/data/digits_X.npy --target-perp 8

python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name digits_match --sweep matched \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 \
  --bar examples/out/exploratory/digits/bar.json

python examples/exploratory/compare_figure.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --Z examples/out/exploratory/digits_match/matched__digits__seed0/Z.npy \
  --out examples/out/exploratory/digits_match/compare_digits.png --cmap tab10
```

The `min_dist` ladder, with the per-epoch uniformity trace:

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --y examples/exploratory/data/s_curve_tbin.npy \
  --name s_curve_mdist --sweep min_dist_scurve \
  --holdout 0.2 --seeds 0 --target-perp 8 --shepard none --monitor 5

# full-N uniformity, comparable to the reference numbers quoted above
python examples/exploratory/uniformity.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --Z examples/out/exploratory/s_curve_mdist/*/Z.npy \
  --intrinsic examples/exploratory/data/s_curve_t.npy \
             examples/exploratory/data/s_curve_w.npy
```

Note `uniformity.py` scores the saved full-N `Z.npy` while `summary.csv` scores
the 20% holdout, and a random subsample of a layout is itself Poisson-like, so
the two disagree in level (0.561 vs 0.308 at `min_dist=0.1`). Compare like with
like; the reference points quoted here are all full-N.

The nonconformity comparison, including each score's own minimum-norm attack and
the EMD adjudication (the `--attack` pass recomputes 4.5M EMD pairs, about 90s at
`--n-jobs 8`):

```bash
python examples/exploratory/nonconformity.py \
  --run examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \
  --X examples/exploratory/data/digits_X.npy \
  --probes examples/exploratory/data/digits_probes_X.npy --attack
```

Sweeps used along the way, all in `axes.py`: `umap_match`, `weights`, `epochs`,
`packing`, `objective`, `optim`, `refine`, `refine2`, `matched`, `s_curve`,
`uniform`, `min_dist_scurve`, `min_dist_digits`, `pdb`, `pdb_weights`.

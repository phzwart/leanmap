# Exploratory visual / metric guide

Single **master** driver that ingests researcher-style arrays and sweeps
high-signal leanmap knobs into a visual + metric atlas.

There are **no** per-dataset scripts here. Prepare `X` (and optional color) once,
then point the driver at the files.

## Quick start

```bash
pip install -e ".[examples,cpu]"

# 1) dump ~2k S-curve, swiss-cone-with-hole, and 8×8 digits
python examples/exploratory/prepare_feeds.py

# 2) dry-run the Phase-1 plan
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --color examples/exploratory/data/s_curve_t.npy \
  --name s_curve \
  --sweep phase1 \
  --dry-run

# 3) train all three feeds (idempotent; writes Z.npy per run)
python examples/exploratory/run_phase1.py --device mps
# or one feed:
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --color examples/exploratory/data/s_curve_t.npy \
  --name s_curve \
  --sweep phase1 \
  --atlas
```

Rebuild atlas later:

```bash
python examples/exploratory/make_atlas.py examples/out/exploratory/s_curve
```

## Ingest contract

| flag | required | meaning |
|------|----------|---------|
| `--X` | yes | `(N, D)` features — `.npy` / `.npz` / `.csv` |
| `--color` / `--y` | no | length-`N` labels for scatter coloring |
| `--name` | no | output tag (default: stem of `--X`) |
| `--sweep` | no | named sweep (`phase1`) |
| `--only` | no | one axis or `run_id` |
| `--dry-run` | no | print planned runs |
| `--force` | no | redo even if `metrics.json` exists |
| `--atlas` | no | write `atlas.png` after the sweep |
| `--monitor N` | no | log layout uniformity every `N` epochs |

The driver never imports sklearn generators or `make_swiss_cone`.

## Phase-1 axes

Baseline matches library defaults (incl. **`λ_geo=0.5`**, frozen landmarks,
soft tau, cohesive pyramid) then 1D ablations:

- **graph:** `n_neighbors`, `pyramid_scales`, `pyramid_coarse_backbone`
- **landmarks:** mode (ambient / geodesic / poisson), `n_landmarks`,
  `learn_landmarks`, `tau_scale`, `learn_tau`
- **structure:** `lambda_geo` around 0.5 (`0 / 0.1 / 0.25 / 1.0`),
  `lambda_frame` (early vs delayed ramp)
- **packing / init:** `min_dist`, `pca_skip`
- **interactions:** geo × delayed frame; poisson × geo

See [`axes.py`](axes.py) for the exact run list.

Re-sweep frame weight with λ_geo held at 0.5::

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --color examples/exploratory/data/s_curve_t.npy \
  --name s_curve_frame_weight --sweep frame_weight --atlas
```

## Ambient ↔ embed density

Each run also writes `density_link.png` (and `.npz`): kNN local density in
ambient vs projection, plus residual map. Standalone::

```bash
python examples/exploratory/density_link.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --Z examples/out/exploratory/s_curve/lambda_geo__0.5/Z.npy \
  --out examples/out/exploratory/s_curve/lambda_geo__0.5/density_link.png
```

## Outputs

Under `examples/out/exploratory/{name}/{run_id}/`:

- `scatter.png`, `shepard_ambient.png`, `shepard_geodesic.png`
- `metrics.json`, `config.json`, `Z.npy`
- with `--monitor N`: `uniformity_trace.csv`, `uniformity_trace.png`
- `split.npz` — the exact `train_idx` / `hold_idx` used. Scoring scripts should
  read this rather than re-deriving the split; re-derivation only stays correct
  while every copy of the rule agrees.
- `model.pt`, `cover.npy`, and with `--probes`, `Z_probe.npy` / `probe_cover.npy`

Aggregates under `examples/out/exploratory/{name}/`:

- `summary.csv`, `atlas.png`, `ingest.json`

## Scoring against EMD instead of pixel L2

Most geometry columns here treat pixel L2 as the truth, and every embedder is
fit from an L2 kNN graph, so those columns partly reward reproducing the input.
Pixel L2 is also blind at range: once two digits stop overlapping, `||a-b||`
saturates and stops ordering anything.

`make_emd.py` builds an Earth Mover's Distance reference — exact `W1` optimal
transport of ink mass over the pixel grid — that no method is fit on. It also
generates the out-of-distribution probes, matched to the median digit's ink mass
so that brightness alone cannot separate them: twelve structured patterns (smile,
frown, ring, bars…) plus two unstructured controls (`noise`, and `shuffled`, a
real digit with its pixels permuted so the intensity histogram is preserved
exactly). Report the two groups separately — on digits the ranking between
methods reverses between them, so a pooled average is misleading.

```bash
# ~70s for 1797 digits + 192 probes at 8 jobs; cached and reused thereafter
python examples/exploratory/make_emd.py \
  --X examples/exploratory/data/digits_X.npy --image-shape 8 8 --n-jobs 8
```

It prints a go/no-go: if L2 already orders far pairs the way EMD does, EMD cannot
arbitrate between two embeddings that were both fit from L2. On digits it does
not (far-band Spearman 0.37), so the reference is worth having.

Pass `--emd` to `master.py` to add `emd_spearman` / `emd_knn_overlap_*` to the
battery, and use `emd_bench.py` for the head-to-head: it scores any set of saved
embeddings in the train, holdout and retrieval regimes with paired bootstrap CIs,
and reports probe AUROC per family against the EMD ceiling. See
`REPORT_umap_match.md` for the digits result.

Two plotting scripts go with it. `plot_embeddings.py` puts the maps side by side
with the probes overlaid on a density field, so you can see which parts of each
map are empty. `plot_conformal.py` covers what the map cannot: leanmap's ambient
landmark-cover score and the conformal test built on it. It splits the holdout
into disjoint calibration and test halves, checks that test-half p-values come out
uniform before claiming any power, scores every probe both ways, and breaks the
result down by pattern with the across-seed spread shown -- which is how you find
out that per-family novelty is reproducible under the ambient score and not under
the map:

```bash
python examples/exploratory/plot_conformal.py \
  --probe-kind examples/exploratory/data/digits_probes_kind.npy \
  --Z seed0=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \
  --out examples/out/exploratory/digits_emd/conformal.png
```

It needs `cover.npy` and `probe_cover.npy`, which `master.py` writes for leanmap
runs, and skips any run that has neither.

`bench_inference.py` measures the other half of the design goal — what it costs to
place a *new* point on an already-fitted model, which is the operation that
actually runs repeatedly. It loads each saved model, warms it up so numba's JIT is
not charged to the first timing, verifies the transform reproduces that run's own
saved `Z_probe.npy`, and reports per-point cost across batch sizes together with
the artefact size:

```bash
python examples/exploratory/bench_inference.py --X examples/exploratory/data/digits_X.npy \
  --verify examples/exploratory/data/digits_probes_X.npy \
  --leanmap leanmap=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \
  --sklearn umap=examples/out/exploratory/digits_holdout/reference/umap_default__none__seed0
```

On digits that is 4.3 us per point for leanmap against 696 us for UMAP at batch
512, and the reason is structural: leanmap's artefact is a fixed-size encoder plus
landmarks, while UMAP's keeps the training data and graph because `transform()`
searches them.

`manifold_repair.py` asks what it would take to make a flagged point acceptable,
with the encoder frozen and only the input moving. No optimiser is needed: cover
is an ambient distance to the nearest landmark, so the acceptance region is a
union of balls and the minimum-norm projection onto it is closed form — move at
the nearest landmark until you touch its surface. The script runs that projection
and then checks the result against EMD, which neither the model nor the repair can
see:

```bash
python examples/exploratory/manifold_repair.py \
  --run examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \
  --X examples/exploratory/data/digits_X.npy \
  --probes examples/exploratory/data/digits_probes_X.npy \
  --probe-kind examples/exploratory/data/digits_probes_kind.npy \
  --out examples/out/exploratory/digits_emd/repair.png
```

The repaired points pass the conformal test by construction and are still 2.15x
farther from real digits than a real digit is, so the check that matters is the
independent one. See `REPORT_umap_match.md` for why the certificate is loose.

Changing the probe set does not mean retraining anything. **Both** families place
new points on an already-trained model — `PLANE.embed()` for leanmap,
`transform()` for UMAP and PCA — so the only requirement is that the fitted model
was kept. `master.py` writes `model.pt` and `reference.py` writes `model.pkl`,
and `embed_probes.py` handles either:

```bash
python examples/exploratory/embed_probes.py \
  --probes examples/exploratory/data/digits_probes_X.npy \
  examples/out/exploratory/digits_emd_lm/matched__digits__seed* \
  examples/out/exploratory/digits_holdout/reference/umap_default__none__seed*
```

Reloading a pickled UMAP and calling `transform()` reproduces its fit-time
placements bit-for-bit, so nothing is lost by deferring. The one exception is
densMAP, whose `transform()` raises `NotImplementedError`.

The real difference is what each artifact contains, not whether it works. A
pickled UMAP carries the training data and its neighbour index, so it grows with
N (619 KB at N=1438). leanmap's `model.pt` is encoder weights plus landmarks,
fixed by the architecture rather than the training-set size (967 KB here —
*larger* at this N, and unchanged if N grew).

## Matching UMAP on a positive control

Digits is the proving ground: UMAP separates its 10 classes cleanly, so there is
a known-good target and ground-truth labels to score against. Three habits make
the numbers mean something, and all three are wired into the driver:

- `--null shuffle` refits the *same configuration* on column-shuffled input.
  Chance levels depend on the config, not just the data, so a null is only valid
  for the run it was produced with. On PDB this is decisive: shuffled input
  already reaches trustworthiness 0.926, so the raw score is nearly
  uninformative and only the null-corrected margin carries signal.
- `--holdout 0.2` scores points the model never trained on. leanmap is
  parametric, and at small N in-sample metrics flatter the embedding.
- `--seeds 0 1 2` gives the run-to-run spread a difference has to clear.

Set the bar first, then derive the configuration instead of guessing at it:

```bash
# reference bar: UMAP / densMAP / PCA-2D plus matched nulls -> bar.json
python examples/exploratory/reference.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy --name digits

# intrinsic dim, kNN radius, n_landmarks bracket, tau_scale by perplexity,
# and the number of pyramid levels that will actually be built
python examples/exploratory/calibrate.py \
  --X examples/exploratory/data/digits_X.npy --target-perp 8

# the matched configuration, scored against the bar
python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name digits_match --sweep matched \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 \
  --bar examples/out/exploratory/digits/bar.json
```

`--target-perp` re-derives `tau_scale` from the anchor geometry of whatever data
is passed, so it is never a carried-over literal. Note `calibrate.py` reports the
level count for the *full* N; with a holdout, fewer points are fit and a level
can disappear (PDB: 4 levels at N=5000, 3 at N_train=4000), so size the weight
tuple to the training split or it gets truncated.

### `pca_skip` and `lr` are one decision, not two

The gap to UMAP on digits was not in the graph or the losses. Weights, epochs
(30-480), `n_neighbors`, `min_dist`, `n_negatives`, `lambda_geo`, `lambda_lm`,
pyramid depth and width/depth were all flat at 0.59-0.71 5-NN label accuracy,
against PCA-2D's 0.603 -- the embedding was a PCA in disguise. The cause is an
interaction that is invisible to any one-axis sweep:

| `pca_skip` | `lr` | 5-NN acc |
|---|---|---|
| on (default) | 1e-3 (default) | 0.65 |
| off | 1e-3 | 0.41 |
| on | 2e-2 | 0.68 |
| **off** | **2e-2** | **0.93** |

With the skip on, the free-scale linear PCA path supplies most of the layout and
pins it near PCA. With it off, the residual head starts from a 1e-4 init and is
undertrained at the default `lr`, scoring *worse* than leaving it on. Only both
changes together escape, so either one alone reads as a dead end.

The resulting `matched` configuration (`pca_skip=False`, `lr=2e-2`,
`lambda_geo=0.15`, 240 epochs, derived `n_landmarks` / `tau_scale`) reaches
0.941 +/- 0.016 against UMAP's 0.987 on held-out digits, and **beats** UMAP on
kNN overlap (0.571 vs 0.538) and density correlation (0.709 vs 0.248). Its
matched null sits at 0.089, i.e. chance for 10 classes.

## Transferring to the s-curve

```bash
python examples/exploratory/quantile_bins.py \
  --values examples/exploratory/data/s_curve_t.npy \
  --out examples/exploratory/data/s_curve_tbin.npy --bins 8
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --y examples/exploratory/data/s_curve_tbin.npy \
  --name s_curve_match --sweep s_curve \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 --cmap Spectral
```

A straight transfer of the digits configuration with `lambda_geo` back at 0.5,
since a smooth sheet wants the global pull that digits did not. Leanmap unrolls
it with the best geodesic fidelity of any method tried (0.985 vs UMAP's 0.955)
and leads UMAP on the null-corrected trustworthiness and kNN-overlap margins.

Watch which column you believe here: PCA-2D scores the *highest* `t`-bin accuracy
(0.984) while managing only 0.306 kNN overlap, because `t` is close to linear and
that column rewards not unrolling. Density is uninformative on this feed for
every method -- uniform sampling means near-constant ambient density, and the
null makes that visible.

## Clumping, and why `min_dist` defaults to 0.5

Leanmap layouts used to come out as knots and voids. On the s-curve, which is
uniformly sampled and therefore has a known target, the kNN-spacing CV was 0.561
against 0.181 for the true flattening, 0.371 for UMAP and 0.299 for PCA-2D.

Exactly one knob touches it. `lambda_lm=0` and `n_negatives=20` changed nothing;
`min_dist=0.5` took it to 0.345. The reason is that `min_dist` is consumed once,
by `find_ab_params`, which fits `1/(1 + a d^2b)` and hands training the exponent
`b`. Attraction near contact goes as `d^(2b-1)`, so below `b = 1` the force
decays slower than the separation, close pairs are pulled proportionally harder,
and neighbourhoods collapse. `b` crosses 1 at `min_dist = 0.197 * spread`, so the
old default of 0.1 (`b = 0.895`) sat in the collapse regime. UMAP ships 0.1 too
but clips its gradients; leanmap differentiates the loss directly and cannot.

A 0.1-0.8 ladder settled the value, and only half of the argument survived it:

| `min_dist` | b | s-curve `spacing_cv` | drift /100ep | digits acc |
|---|---|---|---|---|
| 0.1 | 0.895 | 0.561 | +0.064 | 0.958 |
| 0.2 | 1.003 | 0.486 | +0.069 | **0.969** |
| 0.3 | 1.112 | 0.424 | +0.046 | 0.967 |
| 0.5 | 1.334 | 0.345 | -0.005 | 0.947 |
| 0.8 | 1.681 | 0.291 | -0.006 | 0.939 |

There is **no knee at `b = 1`** -- uniformity improves smoothly straight through
it -- so `b >= 1` is a floor, not a criterion. What flips discretely is the
*drift*: below 0.5 the layout keeps clumping the longer it trains, at and above
0.5 it is flat. Hence the default of 0.5, the smallest value that stops the
runaway. It costs about one seed-sd of digits accuracy (0.941 -> 0.915 over 3
seeds) and halves the area distortion, so lower it toward 0.2 when class
separation matters more than layout geometry, and never go below 0.2.

The drift is invisible end-of-run, which is what `--monitor` is for:

```bash
python examples/exploratory/master.py \
  --X examples/exploratory/data/s_curve_X.npy \
  --y examples/exploratory/data/s_curve_tbin.npy \
  --name s_curve_mdist --sweep min_dist_scurve \
  --holdout 0.2 --seeds 0 --target-perp 8 --monitor 5
```

Per run this writes `uniformity_trace.csv` and a plot of `spacing_cv` vs epoch
against the Poisson floor and the UMAP mark. `calibrate.py` reports the same
regime from the config alone (`a`, `b`, `d_half`, collapse / marginal / stable),
and `find_ab_params` warns when a run is configured below `b = 1`.

Two traps when reading uniformity. `spacing_cv` is only diagnostic where the
sampling density is known: on digits the shuffled null scores 0.434 against the
real data's 0.457, so it measures the algorithm, not the data -- use `area_sd`
there. And the density Spearman is *not* an alarm; leanmap reads 0.709 against
UMAP's 0.248 while being the clumpier of the two, because a rank correlation
cannot see knots and voids.

## Transferring to PDB, and what the weights actually buy

```bash
python examples/exploratory/make_pdb_arrays.py          # X, resolution, quantile bins
python examples/exploratory/reference.py --X .../pdb_X.npy --y .../pdb_resbin.npy --name pdb
python examples/exploratory/master.py --X .../pdb_X.npy --y .../pdb_resbin.npy \
  --name pdb_match --sweep pdb_weights \
  --holdout 0.2 --seeds 0 --null shuffle --target-perp 8 --epochs 120
```

These PDB numbers predate the `min_dist` change and were produced at 0.1, in the
collapse regime. They rest on null-corrected margins rather than on layout
geometry so are unlikely to reverse, but the run has not been repeated.

PDB has no ground truth, so resolution in quantile bins stands in as a proxy
grouping. It has very little headroom: 5-NN accuracy on the raw 9-D features is
0.323 against a chance level of 0.20, and UMAP (0.300), densMAP (0.298) and
PCA-2D (0.291) are indistinguishable from each other. Read every PDB number as a
null-corrected margin, not a raw score.

Held out, real minus matched shuffle:

| config | acc | trust15 | ov15 | geodesic | density |
|---|---|---|---|---|---|
| `flat` (1,1,1) | +0.091 | +0.090 | +0.126 | +0.262 | +0.044 |
| `ramp` (1,2,8) | +0.096 | +0.114 | +0.113 | +0.182 | +0.067 |
| `steep` (1,4,16) | +0.086 | +0.116 | +0.102 | +0.180 | +0.090 |
| `frontload` (8,1,1) | +0.066 | +0.113 | +0.147 | +0.322 | **-0.142** |
| pyramid `off` | +0.064 | +0.095 | +0.126 | +0.337 | **-0.090** |
| UMAP | +0.097 | +0.029 | +0.060 | +0.397 | +0.050 |

Two things fall out. First, the weight tuple trades global against density
fidelity monotonically: coarse emphasis buys density correlation and gives up
geodesic, fine emphasis does the reverse. Second, and this is what the null was
for, `frontload` and pyramid-off have *negative* density margins -- they score
worse on real data than the same configuration scores on shuffled data, so their
apparent density structure is not real. Coarse-weighted pyramids survive
calibration on density; fine-weighted ones do not. `flat` is the balanced choice
here, and the coarse ramp that won on digits over-weights global attraction for
this dataset. Accuracy differences across the table are inside the seed spread
(+/-0.016 on digits) and should not be read as real.

Against UMAP, leanmap wins the trustworthiness, kNN-overlap and density margins
and loses the geodesic one. Note UMAP's raw trustworthiness of 0.954 collapses to
a +0.029 margin because shuffled input already scores 0.926.

Finally, the discrete clusters leanmap used to produce on PDB are gone. Under the
fixed pyramid and a calibrated configuration it agrees with UMAP, densMAP and PCA
that this data is one continuous cloud carrying a resolution gradient -- see
`compare_pdb.png`. The old clusters were an artifact of the truncated weight
tuple and the coarse-backbone MST.

## Reading the guide

Compare scatters along one axis (e.g. all `n_neighbors__*`) and check
`summary.csv` columns `trust_15`, `geodesic_spearman`, `ambient_stress`. Useful
questions:

1. Which knobs bridge a hole vs preserve it?
2. Which knobs untwist / straighten vs curl tips?
3. Which knobs clump vs spread class structure?
4. Do geo / delayed frame / poisson landmarks earn their weight?

# leanmap

A small, self-contained, **deployable parametric UMAP**.

Standard UMAP gives you an embedding of the points you fit on — but no way to
embed *new* points without re-fitting. `leanmap` builds the same fuzzy
topological graph (UMAP's math, reimplemented in vectorized NumPy/SciPy — no
`umap-learn` dependency) using **FAISS** approximate k-NN so it scales, then
trains a small **PCA-anchored neural network** to reproduce the embedding. The
result is a differentiable function you can save, reload, and apply to a stream
of new data.

## Why this design

- **FAISS k-NN** (HNSW by default, optional GPU flat index) instead of exact /
  `pynndescent` — scales to millions of points.
- **PCA-anchored encoder**: `output = PCA_linear(x) + MLP_residual(x)`, with the
  linear part initialized to the true top-2 PCA components and the residual
  initialized near zero. Training *starts* at a sensible PCA embedding and
  refines from there — fast, stable convergence.
- **Ranking loss** (a triplet term over sampled candidates) added on top of the
  usual attract/repel UMAP objective, annealed over training to better preserve
  relative global structure — a common weakness of parametric UMAP.
- **Parametric & portable**: `fit` once, `transform` forever; save to a single
  file, reload anywhere, or grab the raw `torch.nn.Module` to embed in a larger
  model.

## Install

```bash
pip install -e .            # core (numpy, scipy, torch)
pip install -e ".[cpu]"     # + faiss-cpu  (needed to FIT a new model)
pip install -e ".[examples]" # + scikit-learn, matplotlib (for the demo)
```

FAISS is an optional dependency because the correct wheel depends on your
platform: `faiss-cpu` from PyPI, or a CUDA `faiss-gpu` build from conda. FAISS
is only needed to **fit** — a saved model can be **loaded and used to
transform** with just numpy + torch.

## Python API

```python
from leanmap import LeanMap

mapper = LeanMap(n_neighbors=15, min_dist=0.1, epochs=40)
emb = mapper.fit_transform(X_train)     # (n, 2) numpy array

emb_new = mapper.transform(X_test)      # embed unseen points, no re-fit

mapper.save("model.mmap")
emb2 = LeanMap.load("model.mmap").transform(X_other)

module = mapper.torch_module            # raw nn.Module: raw input -> 2D
```

`LeanMap` follows scikit-learn conventions (`fit`, `transform`,
`fit_transform`, `n_features_in_`). All graph and training knobs are constructor
keyword arguments; see `leanmap.MapperConfig` for the full list and defaults.

## Inducing points (landmark out-of-sample extension)

A parametric network is a fixed function: it embeds new points well only where
it generalizes. `umap.transform`, by contrast, re-places each new point using
its *actual* high-dimensional neighbors — which is why it generalizes almost
perfectly. `leanmap` recovers that property **without a network** via
**inducing points**: store a small set of landmarks with known embedding
coordinates, then place any query by its high-D fuzzy membership (UMAP's own
smooth-kNN kernel) to the nearest landmarks.

```python
import umap
ref = umap.UMAP(n_neighbors=15, min_dist=0.0).fit_transform(X_train)  # any 2D layout

mapper = LeanMap(scale_mode="center", n_inducing=300,
                 landmark_method="fps", induce_k=5)
mapper.fit(X_train, reference_coords=ref)   # landmarks inherit ref coords

emb_new = mapper.induce_transform(X_test)   # training-free, generalizes like UMAP
mapper.save("model.mmap")                   # landmarks travel inside the file
```

`reference_coords` is optional — omit it and landmarks inherit the trained
network's own embedding of `X_train`.

**Landmark selection** (`landmark_method`):

- `"fps"` *(default)* — **farthest-point sampling** (greedy k-center). A
  2-approximation to the coverage radius: every point sits within a bounded
  distance of a landmark, so sparse regions and rare classes still get one.
  Coverage-weighted.
- `"kmeans"` — k-means centroids snapped to real points. Density-weighted (more
  landmarks where data is dense); best average silhouette on balanced data.
- `"hexgrid"` — hexagonal lattice over the 2D embedding, pruned to occupied
  cells. Uniform in the embedding *plane*; can under-cover rare classes.

On balanced digits, k-means edges out FPS on silhouette, but **FPS wins on
worst-case coverage and rare-class recall** — the reason to prefer it when
density is uneven. `induce_k` controls sharpness: fewer landmark-neighbors →
tighter clusters (5 is a good default). The extension generalizes with a
near-zero train/test gap and approaches reference-UMAP quality as the landmark
count grows, with no gradient descent — fitting is landmark selection, inference
is one distance computation to a few hundred points.

## Attention conditioning (`conditioning="attention"`)

The inducing points can also *condition a network* instead of only being
averaged. In attention mode each query point cross-attends to the landmark set
(a Set-Transformer readout: keys from landmark high-D positions, values carrying
both position and reference 2D coordinate), and the attention output modulates
an MLP via FiLM (`h → SiLU((1+γ)·h + β)`). Attention logits are biased by
`−β·‖x−landmark‖²` — a learnable locality prior that plays the role of UMAP's
fuzzy kernel and stops the otherwise-global attention from overfitting.

```python
mapper = LeanMap(
    scale_mode="center", conditioning="attention",
    n_inducing=300, landmark_method="fps",
    epochs=120, batch_size=512, learning_rate=5e-3,
    learn_landmarks=True, gram_anchor_weight=1.0,   # defaults
)
mapper.fit(X_train, reference_coords=ref)   # reference_coords REQUIRED here
emb = mapper.transform(X_test)              # a smooth, deployable network
```

`reference_coords` is **required** in attention mode (landmarks are baked into
the encoder before training, so their target coordinates must be supplied).
Inference is a single forward pass, constant per point regardless of the
training-set size.

**Learnable inducing points** (`learn_landmarks=True`, on by default). Landmarks
start at the data-anchored positions but are then optimized jointly with the
network (Titsias-style free inducing inputs) — the best landmarks generally are
*not* real data points. A **Gram-anchor penalty** (`gram_anchor_weight`, default
1.0) keeps their pairwise-distance geometry near its initialization: it forbids
collapse and runaway while leaving rigid motion (translation/rotation) free, so
landmarks reconfigure usefully without running off the manifold. On digits this
reaches the best silhouette and tightest train/test gap of any variant.
`landmark_lr_mult` (default 1.0) can additionally throttle only the landmark
learning rate.

## Generative decoding (embedding → image)

The forward map throws information away (high-D → 2D), so the *inverse* is
one-to-many. A plain regression decoder can only return the **conditional mean**
`E[x|z]` — a blurred, class-typical shape. To draw sharp, varied reconstructions,
`leanmap` models the full conditional `p(x|z)` with a **conditional
normalizing flow**:

```python
mapper.fit(X_train)                       # any conditioning mode
mapper.fit_decoder(X_train, residual_dim=15, flow_layers=10)

z = mapper.transform(X_query)
mean    = mapper.decode_mean(z)                       # E[x|z]  (the blur)
samples = mapper.decode_sample(z, n_per=8)            # (n, 8, n_features) draws
sharp   = mapper.decode_sample(z, temperature=0.7)    # <1 tamer, >1 more varied
score   = mapper.decode_logprob(X_query, z)           # manifold-consistency / novelty
```

Internally: a mean-decoder MLP `D(z)` captures the coarse shape; the residual
`r = x − D(z)` is projected to a small PCA subspace (`residual_dim` coefficients);
a conditional RealNVP flow with `z` injected into every coupling layer learns
`p(c|z)` with exact likelihood. A sample is `x = D(z) + V·c`, `c ~ flow(·|z)`.
Averaging many samples recovers `D(z)` — the flow is a proper conditional
distribution whose mean is the regression mean, so sampling adds the discarded
variance back calibrated, rather than inventing it. `decode_logprob` returns the
residual-space log-density, useful as an out-of-distribution score for a
reconstruction.

## Is a generated sample real? (conformalized discriminator)

A generative decoder is a smooth, lower-rank approximation of the data manifold,
so its samples are **not** exchangeable with real data — a flexible classifier
can always separate them. `LeanmapDiscriminator` turns that into a *calibrated*
one-sided test instead of pretending otherwise:

```python
disc = LeanmapDiscriminator(n_regions=4, min_regional_pool=10)
disc.fit(X_real=Xtr_a, X_generated=gen, X_calib=Xtr_b, Z_calib=mapper.transform(Xtr_b))

p = disc.p_value(X_query, mapper.transform(X_query), mondrian=True)   # "could be real" p-value
ok = disc.could_be_real(X_query, mapper.transform(X_query), alpha=0.10)

# use it as a quality gate: keep only samples that pass
kept, kept_p, n_drawn = disc.rejection_sample(generate_fn, z_star, n_want=8, alpha=0.10)
```

The p-value is conformal against a held-out set of **real** examples, so it is
valid under exchangeability of real calibration/test points: real inputs are
falsely rejected at most `alpha` of the time, while generated samples (the
alternative) collapse toward `p ≈ 0`. **Mondrian** stratification calibrates
within the query's *leanmap region* (a cell of the embedded space), enforcing the
guarantee locally; regions with fewer than `min_regional_pool` calibration points
fall back to the global pool. `rejection_sample` keeps only samples that clear
`alpha`, giving a stream with a calibrated false-keep rate (you trade yield for
fidelity — accept rate ≈ the fraction that pass).

### Decoder fidelity — `residual_mode`

`fit_decoder(residual_mode="full")` models every residual dimension instead of a
low-rank PCA subspace (`"pca"`, default `residual_dim=15`). The PCA truncation is
a *low-rank tell* a discriminator exploits; on sklearn digits the full-rank flow
cut discriminator AUC from ≈0.90 to ≈0.77 and raised the raw pass rate from ≈16%
to ≈66%. (Real and generated remain non-exchangeable — AUC stays above 0.5 — so
the rejection gate above is still the way to guarantee kept-sample fidelity.)

## Supervised axis ordering

`fit(..., order_constraints=[...])` (attention mode) aligns an embedding axis to
a label gradient — so the layout is *readable* along X and/or Y. Each constraint
adds a soft chain-ranking penalty on class centroids along one axis; because X
and Y are penalized independently they compose into orthogonal gradients.

```python
mapper.fit(X_train, reference_coords=ref, order_constraints=[
    {"axis": "x", "kind": "ordinal", "labels": y, "weight": 2.0},
    {"axis": "y", "kind": "separate", "labels": is_prime,
     "order": [0, 1], "weight": 2.0},
])
```

- `kind="ordinal"` — enforce a total order of class centroids along the axis.
  For numeric/ordinal labels (temperature, dose, pseudotime). `order` gives the
  desired low→high id sequence (default: sorted unique labels).
- `kind="separate"` — push one group above/right of another. For binary or
  categorical splits (treated/control, prime/non-prime).

Labels may be numeric or non-numeric (only their rank/group is used). On digits,
`weight=2.0` yields a perfect 0→9 ordering along X (Spearman +1.00) with
essentially no loss of cluster quality (silhouette and kNN unchanged).

## Hardware acceleration

`leanmap` selects the best available device automatically — **CUDA → MPS
(Apple Silicon GPU) → CPU** — so on an M-series Mac `fit`/`transform` run on the
Metal GPU with no arguments:

```python
mapper = LeanMap(...)      # device auto-resolves to "mps" on Apple Silicon
mapper.fit(X)
```

Override explicitly with `LeanMap(device="cpu")` (or `"cuda"`, `"mps"`), or
`--device` on the CLI.

One caveat on macOS: a few PyTorch ops may be missing from older Metal builds.
If a run errors with an "MPS ... not implemented" message, enable the transparent
CPU fallback for unsupported ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The MPS path needs no FAISS change — FAISS k-NN runs on CPU (or a CUDA FAISS
build); only the network training/inference moves to the GPU.

## Command-line interface

```bash
# Fit a model and (optionally) dump the training embedding
leanmap fit data.csv -o model.mmap -e embedding.csv \
    --n-neighbors 15 --min-dist 0.1 --epochs 40 --index-kind hnsw

# Embed new data with a saved model
leanmap transform new_data.csv -m model.mmap -o new_embedding.csv

# Inspect a saved model
leanmap info model.mmap
```

If the `leanmap` console script isn't on your `PATH` (e.g. in some managed
environments), the identical CLI is available as `python -m leanmap ...`.

Data IO supports `.npy`, `.npz` (key `X` or first array), and delimited text
(`.csv`/`.tsv`/`.txt`, with `--delimiter` and `--skip-header`). Embeddings are
written as `.npy` or as a two-column `x,y` text file. Run `leanmap <cmd> -h`
for the full flag list.

## Example

```bash
python examples/demo_digits.py
```

Fits on 70% of scikit-learn's digits, embeds the held-out 30% through the
trained network (never entering the graph build), verifies a save/load
round-trip, and writes `digits_embedding.png`.

## Package layout

```
leanmap/
  _graph.py   FAISS k-NN + fuzzy simplicial-set graph (standardize, smooth_knn_dist, ...)
  _model.py   ParametricMapper, FiLMMapper, AttentionMapper, DeployableMapper
  _train.py   train_parametric_mapper / train_attention_mapper + transform, the UMAP-style loss
  _inducing.py landmark selection (FPS / k-means / hex) + fuzzy extension
  _decoder.py GenerativeDecoder: mean decoder + conditional RealNVP flow p(x|z)
  _discriminator.py LeanmapDiscriminator: conformalized real-vs-generated test + rejection sampling
  _api.py     LeanMap high-level estimator + save/load
  _cli.py     argparse CLI (fit / transform / info)
```

## License

MIT.

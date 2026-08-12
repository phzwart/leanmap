# CellCycle novelty: leanmap on G1/S/G2 with mitotic hold-out

This note documents an end-to-end experiment: **train a 2D leanmap embedding only on interphase cells (G1, S, G2)**, then ask whether mitotic phases (Prophase, Metaphase, Anaphase, Telophase) are **flagged as out-of-distribution / novelty** by leanmap’s conformal landmark-cover test.

Primary run directory:

```text
examples/out/cellcycle_lejepa_cls_full_md0_geo1_L1k_g1sg2/
```

Key artefacts: `final.png`, `novelty.png`, `novelty.txt`, `scores.npz`, `model.pt`.

---

## 1. Scientific question

Cell-cycle imaging datasets label seven morphological stages. The rare mitotic stages are biologically continuous with G2 → mitosis → G1, but they are visually and molecularly distinct (condensed chromosomes, spindle, cytokinesis).

**Question.** If an embedding model is shown *only* G1/S/G2 feature geometry at fit time, do mitotic cells land off the learned support and receive low conformal *p*-values (reject as novel), or do they sit on the same manifold as late G2 and pass as in-distribution?

A positive result (mitotics rejected; held-out G1/S/G2 retained near the nominal false-positive rate) indicates that the LeJEPA feature space + leanmap support model separates mitosis from interphase without having been trained to embed the mitotic classes.

---

## 2. Dataset

| Item | Value |
|------|--------|
| Source | BBBC048 / CellCycle Jurkat IFC (Eulenberg et al., *Nat Commun* 2017; related DeepFlow / theislab) |
| Local path | `~/Projects/cells/CellCycle/` |
| Channels used | Ch3, Ch4, Ch6 (DNA / nuclear structure related IFC channels), stacked as RGB-like `(3, 66, 66)` |
| Labels | 7 phases: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase |
| Full *N* | 32 266 cells |

### Class counts (full zarr)

| Phase | *n* | Role in this experiment |
|-------|----:|-------------------------|
| G1 | 14 333 | in-manifold (train / calib / test) |
| S | 8 616 | in-manifold |
| G2 | 8 601 | in-manifold |
| Prophase | 606 | **novelty probe** (never used in leanmap fit) |
| Metaphase | 68 | novelty probe |
| Anaphase | 15 | novelty probe |
| Telophase | 27 | novelty probe |

In-manifold total: **31 550**. Mitotic hold-out: **716**.

---

## 3. Feature extraction (LeJEPA + class token)

Features are **not** raw pixels. They come from a cheap LeJEPA encoder trained on all phases (including mitosis), then frozen.

| Item | Value |
|------|--------|
| Script | [`cellcycle_lejepa.py`](cellcycle_lejepa.py) |
| Feature store | `examples/out/cellcycle_lejepa_cls.zarr` |
| Method attr | `LeJEPA (cheap ConvNet + SIGReg) + class-token inject + CE` |
| Reference | Balestriero & LeCun, LeJEPA, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) |
| Encoder output | 128-D (`emb_dim=128`), projector 64-D (training only) |
| Exported features | **encoder-only** `(N, 128)` float32 — class token is *not* concatenated into the stored vector |
| Training | 600 epochs, `λ_cls=0.5`, inverse-frequency CE weights, `λ_sigreg=0.05` |
| Views | 2 global + 2 local crops per step |

### Why class-token LeJEPA?

Plain SSL on this dataset produced a strong G1–G2 axis but weak 7-way separation. Adding a learnable per-phase token into the projector path plus supervised CE on the encoder improves phase structure in the encoder features used for leanmap. **Important for interpreting novelty:** the feature extractor *has* seen mitotic labels during pretraining. What is held out here is only the **leanmap embedding / support**, not the SSL encoder. Novelty therefore means “far from the G1/S/G2 landmark cover in this feature space,” not “the encoder never saw mitosis.”

### Geometry for kNN

Before leanmap, features are **L2-normalized** row-wise. Neighbours are built with **Euclidean** distance on those unit vectors ⇒ **cosine-like** geometry.

```text
X ← X / ||X||₂
knn ← NearestNeighbors(metric="euclidean", k=15) on train rows
```

---

## 4. Embedding protocol (this run)

### 4.1 Design choices

| Knob | Value | Rationale |
|------|-------|-----------|
| Output dim | 2 | primary visualisation + novelty overlay |
| `min_dist` | 0.0 | allow tight packing where the graph supports it |
| `λ_geo` | 1.0 | Isomap / landmark-geodesic stress on |
| `n_landmarks` | 1 000 | denser cover for conformal score on ~25k train points |
| `n_neighbors` (k) | 15 | standard fuzzy graph |
| Epochs | 80 | match prior CellCycle leanmap runs |
| Frame dump | every 5 epochs | `live.png` + `frames/` |
| Device | CUDA | |
| Seed | 0 | split + leanmap |

### 4.2 Who is allowed in the embedding?

**Only G1, S, G2.** Prophase / Metaphase / Anaphase / Telophase are excluded from:

- the fuzzy kNN graph,
- landmark FPS,
- encoder training inside leanmap,
- conformal calibration scores.

They are embedded **out of sample** after fit for scoring and plotting only.

### 4.3 Splits (within G1/S/G2 only)

Of the 31 550 in-manifold cells, a single random permutation (`seed=0`) yields:

| Split | Fraction | *n* | Used for |
|-------|----------|----:|----------|
| Train | 80% | 25 240 | fit leanmap (graph + optimisation) |
| Calibration | 10% | 3 155 | conformal exchangeability set |
| Test | 10% | 3 155 | downstream / validity check (never fit) |
| Mitotic OOD | — | 716 | novelty evaluation |

Train phase mix: G1 11 452 / S 6 903 / G2 6 885.  
Test mix: G1 1 427 / S 850 / G2 878.  
OOD mix: Prophase 606 / Metaphase 68 / Anaphase 15 / Telophase 27.

Calibration and test are **exchangeable with train under the in-manifold distribution**. Mitotics are a **structured shift**, not exchangeable with calibration — rejection is the intended signal, not a validity claim for that group.

### 4.4 Conformal novelty score

leanmap’s primary OOD score is **landmark cover**

\[
s(x) = \min_{\ell=1,\ldots,L}\ \|x - M_\ell\|
\]

(ambient distance in the same L2-normalized feature space to the nearest landmark). Higher cover ⇒ farther from support.

After fit, `ConformalCalibrator` stores calibration covers \(\{s_i\}\). For a new point,

\[
p(x) = \frac{1 + \#\{s_i \ge s(x)\}}{n_{\mathrm{cal}} + 1}.
\]

Small \(p\) ⇒ more extreme (more OOD) than most calibration points. At nominal level \(\alpha\), reject when \(p < \alpha\). Under exchangeability, the false-positive rate on a fresh in-manifold test set should be about \(\alpha\).

See also `src/leanmap/README.md` (Conformal / OOD caveats) and [`DIGITS_OOD.md`](DIGITS_OOD.md).

---

## 5. How to reproduce

From the repo root (or `examples/`), with the project venv:

```bash
cd examples

# Features must already exist (LeJEPA + class token → zarr).
# If rebuilding from scratch (long):
#   python cellcycle_lejepa.py --class-token --lambda-cls 0.5 --epochs 600 \
#     --out out/cellcycle_lejepa_cls.zarr --skip-leanmap --device cuda

PYTHONUNBUFFERED=1 python cellcycle_lejepa.py \
  --fit-only \
  --out out/cellcycle_lejepa_cls.zarr \
  --run-dir out/cellcycle_lejepa_cls_full_md0_geo1_L1k_g1sg2 \
  --device cuda \
  --leanmap-epochs 80 \
  --min-dist 0 \
  --lambda-geo 1 \
  --d-out 2 \
  --n-landmarks 1000 \
  --holdout 0.1 \
  --test-frac 0.1 \
  --exclude-phases Prophase Metaphase Anaphase Telophase \
  --frame-every 5 \
  --k 15 \
  --seed 0
```

CLI meaning:

| Flag | Meaning |
|------|---------|
| `--fit-only` | skip LeJEPA; load existing zarr features |
| `--holdout` | calibration fraction of the **in-manifold** set |
| `--test-frac` | downstream test fraction of the in-manifold set |
| `--exclude-phases …` | drop these labels from fit; score them as novelty |
| `--n-landmarks` | override `PLANEConfig.n_landmarks` |

Runtime for this configuration was on the order of **~40 minutes** on a single GPU (kNN build + 80 epochs on 25k points with *L*=1000).

---

## 6. Results (seed 0 run)

### 6.1 Conformal table (`novelty.txt`)

| set | *n* | cover median | *p* median | fraction *p*<0.05 | fraction *p*<0.1 |
|-----|----:|-------------:|-----------:|------------------:|-----------------:|
| train | 25 240 | 1.825 | 0.494 | 0.056 | 0.105 |
| calib | 3 155 | 1.820 | 0.500 | 0.049 | 0.100 |
| test | 3 155 | 1.831 | 0.486 | 0.051 | 0.109 |
| test/G1 | 1 427 | 1.833 | 0.484 | 0.040 | 0.086 |
| test/S | 850 | 1.680 | 0.652 | 0.022 | 0.053 |
| test/G2 | 878 | 1.996 | 0.329 | 0.098 | 0.200 |
| **mitotic** | **716** | **3.683** | **0.001** | **0.922** | **0.954** |
| ood/Prophase | 606 | 3.471 | 0.002 | 0.908 | 0.946 |
| ood/Metaphase | 68 | 5.562 | 0.000 | 1.000 | 1.000 |
| ood/Anaphase | 15 | 5.125 | 0.000 | 1.000 | 1.000 |
| ood/Telophase | 27 | 4.151 | 0.001 | 1.000 | 1.000 |

### 6.2 Interpretation

1. **Validity on interphase.** Held-out G1/S/G2 test rejects at ~5.1% for \(\alpha=0.05\) (nominal 5%) and ~10.9% for \(\alpha=0.1\). Calibration is on-target by construction. This is what you want before trusting OOD calls.

2. **Mitosis is novel under cover.** Median mitotic cover (~3.68) is roughly **2×** the in-manifold median (~1.83). Median *p* ≈ 0.001; **92%** of mitotics reject at \(\alpha=0.05\).

3. **Harder / rarer stages reject more cleanly.** Metaphase, Anaphase, and Telophase are **100%** rejected at \(\alpha=0.05\) (small *n*, but covers are very large). Prophase is slightly softer (~91%), consistent with it being morphologically closer to late G2.

4. **G2 test is the noisiest in-manifold slice.** `test/G2` false-positive rate at \(\alpha=0.05\) is ~9.8% (above nominal). That matches the biological boundary: late G2 and early prophase are adjacent; some G2 cells sit in the sparse tail of cover.

5. **Discrimination.** Landmark-cover AUROC for mitotic vs held-out interphase test is **≈ 0.985** (computed post hoc from `scores.npz`).

### 6.3 Figures

| File | Content |
|------|---------|
| `final.png` | Train embedding colored by phase (G1 / S / G2 only) |
| `live.png` | Last training-frame scatter |
| `novelty.png` | Train scatter + mitotic overlays: ○ = *p*≥0.05, **×** = *p*<0.05 |
| `frames/` | Epoch snapshots every 5 epochs |
| `frames.npz` | Stacked `(epoch, N, 2)` embeddings for animation / replot |

---

## 7. Output layout

```text
examples/out/cellcycle_lejepa_cls_full_md0_geo1_L1k_g1sg2/
├── model.pt          # leanmap inference artefact (encoder + landmarks + calib)
├── final.png         # train scatter
├── novelty.png       # train + mitotic novelty overlay
├── novelty.txt       # conformal summary table
├── scores.npz        # Z / cover / p / y / indices for all splits
├── Z_final.npy       # train embedding (N_train, 2)
├── y_train.npy
├── train_idx.npy     # indices into the zarr (global cell rows)
├── cal_idx.npy
├── test_idx.npy
├── progress.csv      # per-epoch leanmap metrics
├── live.png
├── frames/
└── frames.npz
```

### `scores.npz` keys

| Key | Shape / meaning |
|-----|-----------------|
| `Z_train`, `cover_train`, `p_train`, `y_train`, `train_idx` | fit set |
| `Z_cal`, `cover_cal`, `p_cal`, `y_cal`, `cal_idx` | calibration |
| `Z_test`, `cover_test`, `p_test`, `y_test`, `test_idx` | downstream hold-out |
| `Z_ood`, `cover_ood`, `p_ood`, `y_ood`, `ood_idx` | mitotic probes |
| `phases`, `exclude_phases` | string metadata |

Indices refer to rows of `examples/out/cellcycle_lejepa_cls.zarr` (`features`, `labels`, `images`, `cell_ids`).

### Quick reload

```python
import numpy as np

d = np.load("examples/out/cellcycle_lejepa_cls_full_md0_geo1_L1k_g1sg2/scores.npz", allow_pickle=True)
print((d["p_ood"] < 0.05).mean())          # mitotic reject rate
print((d["p_test"] < 0.05).mean())         # interphase FPR
```

---

## 8. Related runs (context)

Earlier exploratory embeddings (not the novelty protocol):

| Run dir | Notes |
|---------|--------|
| `out/cellcycle_lejepa_cls/` | 10k subsample, `min_dist=0` |
| `out/cellcycle_lejepa_cls_md0p5_geo1/` | 10k, `min_dist=0.5`, `λ_geo=1`, 2D |
| `out/cellcycle_lejepa_cls_md0p5_geo1_3d/` | same, **3D** (`d_out=3`); geodesic Spearman ≈ 0.75 vs ≈ 0.68 in 2D |
| `out/cellcycle_lejepa_cls_umap/` | UMAP baseline on same features |

Those runs included all seven phases in the embedding. The novelty claim **requires** the mitotic exclusion used here.

---

## 9. Caveats and failure modes

1. **Encoder saw mitosis.** LeJEPA+CE was trained on all seven labels. Novelty is w.r.t. the **G1/S/G2 leanmap support**, not a mitosis-naive representation. For a stricter protocol, retrain LeJEPA on G1/S/G2 only, re-export features, then repeat leanmap.

2. **Cover ≠ semantic OOD.** Landmark cover answers “near the landmark cloud?”, not “same biological class.” A novel cell that lands on the interphase manifold would correctly pass.

3. **Triplet retention was weak** during training (warnings that landmark ranking is a poor proxy). Geodesic / ordinal diagnostics were noisy; the conformal cover test can still be useful (and here was highly discriminative), but layout quality metrics should be read cautiously.

4. **Class imbalance in OOD.** Prophase dominates the mitotic pool (606 / 716). Aggregate “92% reject” is mostly a Prophase statement; the rarer stages are stronger but tiny.

5. **G2–Prophase continuum.** Elevated G2 false positives and slightly softer Prophase rejection are expected if features encode a smooth G2→M transition.

6. **Single seed.** Splits and leanmap init use `seed=0` only. Re-run with `--seed 1 2 …` before claiming robustness.

7. **`precomputed_knn` + explicit `X_calib`.** The script builds the graph on train rows only and passes calibration separately, matching leanmap’s contract when `dedup=False`.

---

## 10. Implementation map

| Piece | Location |
|-------|----------|
| LeJEPA train + feature export | `examples/cellcycle_lejepa.py` (`train_lejepa`) |
| Leanmap fit, splits, novelty scoring | `examples/cellcycle_lejepa.py` (`fit_leanmap`) |
| Phase-coloured scatter / frames | `examples/cellcycle_emd.py` (`save_phase_scatter`, `EmbeddingRecorder`) |
| Interactive explorer | `examples/cellcycle_explorer.py` |
| Conformal API | `src/leanmap/conformal.py` (`ConformalCalibrator`) |
| Leanmap fit entry | `src/leanmap/train.py` (`fit`) |

---

## 11. Follow-up: mitosis-naive LeJEPA (train-only, 500 epochs)

To address caveat (1), we retrained LeJEPA **only on the leanmap train split** (25 240 G1/S/G2 cells; no calib/test/mitotic), 500 epochs, class-token CE on the three present classes, then exported encoder features for **all** 32 266 cells and repeated the same novelty leanmap protocol.

| Artefact | Path |
|----------|------|
| Model | `examples/out/cellcycle_lejepa_trainonly_e500.pt` |
| Checkpoints | `examples/out/cellcycle_lejepa_trainonly_e500_ckpts/` |
| Features (all cells) | `examples/out/cellcycle_lejepa_trainonly_e500.zarr` |
| Novelty run | `examples/out/cellcycle_lejepa_trainonly_e500_novelty/` |

### Novelty table (mitosis-naive features)

| set | *n* | cover50 | *p*50 | *p*<0.05 | *p*<0.1 |
|-----|----:|--------:|------:|---------:|--------:|
| test | 3 155 | 0.847 | 0.507 | 0.048 | 0.094 |
| **mitotic** | **716** | **0.924** | **0.329** | **0.148** | **0.223** |
| ood/Prophase | 606 | 0.896 | 0.390 | 0.087 | 0.157 |
| ood/Metaphase | 68 | 1.025 | 0.161 | 0.235 | 0.397 |
| ood/Anaphase | 15 | 1.214 | 0.035 | 0.800 | 0.800 |
| ood/Telophase | 27 | 1.758 | 0.001 | 0.926 | 0.963 |

Cover AUROC mitotic vs test ≈ **0.61** (vs **0.985** when the encoder had seen mitosis).

### Comparison

| Protocol | Encoder saw mitosis? | Mitotic reject @0.05 | Cover AUROC |
|----------|---------------------:|---------------------:|------------:|
| §6 (`cellcycle_lejepa_cls`) | yes (all-phase CE) | **92%** | **0.985** |
| This section (`trainonly_e500`) | **no** | **15%** | **0.61** |

Interphase test FPR stays calibrated (~5%). Late mitosis (Ana/Telo) is still often flagged; **Prophase mostly blends into the G1/S/G2 support** once the encoder is mitosis-naive. So leanmap can still detect *some* novelty from pure geometry, but the dramatic §6 result was largely helped by supervised mitotic structure already present in the features.

Reproduce:

```bash
cd examples
PYTHONUNBUFFERED=1 python cellcycle_lejepa.py \
  --class-token --lambda-cls 0.5 --epochs 500 \
  --out out/cellcycle_lejepa_trainonly_e500.zarr \
  --train-idx out/cellcycle_lejepa_cls_full_md0_geo1_L1k_g1sg2/train_idx.npy \
  --device cuda --ckpt-every 50 \
  --ckpt-dir out/cellcycle_lejepa_trainonly_e500_ckpts \
  --skip-leanmap --seed 0

PYTHONUNBUFFERED=1 python cellcycle_lejepa.py \
  --fit-only \
  --out out/cellcycle_lejepa_trainonly_e500.zarr \
  --run-dir out/cellcycle_lejepa_trainonly_e500_novelty \
  --device cuda --leanmap-epochs 80 \
  --min-dist 0 --lambda-geo 1 --d-out 2 --n-landmarks 1000 \
  --holdout 0.1 --test-frac 0.1 \
  --exclude-phases Prophase Metaphase Anaphase Telophase \
  --frame-every 5 --k 15 --seed 0
```

---

## 12. Bottom line

With LeJEPA class-token features **trained on all phases**, a 2D leanmap fit on **G1/S/G2 only** (`min_dist=0`, `λ_geo=1`, 1 000 landmarks, 10%/10% calib/test) yields:

- roughly **nominal** conformal false-positive rates on held-out interphase,
- **strong novelty detection** of mitotic cells (**~92%** at \(\alpha=0.05\); cover AUROC vs test ≈ **0.985**),
- near-perfect rejection of Metaphase / Anaphase / Telophase, with Prophase slightly softer — consistent with a G2→M continuum in feature space.

With a **mitosis-naive** encoder (§11), novelty is much weaker overall (~15% reject) but late mitotic stages remain detectable — evidence that leanmap’s cover score still picks up geometric shift, while the strongest §6 separation leaned on supervised mitotic signal in the features.

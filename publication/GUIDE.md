# leanmap user guide — embeddings and outlier detection

This is a self-contained guide to using **leanmap** for two coupled goals:

1. **Interpretable 2-D embeddings** of tabular / image-vector data  
2. **Calibrated outlier detection** on the same frozen model  

Artefacts, tables, and exact parameters for the digits record live under
`publication/`. Mathematical definitions are in `docs/math/leanmap.tex`.

---

## 1. Install

```bash
pip install -e ".[examples,cpu]"   # FAISS + sklearn + matplotlib
# GPU torch as appropriate for your platform
```

Entry points: Python API (`leanmap`) and CLI (`leanmap fit|transform|info|mondrian`).

---

## 2. Fit an embedding

```python
from leanmap import PLANEConfig, fit
import numpy as np

X = ...  # (N, D) float32
cfg = PLANEConfig.for_scale(len(X))
# Digits publication recipe (see params/digits_clean.yaml):
cfg.epochs = 80
cfg.min_dist = 0.1          # class packing; use 0.5 for more uniform layouts
cfg.seed = 0

result = fit(X, dist_fn="l2", config=cfg)
Z, cover = result.embed(X)  # cover = min landmark distance
result.save("model.pt")
```

**CLI**

```bash
leanmap fit data.npy -o model.pt --epochs 80 --seed 0
leanmap transform model.pt data.npy -o Z.npy --scores cover.npy
leanmap info model.pt
```

**Digits one-liner**

```bash
python examples/digits.py --device cuda --min-dist 0.1 --epochs 80 --seed 0
# → examples/out/digits.pt   (copied to publication/artefacts/models/digits_clean.pt)
```

### What is saved

The `.pt` artefact holds encoder weights, landmarks, metric scale, and a
**pooled cover** conformal calibrator (held-out real points). The neighbour
graph is discarded; inference is one forward pass.

### Knobs that matter

| knob | digits record | role |
|------|---------------|------|
| `pca_skip=False`, `lr=0.02` | yes | must move together |
| `min_dist` | **0.1** | tighter clusters (default `for_scale` is 0.5) |
| `pyramid_level_weights` | `(1,2,8)` | coarse-heavy multi-scale graph |
| `lambda_geo` | 0.15 | geodesic consistency |
| `n_landmarks` | 128 | support for cover / affinity |
| `epochs` | 80 | sufficient for N≈1.8k |

Full knob encyclopedia: `docs/CONFIGURATION.md`.

---

## 3. Outlier detection (recommended path)

Use a **frozen** clean model. Do **not** rely on trash-basin training for
detection power (see §6).

### 3.1 Scores

| score | formula | when to use |
|-------|---------|-------------|
| `cover` | \(\min_\ell \|x-M_\ell\|\) | simple ambient support test |
| `affinity_entropy` | \(H(a)=-\sum_\ell a_\ell\log a_\ell\) | default Mondrian score |
| `lda` | signed distance to Fisher LDA on \((\mathrm{cover},H)\) | **best digit vs noise** |

Catalog: `tables/nonconformity_catalog.csv`.

### 3.2 Mondrian levels (digit / gauss / shuffle)

Calibrate three groups so each noise family has its own threshold:

```python
from leanmap import MondrianCalibrator, CoverEntropyLDA, make_mondrian_groups, load_plane
import torch

model = load_plane("publication/artefacts/models/digits_clean.pt", device="cuda")

# --- fit LDA on TRAIN pools only ---
g_tr = make_mondrian_groups(X_train, seed=0)
lda = CoverEntropyLDA().fit(
    model, g_tr["digit"], torch.cat([g_tr["gauss"], g_tr["shuffle"]], 0)
)

# --- Mondrian calib on held-out digits (synthesizes gauss/shuffle) ---
cal = MondrianCalibrator(score=lda)           # or score="affinity_entropy"
cal.fit_from_digits(model, X_calib, seed=1)

levels = cal.levels(alphas=(0.01, 0.05, 0.1))  # {group: {α: threshold}}
s = cal.score_points(model, X_test)
p = cal.p_values(s, sided="upper")             # OOD p-values
sets = cal.prediction_set(s, alpha=0.05)       # two-sided sets
```

**CLI**

```bash
leanmap mondrian --list-scores
leanmap mondrian model.pt calib.npy -o mondrian.pt \
  --score affinity_entropy --alphas 0.01,0.05,0.1
leanmap mondrian model.pt --load mondrian.pt \
  --eval test.npy --eval-out eval.npz --alpha 0.05
```

**Semantics**

- `levels` / `threshold`: upper-tailed — reject group \(g\) if \(s > q_g(\alpha)\).  
- `prediction_set`: two-sided — which groups are plausible for this score.  
- Exchangeability requires: score fixed before calib; digit calib unused for LDA fit.

### 3.3 End-to-end demo

```bash
cd examples
python digits_mondrian.py --model out/digits.pt --device cuda
python digits_mondrian.py --model out/digits.pt --score lda   # if wired; else use API above
```

---

## 4. Figure atlas

| file | content |
|------|---------|
| [`artefacts/figures/01_digits_embedding.png`](artefacts/figures/01_digits_embedding.png) | Clean leanmap embedding of digits |
| [`artefacts/figures/02_mondrian_hist.png`](artefacts/figures/02_mondrian_hist.png) | Affinity-entropy scores + Mondrian thresholds |
| [`artefacts/figures/03_mondrian_overlay.png`](artefacts/figures/03_mondrian_overlay.png) | Embedding with OOD overlays / prediction sets |
| [`artefacts/figures/04_cover_vs_entropy.png`](artefacts/figures/04_cover_vs_entropy.png) | Cover vs \(H(a)\): complementary axes |
| [`artefacts/figures/05_lda_plane_dual.png`](artefacts/figures/05_lda_plane_dual.png) | Fisher LDA hyperplane in \((\mathrm{cover},H)\) |
| [`artefacts/figures/06_lda_hist_dual.png`](artefacts/figures/06_lda_hist_dual.png) | LDA score densities |
| [`artefacts/figures/07_dual_basin_overlay.png`](artefacts/figures/07_dual_basin_overlay.png) | Optional: trash-basin junk lobe in Z |
| [`artefacts/figures/08_umap_nn15_reference.png`](artefacts/figures/08_umap_nn15_reference.png) | UMAP reference (nn=15) |

---

## 5. Tables

| file | content |
|------|---------|
| [`tables/ood_detection.md`](tables/ood_detection.md) | Clean vs dual, three scores (human-readable) |
| [`tables/ood_detection.csv`](tables/ood_detection.csv) | Same, machine-readable |
| [`tables/mondrian_levels_affinity_entropy.csv`](tables/mondrian_levels_affinity_entropy.csv) | Thresholds by group × α |
| [`tables/nonconformity_catalog.csv`](tables/nonconformity_catalog.csv) | Score definitions |

**Headline (clean model, α=0.05, matched protocol):** LDA and cover both reach
TPR ≈ 1.0 on gauss/shuffle with FPR ≈ 0.03–0.08; see the markdown table.

---

## 6. Optional: trash basin in Z

`examples/digits_ood_basin.py` parks shuffle and/or Gaussian junk at a learned
anchor **without** putting junk edges in the primary graph. This yields a
visible OOD lobe in the embedding (figure 07) and can improve Mondrian
**type** separation (gauss vs shuffle singletons).

It does **not** improve digit-vs-noise detection on ambient scores — cover TPR
on Gaussian noise drops (1.00 → ~0.91). Prefer the clean model for OOD power;
use a basin only when you need junk geometry in Z.

---

## 7. Reproduce this record

```bash
bash publication/reproduce.sh
```

Exact knobs: [`params/digits_clean.yaml`](params/digits_clean.yaml).  
Math: [`docs/math/leanmap.tex`](../docs/math/leanmap.tex) (§ conformal, Mondrian categories, LDA).

---

## 8. Minimal checklist

1. `PLANEConfig.for_scale(N)` → set `min_dist` / `epochs` for your data.  
2. `fit` → `save("model.pt")`.  
3. Hold out `X_calib` (real in-support rows only).  
4. Fit `CoverEntropyLDA` on a **train** digit vs noise pool.  
5. `MondrianCalibrator(score=lda).fit_from_digits(model, X_calib)`.  
6. Score new points; gate with `p ≤ α` or read `levels`.  
7. Keep the `.pt` model + Mondrian `state_dict` as the long-lived artefact.

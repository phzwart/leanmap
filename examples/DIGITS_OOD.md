# Digits OOD, Mondrian levels, and LDA nonconformity

> **Canonical publication record:** [`../publication/GUIDE.md`](../publication/GUIDE.md)
> (params, figures, tables, frozen models). This file is a compact experiment
> note; prefer the publication guide for a fresh user-facing path.

Reproducible from `examples/` with `pip install -e ".[examples,cpu]"`.

## Models

| artefact | what it is |
|----------|------------|
| `out/digits.pt` | Clean leanmap fit on sklearn digits (`digits.py`, `min_dist=0.1`) |
| `out/digits_ood_basin.pt` | Same primary graph on **real digits only**, plus aux basin loss on **pixel-shuffled** junk |
| `out/digits_ood_basin_dual.pt` | Basin on **shuffle + μ/σ-matched Gaussian** (shared junk anchor) |
| `out/digits_ood_basin_dual_repel.pt` | Dual basin + hinge repel of junk away from real embeddings in Z |

Basin training does **not** put junk edges in the neighbour graph. After each
geom epoch it pulls OOD embeddings to a learned anchor and softly pins real
points (`digits_ood_basin.py`).

```bash
cd examples
python digits.py --device cuda --min-dist 0.1
python digits_ood_basin.py --device cuda --ood shuffle,gauss   # → dual
```

## Nonconformity scores

Conformal validity needs the score fixed before calibration. Built-ins live in
`leanmap.conformal` (`list_nonconformity_scores()`):

| score | definition | notes |
|-------|------------|--------|
| `cover` / `dm_min` | `min_ℓ ‖x − M_ℓ‖` | Ambient landmark cover; default in `embed()` |
| `affinity_entropy` | `H(a) = −∑ a_ℓ log a_ℓ` | Best **unsupervised** separator in the hunt |
| `dm_min+a_ent` | scaled cover + entropy | Simple sum |
| `lda` | signed distance to Fisher LDA on `(cover, H)` | Needs a fitted `CoverEntropyLDA` |
| `emb_cover`, `soft_cover`, … | see registry | Weaker alone for digit OOD |

**Hunt takeaway (frozen model, train/calib/test split):** affinity entropy and
cover are only partly redundant. Cover orders digit ≪ gauss ≪ shuffle along
ambient distance; entropy mostly flags “flat affinity” (any noise). On dual
basin, Spearman(cover, H) ≈ 0.51 overall and ≈ 0.35 within digits — entropy is
not a monotone rehash of cover.

Scatter: `out/digits_cover_vs_entropy.png`.

## Mondrian levels (digit / gauss / shuffle)

`MondrianCalibrator` calibrates **separately** on three groups:

1. **digit** — held-out real rows  
2. **gauss** — i.i.d. Gaussian with the same per-feature μ/σ as the calib digits  
3. **shuffle** — pixel-permuted copies of calib digits  

`levels(α)` returns upper-tailed thresholds per group. `prediction_set` uses
**two-sided** p-values so a typical digit is not accepted as gauss merely
because its score sits below the noise cloud.

### CLI

```bash
leanmap mondrian --list-scores
leanmap mondrian out/digits.pt out/digits_calib.npy -o out/mondrian.pt \
  --score affinity_entropy --alphas 0.01,0.05,0.1
leanmap mondrian out/digits.pt --load out/mondrian.pt \
  --eval out/digits_test.npy --eval-out out/mondrian_eval.npz
```

### Example script

```bash
cd examples
python digits_mondrian.py --model out/digits.pt --device cuda
python digits_mondrian.py --model out/digits_ood_basin_dual.pt --device cuda
# → out/<model_stem>_mondrian_{hist,overlay}.png  .pt  _eval.npz
```

### Python

```python
from leanmap import MondrianCalibrator, CoverEntropyLDA, make_mondrian_groups
import torch

cal = MondrianCalibrator(score="affinity_entropy")  # default
cal.fit_from_digits(model, X_calib)
levels = cal.levels(alphas=(0.01, 0.05, 0.1))
s = cal.score_points(model, X_test)
sets = cal.prediction_set(s, alpha=0.05)
```

## LDA nonconformity

Fit Fisher LDA on features `φ = (cover, H(a))` with labels in-support vs OOD
(train pools only), then score with signed distance to the hyperplane
(oriented **higher ⇒ more OOD**):

```python
g = make_mondrian_groups(X_train, seed=0)
lda = CoverEntropyLDA().fit(
    model, g["digit"], torch.cat([g["gauss"], g["shuffle"]], 0)
)
cal = MondrianCalibrator(score=lda)
cal.fit_from_digits(model, X_calib)
```

On dual basin the learned normal is almost pure entropy
`w ≈ (0.15, 0.99)`. On the clean model it balances both axes
`w ≈ (0.70, 0.71)`.

Plots: `out/digits_ood_basin_dual_lda.png`, `..._lda_hist.png`.

## Clean vs dual basin (matched protocol)

Same train/calib/test split of digits, same gauss/shuffle draws, α = 0.05.
Digit-calib conformal (upper tail) + Mondrian two-sided sets.

| model | score | FPR | TPR gauss | TPR shuffle | AUC_g | digit-only set | OOD leak→digit |
|-------|-------|----:|----------:|------------:|------:|---------------:|---------------:|
| **clean** | cover | 0.08 | **1.00** | 1.00 | **0.999** | 0.95 | 0.00 |
| **clean** | affinity_entropy | 0.04 | 0.995 | 0.94 | 0.999 | 0.92 | ~0.01–0.11 |
| **clean** | lda | 0.03 | **1.00** | 1.00 | **1.000** | 0.97 | ~0 |
| dual basin | cover | 0.07 | 0.91 | 1.00 | 0.981 | 0.77 | 0.13 |
| dual basin | affinity_entropy | 0.05 | 0.99 | 0.94 | 0.997 | 0.93 | higher leak |
| dual basin | lda | 0.04 | 0.998 | 1.00 | 0.999 | 0.95 | ~0 |

**Conclusion:** for **digit vs noise detection**, the clean model is as good or
better. Dual basin **hurts cover** (gauss TPR 1.00 → 0.91) and flattens
affinities so entropy/LDA sit in a narrower band. Dual basin **helps** park
junk in Z and, under Mondrian+LDA, tell gauss from shuffle as singleton sets
(`sing_g` ≈ 0.92 vs ≈ 0.30 on clean) — useful for taxonomy, not for ambient OOD
power.

Basin + Z-repel does **not** fix cover-based detection: cover is ambient
`min_ℓ ‖x−M_ℓ‖` and is unchanged by where points sit in the embedding.

## Practical recipe

1. Fit **clean** `digits.pt` for the map + conformal OOD.  
2. Prefer **`lda`** (fit on train digit vs train noise) or **`affinity_entropy`**
   for Mondrian levels; keep `cover` as the simple baseline.  
3. Use **dual basin** only if you care about a visible junk lobe / separating
   noise *types* in Z — not if the goal is max TPR on μ/σ Gaussian probes.  
4. Always hold out a digit calib pool disjoint from anything that fits LDA
   weights, landmark charts, or basin targets.

## Related code

| path | role |
|------|------|
| `src/leanmap/conformal.py` | `MondrianCalibrator`, `CoverEntropyLDA`, score registry |
| `src/leanmap/_cli.py` | `leanmap mondrian …` |
| `examples/digits_mondrian.py` | End-to-end Mondrian demo |
| `examples/digits_ood_basin.py` | Basin training |
| `docs/METRICS.md` | Short OOD / Mondrian pointer |
| `src/leanmap/README.md` | Design notes (conformal + Mondrian + LDA) |

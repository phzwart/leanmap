# Examples

Toy demos and the exploratory paper harness.

| script | role |
|--------|------|
| `s_curve.py` | Standalone S-curve fit / plot |
| `swiss_roll.py` | Standalone swiss-roll fit / plot |
| `digits.py` | Standalone 8×8 digits fit / plot |
| `digits_emd.py` | Digits with L1-tree → torchemd EMD-rescored kNN |
| `cellcycle_emd.py` | CellCycle merged subset → zarr + L1 (or EMD) kNN + leanmap |
| `cellcycle_explorer.py` | Dash explorer: latent scatter + cell image inspection |
| `cellcycle_lejepa.py` | Cheap LeJEPA on Ch3/4/6 → features → leanmap |
| `cellcycle_celldino.py` | Frozen Cell-DINO (CP ViT-S/8; 3→5ch zero-pad) → leanmap |
| `pistachio_ftir.py` | FTIR spectra (cosine) → leanmap with per-epoch frames |
| `pistachio_ftir_explorer.py` | Dash explorer: latent / spatial / spectrum (+ UMAP) |
| `digits_mondrian.py` | Mondrian levels (digit / gauss / shuffle) on digits |
| `digits_ood_basin.py` | Digits + OOD basin (shuffle / gauss junk parks) |
| `exploratory/` | Paper battery: feeds, sweeps, metrics, EMD, conformal |
| `negative_space.py` | Frozen distance-to-support probe (post-hoc) |
| `reusability.py` | Out-of-sample reuse demo |

Paper documentation lives under [`docs/`](../docs/):

- [CONFIGURATION.md](../docs/CONFIGURATION.md) — practical settings
- [RESULTS.md](../docs/RESULTS.md) — evidence on s-curve, swiss roll, digits, iris
- [METRICS.md](../docs/METRICS.md) — how to read the battery (incl. Mondrian OOD)

Digits OOD / Mondrian / LDA notes: [DIGITS_OOD.md](DIGITS_OOD.md).  
CellCycle G1/S/G2 embedding + mitotic novelty: [CELLCYCLE_NOVELTY.md](CELLCYCLE_NOVELTY.md).  
**Publication record (guide, params, figures, tables):** [`../publication/`](../publication/).

```bash
python examples/exploratory/prepare_feeds.py
python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name paper_digits --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8
```

### Mondrian levels

End-to-end demo (loads `out/digits.pt`, prints levels, writes plots):

```bash
cd examples && python digits_mondrian.py --device cuda
# → out/digits_mondrian_{hist,overlay}.png  out/digits_mondrian.pt
```

Or via the package CLI after exporting a calib `.npy`:

```bash
leanmap mondrian out/digits.pt out/digits_calib.npy -o out/mondrian.pt \
  --score affinity_entropy --alphas 0.01,0.05,0.1
```

See [`exploratory/README.md`](exploratory/README.md) for the harness contract.

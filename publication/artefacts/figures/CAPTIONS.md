# Figure captions

| file | caption |
|------|---------|
| `01_digits_embedding.png` | leanmap embedding of sklearn digits (clean model, `min_dist=0.1`, 80 epochs). Colour = digit class. |
| `02_mondrian_hist.png` | Affinity-entropy score densities for digit / gauss / shuffle with Mondrian thresholds at α=0.05 (dashed). |
| `03_mondrian_overlay.png` | Same embedding with synthetic OOD overlaid; markers indicate whether the two-sided Mondrian set includes `digit`. |
| `04_cover_vs_entropy.png` | Joint distribution of landmark cover vs affinity entropy on clean (left) and dual-basin (right) models. |
| `05_lda_plane_dual.png` | Fisher LDA hyperplane in \((\mathrm{cover},H)\) for the dual-basin model; score = signed distance (↑ OOD). |
| `06_lda_hist_dual.png` | LDA nonconformity histograms with Mondrian group thresholds. |
| `07_dual_basin_overlay.png` | Dual trash-basin embedding: shuffle parks in a tight junk lobe; Gaussian noise spreads through the map. |
| `08_umap_nn15_reference.png` | UMAP reference embedding (n_neighbors=15) for qualitative comparison. |

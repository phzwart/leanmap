# Publication record — leanmap embeddings + outlier detection

Long-lived artefacts for a **fresh** reader: how to embed data with leanmap and
run calibrated outlier detection on the same model. No development diary —
start at the guide.

| path | role |
|------|------|
| **[GUIDE.md](GUIDE.md)** | Concise complete user guide |
| [params/digits_clean.yaml](params/digits_clean.yaml) | Exact digits fit + OOD protocol |
| [tables/](tables/) | OOD detection & Mondrian threshold tables |
| [artefacts/figures/](artefacts/figures/) | Numbered figure atlas |
| [artefacts/models/](artefacts/models/) | Frozen `.pt` models and calibrators |
| [reproduce.sh](reproduce.sh) | Regenerate tables / refresh copies from `examples/out` |
| [../docs/math/leanmap.tex](../docs/math/leanmap.tex) | Mathematics (incl. Mondrian categories & LDA) |

## Recommended artefact set

| artefact | purpose |
|----------|---------|
| `artefacts/models/digits_clean.pt` | Interpretable digits embedding + pooled cover calib |
| `artefacts/models/mondrian_lda_clean.pt` | LDA nonconformity + Mondrian levels (digit/gauss/shuffle) |
| `artefacts/models/mondrian_affinity_entropy.pt` | Entropy-only Mondrian (no LDA fit) |
| `artefacts/figures/01_*.png` … `08_*.png` | Figures for the guide |
| `tables/ood_detection.md` | Clean vs dual-basin detection summary |

## One-command refresh

```bash
bash publication/reproduce.sh
```

Requires a working install (`pip install -e ".[examples,cpu]"`) and, for a full
refit, CUDA. By default the script **refreshes tables** and **copies** figures
from `examples/out/` if models already exist; pass `--fit` to retrain digits.

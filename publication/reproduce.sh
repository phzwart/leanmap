#!/usr/bin/env bash
# Refresh publication tables and curated artefacts.
# Usage:
#   bash publication/reproduce.sh           # tables + copy figures/models
#   bash publication/reproduce.sh --fit     # also retrain digits_clean.pt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi

FIT=0
for arg in "$@"; do
  [[ "$arg" == "--fit" ]] && FIT=1
done

mkdir -p publication/{artefacts/figures,artefacts/models,tables,params}

if [[ "$FIT" -eq 1 ]]; then
  echo "==> fitting clean digits model"
  (cd examples && "$PY" digits.py --device "${LEANMAP_DEVICE:-cuda}" --min-dist 0.1 --epochs 80 --seed 0)
  echo "==> Mondrian demo (affinity entropy)"
  (cd examples && "$PY" digits_mondrian.py --model out/digits.pt --device "${LEANMAP_DEVICE:-cuda}")
fi

echo "==> writing tables + clean LDA Mondrian artefact"
"$PY" - <<'PY'
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from leanmap import load_plane, MondrianCalibrator, CoverEntropyLDA, make_mondrian_groups
from leanmap.conformal import affinity_entropy_score, cover_score

ROOT = Path("publication")
TABLES, MODELS = ROOT / "tables", ROOT / "artefacts" / "models"
TABLES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED, ALPHA = 0, 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"
X = load_digits().data.astype(np.float32)
perm = np.random.default_rng(SEED).permutation(len(X))
n_tr, n_ca, n_te = 800, 200, 400
X_tr, X_ca, X_te = X[perm[:n_tr]], X[perm[n_tr:n_tr+n_ca]], X[perm[n_tr+n_ca:n_tr+n_ca+n_te]]
g_te = make_mondrian_groups(torch.as_tensor(X_te), n_gauss=400, n_shuffle=400, seed=SEED+9)
pools = {"digit": X_te, "gauss": g_te["gauss"].numpy(), "shuffle": g_te["shuffle"].numpy()}

def conformal_p(s_cal, s):
    s_cal = np.sort(np.asarray(s_cal, float))
    n = len(s_cal)
    idx = np.searchsorted(s_cal, np.asarray(s, float), side="left")
    return (1 + (n - idx)) / (n + 1)

def eval_model(path):
    model = load_plane(str(path), device=device)
    g_tr = make_mondrian_groups(torch.as_tensor(X_tr), seed=SEED)
    lda = CoverEntropyLDA().fit(model, torch.as_tensor(X_tr), torch.cat([g_tr["gauss"], g_tr["shuffle"]], 0))
    w, b = lda.hyperplane()
    scorers = {
        "cover": lambda A: cover_score(model, torch.as_tensor(A, device=device)).cpu().numpy(),
        "affinity_entropy": lambda A: affinity_entropy_score(model, torch.as_tensor(A, device=device)).cpu().numpy(),
        "lda": lambda A: lda(model, torch.as_tensor(A)).numpy(),
    }
    rows, levels_rows = [], []
    for name, fn in scorers.items():
        cal = MondrianCalibrator(score=lda if name == "lda" else name)
        cal.fit_from_digits(model, torch.as_tensor(X_ca), seed=SEED+1)
        dig = cal.s_calib["digit"].numpy()
        s_te, s_g, s_s = fn(pools["digit"]), fn(pools["gauss"]), fn(pools["shuffle"])
        rows.append({
            "model": Path(path).stem, "score": name, "alpha": ALPHA,
            "fpr_digit": float((conformal_p(dig, s_te) < ALPHA).mean()),
            "tpr_gauss": float((conformal_p(dig, s_g) < ALPHA).mean()),
            "tpr_shuffle": float((conformal_p(dig, s_s) < ALPHA).mean()),
            "auc_gauss": roc_auc_score(np.r_[np.zeros(len(s_te)), np.ones(len(s_g))], np.r_[s_te, s_g]),
            "auc_shuffle": roc_auc_score(np.r_[np.zeros(len(s_te)), np.ones(len(s_s))], np.r_[s_te, s_s]),
            "frac_digit_singleton": float(np.mean([t == ("digit",) for t in cal.prediction_set(torch.as_tensor(s_te), alpha=ALPHA)])),
            "frac_gauss_has_digit": float(np.mean([("digit" in t) for t in cal.prediction_set(torch.as_tensor(s_g), alpha=ALPHA)])),
            "frac_shuffle_has_digit": float(np.mean([("digit" in t) for t in cal.prediction_set(torch.as_tensor(s_s), alpha=ALPHA)])),
            "frac_gauss_singleton": float(np.mean([t == ("gauss",) for t in cal.prediction_set(torch.as_tensor(s_g), alpha=ALPHA)])),
            "frac_shuffle_singleton": float(np.mean([t == ("shuffle",) for t in cal.prediction_set(torch.as_tensor(s_s), alpha=ALPHA)])),
            "lda_w_cover": float(w[0]), "lda_w_entropy": float(w[1]), "lda_bias": float(b),
        })
        if name == "affinity_entropy":
            for g, d in cal.levels(alphas=(0.01, 0.05, 0.1)).items():
                for a, thr in d.items():
                    levels_rows.append({"model": Path(path).stem, "score": name, "group": g, "alpha": a, "threshold": thr, "n_calib": int(cal.s_calib[g].numel())})
        if name == "lda" and Path(path).stem == "digits":
            torch.save({"lda": lda.state_dict(), "mondrian": cal.state_dict(),
                        "protocol": {"n_train": n_tr, "n_calib": n_ca, "seed_train_ood": SEED, "seed_calib": SEED+1}},
                       MODELS / "mondrian_lda_clean.pt")
    return rows, levels_rows

paths = ["examples/out/digits.pt"]
dual = Path("examples/out/digits_ood_basin_dual.pt")
if dual.is_file():
    paths.append(str(dual))

all_rows, all_levels = [], []
for p in paths:
    r, lv = eval_model(p)
    all_rows.extend(r); all_levels.extend(lv)

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

write_csv(TABLES / "ood_detection.csv", all_rows)
write_csv(TABLES / "mondrian_levels_affinity_entropy.csv", all_levels)
write_csv(TABLES / "nonconformity_catalog.csv", [
    {"score": "cover", "definition": "min_l ||x-M_l||", "needs_fit": "no", "role": "ambient landmark support"},
    {"score": "affinity_entropy", "definition": "H(a)=-sum a_l log a_l", "needs_fit": "no", "role": "default Mondrian nonconformity"},
    {"score": "lda", "definition": "signed distance to Fisher LDA on (cover, H)", "needs_fit": "yes (CoverEntropyLDA)", "role": "recommended digit-vs-noise score"},
    {"score": "dm_min+a_ent", "definition": "cover/med + H/med", "needs_fit": "scales from digit calib", "role": "simple composite"},
])
with open(TABLES / "ood_detection.md", "w") as f:
    f.write("# Digits OOD detection (matched protocol)\n\n")
    f.write(f"Split: train={n_tr}, calib={n_ca}, test={n_te}; α={ALPHA}; seed={SEED}.\n\n")
    f.write("| model | score | FPR | TPR gauss | TPR shuffle | AUC_g | digit singleton |\n|---|---|---:|---:|---:|---:|---:|\n")
    for r in all_rows:
        f.write(f"| {r['model']} | {r['score']} | {r['fpr_digit']:.3f} | {r['tpr_gauss']:.3f} | {r['tpr_shuffle']:.3f} | {r['auc_gauss']:.3f} | {r['frac_digit_singleton']:.3f} |\n")
print("tables OK; LDA artefact:", MODELS / "mondrian_lda_clean.pt")
PY

echo "==> copying curated figures and models"
cp -f examples/out/digits.pt publication/artefacts/models/digits_clean.pt
[[ -f examples/out/digits_mondrian.pt ]] && cp -f examples/out/digits_mondrian.pt publication/artefacts/models/mondrian_affinity_entropy.pt
copy_fig() { [[ -f "$1" ]] && cp -f "$1" "$2" && echo "  $2"; }
copy_fig examples/out/digits.png publication/artefacts/figures/01_digits_embedding.png
copy_fig examples/out/digits_mondrian_hist.png publication/artefacts/figures/02_mondrian_hist.png
copy_fig examples/out/digits_mondrian_overlay.png publication/artefacts/figures/03_mondrian_overlay.png
copy_fig examples/out/digits_cover_vs_entropy.png publication/artefacts/figures/04_cover_vs_entropy.png
copy_fig examples/out/digits_ood_basin_dual_lda.png publication/artefacts/figures/05_lda_plane_dual.png
copy_fig examples/out/digits_ood_basin_dual_lda_hist.png publication/artefacts/figures/06_lda_hist_dual.png
copy_fig examples/out/digits_ood_basin_dual_overlay.png publication/artefacts/figures/07_dual_basin_overlay.png
copy_fig examples/out/umap_nn15.png publication/artefacts/figures/08_umap_nn15_reference.png

echo "==> done. See publication/GUIDE.md"
ls publication/artefacts/models publication/tables | sed 's/^/  /'

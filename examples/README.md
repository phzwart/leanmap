# leanmap examples

Curated toy demos:

```bash
pip install -e ".[examples,cpu]"

python examples/s_curve.py
python examples/swiss_roll.py
python examples/swiss_cone.py
python examples/digits.py
python examples/petiole.py
python examples/reusability.py
```

See [`REUSABILITY.md`](REUSABILITY.md) for a walkthrough of why a fitted leanmap
is a reusable, OOD-aware *model* rather than a one-off layout (with a UMAP
comparison).

Plots write to `examples/out/` (gitignored).

| script | data | default N |
|--------|------|-----------|
| `s_curve.py` | `sklearn.datasets.make_s_curve` (gallery: n=1500, noise=0) | 1500 × 3 |
| `swiss_roll.py` | `sklearn.datasets.make_swiss_roll` | 2000 × 3 |
| `swiss_cone.py` | flaring Swiss *roll* ribbon + parameter-space hole | 5000 × 3 |
| `digits.py` | `sklearn.datasets.load_digits` (8×8) | 1797 × 64 |
| `petiole.py` | 11×11 patches from petiole tomography (256×256) | 5000 × 121 |
| `reusability.py` | Swiss cone; reuse + OOD vs UMAP | 2000 train / 10k test |

Systematic axis sweeps (generic array ingest, visual/metric atlas) live under
[`exploratory/`](exploratory/) — see that README for the master CLI.

Older cell / ligand / sweep scripts and run artifacts live under
`legacy/examples/` (gitignored).

# leanmap examples

## `primitive_orthorhombic.py`

Leanmap of **58,025 primitive (P-centered) orthorhombic PDB unit cells**,
described by the volume-normalized Kurlin cell roots (`rn0..rn5 = r_i / V^(1/3)`)
— a dimensionless, scale-invariant descriptor of lattice *shape*.

```bash
python examples/primitive_orthorhombic.py
```

Fits a plain graph-loss leanmap (no inducing points, no UMAP prior) and writes
`primitive_orthorhombic_embedding.npy` plus a scatter colored by space group.

### Data

`data/primitive_orthorhombic_cells.csv` — 58,025 rows × 22 columns:

| columns | meaning |
|---|---|
| `pdb_id` | PDB identifier |
| `a,b,c,alpha,beta,gamma,volume` | raw cell parameters (all angles 90°) |
| `sg_number,sg_hm` | space group number and Hermann–Mauguin symbol (all `P`) |
| `r0..r5` | raw Kurlin cell roots (Delaunay-reduced lattice invariant) |
| `rn0..rn5` | volume-normalized roots `r_i / V^(1/3)` (shape descriptor) |

Subset of the full PDB cell set: orthorhombic space groups 16–74, primitive
centering only — the clean "primitive" orthorhombic island.

### Note on faiss + torch on macOS

Fitting needs FAISS to build the k-NN graph. On macOS a venv that mixes a
pip `faiss-cpu` with a conda `torch` can hit an OpenMP double-initialization
(`OMP: Error #15`) crash. Use a single-source environment (e.g. install both
from conda-forge) if you see it.

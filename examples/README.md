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

## `data/all_pdb_cells.csv`

The full set — **206,184 PDB unit cells** across all seven crystal systems.
Same columns as above, plus two extra label columns:

| columns | meaning |
|---|---|
| `crystal_system` | triclinic / monoclinic / orthorhombic / tetragonal / trigonal / hexagonal / cubic (from `sg_number`) |
| `centering` | lattice centering from the HM symbol (`A`/`B` collapsed to `C`) |

System counts: orthorhombic 74,017 · monoclinic 57,836 · tetragonal 24,642 ·
trigonal 21,555 · hexagonal 15,454 · triclinic 8,492 · cubic 4,188.
Centering: P 155,726 · C 32,192 · I 11,754 · H 5,457 · F 1,039 · R 16.

### Note on faiss + torch on macOS

Fitting needs FAISS to build the k-NN graph. On macOS a venv that mixes a
pip `faiss-cpu` with a conda `torch` can hit an OpenMP double-initialization
(`OMP: Error #15`) crash. Use a single-source environment (e.g. install both
from conda-forge) if you see it.

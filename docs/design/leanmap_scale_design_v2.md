# leanmap scale design v2 (normative)

**Status:** normative for the 10M refactor PR series  
**Supersedes for implementation:** [`10m_scale_graph_and_train.md`](10m_scale_graph_and_train.md)  
**Departures / choices:** [`departures_from_10m_design.md`](departures_from_10m_design.md)

This document is the implementation contract for scaling leanmap to \(N\sim 10^7\)
while preserving the scientific product: graph build, parametric encoder, and
**first-class core capabilities** (path, class-axis, conformal, density,
conditioning, negative space, evaluate/EMD).

---

## 0. Principles

1. **Code organization mirrors the N/R/L decomposition.** Raw-row work lives in
   streaming-pass / memmap modules; R-bound work under graph/store; L-bound work
   with the model. Reviewers audit “does N leak superlinear?” via imports.
2. **Build / freeze / train are package boundaries.** `fit(...)` remains a
   convenience wrapper: build → freeze → train.
3. **Bit-compat contract:** with \(\delta=\varepsilon\), single process,
   `graph.pt` backend, PRs through the DDP entry at world_size=1 reproduce
   current outputs on golden fixtures (seeded 2k swiss-cone + 10k digits).
4. **Measured sensitivities become assertions:** exact q0.99, deterministic
   union-find roots, same-star \(\ell_x\), recall gate.
5. **Core capabilities stay core.** Path, class-axis, conformal, density,
   conditioning, negative space are not demoted to extras or HPC-only.

---

## 1. Target layout

```text
src/leanmap/
  config.py            # BuildConfig, StoreConfig, TrainConfig, PolicyConfig;
                       # PLANEConfig deprecated shim → four configs
  metrics.py

  build/
    resolution.py      # eps (Def 1) + solve_delta(probe, r_band, alpha_guard)
    landmarks.py
    net.py             # greedy net, max_bucket, halo → union-find
    knn.py             # brute | ivf | ann | ann-gpu; recall_gate
    fuzzy.py           # smooth-kNN, t-conorm, multiplicity, backbone
    pyramid.py         # Galerkin; exact_quantile(w, q) two-pass
    stages.py          # Zarr staging, fingerprints, resume
    pipeline.py        # build_graph orchestration → GraphStore
    bunches.py         # [hpc] distributed build; lazy mpi4py

  store/
    schema.py          # layout + meta.json (versioned)
    base.py            # GraphStore protocol
    ptfile.py          # graph.pt backend (byte-identical to legacy)
    dirstore.py        # directory/Zarr; auto-select on R/T
    fingerprint.py     # streamed xxhash; sampled verify on load

  sampling/
    alias.py           # Vose + two-level alias
    edges.py           # EdgeSampler via alias + memmap CSR
    ordinal.py
    paths.py           # PathTripletSampler (memmap gather)
    policy.py          # ExemplarPolicy p_t

  model/               # affinity, film, heads, warmstart
  losses/              # geom, ordinal, landmark, frame, geo, density, path, ddp_stats
  train/               # fit, ddp, probes
  paths/               # constraint, build, parse
  diagnostics/
    record.py          # typed D → meta.json

  cli.py               # leanmap-graph-build, leanmap-train
```

Legacy modules (`graph.py`, `graph_stages.py`, `sampler.py`, `path.py`,
`train.py`, …) are re-export shims for one deprecation cycle.

---

## 2. Resolution contract

- **ε (Definition 1):** unchanged — \(Q_{0.01}\) of 1-NN; exact for small \(N\);
  subsample + intrinsic-dim correction above. Written into D.
- **δ:** `solve_delta(probe, r_band, alpha_guard) -> (delta, report)`. Net runs
  at δ; ε-cells nest (member CSR composes ε→δ→raw).
- Default `delta="auto"` above an \(N\) threshold; `delta=eps` reproduces today.
- Target **R band:** \(R \in [10^5, 10^6]\) unless overridden.
- Degenerate-fraction warnings when duplicate-heavy.

---

## 3. Build pipeline (§3.2)

Scientific order:

1. Landmarks \(M\) (FPS / geodesic Poisson-disk; top-1 / top-c)
2. ε-net / δ-net per bucket + halo merge (deterministic union-find)
3. kNN on reps (spill to Zarr at large \(R\)); recall gate
4. smooth_knn → fuzzy union → multiplicity → landmark backbone → edges
5. Pyramid Galerkin (`pyramid_scales`); exact quantile squash
6. Freeze → `GraphStore` (ptfile or dirstore)

Lifecycle: **Build → Freeze → Train** (immutable graph).

---

## 4. Store schema

```text
graph_store/          # or single graph.pt via ptfile
  meta.json           # fingerprint, ε, δ, L, k, metric, seed, D record, …
  landmarks.zarr / landmarks tensors
  reps/  knn/  csr/  pyramid/
  alias/              # edge / family alias tables (may be empty early)
  density/
  gauge/              # gauge level + ν
  paths/              # path artefacts (may be empty)
```

**Invalidation (rebuild required):** X fingerprint, metric, ε/δ, k, L, seed,
dedup, pyramid depth.  
**Safe without rebuild:** lr, epochs, `lambda_path`, batches, DDP size, \(p_t\),
probes.

Auto-select directory store when \(R\) or triplet count \(T\) exceeds thresholds
(\(R_{\mathrm{spill}}\sim 5\times 10^4\) for kNN spill; store backend thresholds
in StoreConfig).

---

## 5. Sampling

- Vose alias on edge mass; two-level (shard-mass → in-shard) at freeze.
- `EdgeSampler`: alias + cell→member expansion from memmap CSR.
- Ordinal / path samplers remain core; path uses memmap gather + group-mass alias.
- **ExemplarPolicy \(p_t\):** three families, tilts, coverage floors;
  `reweight=True` default (\(w/p_t\), ratio-capped). CLI:
  `--exemplar-policy {uniform,sufficient_v1}`.

---

## 6. Path (§7.3)

- Vectorized `build_path_triplets` (`searchsorted`);
  `tie_policy ∈ {first, last, drop}` counted into D.
- ε-filter: drop \(\varphi(x_a,x_n)\le\varepsilon\).
- Loss: log-space hinges + distance floors at \(\kappa\cdot s\); feasible set
  \([c,C]\) unchanged; legacy ratio hinge behind a flag for one cycle.

---

## 7. Train / DDP / warm start

- Warm start: Nyström streaming over memmap \(X\) + top-c shortlist; auto layout.
- Schedule defaults keyed on \(N\): warm start + `coarse_first_frac>0` above
  threshold; flat below.
- DDP: `torchrun` entry; seed+rank samplers; allreduce \(\bar a\), density
  moments, path-scale batch mean; geo replicated. Path and class-axis remain
  available under DDP.

---

## 8. Gauge level

Dijkstra on a selectable pyramid level; edge lengths = aggregated **metric**
distances (never squashed weights). Default: level 0 below
\(R\approx 3\times 10^5\), level 1 above. Record level + \(\nu\) in `gauge/`.

---

## 9. Distributed build (hpc; contingency)

`build/bunches.py` behind `leanmap[hpc]`: probe → landmark reconcile → bunch
partition → margin halo → owned nets → kNN fill → distributed union-find →
stitch. Activated by R/N thresholds or `--bunch-partition mpi`.  
**10M at the R band is a single-node job**; core paths must not depend on bunches.

---

## 10. Diagnostics \(\mathcal{D}\)

Typed fields in `diagnostics/record.py` → `meta.json`, including: ε path,
δ report, recall + mode, quantile method, tie-policy counts, \(\nu\), gauge
level, clamp hit rate, ord/lip stats, compression \(N/R\), degenerate fraction.

---

## 11. CLI (App. B)

```text
leanmap-graph-build --X X.npy --out graph_store/ \
  --stages graph_stages/ --knn-mode ann \
  --pyramid-scales 3 --epsilon ... \
  [--bunch-partition mpi|local]

torchrun --nproc_per_node=G leanmap-train \
  --X X.npy --graph-path graph_store/ \
  --exemplar-policy sufficient_v1 \
  --epochs ... --lambda-path ...
```

---

## 12. Thresholds (summary)

| Threshold | Role |
|-----------|------|
| \(R\in[10^5,10^6]\) | Target R band |
| \(R_{\mathrm{spill}}\sim 5\times 10^4\) | Force Zarr kNN spill |
| \(N\le 2\times 10^4\) | Exact vs subsampled ε |
| \(R\approx 3\times 10^5\) | Gauge level 0 → 1 |
| `for_scale`: 5k / 200k | Train presets |
| kNN recall \(<0.9\) | Raise search / brute |
| Degenerate frac \(>0.05\) | Raise LC / ε |
| \(\nu>0.10\) | MDS warn |

---

## 13. PR series (index)

PR-0 docs → PR-1 split+goldens → PR-2 store → PR-3 δ → PR-4 alias → PR-5 path
→ PR-6 warmstart → PR-7 DDP → PR-8 gauge → PR-9 policy → PR-10 bunches →
PR-11 deprecation sweep.

# Departures and choices vs `10m_scale_graph_and_train.md`

**Normative implementation:** [`leanmap_scale_design_v2.md`](leanmap_scale_design_v2.md)  
**Historical / review doc:** [`10m_scale_graph_and_train.md`](10m_scale_graph_and_train.md)

Where the two disagree, **v2 wins**. This file records intentional departures and
closed open questions so later PRs cannot “simplify for scale” by undoing them.

---

## Section-number map

| Topic | v2 (this series) | Old `10m_…` doc |
|-------|------------------|-----------------|
| Resolution / R band | §2 | §2 |
| Build pipeline order | §3 / §3.2 | §4.1–4.2 |
| Store schema | §4 | §5 |
| Sampling / \(p_t\) | §5 | §6 |
| Path mathematics | §7.3 / §6 | (mostly absent; §9.3 non-goal) |
| DDP | §7 | §7 |
| CLI | §11 / App. B | App. B |
| Bunches / MPI | §9 | §4.3–4.5 |

---

## Departures

### 1. δ net radius (new)

**Old:** single ε; scaling via R-band SLO and refuse silent ε no-op at 10M.  
**v2:** keep Def-1 ε; add `solve_delta` with α fidelity guard; nest ε→δ→raw
cells. Default `delta="auto"` above an \(N\) threshold; `delta=eps` bit-compat.

### 2. Expanded store schema

**Old §5.2:** `meta.json`, landmarks, reps, knn, csr, pyramid.  
**v2:** also `alias/`, `density/`, `gauge/`, `paths/` from day one (may be empty).
Directory store earlier in the series (PR-2) while goldens stay on `graph.pt`.

### 3. Path mathematics (intentional exception)

**Old §9.3:** changing path-loss math is *not* a memory strategy / non-goal.  
**v2:** log-space hinges, \(\kappa\cdot s\) floors, ε-filter, vectorized build —
correctness / stability for near-duplicate windows, not a RAM cut. Legacy ratio
hinge kept behind a flag for one cycle. Path remains a **core** capability.

### 4. Gauge on selectable pyramid level (new)

**Old:** gauge stays landmark MDS / symmetrised kNN distance graph.  
**v2:** Dijkstra on a selectable pyramid level; lengths = aggregated **metric**
distances (never squashed weights); default flip at \(R\approx 3\times 10^5\).

### 5. DDP allreduce helpers (closes old §9.4)

**Old open Q:** geo/density rank-0 vs replicated.  
**v2:** allreduce \(\bar a\), density moments, path-scale batch mean; geo
**replicated**. Path and class-axis remain available under DDP (not rank-0 stubs).

### 6. Bunches remain contingency

**Aligned with old design:** 10M at the R band is single-node-first. PR-10 /
`build/bunches.py` is optional `leanmap[hpc]`; no earlier PR may depend on it.

### 7. Core capabilities stay core (product invariant)

**Old doc** focused on graph+train scale and treated path as out of build RAM.  
**v2:** path, class-axis, conformal/Mondrian, density, conditioning, negative
space, evaluate/EMD remain **first-class public API** — not behind
`leanmap[hpc]`, `leanmap[examples]`, or a research extra. Scaling packages must
not demote them. Policy \(p_t\) may *tilt* using path/class signals; it does not
replace those APIs.

---

## Choices locked for implementers

| Choice | Decision |
|--------|----------|
| Spec authority | This series + v2 doc |
| Bit-compat through PR-7 | \(\delta=\varepsilon\), ws=1, `graph.pt`, golden seeds |
| Golden fixtures | swiss-cone \(N=2000\); digits expanded to \(N=10000\); seed 42; CPU threads=1 |
| Store default for goldens | `ptfile` / `graph.pt` |
| Geo under DDP | Replicated |
| Exemplar default | Locked by PR-9 A/B; until then `uniform` reproduces prior behaviour |
| HPC | Opt-in only |

---

## What did *not* change vs the old design

- Build → freeze → train lifecycle.
- Cover vs exemplar stream are separate problems.
- Disk spill + ANN at large \(R\); topology changes are not free compressors.
- DDP for training only; MPI/partition for distributed build.
- Halo expected; no continuous cross-node landmark chatter during train.

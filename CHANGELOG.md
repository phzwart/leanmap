# Changelog

## 0.3.0 — 2026-08-17

10M-scale refactor (design v2). Package layout mirrors N/R/L decomposition;
behaviour-compatible for the small-N recipe with \(\delta=\varepsilon\) and
`graph.pt`.

### Layout
- New packages: `build/`, `store/`, `sampling/`, `paths/`, `losses/`, `model/`,
  `train/`, `diagnostics/`.
- Legacy modules (`graph.py`, `path.py`, `sampler.py`, …) remain as re-export
  shims for one deprecation cycle.

### Symbol map (old → new)

| Old | New |
|-----|-----|
| `leanmap.graph` | `leanmap.build.pipeline` (+ `store.ptfile` for I/O) |
| `leanmap.graph_stages` | `leanmap.build.stages` |
| `leanmap.sampler` | `leanmap.sampling` |
| `leanmap.path` | `leanmap.paths` |
| `leanmap.train` (module) | `leanmap.train.fit` |
| `leanmap.losses` (module) | `leanmap.losses` (package) |
| `leanmap.model` (module) | `leanmap.model` (package) |
| `leanmap.warmstart` | `leanmap.model.warmstart` |
| `leanmap._cli` | `leanmap.cli` |

### Config
- Prefer `BuildConfig` / `StoreConfig` / `TrainConfig` / `PolicyConfig` via
  `compose_plane_config(...)`.
- `PLANEConfig` remains a compatibility facade (deprecated; removal in a
  future major).

### Features (scale path)
- Store abstraction (`ptfile` + `dirstore`), fingerprints, invalidation.
- Resolution contract: `delta` (`None`/`eps`/`auto`/float); default ε.
- Two-level alias sampling; optional memmap EdgeSampler.
- Path v2: vectorized triplets, tie policies, log-space hinges (opt-in).
- Streaming Nyström; large-N schedule defaults.
- DDP helpers + `fit_ddp` (ws=1 ≡ `fit`).
- Gauge level selection (metric edge lengths).
- ExemplarPolicy `uniform` / `sufficient_v1`.
- HPC bunches (`leanmap.build.bunches`, `leanmap[hpc]`).

### Core capabilities
Path, class-axis, conformal, density, conditioning, and negative space remain
first-class public API (not demoted to extras).

### CLI
- `leanmap-graph-build`, `leanmap-train` (plus existing `leanmap`).

### Design docs
- [`docs/design/leanmap_scale_design_v2.md`](docs/design/leanmap_scale_design_v2.md)
- [`docs/design/departures_from_10m_design.md`](docs/design/departures_from_10m_design.md)

## 0.2.0 — 2026-08-12

Parametric rewrite of leanmap as landmark-conditioned neighbour embedding.

- Fit once, embed new points with a single network forward pass (`fit` /
  `PLANEResult.embed` / `load_plane`).
- Multi-scale cohesive graph pyramid with FiLM landmark conditioning.
- densMAP-style density correspondence and auto warm-start for faster training.
- Conformal cover scores and negative-space novelty helpers.
- CLI: `leanmap fit` / `transform` / `info`.
- Paper battery under `examples/exploratory/`; curated research demos
  (SASBDB P(r), digits density, conformal novelty) under `examples/research/`.

## 0.1.0

Initial package lineage (pre-rewrite).

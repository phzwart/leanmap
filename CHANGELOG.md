# Changelog

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

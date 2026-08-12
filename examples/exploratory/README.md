# Exploratory harness

Array-in, sweep-out driver for reproducing the paper battery. Prepare feeds once,
then point `master.py` at the files.

## Quick start

```bash
pip install -e ".[examples,cpu]"
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS OpenMP duplicate workaround

python examples/exploratory/prepare_feeds.py   # s-curve, swiss_roll, digits, iris

python examples/exploratory/calibrate.py \
  --X examples/exploratory/data/digits_X.npy --target-perp 8

python examples/exploratory/reference.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy --name paper_digits \
  --holdout 0.2 --seeds 0 1 2 --null shuffle

python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name paper_digits --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8 --atlas
```

## Paper sweeps

| sweep | purpose |
|-------|---------|
| `canonical` | Recommended + `min_dist` / `lambda_geo` / `lambda_frame` / weights ladders |
| `iris_canonical` | Small-N recipe (no pyramid); same ladders without weight arms |
| `iris_pyramid_weights` | Didactic iris weights panel (`pyramid_min_reps=16` so levels exist) |
| `swiss_roll_frame` | Fold-back frame stress test (`lambda_geo=0.5`) |

Legacy named sweeps (`matched`, `phase1`, `min_dist_*`, …) remain for
reproducing older runs but are not part of the documented battery.

## Ingest contract

| flag | required | meaning |
|------|----------|---------|
| `--X` | yes | `(N, D)` features |
| `--y` / `--color` | no | labels / colour |
| `--sweep` | no | named sweep (default `phase1`) |
| `--only` | no | one axis or `run_id` |
| `--holdout` | no | OOS fraction |
| `--null` | no | `none` / `shuffle` / `gauss` |
| `--seeds` | no | repeat seeds |
| `--target-perp` | no | derive `tau_scale` from anchors |
| `--monitor N` | no | uniformity every N epochs |
| `--bar` | no | reference `bar.json` for gap printout |
| `--probes` / `--emd` | no | OOD probes / EMD matrix |

## Outputs

Per run under `examples/out/exploratory/{name}/{run_id}/`:
`Z.npy`, `metrics.json`, `config.json`, `scatter.png`, `model.pt`, `cover.npy`,
optional `uniformity_trace.*`, `Z_probe.npy`.

Aggregates: `summary.csv`, `atlas.png`, `ingest.json`.

## Scripts in this directory

| script | role |
|--------|------|
| `prepare_feeds.py` | Write paper feeds to `data/` |
| `calibrate.py` | Landmark / temperature calibration |
| `reference.py` | UMAP / PCA reference bar |
| `master.py` | Sweep driver |
| `bench_inference.py` | OOS latency vs UMAP / PCA |
| `make_emd.py` | Digits EMD reference geometry |
| `axes.py`, `ingest.py`, `metrics_run.py`, `nulls.py`, `splits.py`, `monitor.py`, `make_atlas.py`, `quantile_bins.py` | Harness internals |

Research demos (SAXS P(r), density, novelty) live under
[`../research/`](../research/).

## Further reading

- User guide: [`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md)
- Results: [`docs/RESULTS.md`](../../docs/RESULTS.md)
- Metrics: [`docs/METRICS.md`](../../docs/METRICS.md)
- Design notes: [`src/leanmap/README.md`](../../src/leanmap/README.md)

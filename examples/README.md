# Examples

Toy demos and the exploratory paper harness.

| script | role |
|--------|------|
| `s_curve.py` | Standalone S-curve fit / plot |
| `swiss_roll.py` | Standalone swiss-roll fit / plot |
| `digits.py` | Standalone 8×8 digits fit / plot |
| `exploratory/` | Paper battery: feeds, sweeps, metrics, EMD, conformal |
| `negative_space.py` | Frozen distance-to-support probe (post-hoc) |
| `reusability.py` | Out-of-sample reuse demo |

Paper documentation lives under [`docs/`](../docs/):

- [CONFIGURATION.md](../docs/CONFIGURATION.md) — practical settings
- [RESULTS.md](../docs/RESULTS.md) — evidence on s-curve, swiss roll, digits, iris
- [METRICS.md](../docs/METRICS.md) — how to read the battery

```bash
python examples/exploratory/prepare_feeds.py
python examples/exploratory/master.py \
  --X examples/exploratory/data/digits_X.npy \
  --y examples/exploratory/data/digits_y.npy \
  --name paper_digits --sweep canonical --only recommended \
  --holdout 0.2 --seeds 0 1 2 --null shuffle --target-perp 8
```

See [`exploratory/README.md`](exploratory/README.md) for the harness contract.

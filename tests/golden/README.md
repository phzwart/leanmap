# Golden fixtures (10M refactor bit-compat net)

Protocol:

- CPU, single process
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`
- `seed=42`
- `delta=eps` (legacy path)
- `graph.pt` / ptfile backend
- Fixtures: swiss-cone `N=2000`; digits expanded to `N=10000`

Regenerate expected digests:

```bash
LEANMAP_GOLDEN_WRITE=1 pytest tests/golden/test_bitcompat.py -q
```

`expected.json` stores SHA-256 digests of graph tensors and embedding summary
scalars — not full `.pt` blobs.

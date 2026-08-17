"""Streaming Nyström targets and N-keyed schedule defaults."""

import numpy as np
import torch

from leanmap.config import PLANEConfig, apply_scale_train_defaults
from leanmap.metrics import get_metric
from leanmap.warmstart import (
    load_shortlist,
    nystrom_targets,
    nystrom_targets_streaming,
    save_shortlist,
)

L2 = get_metric("l2").fn


def _landmarks_on_a_line(n=200, L=16):
    t = np.linspace(0, 1, n, dtype=np.float32)
    X = np.stack([t, t**2, np.zeros_like(t)], axis=1)
    idx = np.linspace(0, n - 1, L).astype(int)
    return torch.as_tensor(X), torch.as_tensor(X[idx]), torch.as_tensor(t[idx, None])


def test_streaming_nystrom_matches_dense():
    X, X_lm, Z_lm = _landmarks_on_a_line(n=200, L=16)
    dense = nystrom_targets(X, X_lm, Z_lm, L2, min_dist=0.5, seed=0)
    # Same chunk as the dense wrapper so the shared kernel is bit-identical.
    stream = nystrom_targets_streaming(
        X, X_lm, Z_lm, L2, min_dist=0.5, seed=0, chunk=8192
    )
    assert stream.shape == dense.shape
    assert torch.allclose(stream, dense, atol=1e-5, rtol=1e-5)


def test_streaming_nystrom_matches_on_memmap(tmp_path):
    X, X_lm, Z_lm = _landmarks_on_a_line(n=200, L=16)
    path = tmp_path / "X.dat"
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=tuple(X.shape))
    mm[:] = X.numpy()
    mm.flush()
    chunk = 40
    # Tensor vs memmap with the same chunk size must agree.
    from_tensor = nystrom_targets_streaming(
        X, X_lm, Z_lm, L2, min_dist=0.25, seed=1, chunk=chunk
    )
    from_memmap = nystrom_targets_streaming(
        np.memmap(path, dtype=np.float32, mode="r", shape=tuple(X.shape)),
        X_lm,
        Z_lm,
        L2,
        min_dist=0.25,
        seed=1,
        chunk=chunk,
    )
    assert torch.allclose(from_memmap, from_tensor, atol=1e-5, rtol=1e-5)
    dense = nystrom_targets(X, X_lm, Z_lm, L2, min_dist=0.25, seed=1)
    assert torch.allclose(
        nystrom_targets_streaming(
            X, X_lm, Z_lm, L2, min_dist=0.25, seed=1, chunk=8192
        ),
        dense,
        atol=1e-5,
        rtol=1e-5,
    )


def test_shortlist_roundtrip(tmp_path):
    idx = torch.randint(0, 128, (50, 8), dtype=torch.int64)
    path = save_shortlist(tmp_path / "shortlist", idx)
    got = load_shortlist(path)
    assert torch.equal(got, idx)


def test_for_scale_small_n_unchanged_snapshot():
    """N=1000 must keep the measured small-N recipe (no aggressive schedule)."""
    cfg = PLANEConfig.for_scale(1000)
    assert cfg.width == 384
    assert cfg.depth == 3
    assert cfg.n_landmarks == 128
    assert cfg.n_neighbors == 15
    assert cfg.epochs == 240
    assert cfg.calib_max == 200
    assert cfg.pca_skip is False
    assert cfg.lr == 2e-2
    assert cfg.lambda_geo == 0.15
    assert cfg.pyramid_level_weights == (1.0, 2.0, 8.0)
    assert cfg.coarse_first_frac == 0.0
    assert cfg.warm_start_steps == 0
    assert cfg.apply_large_n_schedule is False


def test_for_scale_large_n_sets_coarse_first():
    cfg = PLANEConfig.for_scale(250_000)
    assert cfg.coarse_first_frac > 0
    assert cfg.warm_start_steps > 0
    assert cfg.warm_start_layout == "auto"


def test_apply_scale_train_defaults_opt_in_only():
    # Flag off: untouched even at large N.
    cfg = PLANEConfig()
    apply_scale_train_defaults(cfg, 100_000)
    assert cfg.coarse_first_frac == 0.0
    assert cfg.warm_start_steps == 0

    # Flag on + large N: fill unset sentinels.
    cfg = PLANEConfig(apply_large_n_schedule=True)
    apply_scale_train_defaults(cfg, 100_000)
    assert cfg.coarse_first_frac > 0
    assert cfg.warm_start_steps > 0

    # Flag on + small N: leave alone.
    cfg = PLANEConfig(apply_large_n_schedule=True)
    apply_scale_train_defaults(cfg, 1000)
    assert cfg.coarse_first_frac == 0.0
    assert cfg.warm_start_steps == 0

    # Explicit choices are not overwritten.
    cfg = PLANEConfig(
        apply_large_n_schedule=True,
        coarse_first_frac=0.5,
        warm_start_steps=10,
    )
    apply_scale_train_defaults(cfg, 100_000)
    assert cfg.coarse_first_frac == 0.5
    assert cfg.warm_start_steps == 10

"""PR-7 DDP helpers and world_size=1 identity."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from leanmap import PLANEConfig, fit
from leanmap.losses.ddp_stats import (
    allreduce_density_moments,
    allreduce_mean,
    allreduce_mean_affinity,
    allreduce_path_scale,
)
from leanmap.train.ddp import (
    ClassAxis,
    PathConstraint,
    fit_ddp,
    init_distributed,
    ordinal_class_axis,
    seed_for_rank,
    sync_train_stats,
)


def _tiny_X(n: int = 40, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def _tiny_cfg(n: int, seed: int = 0) -> PLANEConfig:
    cfg = PLANEConfig.for_scale(n)
    cfg.epochs = 2
    cfg.device = "cpu"
    cfg.seed = seed
    cfg.n_landmarks = 6
    return cfg


# ---------------------------------------------------------------------------
# world_size=1 identity
# ---------------------------------------------------------------------------


def test_fit_ddp_world_size_1_matches_fit():
    X = _tiny_X()
    cfg = _tiny_cfg(X.shape[0], seed=7)
    r_fit = fit(X, config=cfg)
    Z_fit, _ = r_fit.embed(X)

    cfg2 = _tiny_cfg(X.shape[0], seed=7)
    r_ddp = fit_ddp(X, config=cfg2)
    Z_ddp, _ = r_ddp.embed(X)

    assert Z_fit.shape == Z_ddp.shape
    np.testing.assert_allclose(Z_fit, Z_ddp, rtol=1e-5, atol=1e-5)


def test_path_and_class_axis_importable_from_ddp():
    assert callable(PathConstraint)
    assert callable(ClassAxis)
    assert callable(ordinal_class_axis)


def test_seed_for_rank():
    assert seed_for_rank(42, 0) == 42
    assert seed_for_rank(42, 3) == 45


def test_init_distributed_noop_single_process(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    rank, ws = init_distributed()
    assert (rank, ws) == (0, 1)


# ---------------------------------------------------------------------------
# Formula / no-op unit tests (no process group)
# ---------------------------------------------------------------------------


def test_allreduce_helpers_noop_without_dist():
    t = torch.tensor([1.0, 3.0])
    assert torch.equal(allreduce_mean(t), t)
    assert torch.equal(allreduce_mean_affinity(t), t)
    assert torch.equal(allreduce_path_scale(torch.tensor(2.5)), torch.tensor(2.5))
    m, s = allreduce_density_moments(torch.tensor(1.0), torch.tensor(2.0), 4)
    assert float(m) == 1.0 and float(s) == 2.0


def test_affinity_mean_formula_two_fake_ranks():
    """World ā is the mean of per-rank ā when batch sizes match."""
    a0 = torch.tensor([0.2, 0.8])
    a1 = torch.tensor([0.6, 0.4])
    expected = (a0 + a1) / 2.0
    # Simulate equal-shard allreduce_mean without a live process group.
    simulated = (a0 + a1) / 2.0
    torch.testing.assert_close(simulated, expected)
    # sync_train_stats no-op path returns local ā unchanged
    out = sync_train_stats(a_bar_local=a0)
    torch.testing.assert_close(out["a_bar"], a0)


def test_density_moments_formula_two_fake_ranks():
    """Count-weighted global moments beat naive mean-of-means."""
    # Rank 0: values [0, 0] → mean 0, sq_mean 0, n=2
    # Rank 1: values [2, 4] → mean 3, sq_mean 10, n=2
    # Global: [0, 0, 2, 4] → mean 1.5, sq_mean 5.0
    mean0, sq0, n0 = torch.tensor(0.0), torch.tensor(0.0), 2
    mean1, sq1, n1 = torch.tensor(3.0), torch.tensor(10.0), 2
    n_tot = n0 + n1
    g_mean = (mean0 * n0 + mean1 * n1) / n_tot
    g_sq = (sq0 * n0 + sq1 * n1) / n_tot
    assert float(g_mean) == pytest.approx(1.5)
    assert float(g_sq) == pytest.approx(5.0)
    # Naive mean-of-means would give mean 1.5 here too; unbalanced counts:
    mean0b, sq0b, n0b = torch.tensor(0.0), torch.tensor(0.0), 1
    mean1b, sq1b, n1b = torch.tensor(3.0), torch.tensor(10.0), 3
    n_tot_b = n0b + n1b
    g_mean_b = (mean0b * n0b + mean1b * n1b) / n_tot_b
    naive = (mean0b + mean1b) / 2.0
    assert float(g_mean_b) == pytest.approx(2.25)
    assert float(naive) == pytest.approx(1.5)
    assert float(g_mean_b) != float(naive)


def test_path_scale_formula_two_fake_ranks():
    b0 = torch.tensor(1.0)
    b1 = torch.tensor(3.0)
    expected = (b0 + b1) / 2.0
    assert float(expected) == pytest.approx(2.0)
    out = sync_train_stats(path_batch_mean=b0)
    assert float(out["path_batch_mean"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Optional 2-rank gloo (LEANMAP_DDP_TEST=1)
# ---------------------------------------------------------------------------


def _ddp_worker(rank: int, world_size: int, port: str, results: dict) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # allreduce_mean / affinity
        local_a = torch.tensor([float(rank), 1.0 - float(rank)])
        a_bar = allreduce_mean_affinity(local_a)
        # density moments: rank0 n=1 mean=0 sq=0; rank1 n=3 mean=3 sq=10
        if rank == 0:
            m_loc, sq_loc, n = torch.tensor(0.0), torch.tensor(0.0), 1
        else:
            m_loc, sq_loc, n = torch.tensor(3.0), torch.tensor(10.0), 3
        g_m, g_sq = allreduce_density_moments(m_loc, sq_loc, n)
        # path scale
        p = allreduce_path_scale(torch.tensor(float(rank + 1)))
        synced = sync_train_stats(
            a_bar_local=local_a,
            dens_mean_local=m_loc,
            dens_sq_mean_local=sq_loc,
            dens_count=n,
            path_batch_mean=torch.tensor(float(rank + 1)),
        )
        if rank == 0:
            results["a_bar"] = a_bar.tolist()
            results["dens_mean"] = float(g_m)
            results["dens_sq"] = float(g_sq)
            results["path"] = float(p)
            results["sync_a"] = synced["a_bar"].tolist()
            results["sync_dens"] = float(synced["dens_mean"])
            results["sync_path"] = float(synced["path_batch_mean"])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.ddp
@pytest.mark.skipif(
    os.environ.get("LEANMAP_DDP_TEST") != "1",
    reason="set LEANMAP_DDP_TEST=1 to run 2-rank gloo integration",
)
def test_allreduce_helpers_two_rank_gloo():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import multiprocessing as mp

    port = str(29510 + (os.getpid() % 1000))
    manager = mp.Manager()
    results = manager.dict()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_ddp_worker, args=(r, 2, port, results)) for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"rank process failed exitcode={p.exitcode}"

    # ranks had ā=[0,1] and [1,0] → world mean [0.5, 0.5]
    assert results["a_bar"] == pytest.approx([0.5, 0.5])
    assert results["sync_a"] == pytest.approx([0.5, 0.5])
    # weighted moments: (0*1 + 3*3)/4 = 2.25; (0*1 + 10*3)/4 = 7.5
    assert results["dens_mean"] == pytest.approx(2.25)
    assert results["dens_sq"] == pytest.approx(7.5)
    assert results["sync_dens"] == pytest.approx(2.25)
    # path batch means 1 and 2 → 1.5
    assert results["path"] == pytest.approx(1.5)
    assert results["sync_path"] == pytest.approx(1.5)

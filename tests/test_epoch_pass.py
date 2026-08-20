"""Tests for overlapping epoch active sets and cover estimates."""

from __future__ import annotations

import numpy as np
import pytest

from leanmap.sampling.epoch_pass import (
    estimate_cover_passes,
    format_cover_passes,
    next_epoch_active_set,
)


def test_overlap_fraction_approximately_held():
    rng = np.random.default_rng(0)
    n, B, rho = 1000, 200, 0.2
    a0 = next_epoch_active_set(n, B, None, rho, rng)
    assert a0.size == B
    a1 = next_epoch_active_set(n, B, a0, rho, rng)
    assert a1.size == B
    ov = len(np.intersect1d(a0, a1)) / float(B)
    assert ov == pytest.approx(rho, abs=0.05)


def test_fresh_rows_come_from_outside_kept():
    rng = np.random.default_rng(1)
    n, B, rho = 500, 100, 0.2
    prev = next_epoch_active_set(n, B, None, rho, rng)
    cur = next_epoch_active_set(n, B, prev, rho, rng)
    kept = np.intersect1d(prev, cur)
    fresh = np.setdiff1d(cur, prev)
    assert kept.size == pytest.approx(round(rho * B), abs=1)
    assert fresh.size == B - kept.size
    # Fresh should not include kept (by construction they are outside keep)
    assert np.intersect1d(fresh, kept).size == 0


def test_cover_passes_formula():
    # N=1000, B=200, overlap=0.2 → fresh=160 → 1× cover ≈ ceil(1000/160)=7
    r = estimate_cover_passes(1000, 200, 0.2, n_visits=1)
    assert r["fresh_per_epoch"] == pytest.approx(160.0)
    assert r["epochs"] == 7
    assert r["epochs_slots"] == 5  # ceil(1000/200)
    r3 = estimate_cover_passes(1000, 200, 0.2, n_visits=3)
    assert r3["epochs"] == 19  # ceil(3000/160)
    s = format_cover_passes(r)
    assert "cover≈7" in s


def test_full_data_active_set():
    rng = np.random.default_rng(2)
    a = next_epoch_active_set(50, 50, None, 0.2, rng)
    assert a.size == 50
    b = next_epoch_active_set(50, 50, a, 0.2, rng)
    assert b.size == 50
    # With B=N, overlap estimate still defined
    r = estimate_cover_passes(50, 50, 0.2, n_visits=1)
    assert r["epochs"] == 2  # fresh=40 → ceil(50/40)=2

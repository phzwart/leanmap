"""Deterministic synthetic datasets for golden / smoke tests."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.datasets import load_digits


def make_swiss_cone(
    n_samples: int = 5000,
    *,
    noise: float = 0.05,
    hole: bool = True,
    n_turns: float = 1.5,
    width0: float = 4.0,
    width1: float = 14.0,
    hole_t: Tuple[float, float] = (0.35, 0.65),
    hole_h: Tuple[float, float] = (0.25, 0.75),
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a flaring Swiss-roll ribbon (cone) with an optional hole.

    Vendored from the historical ``examples/swiss_cone.py`` (removed in 0.2.0).
    """
    rng = np.random.default_rng(random_state)
    t_min = 1.5 * np.pi
    t_max = t_min + n_turns * 2.0 * np.pi

    pts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    batch = max(n_samples * 3, 1024)
    while sum(map(len, pts)) < n_samples:
        t = rng.uniform(t_min, t_max, size=batch)
        h = rng.uniform(0.0, 1.0, size=batch)
        if hole:
            t_n = (t - t_min) / (t_max - t_min)
            in_hole = (
                (t_n >= hole_t[0])
                & (t_n <= hole_t[1])
                & (h >= hole_h[0])
                & (h <= hole_h[1])
            )
            keep = ~in_hole
            t, h = t[keep], h[keep]
        if t.size == 0:
            continue
        frac = (t - t_min) / (t_max - t_min)
        width = width0 + (width1 - width0) * frac
        x = t * np.cos(t)
        y = (h - 0.5) * width
        z = t * np.sin(t)
        pts.append(np.column_stack([x, y, z]))
        colors.append(t)

    X = np.concatenate(pts, axis=0)[:n_samples]
    t = np.concatenate(colors, axis=0)[:n_samples]
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32), t.astype(np.float64)


def make_digits_expanded(
    n_samples: int = 10_000,
    *,
    noise: float = 1e-3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand ``sklearn`` digits to ``n_samples`` with fixed-seed resampling."""
    data = load_digits()
    X0 = np.asarray(data.data, dtype=np.float32)
    y0 = np.asarray(data.target, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, X0.shape[0], size=n_samples)
    X = X0[idx].copy()
    y = y0[idx].copy()
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape).astype(np.float32)
    return X, y


def make_swiss_cone_2k(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    return make_swiss_cone(2000, noise=0.05, random_state=seed)


def make_digits_10k(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    return make_digits_expanded(10_000, noise=1e-3, random_state=seed)

"""Null datasets for calibrating embedding metrics.

Every embedding metric has a chance level, and that level depends on the
*configuration* as much as on the data: a coarse-heavy graph pyramid imposes a
smooth global layout whether or not the data has one, which inflates geodesic
and density scores on pure noise. So a null must be refit with the same
hyperparameters as the run it calibrates, and re-derived whenever the config
changes. Comparing against a stale null from a different config is how you
manufacture a result.

``shuffle`` is the null of choice: permuting each feature independently destroys
all dependence between columns while preserving every marginal exactly, so
anything the embedding still shows is produced by the method rather than by
structure in the data.
"""

from __future__ import annotations

import numpy as np

KINDS = ("none", "shuffle", "gauss")


def shuffle_null(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute each feature independently: marginals kept, joint destroyed."""
    rng = np.random.default_rng(seed)
    out = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return np.ascontiguousarray(out, dtype=np.float32)


def gaussian_null(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Matched mean/covariance Gaussian: keeps second moments, nothing else."""
    rng = np.random.default_rng(seed)
    cov = np.cov(np.asarray(X, dtype=np.float64), rowvar=False)
    cov = np.atleast_2d(cov)
    out = rng.multivariate_normal(np.asarray(X, dtype=np.float64).mean(0), cov, size=len(X))
    return np.ascontiguousarray(out, dtype=np.float32)


def make_null(X: np.ndarray, kind: str = "shuffle", seed: int = 0) -> np.ndarray:
    """Dispatch to a null of type ``kind`` (``none`` returns ``X`` unchanged)."""
    if kind not in KINDS:
        raise ValueError(f"unknown null kind {kind!r}; choose from {KINDS}")
    if kind == "none":
        return np.ascontiguousarray(X, dtype=np.float32)
    if kind == "shuffle":
        return shuffle_null(X, seed=seed)
    return gaussian_null(X, seed=seed)


def describe(kind: str) -> str:
    return {
        "none": "real data",
        "shuffle": "features permuted independently (marginals kept, joint destroyed)",
        "gauss": "matched-covariance Gaussian (second moments only)",
    }[kind]

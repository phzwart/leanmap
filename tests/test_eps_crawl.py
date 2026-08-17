"""Random-subsample ε crawl / browser."""
from __future__ import annotations

import numpy as np
import torch

from leanmap.build.resolution import crawl_epsilon, format_epsilon_crawl
from leanmap.distance import EuclideanDistance


def test_crawl_epsilon_reports_monotonic_compression():
    rng = np.random.default_rng(0)
    # Tight clusters → larger ε merges more.
    centers = rng.normal(size=(20, 3))
    X = np.vstack([c + 0.01 * rng.normal(size=(30, 3)) for c in centers]).astype(
        np.float32
    )
    report = crawl_epsilon(
        torch.as_tensor(X),
        EuclideanDistance(),
        n_sample=200,
        epsilons=[0.001, 0.05, 0.5, 5.0],
        seed=0,
    )
    rows = report["rows"]
    assert len(rows) == 4
    # R should not increase as ε grows.
    for a, b in zip(rows, rows[1:]):
        assert b["R_sub"] <= a["R_sub"]
    assert report["recommend"] is not None
    text = format_epsilon_crawl(report)
    assert "recommend epsilon=" in text

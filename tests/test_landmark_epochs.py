"""Landmark-basin epoch unit and edge reweighting."""
from __future__ import annotations

import numpy as np

from leanmap.config import PLANEConfig
from leanmap.sampling.edges import basin_balanced_edge_weights, landmark_epoch_steps
from leanmap.train.fit import coarse_to_fine_plan


def test_landmark_epoch_steps_scales_with_L_not_E():
    assert landmark_epoch_steps(407, 4096, samples_per_landmark=128) == 13
    assert landmark_epoch_steps(512, 4096, samples_per_landmark=128) == 16
    # Independent of any edge count — large E would be ~140 steps at R~56k.
    assert landmark_epoch_steps(407, 4096, samples_per_landmark=128) < 50


def test_basin_balanced_boosts_light_basins():
    # Two landmarks: basin 0 has heavy edge mass, basin 1 light.
    edges = np.array([[0, 1], [2, 3], [0, 2]], dtype=np.int64)
    # cells 0,1 → lm 0; cells 2,3 → lm 1
    cell_lm = np.array([0, 0, 1, 1], dtype=np.int64)
    w = np.array([10.0, 1.0, 1.0], dtype=np.float64)
    out = basin_balanced_edge_weights(edges, w, cell_lm, mix=1.0)
    # Edge within light basin (2,3) should be upweighted relative to (0,1).
    assert out[1] / w[1] > out[0] / w[0]


def test_coarse_to_fine_plan_landmark_unit_ignores_edge_count():
    plan_e = coarse_to_fine_plan(
        2, [100_000, 10_000], 4096, [1.0, 1.0], 0.0, epoch_unit="edges"
    )
    plan_l = coarse_to_fine_plan(
        2,
        [100_000, 10_000],
        4096,
        [1.0, 1.0],
        0.0,
        epoch_unit="landmarks",
        n_landmarks=407,
        landmark_epoch_samples=128,
    )
    assert plan_e[0][1] > 20
    assert plan_l[0][1] == landmark_epoch_steps(407, 4096, samples_per_landmark=128)


def test_for_scale_large_n_opts_into_landmark_epochs():
    cfg = PLANEConfig.for_scale(300_000)
    assert cfg.epoch_unit == "landmarks"
    assert cfg.landmark_sample_mix > 0
    cfg_s = PLANEConfig.for_scale(3_000)
    assert cfg_s.epoch_unit == "edges"

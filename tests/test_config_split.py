"""Split config composition (PR-11)."""
from __future__ import annotations

from leanmap import (
    BuildConfig,
    PolicyConfig,
    TrainConfig,
    compose_plane_config,
)


def test_compose_plane_config_merges_fields():
    cfg = compose_plane_config(
        build=BuildConfig(n_landmarks=64, pyramid_scales=1),
        train=TrainConfig(epochs=7, lr=1e-2),
        policy=PolicyConfig(exemplar_policy="sufficient_v1"),
    )
    assert cfg.n_landmarks == 64
    assert cfg.pyramid_scales == 1
    assert cfg.epochs == 7
    assert cfg.lr == 1e-2
    assert cfg.exemplar_policy == "sufficient_v1"

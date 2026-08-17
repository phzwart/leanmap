"""Shim — implementation in :mod:`leanmap.model.warmstart`."""
from __future__ import annotations

from leanmap.model.warmstart import (
    LAYOUTS,
    SPACING_SAMPLE,
    load_shortlist,
    nystrom_targets,
    nystrom_targets_streaming,
    pretrain_to_targets,
    rank_inits,
    save_shortlist,
    spectral_layout,
    warm_start,
)

__all__ = [
    "LAYOUTS",
    "SPACING_SAMPLE",
    "load_shortlist",
    "nystrom_targets",
    "nystrom_targets_streaming",
    "pretrain_to_targets",
    "rank_inits",
    "save_shortlist",
    "spectral_layout",
    "warm_start",
]

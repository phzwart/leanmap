"""Shim — canonical CLI is :mod:`leanmap.cli`."""
from __future__ import annotations

from leanmap.cli import main, main_graph_build, main_train

__all__ = ["main", "main_graph_build", "main_train"]

"""One definition of the train/holdout split, shared by every runner.

Comparing two embedders only means something if they are fit on the same rows
and scored on the same rows. That is easy to get wrong when each runner derives
its own split: the derivations agree until one of them changes, and then the
comparison silently starts measuring two different experiments.

So the derivation lives here, and every run also *records* the indices it used
next to its embedding. A scoring script should read the recorded split rather
than re-deriving it -- re-derivation only works while every copy of the rule
still agrees.

The rule itself is unchanged from the original inline version in ``master.py``,
so runs produced before this module existed remain comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

__all__ = ["split_indices", "save_split", "load_split"]


def split_indices(n: int, holdout: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(train_idx, hold_idx)`` for ``n`` rows.

    ``holdout <= 0`` returns every index in both, matching the existing
    convention where "no holdout" means in-sample scoring over everything.
    """
    n = int(n)
    if holdout and holdout > 0:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        n_hold = max(1, int(float(holdout) * n))
        return perm[n_hold:], perm[:n_hold]
    everything = np.arange(n)
    return everything, everything


def save_split(
    out_dir: PathLike,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    holdout: float,
    seed: int,
) -> Path:
    """Persist the split next to a run's ``Z.npy``."""
    path = Path(out_dir) / "split.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        hold_idx=np.asarray(hold_idx, dtype=np.int64),
        holdout=np.float64(holdout),
        seed=np.int64(seed),
    )
    return path


def load_split(
    run_dir: PathLike,
    *,
    n: Optional[int] = None,
    holdout: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a recorded split, falling back to re-deriving it for older runs.

    The fallback exists only so that runs made before ``split.npz`` was written
    can still be scored; it needs ``n``, ``holdout`` and ``seed`` to match what
    the run actually used.
    """
    path = Path(run_dir) / "split.npz"
    if path.is_file():
        d = np.load(path)
        return d["train_idx"], d["hold_idx"]
    if n is None or holdout is None or seed is None:
        raise FileNotFoundError(
            f"no split.npz in {run_dir} and not enough information to re-derive it"
        )
    return split_indices(n, holdout, seed)

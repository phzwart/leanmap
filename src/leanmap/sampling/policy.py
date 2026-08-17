"""ExemplarPolicy \(p_t\) — within-epoch sampling measure (PR-9).

Universe = all rows available for embed / eval / refresh. Within an epoch the
stream is drawn from an explicit measure \(p_t\) over exemplar families
(edges, paths, class/ordinal). ``uniform`` reproduces prior EdgeSampler
behaviour (alias ∝ edge mass). ``sufficient_v1`` applies visit / violation
tilts and coverage floors. Importance weights \(w/p_t\) are ratio-capped when
``reweight=True`` (default); ``reweight=False`` is the unweighted stream.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .alias import _alias_draw, build_edge_alias

ArrayLike = Union[np.ndarray, Any]

MODES: Tuple[str, ...] = ("uniform", "sufficient_v1")
# Three exemplar families. ``negatives`` is accepted as an alias of
# ``class_ordinal`` (uniform-over-cells base) for callers that prefer the
# edges / negatives / paths naming.
FAMILIES: Tuple[str, ...] = ("edges", "paths", "class_ordinal")
RATIO_CAP_DEFAULT: float = 10.0
_EPS: float = 1e-12


def _as_f64(x: ArrayLike) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float64).reshape(-1), _EPS)


def _normalize(mass: np.ndarray) -> np.ndarray:
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    m = np.maximum(m, 0.0)
    s = float(m.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(m, dtype=np.float64) / max(m.size, 1)
    return m / s


def _cap_ratio(ratio: np.ndarray, cap: float) -> np.ndarray:
    """Clip importance ratios into ``[1/cap, cap]`` (cap >= 1)."""
    c = float(max(cap, 1.0))
    return np.clip(np.asarray(ratio, dtype=np.float64), 1.0 / c, c)


def _family_key(family: str) -> str:
    f = str(family).lower().strip()
    if f in ("negatives", "negative", "class", "ordinal", "class_ordinal"):
        return "class_ordinal"
    if f in ("path", "paths"):
        return "paths"
    if f in ("edge", "edges"):
        return "edges"
    raise ValueError(
        f"unknown exemplar family {family!r}; expected one of {FAMILIES} "
        f"(or 'negatives')"
    )


class ExemplarPolicy:
    """Sampling measure \(p_t\) over edge / path / class-ordinal exemplars.

    Parameters
    ----------
    mode
        ``"uniform"`` — \(p_t\) follows base family mass (edge mass for edges).
        ``"sufficient_v1"`` — inverse-visit + optional violation tilts with
        cell / landmark coverage floors.
    reweight
        If True (default), draws carry importance weights ``w/p_t`` ratio-capped
        at ``ratio_cap``. If False, weights are ones (unweighted stream).
    """

    def __init__(
        self,
        *,
        mode: str = "uniform",
        edge_mass: ArrayLike,
        edges: Optional[ArrayLike] = None,
        n_cells: int = 0,
        path_mass: Optional[ArrayLike] = None,
        class_mass: Optional[ArrayLike] = None,
        landmark_of_cell: Optional[ArrayLike] = None,
        reweight: bool = True,
        ratio_cap: float = RATIO_CAP_DEFAULT,
        cell_floor: float = 1e-6,
        landmark_mass_floor: float = 1e-6,
        seed: int = 0,
    ):
        mode_s = str(mode).lower().strip()
        if mode_s not in MODES:
            raise ValueError(f"exemplar mode {mode!r}; expected one of {MODES}")
        self.mode = mode_s
        self.reweight = bool(reweight)
        self.ratio_cap = float(ratio_cap)
        self.cell_floor = float(cell_floor)
        self.landmark_mass_floor = float(landmark_mass_floor)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self._base: Dict[str, np.ndarray] = {
            "edges": _as_f64(edge_mass),
        }
        self.n_edges = int(self._base["edges"].shape[0])
        if edges is not None:
            self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            if self.edges.shape[0] != self.n_edges:
                raise ValueError(
                    f"edges length {self.edges.shape[0]} != edge_mass {self.n_edges}"
                )
        else:
            self.edges = np.zeros((self.n_edges, 2), dtype=np.int64)

        if n_cells <= 0 and self.n_edges > 0:
            n_cells = int(self.edges.max()) + 1 if self.edges.size else 1
        self.n_cells = int(max(n_cells, 1))

        if path_mass is None:
            self._base["paths"] = np.ones(1, dtype=np.float64)
            self._path_active = False
        else:
            self._base["paths"] = _as_f64(path_mass)
            self._path_active = True

        if class_mass is None:
            # Uniform over cells — negatives / class-ordinal base.
            self._base["class_ordinal"] = np.ones(self.n_cells, dtype=np.float64)
        else:
            cm = _as_f64(class_mass)
            if cm.shape[0] != self.n_cells:
                raise ValueError(
                    f"class_mass length {cm.shape[0]} != n_cells {self.n_cells}"
                )
            self._base["class_ordinal"] = cm

        if landmark_of_cell is None:
            self.landmark_of_cell = None
            self.n_landmarks = 0
        else:
            loc = np.asarray(landmark_of_cell, dtype=np.int64).reshape(-1)
            if loc.shape[0] != self.n_cells:
                raise ValueError(
                    f"landmark_of_cell length {loc.shape[0]} != n_cells {self.n_cells}"
                )
            self.landmark_of_cell = loc
            self.n_landmarks = int(loc.max()) + 1 if loc.size else 0

        # Visit / violation state (updated via refresh).
        self.edge_visits = np.ones(self.n_edges, dtype=np.float64)
        self.cell_visits = np.ones(self.n_cells, dtype=np.float64)
        self.path_violation: Optional[np.ndarray] = None
        self.class_violation: Optional[np.ndarray] = None
        self._last_stats: Dict[str, Any] = {}

        self._p: Dict[str, np.ndarray] = {}
        self._alias: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._rebuild()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        mode: str = "uniform",
        *,
        reweight: bool = True,
        ratio_cap: float = RATIO_CAP_DEFAULT,
        path_mass: Optional[ArrayLike] = None,
        class_mass: Optional[ArrayLike] = None,
        landmark_of_cell: Optional[ArrayLike] = None,
        constraints: Any = None,
        stats: Optional[Mapping[str, Any]] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> "ExemplarPolicy":
        """Build a policy from a frozen :class:`~leanmap.build.pipeline.Graph`.

        ``constraints`` / ``stats`` are optional hooks for path tables and an
        initial refresh. Uniform mode ignores tilts and matches EdgeSampler
        edge-mass alias sampling.
        """
        del constraints  # reserved; path_mass may be passed explicitly
        weights = graph.weights
        if hasattr(weights, "detach"):
            weights = weights.detach().cpu().numpy()
        edges = graph.edges
        if hasattr(edges, "detach"):
            edges = edges.detach().cpu().numpy()
        n_cells = int(graph.reps.rep_idx.shape[0])
        pol = cls(
            mode=mode,
            edge_mass=weights,
            edges=edges,
            n_cells=n_cells,
            path_mass=path_mass,
            class_mass=class_mass,
            landmark_of_cell=landmark_of_cell,
            reweight=reweight,
            ratio_cap=ratio_cap,
            seed=seed,
            **kwargs,
        )
        if stats is not None:
            pol.refresh(stats)
        return pol

    # ------------------------------------------------------------------
    # Measure rebuild / refresh
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        for fam in FAMILIES:
            base = self._base[fam]
            if self.mode == "uniform":
                mass = base.copy()
            else:
                mass = self._tilt_family(fam, base)
            if fam == "edges":
                mass = self._apply_coverage_floors(mass)
            self._p[fam] = _normalize(mass)
            self._alias[fam] = build_edge_alias(self._p[fam])

    def _tilt_family(self, fam: str, base: np.ndarray) -> np.ndarray:
        """sufficient_v1 tilts: edge mass, inverse visits, optional violations."""
        mass = base.astype(np.float64, copy=True)
        if fam == "edges":
            mass = mass / np.maximum(self.edge_visits, 1.0)
            # Soft cell-visit tilt via endpoints.
            if self.edges.shape[0] == mass.shape[0] and self.edges.size:
                ci = self.edges[:, 0]
                cj = self.edges[:, 1]
                cell_tilt = 1.0 / np.maximum(
                    0.5 * (self.cell_visits[ci] + self.cell_visits[cj]), 1.0
                )
                mass = mass * cell_tilt
            if self.path_violation is not None and self.path_violation.shape[0] == mass.shape[0]:
                mass = mass * (1.0 + np.maximum(self.path_violation, 0.0))
        elif fam == "paths":
            if self.path_violation is not None and self.path_violation.shape[0] == mass.shape[0]:
                mass = mass * (1.0 + np.maximum(self.path_violation, 0.0))
        elif fam == "class_ordinal":
            mass = mass / np.maximum(self.cell_visits[: mass.shape[0]], 1.0)
            if (
                self.class_violation is not None
                and self.class_violation.shape[0] == mass.shape[0]
            ):
                mass = mass * (1.0 + np.maximum(self.class_violation, 0.0))
        return np.maximum(mass, _EPS)

    def _apply_coverage_floors(self, edge_mass: np.ndarray) -> np.ndarray:
        """Keep ≥1 exemplar worth of mass per occupied cell; landmark floors."""
        mass = np.asarray(edge_mass, dtype=np.float64).copy()
        if self.edges.shape[0] != mass.shape[0] or self.edges.size == 0:
            return mass

        # Per-cell incident mass floor.
        cell_mass = np.zeros(self.n_cells, dtype=np.float64)
        for e, (a, b) in enumerate(self.edges):
            a_i, b_i = int(a), int(b)
            if 0 <= a_i < self.n_cells:
                cell_mass[a_i] += mass[e]
            if 0 <= b_i < self.n_cells and b_i != a_i:
                cell_mass[b_i] += mass[e]

        floor = float(self.cell_floor)
        occupied = cell_mass > 0
        for c in np.where(occupied & (cell_mass < floor))[0]:
            # Boost the heaviest incident edge enough to clear the floor.
            inc = np.where((self.edges[:, 0] == c) | (self.edges[:, 1] == c))[0]
            if inc.size == 0:
                continue
            j = int(inc[np.argmax(mass[inc])])
            need = floor - float(cell_mass[c])
            mass[j] += max(need, 0.0)

        # Landmark mass floors (optional).
        if self.landmark_of_cell is not None and self.n_landmarks > 0:
            lm_floor = float(self.landmark_mass_floor)
            cell_mass2 = np.zeros(self.n_cells, dtype=np.float64)
            for e, (a, b) in enumerate(self.edges):
                cell_mass2[int(a)] += mass[e]
                cell_mass2[int(b)] += mass[e]
            for ell in range(self.n_landmarks):
                cells = np.where(self.landmark_of_cell == ell)[0]
                if cells.size == 0:
                    continue
                lm_mass = float(cell_mass2[cells].sum())
                if lm_mass >= lm_floor:
                    continue
                # Spread deficit across edges touching those cells.
                mask = np.isin(self.edges[:, 0], cells) | np.isin(self.edges[:, 1], cells)
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue
                mass[idx] += (lm_floor - lm_mass) / float(idx.size)

        return np.maximum(mass, _EPS)

    def refresh(self, stats: Optional[Mapping[str, Any]] = None) -> None:
        """Update visit / violation tilts from probe or epoch stats, then rebuild.

        Recognised keys (all optional):

        - ``edge_visits``, ``cell_visits`` — non-negative counts
        - ``path_violation``, ``class_violation`` — non-negative tilt hooks
        - ``path_mass``, ``class_mass`` — replace base family masses
        """
        stats = dict(stats or {})
        self._last_stats = stats

        if "edge_visits" in stats:
            v = np.asarray(stats["edge_visits"], dtype=np.float64).reshape(-1)
            if v.shape[0] == self.n_edges:
                self.edge_visits = np.maximum(v, 1.0)
        if "cell_visits" in stats:
            v = np.asarray(stats["cell_visits"], dtype=np.float64).reshape(-1)
            if v.shape[0] == self.n_cells:
                self.cell_visits = np.maximum(v, 1.0)
        if "path_violation" in stats and stats["path_violation"] is not None:
            self.path_violation = np.asarray(
                stats["path_violation"], dtype=np.float64
            ).reshape(-1)
        if "class_violation" in stats and stats["class_violation"] is not None:
            self.class_violation = np.asarray(
                stats["class_violation"], dtype=np.float64
            ).reshape(-1)
        if "path_mass" in stats and stats["path_mass"] is not None:
            self._base["paths"] = _as_f64(stats["path_mass"])
            self._path_active = True
        if "class_mass" in stats and stats["class_mass"] is not None:
            cm = _as_f64(stats["class_mass"])
            if cm.shape[0] == self.n_cells:
                self._base["class_ordinal"] = cm

        self._rebuild()

    # ------------------------------------------------------------------
    # Queries / sampling
    # ------------------------------------------------------------------

    def sampling_mass(self, family: str = "edges") -> np.ndarray:
        """Unnormalized \(p_t\) mass used for alias construction (normalized copy)."""
        return self._p[_family_key(family)].copy()

    def base_mass(self, family: str = "edges") -> np.ndarray:
        return self._base[_family_key(family)].copy()

    def importance_weights(
        self,
        indices: ArrayLike,
        family: str = "edges",
        *,
        base_w: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return ratio-capped \(w/p_t\) (or ones if ``reweight=False``)."""
        fam = _family_key(family)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = int(idx.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if not self.reweight:
            return np.ones(n, dtype=np.float64)

        p = self._p[fam]
        w = self._base[fam] if base_w is None else _as_f64(base_w)
        # Probability ratio (w/W) / (p/P); equals 1 when p ∝ w.
        w_sum = float(np.sum(w))
        p_sum = float(np.sum(p))
        wi = w[idx]
        pi = p[idx]
        ratio = (wi / max(w_sum, _EPS)) / np.maximum(pi / max(p_sum, _EPS), _EPS)
        return _cap_ratio(ratio, self.ratio_cap)

    def sample_indices(
        self,
        n: int,
        family: str = "edges",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` indices from family \(p_t\); return ``(idx, importance_w)``."""
        fam = _family_key(family)
        gen = self.rng if rng is None else rng
        n_draw = int(n)
        if n_draw <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        prob, alias = self._alias[fam]
        idx = _alias_draw(n_draw, prob, alias, gen)
        return idx, self.importance_weights(idx, fam)

    def sample_edges(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw edge exemplars: ``(edge_idx, endpoints (n,2), importance_w)``."""
        idx, iw = self.sample_indices(n, family="edges", rng=rng)
        endpoints = self.edges[idx] if self.edges.shape[0] else np.zeros((0, 2), dtype=np.int64)
        return idx, endpoints, iw

    def set_violation_hooks(
        self,
        *,
        path_violation: Optional[ArrayLike] = None,
        class_violation: Optional[ArrayLike] = None,
        rebuild: bool = True,
    ) -> None:
        """Optional path / class violation mass hooks (sufficient_v1 tilts)."""
        if path_violation is not None:
            self.path_violation = np.asarray(path_violation, dtype=np.float64).reshape(-1)
        if class_violation is not None:
            self.class_violation = np.asarray(
                class_violation, dtype=np.float64
            ).reshape(-1)
        if rebuild:
            self._rebuild()


__all__ = [
    "ExemplarPolicy",
    "FAMILIES",
    "MODES",
    "RATIO_CAP_DEFAULT",
    "_cap_ratio",
]

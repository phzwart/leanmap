#!/usr/bin/env python
"""Is the leftover density structure fabricated, or just nonlinear?

``pr_clumpiness.py`` regresses embedded log-density on ambient log-density with a
straight line and calls whatever is left "unexplained". That is too strict. A map
that reduces dimension has no reason to relate the two densities linearly, and
when the true relation is curved, a linear fit dumps the curvature into the
residual. Because the ambient density field is strongly spatially organised, that
misfit is *also* spatially organised -- so an organised residual is exactly what a
faithful-but-nonlinear map produces, and it cannot be used as evidence of
fabrication on its own.

This lets the fit be curved and asks whether the residual survives:

``linear``    the straight line the clumpiness audit uses
``monotone``  isotonic regression: any increasing relation, no shape assumed
``spline``    natural cubic spline, a smooth curved relation

All three are scored **out of fold** under 5-fold cross-validation, so a flexible
fit cannot win by memorising points; ``monotone`` in particular would otherwise
chase noise. Read it this way:

* if a curved fit lifts R-squared and collapses the residual's Moran's I, the
  leftover was the straight line's fault and the clumping is real
* if the residual stays organised no matter how flexible the fit, the map is
  putting structure into the layout that the ambient density does not account for

Usage::

    python examples/exploratory/pr_licensed.py
    python examples/exploratory/pr_licensed.py --runs runs/sasbdb_pr_density3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pr_clumpiness import density, intrinsic_dim, knn_dist, morans_i  # noqa: E402

DEFAULT_RUNS = (
    "runs/sasbdb_pr_l1_frozen",
    "runs/sasbdb_pr_density",
    "runs/sasbdb_pr_density3",
)


def local_dim(d: np.ndarray) -> np.ndarray:
    """Per-point Levina-Bickel intrinsic dimension from sorted kNN distances.

    The reason this belongs in the regression: a neighbourhood that is locally
    higher-dimensional cannot be laid into ``d_out`` dimensions without being
    squeezed, and how hard it is squeezed varies smoothly over the manifold. That
    is spatially organised area distortion which ambient *density* cannot predict,
    yet it is forced by the graph rather than invented by the optimiser.
    """
    r = np.maximum(d, 1e-12)
    return 1.0 / np.maximum(np.log(r[:, -1:] / r[:, :-1]).mean(axis=1), 1e-12)


def _fit_predict(kind: str, xa: np.ndarray, ya: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """Fit on ``(xa, ya)``, predict at ``xb``. One fold of the CV.

    ``xa``/``xb`` are ``(n, p)``; ``p > 1`` adds regressors beyond ambient density.
    """
    if kind == "monotone":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip").fit(xa[:, 0], ya).predict(xb[:, 0])

    from sklearn.linear_model import LinearRegression

    if kind == "linear":
        return LinearRegression().fit(xa, ya).predict(xb)
    if kind == "spline":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        m = make_pipeline(
            SplineTransformer(n_knots=8, degree=3, extrapolation="linear"),
            LinearRegression(),
        )
        return m.fit(xa, ya).predict(xb)
    raise KeyError(kind)


def cv_residual(kind: str, x: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5):
    """Out-of-fold predictions, so flexibility cannot buy R-squared."""
    x = x.reshape(len(y), -1)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    pred = np.empty_like(y)
    for f in range(folds):
        te = fold == f
        pred[te] = _fit_predict(kind, x[~te], y[~te], x[te])
    resid = y - pred
    r2 = float(1.0 - resid.var() / y.var())
    return r2, resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.neighbors import NearestNeighbors

    print(f"{'run':<24}{'fit':<14}{'R^2':>8}{'resid Moran I':>16}{'resid sd':>10}")
    print("-" * 72)
    for name in args.runs:
        run = Path(name) if Path(name).is_absolute() else _ROOT / name
        X = np.load(run / "X.npy").astype(np.float64)
        Z = np.load(run / "Z.npy").astype(np.float64)
        d_amb = knn_dist(X, args.k, args.metric)
        dim = intrinsic_dim(knn_dist(X, 10, args.metric))
        la = np.log10(density(d_amb, dim))
        lz = np.log10(density(knn_dist(Z, args.k, "euclidean"), float(Z.shape[1])))
        ldim = local_dim(d_amb)
        nb_z = NearestNeighbors(n_neighbors=args.k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
        # Reference: the organisation of the ambient field itself, which is the
        # most an inherited misfit could possibly show.
        mi_amb = morans_i(la, nb_z)
        both = np.column_stack([la, np.log10(np.clip(ldim, 1e-6, None))])
        for kind, feats, tag in (
            ("linear", la, "linear"),
            ("monotone", la, "monotone"),
            ("spline", la, "spline"),
            ("spline", both, "spline+locdim"),
        ):
            r2, resid = cv_residual(kind, feats, lz, seed=args.seed)
            print(
                f"{run.name:<24}{tag:<14}{r2:>8.3f}{morans_i(resid, nb_z):>16.3f}"
                f"{resid.std():>10.3f}"
            )
        print(
            f"{'':<24}{'(ambient log-density over the embedding graph: '}"
            f"Moran I = {mi_amb:+.3f})"
        )
        # At what scale is the leftover organised? A mixture-of-experts layout
        # fabricates structure at the size of a landmark cell, which holds
        # N / n_landmarks points; broad organisation that outlives that scale is
        # not the tessellation talking. Compared against the ambient field, whose
        # decay profile is what genuinely broad structure looks like here.
        _, resid = cv_residual("spline", both, lz, seed=args.seed)
        print(f"{'':<24}scale of organisation (Moran's I at increasing k):")
        print(f"{'':<26}{'k':>6}{'residual':>11}{'ambient':>10}")
        for kk in (5, 15, 32, 100, 300, 900):
            if kk >= len(Z) - 1:
                continue
            nb = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)[1][:, 1:]
            mark = "   <- landmark cell" if kk == 32 else ""
            print(
                f"{'':<26}{kk:>6}{morans_i(resid, nb):>11.3f}"
                f"{morans_i(la, nb):>10.3f}{mark}"
            )
        print("-" * 72)


if __name__ == "__main__":
    main()

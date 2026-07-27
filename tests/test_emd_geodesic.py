"""EMD as a reference geometry, and where the pixel-L2 shortcut breaks.

The claims under test, in the order they have to hold for EMD to be usable as an
independent arbiter between embeddings:

C1  EMD is correct: translating a blob by ``s`` costs exactly ``s``.
C2  At short range pixel L2 tracks EMD, which is what licenses building a kNN
    graph out of L2 in the first place.
C3  At long range L2 saturates -- non-overlapping images are all equidistant --
    while a geodesic chained through the graph keeps tracking EMD.

C3 is asserted *per band*. An overall correlation is dominated by the many near
pairs where both metrics are perfect, so it hides the divergence entirely; on
the fixture below the overall margin is ~0.00 while the far-band margin is
+0.21. Real 8x8 digits behave differently again, which
``test_digits_emd_is_not_a_relabelled_l2`` pins down.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from leanmap.emd import (
    geodesic_from_matrix,
    grid_cost_matrix,
    image_emd,
    pairwise_emd,
)
from leanmap.probes import (
    PROBE_PATTERNS,
    control_probes,
    render_pattern,
    structured_probes,
)

GRID = 12
FRAMES = 56
SIGMA = 1.0


def _blob(cx: float, cy: float, n: int = GRID, s: float = SIGMA) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))


@pytest.fixture(scope="module")
def blob_path():
    """A blob swept along a curved 1-D path: a manifold with known geometry.

    True EMD between two frames is the straight-line distance between their
    centres, so the reference has an analytic ground truth rather than another
    numerical estimate.
    """
    t = np.linspace(0.0, 1.0, FRAMES)
    cx = 0.18 * GRID + 0.64 * GRID * t
    cy = 0.5 * GRID + 2.2 * np.sin(2 * np.pi * t * 0.8)
    imgs = np.stack([_blob(a, b).ravel() for a, b in zip(cx, cy)])
    centres = np.stack([cx, cy], axis=1)
    D_emd = pairwise_emd(imgs, (GRID, GRID), progress=False)
    D_l2 = squareform(pdist(imgs))
    D_geo = geodesic_from_matrix(D_l2, n_neighbors=6)
    iu = np.triu_indices(FRAMES, k=1)
    return {
        "emd": D_emd[iu],
        "l2": D_l2[iu],
        "geo": D_geo[iu],
        "true": squareform(pdist(centres))[iu],
        "bands": np.quantile(D_emd[iu], [0.0, 1 / 3, 2 / 3, 1.0]),
    }


def _band(d, name):
    e, edges = d["emd"], d["bands"]
    b = {"local": 0, "mid": 1, "global": 2}[name]
    return (e >= edges[b]) & (e <= edges[b + 1])


# --- C1: the reference is correct -----------------------------------------


@pytest.mark.parametrize("shift", [1.0, 2.0, 4.0])
def test_image_emd_matches_translation(shift):
    # W1 between a measure and its translate is exactly the translation length.
    C = grid_cost_matrix((16, 16))
    got = image_emd(_blob(5.0, 8.0, n=16, s=1.4), _blob(5.0 + shift, 8.0, n=16, s=1.4), C)
    assert got == pytest.approx(shift, rel=0.03)


def test_emd_recovers_analytic_path_geometry(blob_path):
    rho = spearmanr(blob_path["emd"], blob_path["true"]).correlation
    assert rho > 0.99


def test_grid_cost_matrix_is_a_metric_on_pixels():
    C = grid_cost_matrix((5, 5))
    assert C.shape == (25, 25)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 0.0)
    # Corner to opposite corner of a 5x5 grid.
    assert C[0, -1] == pytest.approx(np.hypot(4.0, 4.0))


# --- C2: L2 is fine locally ------------------------------------------------


def test_local_l2_tracks_emd(blob_path):
    m = _band(blob_path, "local")
    rho = spearmanr(blob_path["emd"][m], blob_path["l2"][m]).correlation
    assert rho > 0.95


# --- C3: L2 fails globally, the geodesic does not --------------------------


def test_pixel_l2_saturates_at_range(blob_path):
    far = blob_path["l2"][_band(blob_path, "global")]
    # Once the blobs stop overlapping every pair is essentially equidistant.
    assert np.percentile(far, 99) / np.percentile(far, 50) < 1.05


def test_geodesic_beats_l2_in_the_far_band(blob_path):
    m = _band(blob_path, "global")
    rho_l2 = spearmanr(blob_path["emd"][m], blob_path["l2"][m]).correlation
    rho_geo = spearmanr(blob_path["emd"][m], blob_path["geo"][m]).correlation
    assert rho_geo > 0.70
    assert rho_geo - rho_l2 > 0.10


def test_shuffled_pairing_is_uncorrelated(blob_path):
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(blob_path["geo"]))
    rho = spearmanr(blob_path["emd"], blob_path["geo"][perm]).correlation
    assert abs(rho) < 0.2


def test_geodesic_from_matrix_is_a_metric():
    D = squareform(pdist(np.random.default_rng(0).normal(size=(40, 3))))
    G = geodesic_from_matrix(D, n_neighbors=5)
    assert np.allclose(G, G.T, atol=1e-9)
    assert np.allclose(np.diag(G), 0.0)
    assert (G <= G[:, :1] + G[:1, :] + 1e-9).all()  # triangle inequality via node 0
    # Detours through kNN edges can only be longer than going straight, which is
    # exactly why a geodesic can express structure that the direct distance cannot.
    assert (G >= D - 1e-9).all()


# --- the same questions on real digits ------------------------------------


def test_digits_emd_is_not_a_relabelled_l2():
    """The gate for the whole comparison.

    If pixel L2 already ordered digit pairs the way EMD does, EMD could not
    arbitrate between two embeddings that were both fit from L2. Measured on the
    full 1797-image matrix: 0.76 overall, 0.69 local, 0.37 global.

    Note this fixture does *not* assert that a geodesic recovers EMD on digits.
    It does not: on the full matrix the L2-graph geodesic scores 0.59 against
    EMD versus raw L2's 0.76. Digits are too sparsely sampled in 64-D for
    chaining to help, unlike the dense blob path above.
    """
    sklearn = pytest.importorskip("sklearn")
    X = sklearn.datasets.load_digits().data.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=150, replace=False)
    Xs = X[idx]

    D_emd = pairwise_emd(Xs, (8, 8), progress=False)
    D_l2 = squareform(pdist(Xs))
    iu = np.triu_indices(len(Xs), k=1)
    e, d2 = D_emd[iu], D_l2[iu]

    overall = spearmanr(e, d2).correlation
    assert 0.4 < overall < 0.90, f"L2 vs EMD Spearman {overall:.3f} leaves no room to arbitrate"

    edges = np.quantile(e, [0.0, 1 / 3, 2 / 3, 1.0])
    lo = (e >= edges[0]) & (e <= edges[1])
    hi = (e >= edges[2]) & (e <= edges[3])
    rho_local = spearmanr(e[lo], d2[lo]).correlation
    rho_global = spearmanr(e[hi], d2[hi]).correlation
    # L2 is a decent local proxy and a poor global one -- the whole premise.
    assert rho_local - rho_global > 0.10


# --- structured probes -----------------------------------------------------


def test_probes_are_deterministic_and_mass_matched():
    X1, k1 = structured_probes((8, 8), n_variants=4, seed=0, mass_match=313.0)
    X2, k2 = structured_probes((8, 8), n_variants=4, seed=0, mass_match=313.0)
    assert np.array_equal(X1, X2)
    assert list(k1) == list(k2)
    assert X1.shape == (len(PROBE_PATTERNS) * 4, 64)
    # Ink mass must not leak the label: an unmatched probe would be separable
    # from digits on total intensity alone, with no geometry involved.
    assert np.allclose(X1.sum(axis=1), 313.0, rtol=1e-4)
    assert (X1 >= 0).all()


def test_probe_patterns_are_distinct_under_emd():
    X, kinds = structured_probes((8, 8), n_variants=3, seed=0, mass_match=313.0)
    D = pairwise_emd(X, (8, 8), progress=False)
    kinds = np.asarray([str(k) for k in kinds])
    within, between = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            (within if kinds[i] == kinds[j] else between).append(D[i, j])
    # Variants of one pattern must stay closer to each other than to other
    # patterns, or the probe set is measuring noise rather than shape.
    assert np.mean(within) < np.mean(between)
    assert np.max(within) < np.percentile(between, 90)


def test_shuffled_control_preserves_the_intensity_histogram():
    """The point of the pixel-shuffled control is that only layout changes.

    If the histogram shifted, the control would be separable on brightness or
    sparsity and would stop being a test of spatial structure.
    """
    sklearn = pytest.importorskip("sklearn")
    X = sklearn.datasets.load_digits().data.astype(np.float64)
    P, kinds = control_probes(
        (8, 8), source=X, kinds=("shuffled",), n_variants=8, seed=0, mass_match=None
    )
    assert list(kinds) == ["shuffled"] * 8
    sources = {tuple(np.sort(row)) for row in X}
    for row in P:
        assert tuple(np.sort(row.astype(np.float64))) in sources


def test_noise_control_is_mass_matched_and_unstructured():
    P, kinds = control_probes(
        (8, 8), kinds=("noise",), n_variants=6, seed=0, mass_match=313.0
    )
    assert P.shape == (6, 64)
    assert list(kinds) == ["noise"] * 6
    assert np.allclose(P.sum(axis=1), 313.0, rtol=1e-4)
    # No two draws should coincide, and none should be a constant image.
    assert len(np.unique(P, axis=0)) == 6
    assert (P.std(axis=1) > 0).all()


def test_shuffled_control_requires_a_source():
    with pytest.raises(ValueError, match="source"):
        control_probes((8, 8), kinds=("shuffled",), n_variants=2)


def test_render_pattern_is_normalised_and_shaped():
    for name in PROBE_PATTERNS:
        img = render_pattern(name, (8, 8))
        assert img.shape == (8, 8)
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert img.sum() > 0.5, f"{name} rendered nearly empty"

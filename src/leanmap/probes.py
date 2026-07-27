"""Structured synthetic images for probing an embedding out of sample.

Held-out digits test how a map handles *more of the same*. They cannot test what
it does with something structured but genuinely new, which is the case that
matters when a map is used to triage unseen data. These probes fill that gap:
recognisable shapes -- faces, a cross, a ring, bars -- drawn on the same grid as
the data, never shown during training.

Two properties are deliberate.

**Mass matching.** Every probe is rescaled to a target ink mass, by default the
median of the real data. Without this, a probe carrying twice a digit's ink is
detectable from total intensity alone and every out-of-distribution score is
inflated for a reason that has nothing to do with geometry. Pass
``mass_match=None`` to get the unmatched version as a control.

**No claimed ground truth.** These points are off-manifold by construction, so
there is no correct 2-D location for a smiley and nothing here asserts one. They
support only narrow claims: that a probe is farther from the data than a held-out
digit is, that distinct patterns stay distinct, and that they do not collapse.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PROBE_PATTERNS",
    "CONTROL_PROBES",
    "probe_pattern_names",
    "render_pattern",
    "structured_probes",
    "control_probes",
]

PROBE_PATTERNS: Tuple[str, ...] = (
    "smile",
    "frown",
    "neutral",
    "surprised",
    "cross",
    "ex",
    "ring",
    "checker",
    "hbars",
    "vbars",
    "dot",
    "block",
)


def probe_pattern_names() -> Tuple[str, ...]:
    """Names of the available probe patterns, in a stable order."""
    return PROBE_PATTERNS


def _disc(xx, yy, cx, cy, r):
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _annulus(xx, yy, cx, cy, r, t):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (d <= r + 0.5 * t) & (d >= r - 0.5 * t)


def _segment(xx, yy, x0, y0, x1, y1, t):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return _disc(xx, yy, x0, y0, 0.5 * t)
    s = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0.0, 1.0)
    px, py = x0 + s * dx, y0 + s * dy
    return ((xx - px) ** 2 + (yy - py) ** 2) <= (0.5 * t) ** 2


def _arc(xx, yy, cx, cy, r, t, a0, a1):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
    band = (d <= r + 0.5 * t) & (d >= r - 0.5 * t)
    if a0 <= a1:
        return band & (ang >= a0) & (ang <= a1)
    return band & ((ang >= a0) | (ang <= a1))


def _face(xx, yy, mouth) -> np.ndarray:
    m = _disc(xx, yy, 2.3, 2.6, 0.72) | _disc(xx, yy, 5.7, 2.6, 0.72)
    return m | mouth


def _pattern_mask(name: str, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Boolean coverage of ``name`` on continuous coordinates in ``[0, 8]``."""
    if name == "smile":
        return _face(xx, yy, _arc(xx, yy, 4.0, 3.6, 2.3, 0.85, 25.0, 155.0))
    if name == "frown":
        return _face(xx, yy, _arc(xx, yy, 4.0, 7.2, 2.3, 0.85, 205.0, 335.0))
    if name == "neutral":
        return _face(xx, yy, _segment(xx, yy, 2.0, 5.6, 6.0, 5.6, 0.85))
    if name == "surprised":
        return _face(xx, yy, _annulus(xx, yy, 4.0, 5.4, 1.15, 0.8))
    if name == "cross":
        return _segment(xx, yy, 4.0, 0.8, 4.0, 7.2, 1.1) | _segment(
            xx, yy, 0.8, 4.0, 7.2, 4.0, 1.1
        )
    if name == "ex":
        return _segment(xx, yy, 1.0, 1.0, 7.0, 7.0, 1.1) | _segment(
            xx, yy, 7.0, 1.0, 1.0, 7.0, 1.1
        )
    if name == "ring":
        return _annulus(xx, yy, 4.0, 4.0, 2.4, 1.0)
    if name == "checker":
        return ((np.floor(xx / 2.0) + np.floor(yy / 2.0)) % 2.0) < 0.5
    if name == "hbars":
        return (np.floor(yy / 1.5) % 2.0) < 0.5
    if name == "vbars":
        return (np.floor(xx / 1.5) % 2.0) < 0.5
    if name == "dot":
        return _disc(xx, yy, 4.0, 4.0, 1.35)
    if name == "block":
        return (np.abs(xx - 4.0) <= 1.8) & (np.abs(yy - 4.0) <= 1.8)
    raise ValueError(f"unknown probe pattern {name!r}; choose from {PROBE_PATTERNS}")


def render_pattern(
    name: str,
    shape: Tuple[int, int] = (8, 8),
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
    supersample: int = 8,
) -> np.ndarray:
    """Render one pattern as a grayscale ``shape`` image with values in ``[0, 1]``.

    Shapes are rasterised at ``supersample`` times the target resolution and
    block-averaged, which gives the soft edges that make an 8x8 render legible
    instead of a handful of hard squares.
    """
    h, w = int(shape[0]), int(shape[1])
    s = int(supersample)
    # Continuous coordinates on a nominal 8x8 canvas, so pattern geometry is
    # independent of the output resolution.
    jj, ii = np.meshgrid(np.arange(w * s), np.arange(h * s), indexing="xy")
    xx = (jj + 0.5) / (w * s) * 8.0
    yy = (ii + 0.5) / (h * s) * 8.0
    xx = (xx - 4.0) / max(scale, 1e-6) + 4.0 - dx
    yy = (yy - 4.0) / max(scale, 1e-6) + 4.0 - dy
    mask = _pattern_mask(name, xx, yy).astype(np.float64)
    return mask.reshape(h, s, w, s).mean(axis=(1, 3))


def structured_probes(
    shape: Tuple[int, int] = (8, 8),
    *,
    patterns: Optional[Sequence[str]] = None,
    n_variants: int = 16,
    seed: int = 0,
    mass_match: Optional[float] = None,
    peak: float = 16.0,
    shift: float = 0.6,
    scale_jitter: float = 0.08,
    intensity_jitter: float = 0.12,
    noise: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the probe set.

    Parameters
    ----------
    mass_match : total ink each probe is rescaled to, typically
        ``float(np.median(X.sum(1)))`` of the real data. ``None`` scales each
        probe to ``peak`` instead, which leaves brightness free -- useful only as
        a control for how much of a detection is explained by ink alone.
    shift, scale_jitter, intensity_jitter, noise : variation across the
        ``n_variants`` copies of each pattern, so each pattern is a small cloud
        rather than a single point.

    Returns
    -------
    X : ``(n_patterns * n_variants, H * W)`` float32, flattened like the data
    names : ``(n_patterns * n_variants,)`` str, the pattern each row came from
    """
    names = tuple(patterns) if patterns is not None else PROBE_PATTERNS
    for nm in names:
        if nm not in PROBE_PATTERNS:
            raise ValueError(f"unknown probe pattern {nm!r}")
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for nm in names:
        for _ in range(int(n_variants)):
            img = render_pattern(
                nm,
                shape,
                dx=float(rng.uniform(-shift, shift)),
                dy=float(rng.uniform(-shift, shift)),
                scale=float(1.0 + rng.uniform(-scale_jitter, scale_jitter)),
            )
            img = img * float(1.0 + rng.uniform(-intensity_jitter, intensity_jitter))
            if noise > 0:
                img = img + rng.normal(0.0, noise, size=img.shape)
            img = np.clip(img, 0.0, None)
            total = img.sum()
            if total <= 0:
                raise RuntimeError(f"probe {nm!r} rendered empty")
            if mass_match is not None:
                img = img * (float(mass_match) / total)
            else:
                img = img * (float(peak) / max(img.max(), 1e-12))
            rows.append(img.ravel())
            labels.append(nm)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=object)


CONTROL_PROBES: Tuple[str, ...] = ("noise", "shuffled")


def control_probes(
    shape: Tuple[int, int] = (8, 8),
    *,
    source: Optional[np.ndarray] = None,
    kinds: Sequence[str] = CONTROL_PROBES,
    n_variants: int = 16,
    seed: int = 0,
    mass_match: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unstructured controls that bound what the structured probes can mean.

    A detector that scores highly on smileys has only shown something if it also
    behaves sensibly on images with no structure at all. Two kinds:

    ``noise``
        Independent random pixels. Off-manifold in every sense, so this is the
        floor: any detector that misses it is broken, and the number it gets
        here is the ceiling that the structured-probe score should be read
        against.

    ``shuffled``
        A real image with its pixels spatially permuted. This preserves the
        intensity histogram and the total ink **exactly** -- the multiset of
        pixel values is untouched -- so nothing about brightness, contrast or
        sparsity can separate it from a real digit. Only spatial arrangement
        can, which makes it the sharp test of whether a map encodes layout or
        merely intensity statistics. Requires ``source``.
    """
    for nm in kinds:
        if nm not in CONTROL_PROBES:
            raise ValueError(f"unknown control probe {nm!r}; choose from {CONTROL_PROBES}")
    if "shuffled" in kinds and source is None:
        raise ValueError("the 'shuffled' control needs a `source` array to permute")
    p = int(shape[0]) * int(shape[1])
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for nm in kinds:
        for _ in range(int(n_variants)):
            if nm == "noise":
                img = rng.random(p)
            else:
                src = np.asarray(source, dtype=np.float64)
                src = src.reshape(len(src), -1)
                row = src[rng.integers(0, len(src))]
                img = rng.permutation(row)
            img = np.clip(img, 0.0, None)
            total = img.sum()
            if total <= 0:
                continue
            if mass_match is not None:
                img = img * (float(mass_match) / total)
            rows.append(img)
            labels.append(nm)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=object)

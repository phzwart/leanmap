#!/usr/bin/env python
"""Build the EMD reference geometry for an image feed, and gate it.

Every geometry metric in this harness normally calls pixel L2 the truth, and
every embedder here is fit from a pixel-L2 kNN graph, so those metrics partly
reward reproducing the input. EMD is an independent reference: no method sees
it, and unlike L2 it does not saturate once two images stop overlapping.

This writes the reference once -- it is reused to score every embedding, seed
and null -- along with the structured probes, then reports whether the reference
is worth having at all:

* ``C2`` local agreement: at short range L2 should track EMD, which is what
  licenses building the kNN graph from L2 in the first place.
* ``C3`` global divergence: at long range L2 should saturate while the graph
  geodesic still tracks EMD. **If C3 fails, EMD is a relabelled copy of L2 and
  cannot arbitrate between embeddings** -- that is a go/no-go, not a detail.

Usage::

    python examples/exploratory/make_emd.py \\
      --X examples/exploratory/data/digits_X.npy --image-shape 8 8 --n-jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (_EXAMPLES, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_run import write_json  # noqa: E402

DEFAULT_DATA = _HERE / "data"


def gate_report(
    X: np.ndarray,
    D_emd: np.ndarray,
    *,
    n_neighbors: int = 15,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """Measure how much EMD adds over L2, overall and by distance band."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    from leanmap.emd import geodesic_from_matrix

    n = len(X)
    D_l2 = squareform(pdist(np.asarray(X, dtype=np.float64)))
    D_geo = geodesic_from_matrix(D_l2, n_neighbors=n_neighbors)

    iu = np.triu_indices(n, k=1)
    emd = D_emd[iu]
    l2 = D_l2[iu]
    geo = D_geo[iu]
    finite = np.isfinite(emd) & np.isfinite(l2) & np.isfinite(geo)
    emd, l2, geo = emd[finite], l2[finite], geo[finite]
    if emd.size > max_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(emd.size, size=max_pairs, replace=False)
        emd, l2, geo = emd[sel], l2[sel], geo[sel]

    out = {
        "n_points": int(n),
        "n_pairs": int(emd.size),
        "n_neighbors": int(n_neighbors),
        "unreachable_frac": float(1.0 - finite.mean()),
        "l2_spearman": float(spearmanr(emd, l2).correlation),
        "geo_spearman": float(spearmanr(emd, geo).correlation),
    }
    edges = np.quantile(emd, [0.0, 1 / 3, 2 / 3, 1.0])
    for b, band in enumerate(("local", "mid", "global")):
        m = (emd >= edges[b]) & (emd <= edges[b + 1])
        out[f"l2_spearman_{band}"] = float(spearmanr(emd[m], l2[m]).correlation)
        out[f"geo_spearman_{band}"] = float(spearmanr(emd[m], geo[m]).correlation)
    out["margin_global"] = out["geo_spearman_global"] - out["l2_spearman_global"]
    out["margin_overall"] = out["geo_spearman"] - out["l2_spearman"]

    # L2 saturation: how compressed L2 is among the pairs EMD calls most distant.
    top = l2[emd >= edges[2]]
    out["l2_saturation_ratio"] = float(np.percentile(top, 99) / max(np.percentile(top, 50), 1e-12))

    # Two independent questions that must not be conflated. The gate is only
    # about whether EMD is its own geometry; whether a *geodesic* recovers it is
    # a separate claim that can fail without invalidating the comparison.
    out["gate_pass"] = bool(out["l2_spearman_global"] < 0.90 and out["l2_spearman"] < 0.95)
    out["gate_verdict"] = (
        "GO: EMD disagrees with L2 enough to arbitrate between embeddings"
        if out["gate_pass"]
        else "NO-GO: EMD is a relabelled copy of L2 on this feed"
    )
    out["geodesic_tracks_emd"] = bool(out["margin_global"] > 0.05)
    out["geodesic_verdict"] = (
        "geodesic on the L2 graph tracks EMD better than raw L2"
        if out["geodesic_tracks_emd"]
        else "geodesic on the L2 graph is a WORSE proxy for EMD than raw L2"
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--X", required=True, help="image features (N, H*W) .npy")
    ap.add_argument("--image-shape", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--out", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--name", default=None, help="cache prefix (default: X stem)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-variants", type=int, default=16, help="probes per pattern")
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument(
        "--no-controls",
        action="store_true",
        help="skip the unstructured noise / pixel-shuffled control probes",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from leanmap.emd import pairwise_emd
    from leanmap.probes import control_probes, structured_probes

    X = np.load(args.X).astype(np.float32)
    shape = (int(args.image_shape[0]), int(args.image_shape[1]))
    if X.shape[1] != shape[0] * shape[1]:
        raise SystemExit(f"X has {X.shape[1]} columns, incompatible with shape {shape}")
    name = args.name or Path(args.X).stem.replace("_X", "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    emd_path = out_dir / f"{name}_emd.npy"
    meta_path = out_dir / f"{name}_emd.json"

    # Probes are matched to the median ink mass of the real data so that total
    # intensity alone cannot separate them from digits later.
    mass = float(np.median(X.sum(axis=1)))
    if args.no_probes:
        P = np.zeros((0, X.shape[1]), dtype=np.float32)
        kinds = np.asarray([], dtype=object)
    else:
        P, kinds = structured_probes(
            shape,
            n_variants=args.n_variants,
            seed=args.probe_seed,
            mass_match=mass,
        )
        if not args.no_controls:
            # Unstructured controls bound what the structured probes can mean:
            # noise is the floor every detector should clear, and a
            # pixel-permuted digit has an identical intensity histogram, so only
            # spatial layout can give it away.
            Pc, kc = control_probes(
                shape,
                source=X,
                n_variants=args.n_variants,
                seed=args.probe_seed,
                mass_match=mass,
            )
            P = np.concatenate([P, Pc], axis=0)
            kinds = np.concatenate([kinds, kc])
        np.save(out_dir / f"{name}_probes_X.npy", P)
        np.save(out_dir / f"{name}_probes_kind.npy", kinds)
        print(
            f"probes: {len(P)} rows over {len(set(kinds.tolist()))} families, "
            f"ink matched to median digit mass {mass:.1f}"
        )

    A = np.concatenate([X, P], axis=0) if len(P) else X
    print(f"{name}: {len(X)} images + {len(P)} probes = {len(A)} rows, shape {shape}")

    if emd_path.is_file() and not args.force:
        print(f"reusing {emd_path} (pass --force to rebuild)")
        D = np.load(emd_path).astype(np.float64)
        if len(D) != len(A):
            raise SystemExit(
                f"cached matrix is {len(D)}x{len(D)} but {len(A)} rows were requested; "
                "rerun with --force"
            )
    else:
        t0 = time.perf_counter()
        D = pairwise_emd(A, shape, n_jobs=args.n_jobs, progress=True)
        dt = time.perf_counter() - t0
        np.save(emd_path, D.astype(np.float32))
        print(
            f"wrote {emd_path} ({D.nbytes / 4 / 1e6:.1f} MB as float32) in {dt:.0f}s "
            f"-> {dt / max(len(A) ** 2, 1) * 1e6:.0f} us/pair"
        )

    gate = gate_report(X, D[: len(X), : len(X)], n_neighbors=args.n_neighbors)
    write_json(
        meta_path,
        {
            "name": name,
            "X": str(Path(args.X).resolve()),
            "image_shape": list(shape),
            "n_images": int(len(X)),
            "n_probes": int(len(P)),
            "probe_mass": mass,
            "probe_seed": int(args.probe_seed),
            "emd": str(emd_path.resolve()),
            "gate": gate,
        },
    )

    print("\nreference gate: Spearman against EMD, by EMD distance band")
    print(f"  {'band':<8} {'L2':>8} {'geodesic':>10} {'margin':>8}")
    for band in ("local", "mid", "global"):
        a, b = gate[f"l2_spearman_{band}"], gate[f"geo_spearman_{band}"]
        print(f"  {band:<8} {a:>8.3f} {b:>10.3f} {b - a:>8.3f}")
    print(
        f"  {'overall':<8} {gate['l2_spearman']:>8.3f} {gate['geo_spearman']:>10.3f} "
        f"{gate['margin_overall']:>8.3f}"
    )
    print(f"  L2 saturation (p99/p50 among far pairs): {gate['l2_saturation_ratio']:.3f}")
    print(f"  unreachable pairs: {gate['unreachable_frac']:.1%}")
    print(f"\n{gate['gate_verdict']}")
    print(f"C3: {gate['geodesic_verdict']}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

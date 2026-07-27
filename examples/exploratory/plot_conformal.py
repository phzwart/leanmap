#!/usr/bin/env python
"""leanmap's own OOD scores, plotted -- landmark cover and the conformal test.

The map panels in ``plot_embeddings.py`` show what the 2-D geometry can see. This
shows what leanmap actually ships for the job: ``cover``, the ambient distance to
the nearest landmark, and the conformal p-value calibrated on real held-out
digits. The two are worth plotting side by side because they disagree -- the map
scores AUROC ~0.81 on the probes and cover scores 1.000.

The holdout is split in half so the conformal test is honest: one half calibrates,
the other is a genuine test set whose p-values must come out uniform. Probes are
never involved in calibration.

Usage::

    python examples/exploratory/plot_conformal.py \\
      --probe-kind examples/exploratory/data/digits_probes_kind.npy \\
      --Z seed0=examples/out/exploratory/digits_emd_lm/matched__digits__seed0 \\
      --Z seed1=examples/out/exploratory/digits_emd_lm/matched__digits__seed1 \\
      --Z seed2=examples/out/exploratory/digits_emd_lm/matched__digits__seed2 \\
      --out examples/out/exploratory/digits_emd/conformal.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from splits import load_split


def _parse_z(spec: str) -> Tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)), ties counted as half."""
    pooled = np.concatenate([pos, neg])
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty(len(pooled), dtype=np.float64)
    ranks[order] = np.arange(1, len(pooled) + 1)
    # Cover scores tie often (the landmarks all sit at 0), so tied ranks have to
    # be averaged or the AUROC picks up an ordering the score never expressed.
    s = pooled[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _roc(pos: np.ndarray, neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0]


def _pvalues(scores: np.ndarray, calib: np.ndarray) -> np.ndarray:
    """``(1 + #{calib >= s}) / (n + 1)`` -- ConformalCalibrator.p_value."""
    s_calib = np.sort(calib)
    n_ge = len(s_calib) - np.searchsorted(s_calib, scores, side="left")
    return (1.0 + n_ge) / (len(s_calib) + 1.0)


def load_run(path: Path) -> dict | None:
    """A leanmap run needs both cover arrays; anything else has no score to plot."""
    need = ("Z.npy", "cover.npy", "Z_probe.npy", "probe_cover.npy")
    if not all((path / f).is_file() for f in need):
        return None
    Z = np.load(path / "Z.npy").astype(np.float64)
    train_idx, hold_idx = load_split(path, n=len(Z))
    return {
        "Z": Z,
        "cover": np.load(path / "cover.npy").astype(np.float64),
        "Z_probe": np.load(path / "Z_probe.npy").astype(np.float64),
        "probe_cover": np.load(path / "probe_cover.npy").astype(np.float64),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "hold_idx": np.asarray(hold_idx, dtype=np.int64),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Z", action="append", required=True, help="name=run_dir")
    ap.add_argument("--probe-kind", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="calib/test split of the holdout")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = (
        np.asarray([str(v) for v in np.load(args.probe_kind, allow_pickle=True)])
        if args.probe_kind
        else None
    )

    runs: List[Tuple[str, dict]] = []
    for spec in args.Z:
        name, path = _parse_z(spec)
        run = load_run(path)
        if run is None:
            print(f"skipping {name}: no cover.npy / probe_cover.npy in {path}")
            continue
        runs.append((name, run))
    if not runs:
        raise SystemExit("no run had leanmap cover scores to plot")

    rng = np.random.default_rng(args.seed)
    # Pooled across seeds. Raw cover is in ambient units and differs a little
    # between runs, so anything pooled is expressed relative to that run's own
    # calibration median; p-values and AUROC are already scale-free.
    pool = {"train": [], "test": [], "probe": []}
    p_test, p_probe = [], []
    roc_cover, roc_map = [], []
    # Both scores expressed as a multiple of what a real held-out digit gets, so
    # the ambient score and the map distance become directly comparable.
    rel = {"probe_cover": [], "probe_map": [], "test_cover": [], "test_map": []}
    fam_rows: dict[str, list] = {}
    fam_rel: dict[str, list] = {}

    for _, run in runs:
        hold = run["hold_idx"].copy()
        rng.shuffle(hold)
        cut = len(hold) // 2
        calib_idx, test_idx = hold[:cut], hold[cut:]
        cov = run["cover"]
        c_calib, c_test = cov[calib_idx], cov[test_idx]
        c_train, c_probe = cov[run["train_idx"]], run["probe_cover"]

        med = float(np.median(c_calib)) or 1e-12
        pool["train"].append(c_train / med)
        pool["test"].append(c_test / med)
        pool["probe"].append(c_probe / med)

        p_test.append(_pvalues(c_test, c_calib))
        pp = _pvalues(c_probe, c_calib)
        p_probe.append(pp)

        # The 2-D map's own score, for the same points, as the comparison.
        Zt = run["Z"][run["train_idx"]]
        from scipy.spatial import cKDTree

        tree = cKDTree(Zt)
        d_probe, _ = tree.query(run["Z_probe"], k=1)
        d_test, _ = tree.query(run["Z"][test_idx], k=1)
        roc_cover.append((c_probe, c_test))
        roc_map.append((d_probe, d_test))

        d_med = float(np.median(d_test)) or 1e-12
        rel["probe_cover"].append(c_probe / med)
        rel["probe_map"].append(d_probe / d_med)
        rel["test_cover"].append(c_test / med)
        rel["test_map"].append(d_test / d_med)

        if kinds is not None:
            k = kinds[: len(c_probe)]
            for fam in np.unique(k):
                m = k == fam
                fam_rows.setdefault(fam, []).append(
                    (
                        _auroc(c_probe[m], c_test),
                        float(np.median(pp[m])),
                        float(np.mean(pp[m] <= args.alpha)),
                    )
                )
                fam_rel.setdefault(fam, []).append(
                    (
                        float(np.median(c_probe[m] / med)),
                        float(np.median(d_probe[m] / d_med)),
                    )
                )

    cat = {k: np.concatenate(v) for k, v in pool.items()}
    relc = {k: np.concatenate(v) for k, v in rel.items()}
    p_test_all = np.concatenate(p_test)
    p_probe_all = np.concatenate(p_probe)
    n_calib = len(runs[0][1]["hold_idx"]) // 2
    p_floor = 1.0 / (n_calib + 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.2))

    # (a) the score itself
    ax = axes[0, 0]
    bins = np.linspace(0, max(np.percentile(cat["probe"], 99.5), 2.0), 60)
    for key, label, color in (
        ("train", "training digits", "tab:blue"),
        ("test", "held-out digits (test half)", "tab:green"),
        ("probe", "probes", "tab:red"),
    ):
        ax.hist(
            np.clip(cat[key], bins[0], bins[-1]), bins=bins, density=True,
            alpha=0.55, color=color, label=label,
        )
    thr = float(np.quantile(cat["test"], 1.0 - args.alpha))
    ax.axvline(thr, color="k", ls="--", lw=1.2, label=f"alpha={args.alpha} threshold")
    # Landmarks are drawn from the training set, so they sit at cover exactly 0.
    n_zero = float(np.mean(cat["train"] <= 1e-9))
    if n_zero > 0.01:
        ax.annotate(
            f"{n_zero:.0%} of training points\nare the landmarks themselves\n(cover = 0 by construction)",
            xy=(0.02, 1.7), xytext=(0.55, 2.1), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
    ax.set_xlabel("landmark cover, in units of the calibration median")
    ax.set_ylabel("density")
    ax.set_title("(a) leanmap cover score: ambient distance to nearest landmark")
    ax.legend(fontsize=8)

    # (b) is the test valid, and is it powerful?
    ax = axes[0, 1]
    for p, label, color in (
        (p_test_all, "held-out digits", "tab:green"),
        (p_probe_all, "probes", "tab:red"),
    ):
        xs = np.sort(p)
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=color, lw=2, label=label)
    # Uniform is y = x; on a log x-axis it has to be drawn as a curve, not a line.
    grid = np.logspace(np.log10(p_floor), 0, 200)
    ax.plot(grid, grid, color="grey", ls=":", lw=1.2, label="uniform (a valid test)")
    ax.axvline(args.alpha, color="k", ls="--", lw=1.2)
    hit = float(np.mean(p_test_all <= args.alpha))
    ax.annotate(
        f"real digits flagged at alpha={args.alpha}: {hit:.3f}\n"
        f"(nominal {args.alpha} -- the test is calibrated)",
        xy=(args.alpha, hit), xytext=(0.011, 0.40), fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.annotate(
        f"every probe pinned at the floor\np = 1/(n+1) = {p_floor:.4f}, n_calib = {n_calib}\n"
        "out of resolution, not out of power",
        xy=(p_floor, 0.88), xytext=(0.011, 0.72), fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xscale("log")
    ax.set_xlim(p_floor * 0.8, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("conformal p-value (log)")
    ax.set_ylabel("fraction with p <= x")
    ax.set_title("(b) conformal test: calibrated on real held-out digits")
    ax.legend(fontsize=8, loc="lower right")

    # (c) the same colour scale for digits and probes. Putting probes on the
    # digits' scale is the point: they saturate it wherever they happen to land,
    # so position on the map carries none of the information the score has.
    name0, run0 = runs[0]
    ax = axes[0, 2]
    Z, cov = run0["Z"], run0["cover"]
    Zp, covp = run0["Z_probe"], run0["probe_cover"]
    ok = np.isfinite(Z).all(axis=1)
    okp = np.isfinite(Zp).all(axis=1)
    vmax = float(np.percentile(np.concatenate([cov, covp]), 99))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    ax.scatter(Z[ok, 0], Z[ok, 1], c=cov[ok], cmap="magma", norm=norm, s=7, linewidths=0)
    sc = ax.scatter(
        Zp[okp, 0], Zp[okp, 1], c=covp[okp], cmap="magma", norm=norm, s=52,
        marker="o", linewidths=0.7, edgecolors="k",
    )
    fig.colorbar(sc, ax=ax, fraction=0.046).set_label("cover score (shared scale)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"(c) {name0}: one colour scale, digits (small) and probes (ringed)",
        fontsize=10,
    )

    # (d) the ambient score against the map's own geometry
    ax = axes[1, 0]
    for pairs, label, color in (
        (roc_cover, "leanmap cover (ambient)", "tab:purple"),
        (roc_map, "distance to nearest train point in the 2-D map", "tab:orange"),
    ):
        pos = np.concatenate([p for p, _ in pairs])
        neg = np.concatenate([n for _, n in pairs])
        fpr, tpr = _roc(pos, neg)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} -- AUROC {_auroc(pos, neg):.3f}")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.set_xlabel("false positive rate on real held-out digits")
    ax.set_ylabel("probes detected")
    ax.set_title("(d) what the score sees vs what the picture sees")
    ax.legend(fontsize=8, loc="lower right")

    # (e) the two scores against each other, per point. This is where the AUROC
    # gap in (d) comes from: cover is uniformly high, map distance is not.
    ax = axes[1, 1]
    ax.scatter(
        relc["test_map"], relc["test_cover"], s=9, c="tab:green", linewidths=0,
        alpha=0.5, label="held-out digits",
    )
    ax.scatter(
        relc["probe_map"], relc["probe_cover"], s=16, c="tab:red", linewidths=0,
        alpha=0.65, label="probes",
    )
    map_thr = float(np.quantile(relc["test_map"], 0.95))
    cov_thr = float(np.quantile(relc["test_cover"], 1.0 - args.alpha))
    ax.axvline(map_thr, color="tab:orange", ls="--", lw=1.2)
    ax.axhline(cov_thr, color="tab:purple", ls="--", lw=1.2)
    missed = (relc["probe_map"] < map_thr) & (relc["probe_cover"] > cov_thr)
    ax.text(
        0.03, 0.97,
        f"{missed.mean():.0%} of probes sit left of the orange line\n"
        "and above the purple one: the map cannot\n"
        "tell them from a digit, the score can",
        transform=ax.transAxes, fontsize=8, color="tab:purple", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:purple", alpha=0.85),
    )
    ax.set_xscale("log")
    ax.set_xlabel("distance to nearest training point in the map (x a real digit's)")
    ax.set_ylabel("cover score (x a real digit's)")
    ax.set_title("(e) the same probe, scored both ways")
    ax.legend(fontsize=8, loc="lower right")

    # (f) which patterns are novel, and does the map agree about the ordering?
    ax = axes[1, 2]
    if fam_rel:
        fams = sorted(fam_rel, key=lambda f: np.mean(fam_rel[f], axis=0)[0])
        arr = {f: np.asarray(fam_rel[f]) for f in fams}
        cvals = np.array([arr[f][:, 0].mean() for f in fams])
        mvals = np.array([arr[f][:, 1].mean() for f in fams])
        # Range across seeds. Whether an ordering is reproducible matters more
        # than the ordering itself, and the two series differ sharply on that.
        cerr = np.array([[cvals[i] - arr[f][:, 0].min(), arr[f][:, 0].max() - cvals[i]]
                         for i, f in enumerate(fams)]).T
        merr = np.array([[mvals[i] - arr[f][:, 1].min(), arr[f][:, 1].max() - mvals[i]]
                         for i, f in enumerate(fams)]).T
        ypos = np.arange(len(fams))
        ax.barh(
            ypos, cvals, 0.62, xerr=cerr, color="tab:purple", alpha=0.85,
            error_kw=dict(ecolor="k", lw=1.0, capsize=2), label="cover (ambient)",
        )
        ax.errorbar(
            mvals, ypos, xerr=merr, fmt="D", color="tab:orange", markersize=6,
            lw=1.0, capsize=2, label="map distance",
        )
        ax.set_yticks(ypos)
        ax.set_yticklabels(fams, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.85, float((mvals + merr[1]).max()) * 1.5)
        ax.axvline(1.0, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("median, as a multiple of a real held-out digit's")
        cv_c = float(np.mean([arr[f][:, 0].std() / arr[f][:, 0].mean() for f in fams]))
        cv_m = float(np.mean([arr[f][:, 1].std() / arr[f][:, 1].mean() for f in fams]))
        ax.set_title(
            f"(f) novelty by pattern -- seed spread {cv_c:.0%} vs {cv_m:.0%}", fontsize=10
        )
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.axis("off")

    fig.suptitle(
        f"leanmap OOD scores on structured probes -- {len(runs)} seed(s), "
        f"{len(p_probe_all) // len(runs)} probes, calibration never sees a probe",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=145)
    plt.close(fig)
    print(f"wrote {args.out}")

    print(
        f"probes the map cannot separate but the score can: {missed.mean():.1%} "
        f"(map threshold at 5% FPR, cover at alpha={args.alpha})"
    )
    if fam_rows:
        print(
            f"\n{'family':<12}{'cover AUROC':>13}{'median p':>11}{'flagged':>10}"
            f"{'cover x digit':>15}{'map x digit':>13}"
        )
        for fam in sorted(fam_rows):
            a, p, f = np.mean(fam_rows[fam], axis=0)
            c, m = np.mean(fam_rel[fam], axis=0) if fam in fam_rel else (np.nan, np.nan)
            print(f"{fam:<12}{a:>13.3f}{p:>11.4f}{f:>10.0%}{c:>15.2f}{m:>13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

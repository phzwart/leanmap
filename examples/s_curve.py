#!/usr/bin/env python
"""leanmap on the classic S-curve manifold."""

from __future__ import annotations

import argparse

from sklearn.datasets import make_s_curve

from _demo import OUT_DIR, fit_embed, save_density, save_scatter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Match sklearn manifold gallery (plot_compare_methods.py):
    # n_samples=1500, noise=0.0, random_state=0
    ap.add_argument("--n", type=int, default=1500, help="number of points")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="initial learning rate",
    )
    ap.add_argument(
        "--lr-after",
        type=float,
        default=5e-3,
        help="learning rate after --lr-switch-epochs",
    )
    ap.add_argument(
        "--lr-switch-epochs",
        type=int,
        default=5,
        help="epochs at --lr before switching to --lr-after",
    )
    ap.add_argument(
        "--batch-edges",
        type=int,
        default=512,
        help="edges per step (smaller => more steps/epoch)",
    )
    ap.add_argument(
        "--min-dist",
        type=float,
        default=0.3,
        help="UMAP-style min_dist (larger => less clumpy)",
    )
    ap.add_argument(
        "--n-negatives",
        type=int,
        default=15,
        help="negatives per edge (higher => stronger repulsion)",
    )
    ap.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="kNN graph neighbors (default: 10)",
    )
    ap.add_argument(
        "--device",
        default="mps",
        help="torch device (default: mps; use cpu/cuda to override)",
    )
    ap.add_argument(
        "--pyramid-scales",
        type=int,
        default=3,
        help="coarse levels (default: 3 = cohesive pyramid; 0 = single-scale)",
    )
    ap.add_argument(
        "--pyramid-level-weights",
        type=str,
        default="1,1,1,1",
        help="comma-separated per-level attraction weights (default: equal 1,1,1,1)",
    )
    ap.add_argument(
        "--n-landmarks",
        type=int,
        default=250,
        help="number of landmarks (default: 250)",
    )
    ap.add_argument(
        "--local-connectivity",
        type=int,
        default=2,
        help="graph local_connectivity (rho = k-th NN dist; higher => smoother local packing)",
    )
    ap.add_argument(
        "--lambda-lm",
        type=float,
        default=None,
        help="landmark soft-quantization weight (0 disables landmark clumping)",
    )
    ap.add_argument(
        "--tau-scale",
        type=float,
        default=2.0,
        help="scale on default anchor temperature (>1 => spread over more landmarks)",
    )
    ap.add_argument(
        "--learn-tau",
        dest="learn_tau",
        action="store_true",
        help="learn anchor temperature (default off: learnable tau sharpens to one-hot => clumps)",
    )
    ap.add_argument(
        "--no-learn-landmarks",
        dest="learn_landmarks",
        action="store_false",
        help="freeze landmark positions at FPS init (default: frozen)",
    )
    ap.add_argument(
        "--learn-landmarks",
        dest="learn_landmarks",
        action="store_true",
        help="learn landmark positions (default off: drift twists high-curvature tips)",
    )
    ap.add_argument(
        "--geodesic-landmarks",
        dest="landmark_geodesic",
        action="store_true",
        help="pick landmarks by geodesic FPS (kNN shortest-path) instead of ambient",
    )
    ap.add_argument(
        "--poisson-landmarks",
        dest="landmark_poisson",
        action="store_true",
        help="pick landmarks by geodesic Poisson-disk (blue-noise) sampling "
        "instead of FPS (more uniform interior coverage)",
    )
    ap.set_defaults(landmark_poisson=False)
    ap.add_argument(
        "--lambda-frame",
        type=float,
        default=0.5,
        help="weight for the local-rigidity (ARAP) loss on fine-graph "
        "neighbourhoods (opposes the frame-rotation twist/pinch); 0 = off",
    )
    ap.add_argument(
        "--frame-neighbors",
        type=int,
        default=6,
        help="neighbours per star for the local-rigidity loss (default: 6)",
    )
    ap.add_argument(
        "--lambda-geo",
        type=float,
        default=0.5,
        help="weight for coarse geodesic (Isomap) MDS backbone on landmarks "
        "(straightens global banana bends); 0 = off (default: 0.5)",
    )
    ap.set_defaults(learn_tau=False, learn_landmarks=False, landmark_geodesic=False)
    ap.add_argument(
        "--pca-skip",
        action="store_true",
        help="enable PCA linear skip init (default: off)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="generate fresh S-curve points (new seed), embed via the trained "
        "encoder, and overlay them in red (out-of-sample test)",
    )
    ap.add_argument(
        "--fresh-n",
        type=int,
        default=300,
        help="number of fresh out-of-sample points to overlay",
    )
    ap.add_argument(
        "--fresh-seed",
        type=int,
        default=None,
        help="random seed for the fresh points (default: train seed + 1000)",
    )
    ap.add_argument(
        "--save-encoder",
        type=str,
        default=None,
        help="path to save the trained encoder (.pt) for future use",
    )
    ap.add_argument(
        "--density",
        type=int,
        default=0,
        help="reload the saved encoder, embed N fresh points, and plot a 2-D "
        "density (hexbin) map (e.g. --density 10000)",
    )
    args = ap.parse_args()

    X, t = make_s_curve(n_samples=args.n, noise=args.noise, random_state=args.seed)
    X = X.astype("float32")
    weights = [float(x) for x in args.pyramid_level_weights.split(",") if x.strip()]
    print(
        f"fit: N={len(X)} device={args.device} epochs={args.epochs} "
        f"lr={args.lr}->{args.lr_after} after {args.lr_switch_epochs} ep "
        f"batch_edges={args.batch_edges} min_dist={args.min_dist} "
        f"n_negatives={args.n_negatives} n_neighbors={args.n_neighbors} "
        f"pca_skip={args.pca_skip} n_landmarks={args.n_landmarks} "
        f"learn_landmarks={args.learn_landmarks} pyramid_scales={args.pyramid_scales} "
        f"lambda_lm={args.lambda_lm} local_connectivity={args.local_connectivity} "
        f"tau_scale={args.tau_scale} learn_tau={args.learn_tau} "
        f"geodesic_landmarks={args.landmark_geodesic}",
        flush=True,
    )
    result, Z, _ = fit_embed(
        X,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        pyramid_scales=args.pyramid_scales,
        pyramid_level_weights=None if args.pyramid_scales == 0 else weights,
        pca_skip=args.pca_skip,
        n_landmarks=args.n_landmarks,
        learn_landmarks=args.learn_landmarks,
        lr=args.lr,
        lr_after=args.lr_after,
        lr_switch_epochs=args.lr_switch_epochs,
        batch_edges=args.batch_edges,
        min_dist=args.min_dist,
        n_negatives=args.n_negatives,
        n_neighbors=args.n_neighbors,
        local_connectivity=args.local_connectivity,
        lambda_lm=args.lambda_lm,
        tau_scale=args.tau_scale,
        learn_tau=args.learn_tau,
        landmark_geodesic=args.landmark_geodesic,
        landmark_poisson=args.landmark_poisson,
        lambda_frame=args.lambda_frame,
        frame_neighbors=args.frame_neighbors,
        lambda_geo=args.lambda_geo,
    )
    overlay = None
    if args.fresh:
        import torch

        fresh_seed = args.fresh_seed if args.fresh_seed is not None else args.seed + 1000
        Xf, tf = make_s_curve(n_samples=args.fresh_n, noise=args.noise, random_state=fresh_seed)
        Xf = Xf.astype("float32")
        with torch.no_grad():
            Zf, _ = result.embed(Xf)
        overlay = Zf.detach().cpu().numpy()
        print(
            f"fresh out-of-sample: N={len(Xf)} seed={fresh_seed} -> embedding {overlay.shape}",
            flush=True,
        )

    out = save_scatter(
        Z,
        t,
        title="leanmap — S-curve",
        path=OUT_DIR / "s_curve.png",
        colorbar_label="manifold parameter",
        overlay=overlay,
    )
    print(f"N={len(X)} d={X.shape[1]} -> embedding {Z.shape}")
    print(
        f"pca_skip={result.config.pca_skip} n_landmarks={result.config.n_landmarks} "
        f"learn_landmarks={result.config.learn_landmarks}"
    )
    print(
        f"pyramid_scales={result.config.pyramid_scales} "
        f"level_weights={result.config.pyramid_level_weights} "
        f"coarse_backbone={result.config.pyramid_coarse_backbone}"
    )
    print(f"saved {out}")

    # Persist the encoder + optional 10k out-of-sample density map.
    enc_path = args.save_encoder
    if enc_path is None and args.density > 0:
        enc_path = str(OUT_DIR / "s_curve_encoder.pt")
    if enc_path is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result.save(enc_path)
        print(f"saved encoder {enc_path}", flush=True)

    if args.density > 0:
        import torch

        from leanmap import load_plane

        # Reload the persisted encoder ("future use") and stream fresh points.
        model = load_plane(enc_path, device=args.device)
        dens_seed = (args.fresh_seed if args.fresh_seed is not None else args.seed + 1000) + 1
        Xd, _ = make_s_curve(n_samples=args.density, noise=args.noise, random_state=dens_seed)
        Xd = Xd.astype("float32")
        with torch.no_grad():
            Zd, _ = model.embed(torch.as_tensor(Xd), return_score=False)
        Zd = Zd.detach().cpu().numpy()
        dpath = save_density(
            Zd,
            title=f"leanmap — S-curve density ({args.density:,} OOS points)",
            path=OUT_DIR / "s_curve_density.png",
        )
        print(
            f"density: embedded {len(Xd):,} fresh points via reloaded encoder "
            f"(seed={dens_seed}) -> saved {dpath}",
            flush=True,
        )


if __name__ == "__main__":
    main()

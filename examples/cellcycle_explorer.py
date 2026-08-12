#!/usr/bin/env python
"""Dash explorer for CellCycle leanmap embeddings.

Shows the 2-D latent space (scrubbable by epoch), click/lasso cells, and
inspect RGB crops + neighbors for the selection.

    python examples/cellcycle_explorer.py
    python examples/cellcycle_explorer.py --port 8051 --run-dir examples/out/cellcycle_l1

Requires: dash, plotly, zarr, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZARR = ROOT / "examples" / "out" / "cellcycle_l1.zarr"
DEFAULT_RUN = ROOT / "examples" / "out" / "cellcycle_l1"

PHASES = ("G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase")
PHASE_COLORS = {
    "G1": "#4C78A8",
    "S": "#F58518",
    "G2": "#54A24B",
    "Prophase": "#E45756",
    "Metaphase": "#72B7B2",
    "Anaphase": "#B279A2",
    "Telophase": "#FF9DA6",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _phase_heatmap_colorscale(hex_color: str) -> list[list]:
    """Transparent → phase color colorscale for density overlays."""
    r, g, b = _hex_to_rgb(hex_color)
    return [
        [0.0, f"rgba({r},{g},{b},0.0)"],
        [0.35, f"rgba({r},{g},{b},0.15)"],
        [0.65, f"rgba({r},{g},{b},0.45)"],
        [1.0, f"rgba({r},{g},{b},0.85)"],
    ]


def _train_idx(n: int, *, seed: int, holdout: float) -> np.ndarray:
    """Reproduce the train split used by cellcycle_emd.fit_leanmap."""
    rng = np.random.default_rng(int(seed))
    n_cal = max(1, int(round(float(holdout) * n)))
    perm = rng.permutation(n)
    return perm[n_cal:].astype(np.int64)


def _list_epochs(run_dir: Path) -> list[int]:
    frames = sorted((run_dir / "frames").glob("epoch_*.npy"))
    out = []
    for p in frames:
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def _load_Z(run_dir: Path, epoch: int | None) -> tuple[np.ndarray, int]:
    final = run_dir / "Z_final.npy"
    epochs = _list_epochs(run_dir)
    if epoch is None:
        if final.exists() and not epochs:
            return np.load(final).astype(np.float32), -1
        if not epochs:
            raise FileNotFoundError(f"no embeddings under {run_dir}")
        epoch = epochs[-1]
    path = run_dir / "frames" / f"epoch_{int(epoch):04d}.npy"
    if not path.exists():
        matches = list((run_dir / "frames").glob(f"epoch_*{int(epoch)}.npy"))
        if not matches:
            raise FileNotFoundError(path)
        path = matches[0]
    return np.load(path).astype(np.float32), int(epoch)


def _as_display_rgb(images: np.ndarray) -> np.ndarray:
    """Normalize stored crops to uint8 RGB (N, H, W, 3) for the image route."""
    arr = np.asarray(images)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4, 5) and arr.shape[-1] not in (1, 3, 4):
        # (N, C, H, W) → take first 3 channels as RGB
        c = min(3, arr.shape[1])
        arr = np.transpose(arr[:, :c], (0, 2, 3, 1))
        if c == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif c == 2:
            z = np.zeros_like(arr[..., :1])
            arr = np.concatenate([arr, z], axis=-1)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if float(np.nanmax(a)) <= 1.5:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def _center_l2(
    feats: np.ndarray, *, mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center then row-L2-normalize. Returns (X_unit, feat_l2_of_centered)."""
    X = np.asarray(feats, dtype=np.float32)
    mu = X.mean(axis=0) if mean is None else np.asarray(mean, dtype=np.float32)
    Xc = X - mu.reshape(1, -1)
    feat_l2 = np.linalg.norm(Xc, axis=1).astype(np.float32)
    Xn = (Xc / np.clip(feat_l2[:, None], 1e-8, None)).astype(np.float32)
    return Xn, feat_l2


def _knn_from_features(
    feats: np.ndarray,
    *,
    k: int = 15,
    metric: str = "cosine",
    mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (knn_idx, knn_dist, feat_l2) after mean-center + L2 normalize."""
    from sklearn.neighbors import NearestNeighbors

    X, feat_l2 = _center_l2(feats, mean=mean)
    kk = min(int(k), len(X) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    return idx[:, 1:].astype(np.int64), dist[:, 1:].astype(np.float32), feat_l2


def load_bundle(zarr_path: Path, run_dir: Path, *, seed: int, holdout: float) -> dict:
    root = zarr.open_group(str(zarr_path), mode="r")
    # Prefer processed crops (CLAHE+Otsu) for display — matches L1 features.
    if "images_proc" in root:
        images = np.asarray(root["images_proc"])
    elif "images_clahe" in root:
        images = np.asarray(root["images_clahe"])
    else:
        images = np.asarray(root["images"])
    images = _as_display_rgb(images)
    labels = np.asarray(root["labels"]).astype(np.int64)
    cell_ids = np.asarray(root["cell_ids"]).astype(np.int64)
    attrs = dict(root.attrs)
    phases = tuple(attrs.get("phases", PHASES))
    n = int(images.shape[0])

    train_path = run_dir / "train_idx.npy"
    if train_path.exists():
        train_idx = np.load(train_path).astype(np.int64)
    else:
        train_idx = _train_idx(
            n,
            seed=int(attrs.get("seed", seed)),
            holdout=holdout,
        )

    feat_l2: np.ndarray | None = None
    if "features" in root:
        feats = np.asarray(root["features"], dtype=np.float32)
        # Center with the leanmap *train* mean so ‖x‖₂ matches the fit geometry.
        mu = feats[train_idx].mean(axis=0).astype(np.float32)
        _, feat_l2 = _center_l2(feats, mean=mu)
        if "knn_idx" in root and "knn_dist" in root:
            knn_idx = np.asarray(root["knn_idx"]).astype(np.int64)
            knn_dist = np.asarray(root["knn_dist"]).astype(np.float32)
            knn_label = "zarr"
        else:
            k = int(attrs.get("k", 15))
            knn_idx, knn_dist, _ = _knn_from_features(
                feats, k=k, metric="cosine", mean=mu
            )
            knn_label = "features/cosine (mean-center+L2)"
        print(
            f"features mean-centered on train (||μ||₂={float(np.linalg.norm(mu)):.4f}); "
            f"‖x-μ‖₂ med={float(np.median(feat_l2)):.3f}; knn={knn_label}"
        )
    elif "knn_idx" in root and "knn_dist" in root:
        knn_idx = np.asarray(root["knn_idx"]).astype(np.int64)
        knn_dist = np.asarray(root["knn_dist"]).astype(np.float32)
        knn_label = "zarr"
    else:
        knn_idx = np.zeros((n, 1), dtype=np.int64)
        knn_dist = np.zeros((n, 1), dtype=np.float32)
        knn_label = "none"
    attrs = {**attrs, "knn": knn_label}

    epochs = _list_epochs(run_dir)
    Z, epoch = _load_Z(run_dir, epochs[-1] if epochs else None)
    if len(Z) != len(train_idx):
        raise ValueError(
            f"embedding rows ({len(Z)}) != train_idx ({len(train_idx)}); "
            "pass matching --seed/--holdout or provide train_idx.npy"
        )

    phase_names = np.asarray([phases[int(i)] for i in labels], dtype=object)
    return {
        "images": images,
        "labels": labels,
        "cell_ids": cell_ids,
        "knn_idx": knn_idx,
        "knn_dist": knn_dist,
        "phases": phases,
        "phase_names": phase_names,
        "train_idx": train_idx,
        "epochs": epochs,
        "Z": Z,
        "epoch": epoch,
        "attrs": attrs,
        "zarr_path": zarr_path,
        "run_dir": run_dir,
        "feat_l2": feat_l2,
    }


def _point_train_index(point: dict) -> int | None:
    """Extract train-row index from a Plotly event point."""
    cd = point.get("customdata")
    if cd is None:
        return None
    # customdata may be a scalar, list, or nested list depending on Plotly version
    if isinstance(cd, (list, tuple, np.ndarray)):
        if len(cd) == 0:
            return None
        return int(cd[0])
    return int(cd)


def build_app(bundle: dict):
    import dash
    from dash import Dash, Input, Output, State, dcc, html, no_update
    import plotly.graph_objects as go
    from flask import send_file

    phases = bundle["phases"]
    train_idx = bundle["train_idx"]
    y_train = bundle["labels"][train_idx]
    names_train = bundle["phase_names"][train_idx]
    ids_train = bundle["cell_ids"][train_idx]
    images = bundle["images"]
    knn_idx = bundle["knn_idx"]
    knn_dist = bundle["knn_dist"]
    epochs = bundle["epochs"] or [bundle["epoch"]]
    colors_train = [PHASE_COLORS.get(str(n), "#888") for n in names_train]
    feat_l2 = bundle.get("feat_l2")
    feat_l2_train = (
        None if feat_l2 is None else np.asarray(feat_l2, dtype=np.float32)[train_idx]
    )
    color_options = [{"label": "Phase", "value": "phase"}]
    if feat_l2_train is not None:
        color_options.append(
            {"label": "Feature ‖x−μ‖₂ (train-centered)", "value": "feat_l2"}
        )

    Z_cache: dict[int, np.ndarray] = {bundle["epoch"]: bundle["Z"]}

    def get_Z(epoch: int) -> np.ndarray:
        if epoch not in Z_cache:
            Z_cache[epoch], _ = _load_Z(bundle["run_dir"], epoch)
        return Z_cache[epoch]

    def scatter_fig(
        epoch: int,
        selected: list[int] | None = None,
        *,
        color_by: str = "phase",
        heatmap_phases: list[str] | None = None,
    ) -> go.Figure:
        Z = get_Z(epoch)
        n = len(Z)
        l2 = (
            feat_l2_train[:n]
            if feat_l2_train is not None
            else np.full(n, np.nan, dtype=np.float32)
        )
        # JSON-safe python ints in customdata (int64 often drops in the browser)
        custom = np.column_stack(
            [
                np.arange(n, dtype=np.int32),
                train_idx[:n].astype(np.int32),
                ids_train[:n].astype(np.int32),
                l2.astype(np.float32),
            ]
        ).tolist()
        selected = [int(i) for i in (selected or []) if 0 <= int(i) < n]
        heat_phases = [str(p) for p in (heatmap_phases or []) if str(p) in phases]
        use_heat = len(heat_phases) > 0
        use_l2 = color_by == "feat_l2" and feat_l2_train is not None
        if use_l2:
            marker = dict(
                size=8,
                color=l2.astype(np.float64),
                colorscale="Viridis",
                opacity=0.9,
                line=dict(width=0),
                colorbar=dict(title="‖x−μ‖₂", thickness=14, len=0.7),
                cmin=float(np.nanpercentile(l2, 1)),
                cmax=float(np.nanpercentile(l2, 99)),
            )
            hover = (
                "%{text}<br>train=#%{customdata[0]}"
                "<br>zarr=%{customdata[1]}"
                "<br>cell_id=%{customdata[2]}"
                "<br>‖x−μ‖₂=%{customdata[3]:.3f}"
                "<br>z=(%{x:.3f}, %{y:.3f})<extra></extra>"
            )
        else:
            marker = dict(
                size=8,
                color=colors_train[:n],
                opacity=0.9,
                line=dict(width=0),
            )
            hover = (
                "%{text}<br>train=#%{customdata[0]}"
                "<br>zarr=%{customdata[1]}"
                "<br>cell_id=%{customdata[2]}"
                "<br>‖x−μ‖₂=%{customdata[3]:.3f}"
                "<br>z=(%{x:.3f}, %{y:.3f})<extra></extra>"
            )
        fig = go.Figure()
        # Shared axis range so heatmaps / points stay aligned across epochs
        pad = 0.05
        x0, x1 = float(Z[:, 0].min()), float(Z[:, 0].max())
        y0, y1 = float(Z[:, 1].min()), float(Z[:, 1].max())
        dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        xrange = [x0 - pad * dx, x1 + pad * dx]
        yrange = [y0 - pad * dy, y1 + pad * dy]
        nbins = 40
        for pname in heat_phases:
            mask = np.asarray(names_train[:n]) == pname
            if int(mask.sum()) < 3:
                continue
            fig.add_trace(
                go.Histogram2dContour(
                    x=Z[mask, 0].astype(np.float64),
                    y=Z[mask, 1].astype(np.float64),
                    name=f"{pname} density",
                    colorscale=_phase_heatmap_colorscale(
                        PHASE_COLORS.get(pname, "#888888")
                    ),
                    showscale=False,
                    contours=dict(coloring="heatmap", showlines=False),
                    ncontours=18,
                    histnorm="probability density",
                    xbins=dict(start=xrange[0], end=xrange[1], size=(xrange[1] - xrange[0]) / nbins),
                    ybins=dict(start=yrange[0], end=yrange[1], size=(yrange[1] - yrange[0]) / nbins),
                    hoverinfo="skip",
                    showlegend=True,
                    opacity=0.95,
                )
            )
        # Heatmap mode: density only (no cell markers)
        if not use_heat:
            fig.add_trace(
                go.Scattergl(
                    x=Z[:, 0].astype(np.float64),
                    y=Z[:, 1].astype(np.float64),
                    mode="markers",
                    name="cells",
                    customdata=custom,
                    text=[str(nm) for nm in names_train[:n]],
                    marker=marker,
                    selectedpoints=selected if selected else None,
                    selected=dict(marker=dict(size=12, opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.35)),
                    hovertemplate=hover,
                    showlegend=False,
                )
            )
        if not use_l2 and not use_heat:
            # legend swatches via dummy traces
            for pname in phases:
                fig.add_trace(
                    go.Scattergl(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=pname,
                        marker=dict(size=8, color=PHASE_COLORS.get(pname, "#888")),
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
        bits = []
        if use_l2:
            bits.append("color=‖x−μ‖₂")
        else:
            bits.append("color=phase")
        if use_heat:
            bits.append("heatmap=" + ",".join(heat_phases))
        title_extra = " · " + " · ".join(bits)
        heat_key = ",".join(heat_phases)
        fig.update_layout(
            title=f"CellCycle latent space — epoch {epoch} (N={n}){title_extra}",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(
                title="z1",
                zeroline=False,
                showgrid=True,
                gridcolor="#eee",
                range=xrange,
            ),
            yaxis=dict(
                title="z2",
                zeroline=False,
                showgrid=True,
                gridcolor="#eee",
                scaleanchor="x",
                scaleratio=1,
                range=yrange,
            ),
            plot_bgcolor="#fafafa",
            dragmode="lasso",
            clickmode="event+select",
            uirevision=f"epoch-{epoch}-{color_by}-{heat_key}",
            height=640,
        )
        return fig

    def cell_card(global_i: int, *, badge: str = "") -> html.Div:
        phase = str(bundle["phase_names"][global_i])
        cid = int(bundle["cell_ids"][global_i])
        kids = [
            html.Img(
                src=f"/img/{int(global_i)}.png?s=4",
                style={
                    "width": "132px",
                    "height": "132px",
                    "imageRendering": "pixelated",
                    "display": "block",
                    "background": "#111",
                },
            ),
            html.Div(phase, style={"fontWeight": 600, "marginTop": "4px"}),
            html.Div(f"id {cid}", style={"color": "#555", "fontSize": "12px"}),
            html.Div(f"row {global_i}", style={"color": "#888", "fontSize": "11px"}),
        ]
        if feat_l2 is not None:
            kids.append(
                html.Div(
                    f"‖x−μ‖₂={float(feat_l2[global_i]):.3f}",
                    style={"color": "#333", "fontSize": "11px"},
                )
            )
        if badge:
            kids.append(html.Div(badge, style={"color": "#333", "fontSize": "11px"}))
        return html.Div(
            kids,
            style={
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "8px",
                "background": "#fff",
                "width": "148px",
            },
        )

    app = Dash(__name__)
    app.title = "CellCycle latent explorer"

    @app.server.route("/img/<int:idx>.png")
    def serve_image(idx: int, s: int = 4):
        from flask import request

        if idx < 0 or idx >= len(images):
            return ("not found", 404)
        scale = int(request.args.get("s", 4))
        scale = max(1, min(scale, 8))
        arr = np.asarray(images[idx], dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("CellCycle latent explorer", style={"margin": "0 0 4px 0"}),
                    html.Div(
                        f"zarr={bundle['zarr_path'].name} · run={bundle['run_dir'].name} · "
                        f"train N={len(train_idx)} · knn={bundle['attrs'].get('knn', '?')}",
                        style={"color": "#666", "fontSize": "13px"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Epoch", style={"fontWeight": 600, "marginRight": "10px"}),
                    dcc.Slider(
                        id="epoch",
                        min=int(min(epochs)),
                        max=int(max(epochs)),
                        step=1,
                        value=int(bundle["epoch"] if bundle["epoch"] >= 0 else max(epochs)),
                        marks={
                            int(e): str(int(e))
                            for e in epochs
                            if e % max(1, len(epochs) // 8) == 0
                            or e in (epochs[0], epochs[-1])
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Label(
                        "Color by",
                        style={"fontWeight": 600, "marginRight": "10px"},
                    ),
                    dcc.Dropdown(
                        id="color-by",
                        options=color_options,
                        value="phase",
                        clearable=False,
                        style={"width": "280px"},
                    ),
                    html.Label(
                        "Class heatmap",
                        style={"fontWeight": 600, "marginLeft": "16px", "marginRight": "10px"},
                    ),
                    dcc.Dropdown(
                        id="heatmap-phases",
                        options=[{"label": p, "value": p} for p in phases],
                        value=[],
                        multi=True,
                        placeholder="Select class(es) for 2D density…",
                        style={"width": "360px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "8px",
                    "gap": "8px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="latent",
                            figure=scatter_fig(int(bundle["epoch"]), color_by="phase"),
                            config={"displayModeBar": True},
                        ),
                        style={"flex": "1.2", "minWidth": "480px"},
                    ),
                    html.Div(
                        [
                            html.H4("Selection", style={"marginTop": 0}),
                            html.Button("Clear", id="clear-sel", n_clicks=0, style={"marginBottom": "8px"}),
                            html.Div(id="sel-summary", style={"marginBottom": "8px", "color": "#444"}),
                            html.Div(
                                id="sel-gallery",
                                style={"display": "flex", "flexWrap": "wrap", "gap": "10px"},
                            ),
                            html.Hr(),
                            html.H4("Neighbors (full-set L1 knn)"),
                            html.Div(
                                id="nbr-gallery",
                                style={"display": "flex", "flexWrap": "wrap", "gap": "10px"},
                            ),
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "360px",
                            "maxHeight": "720px",
                            "overflowY": "auto",
                            "padding": "8px 12px",
                            "background": "#f7f7f7",
                            "borderRadius": "8px",
                            "border": "1px solid #e5e5e5",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
            ),
            dcc.Store(id="selected-train", data=[]),
        ],
        style={
            "fontFamily": "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
            "padding": "16px 20px",
            "maxWidth": "1400px",
            "margin": "0 auto",
        },
    )

    @app.callback(
        Output("selected-train", "data"),
        Input("latent", "selectedData"),
        Input("latent", "clickData"),
        Input("clear-sel", "n_clicks"),
        State("selected-train", "data"),
        prevent_initial_call=True,
    )
    def update_selection(selected_data, click_data, clear_clicks, current):
        from dash import ctx

        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if prop == "clear-sel.n_clicks":
            return []
        if prop == "latent.selectedData":
            if not selected_data or not selected_data.get("points"):
                return no_update
            idx = sorted(
                {
                    ti
                    for p in selected_data["points"]
                    if (ti := _point_train_index(p)) is not None
                }
            )
            return idx
        if prop == "latent.clickData":
            if not click_data or not click_data.get("points"):
                return no_update
            ti = _point_train_index(click_data["points"][0])
            if ti is None:
                return no_update
            cur = list(current or [])
            if ti in cur:
                return [x for x in cur if x != ti]
            return cur + [ti]
        return no_update

    @app.callback(
        Output("latent", "figure"),
        Input("epoch", "value"),
        Input("color-by", "value"),
        Input("heatmap-phases", "value"),
        State("selected-train", "data"),
    )
    def update_epoch_figure(epoch, color_by, heatmap_phases, selected):
        return scatter_fig(
            int(epoch),
            list(selected or []),
            color_by=str(color_by or "phase"),
            heatmap_phases=list(heatmap_phases or []),
        )

    @app.callback(
        Output("sel-summary", "children"),
        Output("sel-gallery", "children"),
        Output("nbr-gallery", "children"),
        Input("selected-train", "data"),
        Input("epoch", "value"),
    )
    def update_galleries(selected, _epoch):
        selected = [int(i) for i in (selected or [])]
        if not selected:
            return (
                "Click a point or lasso a region. Click again to toggle. Use Clear to reset.",
                [],
                [html.Div("Select a cell to see its L1 neighbors.", style={"color": "#888"})],
            )

        cards = []
        counts = {p: 0 for p in phases}
        for ti in selected[:48]:
            if ti < 0 or ti >= len(train_idx):
                continue
            gi = int(train_idx[ti])
            counts[phases[int(bundle["labels"][gi])]] += 1
            cards.append(cell_card(gi, badge=f"train #{ti}"))
        extra = f" (showing 48/{len(selected)})" if len(selected) > 48 else ""
        summary = (
            f"{len(selected)} selected{extra}: "
            + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        )

        gi0 = int(train_idx[selected[0]])
        nbrs = [cell_card(gi0, badge="query")]
        for j, (ni, nd) in enumerate(zip(knn_idx[gi0], knn_dist[gi0])):
            nbrs.append(cell_card(int(ni), badge=f"nn{j + 1}  d={float(nd):.3f}"))
        return summary, cards, nbrs

    # keep marker highlight in sync without wiping uirevision on every click
    @app.callback(
        Output("latent", "figure", allow_duplicate=True),
        Input("selected-train", "data"),
        State("epoch", "value"),
        State("color-by", "value"),
        State("heatmap-phases", "value"),
        prevent_initial_call=True,
    )
    def highlight_selection(selected, epoch, color_by, heatmap_phases):
        return scatter_fig(
            int(epoch),
            list(selected or []),
            color_by=str(color_by or "phase"),
            heatmap_phases=list(heatmap_phases or []),
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    bundle = load_bundle(
        args.zarr.expanduser().resolve(),
        args.run_dir.expanduser().resolve(),
        seed=args.seed,
        holdout=args.holdout,
    )
    print(
        f"loaded N_full={int(bundle['images'].shape[0])} N_train={len(bundle['train_idx'])} "
        f"epochs={bundle['epochs'][:3]}…{bundle['epochs'][-1:] if bundle['epochs'] else []} "
        f"current_epoch={bundle['epoch']}"
    )
    app = build_app(bundle)
    app.run(host=args.host, port=int(args.port), debug=bool(args.debug))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Dash explorer for pistachio FTIR leanmap embeddings.

Three linked views:
  • latent scatter (scrubbable by epoch)
  • spatial map coloured by a chosen spectral band
  • spectrum of the current selection

    python examples/pistachio_ftir_explorer.py
    python examples/pistachio_ftir_explorer.py --port 8052 \\
        --run-dir examples/out/pistachio_ftir_cosine

Requires: dash, plotly, zarr, numpy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZARR = ROOT / "examples" / "GD_Pistachio_Stem_Ctr_whole_10um__ftir.zarr"
DEFAULT_RUN = ROOT / "examples" / "out" / "pistachio_ftir_cosine"


def _list_epochs(run_dir: Path) -> list[int]:
    frames = sorted((run_dir / "frames").glob("epoch_*.npy"))
    out: list[int] = []
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


def _train_idx(n: int, *, seed: int, holdout: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n_cal = max(1, int(round(float(holdout) * n)))
    perm = rng.permutation(n)
    return perm[n_cal:].astype(np.int64)


def _to_absorbance(spec: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(spec, dtype=np.float32), 1e-3, None) / 100.0
    return (-np.log10(t)).astype(np.float32)


def load_bundle(
    zarr_path: Path,
    run_dir: Path,
    *,
    seed: int,
    holdout: float,
    absorbance: bool,
) -> dict:
    root = zarr.open_group(str(zarr_path), mode="r")
    wn = np.asarray(root["wavenumbers"][:], dtype=np.float64)
    x = np.asarray(root["x"][:], dtype=np.float64)
    y = np.asarray(root["y"][:], dtype=np.float64)
    row = np.asarray(root["row"][:], dtype=np.int32)
    col = np.asarray(root["col"][:], dtype=np.int32)
    n = int(len(x))
    attrs = dict(root.attrs)
    ny = int(attrs.get("grid_shape_yx", [int(row.max()) + 1, int(col.max()) + 1])[0])
    nx = int(attrs.get("grid_shape_yx", [ny, int(col.max()) + 1])[1])

    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        absorbance = bool(meta.get("absorbance", absorbance))
        holdout = float(meta.get("holdout", holdout)) if "holdout" in meta else holdout

    train_path = run_dir / "train_idx.npy"
    if train_path.exists():
        train_idx = np.load(train_path).astype(np.int64)
    else:
        train_idx = _train_idx(n, seed=seed, holdout=holdout)

    epochs = _list_epochs(run_dir)
    Z, epoch = _load_Z(run_dir, epochs[-1] if epochs else None)
    if len(Z) != len(train_idx):
        raise ValueError(
            f"embedding rows ({len(Z)}) != train_idx ({len(train_idx)}); "
            "pass matching --seed/--holdout or provide train_idx.npy"
        )

    Z_umap = None
    umap_meta = {}
    umap_path = run_dir / "Z_umap.npy"
    if umap_path.exists():
        Z_umap = np.load(umap_path).astype(np.float32)
        if len(Z_umap) != len(train_idx):
            raise ValueError(
                f"Z_umap rows ({len(Z_umap)}) != train_idx ({len(train_idx)})"
            )
        umap_meta_path = run_dir / "umap_meta.json"
        if umap_meta_path.exists():
            umap_meta = json.loads(umap_meta_path.read_text())

    # µm from OMNIC units (10000 = 10 µm)
    x_um = (x / 1000.0).astype(np.float32)
    y_um = (y / 1000.0).astype(np.float32)

    return {
        "root": root,
        "wn": wn,
        "x_um": x_um,
        "y_um": y_um,
        "row": row,
        "col": col,
        "ny": ny,
        "nx": nx,
        "train_idx": train_idx,
        "epochs": epochs,
        "Z": Z,
        "epoch": epoch,
        "Z_umap": Z_umap,
        "umap_meta": umap_meta,
        "absorbance": absorbance,
        "zarr_path": zarr_path,
        "run_dir": run_dir,
        "attrs": attrs,
        "meta": meta,
        "band_cache": {},
    }


def _band_index(wn: np.ndarray, cm: float) -> int:
    return int(np.argmin(np.abs(wn - float(cm))))


def _load_band(bundle: dict, cm: float) -> np.ndarray:
    """Band intensities for all pixels (absorbance or %T). Cached by index."""
    i = _band_index(bundle["wn"], cm)
    cache = bundle["band_cache"]
    if i in cache:
        return cache[i]
    col = np.asarray(bundle["root"]["spectra"][:, i], dtype=np.float32)
    if bundle["absorbance"]:
        col = _to_absorbance(col)
    cache[i] = col
    # keep cache small
    if len(cache) > 12:
        cache.pop(next(iter(cache)))
    return col


def _load_spectrum(bundle: dict, global_i: int) -> np.ndarray:
    spec = np.asarray(bundle["root"]["spectra"][int(global_i)], dtype=np.float32)
    if bundle["absorbance"]:
        return _to_absorbance(spec)
    return spec


def _point_train_index(point: dict) -> int | None:
    cd = point.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, (list, tuple, np.ndarray)):
        if len(cd) == 0:
            return None
        return int(cd[0])
    return int(cd)


def _view_from_relayout(relayout: dict | None, prev: dict | None = None) -> dict | None:
    """Extract axis ranges from a Plotly relayout event (merge with previous)."""
    if not relayout:
        return prev
    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return None  # double-click reset
    out = dict(prev or {})
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        out["xaxis"] = [float(relayout["xaxis.range[0]"]), float(relayout["xaxis.range[1]"])]
    elif "xaxis.range" in relayout:
        r = relayout["xaxis.range"]
        out["xaxis"] = [float(r[0]), float(r[1])]
    if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
        out["yaxis"] = [float(relayout["yaxis.range[0]"]), float(relayout["yaxis.range[1]"])]
    elif "yaxis.range" in relayout:
        r = relayout["yaxis.range"]
        out["yaxis"] = [float(r[0]), float(r[1])]
    return out or prev


def _apply_view(fig, view: dict | None) -> None:
    """Re-apply stored zoom/pan so full figure rebuilds do not reset the viewport."""
    if not view:
        return
    if "xaxis" in view:
        fig.update_layout(xaxis_range=list(view["xaxis"]), xaxis_autorange=False)
    if "yaxis" in view:
        fig.update_layout(yaxis_range=list(view["yaxis"]), yaxis_autorange=False)


def build_app(bundle: dict):
    import dash
    from dash import Dash, Input, Output, State, dcc, html, no_update
    import plotly.graph_objects as go

    train_idx = bundle["train_idx"]
    wn = bundle["wn"]
    x_um = bundle["x_um"]
    y_um = bundle["y_um"]
    row = bundle["row"]
    col = bundle["col"]
    ny, nx = bundle["ny"], bundle["nx"]
    unit = "A" if bundle["absorbance"] else "%T"
    epochs = bundle["epochs"] or [bundle["epoch"]]
    Z_cache: dict[int, np.ndarray] = {bundle["epoch"]: bundle["Z"]}
    Z_umap = bundle.get("Z_umap")
    has_umap = Z_umap is not None

    # spatial axis in µm for full grid
    xs_um = (np.arange(nx, dtype=np.float64) * 10.0).astype(np.float32)
    ys_um = (np.arange(ny, dtype=np.float64) * 10.0).astype(np.float32)

    def get_Z(epoch: int, method: str = "leanmap") -> np.ndarray:
        if method == "umap":
            if Z_umap is None:
                raise FileNotFoundError("Z_umap.npy not found in run dir")
            return Z_umap
        if epoch not in Z_cache:
            Z_cache[epoch], _ = _load_Z(bundle["run_dir"], epoch)
        return Z_cache[epoch]

    def _clim(vals: np.ndarray) -> tuple[float, float]:
        lo, hi = np.nanpercentile(vals, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1e-6
        return float(lo), float(hi)

    def latent_fig(
        epoch: int,
        band_cm: float,
        selected: list[int] | None = None,
        view: dict | None = None,
        method: str = "leanmap",
    ) -> go.Figure:
        method = "umap" if method == "umap" else "leanmap"
        Z = get_Z(epoch, method=method)
        n = len(Z)
        band = _load_band(bundle, band_cm)[train_idx[:n]]
        cmin, cmax = _clim(band)
        selected = [int(i) for i in (selected or []) if 0 <= int(i) < n]
        custom = np.arange(n, dtype=np.int32).tolist()
        fig = go.Figure(
            go.Scattergl(
                x=Z[:, 0].astype(np.float64),
                y=Z[:, 1].astype(np.float64),
                mode="markers",
                name="latent",
                customdata=custom,
                marker=dict(
                    size=6,
                    color=band.astype(np.float64),
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.85,
                    line=dict(width=0),
                    colorbar=dict(
                        title=f"{unit}({band_cm:.0f})",
                        thickness=12,
                        len=0.65,
                    ),
                ),
                selectedpoints=selected if selected else None,
                selected=dict(marker=dict(size=11, opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.25)),
                hovertemplate=(
                    "train=#%{customdata}"
                    "<br>z=(%{x:.3f}, %{y:.3f})"
                    f"<br>{unit}={band_cm:.0f}: %{{marker.color:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
        if method == "umap":
            title = f"UMAP (cosine) — N={n} · colour {unit}({band_cm:.0f} cm⁻¹)"
        else:
            title = (
                f"leanmap — epoch {epoch} · N={n} · colour {unit}({band_cm:.0f} cm⁻¹)"
            )
        # No scaleanchor: it fights zoom persistence on Scattergl rebuilds.
        # Separate uirevision per method so switching embeddings doesn't keep
        # a stale zoom from the other layout.
        fig.update_layout(
            title=title,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(
                title="z1",
                zeroline=False,
                showgrid=True,
                gridcolor="#eee",
            ),
            yaxis=dict(
                title="z2",
                zeroline=False,
                showgrid=True,
                gridcolor="#eee",
            ),
            plot_bgcolor="#fafafa",
            dragmode="lasso",
            clickmode="event+select",
            uirevision=f"pistachio-latent-{method}",
            height=520,
        )
        _apply_view(fig, view)
        return fig

    def spatial_fig(
        band_cm: float,
        selected: list[int] | None = None,
        view: dict | None = None,
    ) -> go.Figure:
        band_all = _load_band(bundle, band_cm)
        grid = np.full((ny, nx), np.nan, dtype=np.float32)
        grid[row, col] = band_all
        cmin, cmax = _clim(band_all)
        fig = go.Figure(
            go.Heatmap(
                z=grid.astype(np.float64),
                x=xs_um.astype(np.float64),
                y=ys_um.astype(np.float64),
                colorscale="Viridis",
                zmin=cmin,
                zmax=cmax,
                colorbar=dict(title=f"{unit}({band_cm:.0f})", thickness=12, len=0.65),
                hovertemplate=(
                    "x=%{x:.0f} µm<br>y=%{y:.0f} µm"
                    f"<br>{unit}={band_cm:.0f}: %{{z:.3f}}"
                    "<extra></extra>"
                ),
                name="map",
            )
        )
        # train points for selection (invisible until selected, then ring markers)
        n = len(train_idx)
        custom = np.arange(n, dtype=np.int32)
        sel = [int(i) for i in (selected or []) if 0 <= int(i) < n]
        # all train as faint clickable overlay
        fig.add_trace(
            go.Scattergl(
                x=x_um[train_idx].astype(np.float64),
                y=y_um[train_idx].astype(np.float64),
                mode="markers",
                name="train",
                customdata=custom.tolist(),
                marker=dict(size=5, color="rgba(255,255,255,0.05)"),
                selectedpoints=sel if sel else None,
                selected=dict(marker=dict(size=11, color="#ff5050", opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.0)),
                hovertemplate=(
                    "train=#%{customdata}"
                    "<br>x=%{x:.0f} µm · y=%{y:.0f} µm"
                    "<extra></extra>"
                ),
            )
        )
        if sel:
            fig.add_trace(
                go.Scatter(
                    x=x_um[train_idx[sel]].astype(np.float64),
                    y=y_um[train_idx[sel]].astype(np.float64),
                    mode="markers",
                    name="selected",
                    marker=dict(
                        size=11,
                        color="#ff5050",
                        line=dict(width=1.2, color="#111"),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.update_layout(
            title=f"Spatial map — {unit}({band_cm:.0f} cm⁻¹)",
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(title="x (µm)", constrain="domain"),
            yaxis=dict(
                title="y (µm)",
                scaleanchor="x",
                scaleratio=1,
                constrain="domain",
            ),
            plot_bgcolor="#111",
            dragmode="lasso",
            clickmode="event+select",
            uirevision="pistachio-spatial",
            height=520,
        )
        _apply_view(fig, view)
        return fig

    def spectrum_fig(selected: list[int] | None, band_cm: float) -> go.Figure:
        selected = [int(i) for i in (selected or []) if 0 <= int(i) < len(train_idx)]
        fig = go.Figure()
        if not selected:
            fig.update_layout(
                title="Spectrum — click a point in latent or spatial",
                xaxis_title="wavenumber (cm⁻¹)",
                yaxis_title=unit,
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
                plot_bgcolor="#fafafa",
                annotations=[
                    dict(
                        text="No selection",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="#888", size=14),
                    )
                ],
            )
            return fig

        # mean of selection + up to 8 individuals
        specs = []
        for ti in selected[:48]:
            specs.append(_load_spectrum(bundle, int(train_idx[ti])))
        specs_a = np.stack(specs, axis=0)
        mean = specs_a.mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=wn,
                y=mean.astype(np.float64),
                mode="lines",
                name=f"mean (n={len(selected)})",
                line=dict(color="#1f4e79", width=2.2),
            )
        )
        show = selected[:6]
        palette = ["#e45756", "#4c78a8", "#54a24b", "#f58518", "#b279a2", "#72b7b2"]
        for k, ti in enumerate(show):
            if len(selected) == 1:
                continue
            fig.add_trace(
                go.Scatter(
                    x=wn,
                    y=specs_a[k].astype(np.float64),
                    mode="lines",
                    name=f"train #{ti}",
                    line=dict(color=palette[k % len(palette)], width=1),
                    opacity=0.55,
                )
            )
        if len(selected) == 1:
            fig.data[0].name = f"train #{selected[0]}"
            fig.data[0].line = dict(color="#1f4e79", width=1.8)

        # band marker
        fig.add_vline(
            x=float(band_cm),
            line_width=1,
            line_dash="dot",
            line_color="#c44",
            annotation_text=f"{band_cm:.0f}",
            annotation_position="top",
        )
        fig.update_layout(
            title=f"Spectrum — {len(selected)} selected · {unit}",
            xaxis=dict(title="wavenumber (cm⁻¹)", autorange="reversed"),
            yaxis=dict(title=unit),
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor="#fafafa",
            legend=dict(orientation="h", y=1.12),
            uirevision="spectrum",
        )
        return fig

    app = Dash(__name__)
    app.title = "Pistachio FTIR explorer"

    default_band = 2920.0
    ep0 = int(bundle["epoch"] if bundle["epoch"] >= 0 else max(epochs))

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Pistachio FTIR explorer", style={"margin": "0 0 4px 0"}),
                    html.Div(
                        f"zarr={bundle['zarr_path'].name} · run={bundle['run_dir'].name} · "
                        f"train N={len(train_idx)} · {'absorbance' if bundle['absorbance'] else '%T'} · "
                        f"bands={len(wn)} ({wn[0]:.0f}–{wn[-1]:.0f} cm⁻¹)",
                        style={"color": "#666", "fontSize": "13px"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Embedding", style={"fontWeight": 600}),
                            dcc.Dropdown(
                                id="embed-method",
                                options=(
                                    [
                                        {"label": "leanmap", "value": "leanmap"},
                                        {
                                            "label": "UMAP (cosine)",
                                            "value": "umap",
                                            "disabled": not has_umap,
                                        },
                                    ]
                                ),
                                value="leanmap",
                                clearable=False,
                                style={"width": "200px"},
                            ),
                            html.Div(
                                (
                                    "UMAP ready"
                                    if has_umap
                                    else "UMAP pending (Z_umap.npy)"
                                ),
                                style={
                                    "color": "#2a7" if has_umap else "#a60",
                                    "fontSize": "12px",
                                    "marginTop": "4px",
                                },
                            ),
                        ],
                        style={"minWidth": "200px"},
                    ),
                    html.Div(
                        [
                            html.Label("Epoch", style={"fontWeight": 600}),
                            dcc.Slider(
                                id="epoch",
                                min=int(min(epochs)),
                                max=int(max(epochs)),
                                step=1,
                                value=ep0,
                                marks={
                                    int(e): str(int(e))
                                    for e in epochs
                                    if e % max(1, len(epochs) // 8) == 0
                                    or e in (epochs[0], epochs[-1])
                                },
                                tooltip={"placement": "bottom"},
                                disabled=False,
                            ),
                        ],
                        id="epoch-wrap",
                        style={"flex": "1", "minWidth": "280px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Band (cm⁻¹)",
                                style={"fontWeight": 600, "marginRight": "8px"},
                            ),
                            dcc.Input(
                                id="band-cm",
                                type="number",
                                value=default_band,
                                min=float(wn.min()),
                                max=float(wn.max()),
                                step=1.0,
                                style={"width": "100px"},
                            ),
                            dcc.Slider(
                                id="band-slider",
                                min=float(wn.min()),
                                max=float(wn.max()),
                                step=1.0,
                                value=default_band,
                                marks={
                                    int(v): str(int(v))
                                    for v in np.linspace(wn.min(), wn.max(), 6)
                                },
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "1.4", "minWidth": "320px"},
                    ),
                    html.Button(
                        "Refresh epochs",
                        id="refresh-epochs",
                        n_clicks=0,
                        style={"alignSelf": "flex-end", "marginBottom": "8px"},
                    ),
                    html.Button(
                        "Clear selection",
                        id="clear-sel",
                        n_clicks=0,
                        style={"alignSelf": "flex-end", "marginBottom": "8px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                id="sel-summary",
                style={"color": "#444", "fontSize": "13px", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="latent", config={"displayModeBar": True}),
                        style={"flex": "1", "minWidth": "420px"},
                    ),
                    html.Div(
                        dcc.Graph(id="spatial", config={"displayModeBar": True}),
                        style={"flex": "1", "minWidth": "420px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="spectrum", config={"displayModeBar": True}),
            dcc.Store(id="selected-train", data=[]),
            dcc.Store(id="band-store", data=default_band),
            dcc.Store(id="latent-view", data=None),
            dcc.Store(id="spatial-view", data=None),
            dcc.Store(
                id="latent-views",
                data={"leanmap": None, "umap": None},
            ),
            dcc.Interval(id="tick", interval=8000, n_intervals=0),
        ],
        style={
            "fontFamily": "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
            "padding": "16px 20px",
            "maxWidth": "1500px",
            "margin": "0 auto",
        },
    )

    @app.callback(
        Output("band-store", "data"),
        Output("band-cm", "value"),
        Output("band-slider", "value"),
        Input("band-cm", "value"),
        Input("band-slider", "value"),
    )
    def sync_band(cm_input, cm_slider):
        from dash import ctx

        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else "band-cm.value"
        if prop.startswith("band-slider"):
            v = float(cm_slider if cm_slider is not None else default_band)
        else:
            v = float(cm_input if cm_input is not None else default_band)
        v = float(np.clip(v, wn.min(), wn.max()))
        v = float(wn[_band_index(wn, v)])
        return v, v, v

    @app.callback(
        Output("epoch", "min"),
        Output("epoch", "max"),
        Output("epoch", "marks"),
        Output("epoch", "value"),
        Input("refresh-epochs", "n_clicks"),
        Input("tick", "n_intervals"),
        State("epoch", "value"),
    )
    def refresh_epochs(_btn, _tick, cur):
        from dash import ctx

        eps = _list_epochs(bundle["run_dir"])
        if not eps:
            eps = list(epochs)
        bundle["epochs"] = eps
        marks = {
            int(e): str(int(e))
            for e in eps
            if e % max(1, len(eps) // 8) == 0 or e in (eps[0], eps[-1])
        }
        cur_i = int(cur) if cur is not None else eps[-1]
        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        # Follow the tip only when the user asks, or was already on the previous tip.
        if prop.startswith("refresh-epochs") or cur_i >= max(eps) - 1:
            new_i = eps[-1]
        elif cur_i not in eps:
            new_i = eps[-1]
        else:
            new_i = cur_i
        # Avoid re-firing figure callbacks when nothing changed.
        val_out = int(new_i) if new_i != cur_i else no_update
        return int(min(eps)), int(max(eps)), marks, val_out

    @app.callback(
        Output("latent-views", "data"),
        Output("latent-view", "data"),
        Input("latent", "relayoutData"),
        Input("embed-method", "value"),
        State("latent-views", "data"),
        prevent_initial_call=True,
    )
    def store_latent_view(relayout, method, views):
        from dash import ctx

        views = dict(views or {"leanmap": None, "umap": None})
        method = "umap" if method == "umap" else "leanmap"
        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if prop.startswith("latent.relayoutData"):
            views[method] = _view_from_relayout(relayout, views.get(method))
        return views, views.get(method)

    @app.callback(
        Output("spatial-view", "data"),
        Input("spatial", "relayoutData"),
        State("spatial-view", "data"),
        prevent_initial_call=True,
    )
    def store_spatial_view(relayout, prev):
        return _view_from_relayout(relayout, prev)

    @app.callback(
        Output("epoch", "disabled"),
        Input("embed-method", "value"),
    )
    def toggle_epoch(method):
        return str(method) == "umap"

    @app.callback(
        Output("selected-train", "data"),
        Input("latent", "selectedData"),
        Input("latent", "clickData"),
        Input("spatial", "selectedData"),
        Input("spatial", "clickData"),
        Input("clear-sel", "n_clicks"),
        State("selected-train", "data"),
        prevent_initial_call=True,
    )
    def update_selection(lat_sel, lat_click, sp_sel, sp_click, _clear, current):
        from dash import ctx

        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if prop == "clear-sel.n_clicks":
            return []

        def from_sel(data):
            if not data or not data.get("points"):
                return None
            return sorted(
                {
                    ti
                    for p in data["points"]
                    if (ti := _point_train_index(p)) is not None
                }
            )

        def from_click(data):
            if not data or not data.get("points"):
                return None
            return _point_train_index(data["points"][0])

        if prop in ("latent.selectedData", "spatial.selectedData"):
            data = lat_sel if prop.startswith("latent") else sp_sel
            idx = from_sel(data)
            return idx if idx is not None else no_update

        if prop in ("latent.clickData", "spatial.clickData"):
            data = lat_click if prop.startswith("latent") else sp_click
            ti = from_click(data)
            if ti is None:
                return no_update
            cur = list(current or [])
            if ti in cur:
                return [x for x in cur if x != ti]
            return cur + [ti]
        return no_update

    @app.callback(
        Output("latent", "figure"),
        Output("spatial", "figure"),
        Output("spectrum", "figure"),
        Output("sel-summary", "children"),
        Input("epoch", "value"),
        Input("band-store", "data"),
        Input("selected-train", "data"),
        Input("embed-method", "value"),
        State("spatial-view", "data"),
        State("latent-views", "data"),
    )
    def update_figures(epoch, band_cm, selected, method, spatial_view, latent_views):
        epoch = int(epoch)
        band_cm = float(band_cm)
        method = "umap" if method == "umap" else "leanmap"
        selected = list(selected or [])
        i_wn = _band_index(wn, band_cm)
        actual = float(wn[i_wn])
        views = latent_views or {}
        view = views.get(method)
        lat = latent_fig(
            epoch, actual, selected, view=view, method=method
        )
        spat = spatial_fig(actual, selected, view=spatial_view)
        spec = spectrum_fig(selected, actual)
        if selected:
            gi = int(train_idx[selected[0]])
            summary = (
                f"{method} · {len(selected)} selected · first train #{selected[0]} "
                f"(zarr row {gi}, x={x_um[gi]:.0f} µm, y={y_um[gi]:.0f} µm) · "
                f"band {actual:.1f} cm⁻¹"
            )
        else:
            summary = (
                f"{method} · click or lasso in latent / spatial. "
                "Band colour updates both maps; spectrum shows the selection mean. "
                "Double-click a plot to reset zoom."
            )
        return lat, spat, spec, summary

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument(
        "--no-absorbance",
        action="store_true",
        help="colour/plot %%T (default follows training: absorbance)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8052)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    bundle = load_bundle(
        args.zarr.expanduser().resolve(),
        args.run_dir.expanduser().resolve(),
        seed=args.seed,
        holdout=args.holdout,
        absorbance=not args.no_absorbance,
    )
    print(
        f"loaded N={len(bundle['x_um'])} N_train={len(bundle['train_idx'])} "
        f"epochs={bundle['epochs'][:3]}…{bundle['epochs'][-1:] if bundle['epochs'] else []} "
        f"epoch={bundle['epoch']} absorbance={bundle['absorbance']}"
    )
    app = build_app(bundle)
    app.run(host=args.host, port=int(args.port), debug=bool(args.debug))


if __name__ == "__main__":
    main()

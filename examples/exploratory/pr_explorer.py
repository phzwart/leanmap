#!/usr/bin/env python
"""Interactive P(r) explorer: a self-contained Plotly page.

Left panel is the embedding; lasso a region and the right panel shows the
P(r) profiles of the selected points. Two display modes:

``curves``  every selected profile drawn individually
``bands``   median with a 5-95 percentile band, plus the mean

Any number of lassos can be stored as named, colour-coded selections that stay
on the plot and are overlaid in the right panel, so regions can be compared
against each other. Stored selections survive a page reload (localStorage) and
export to CSV with a group column.

The grey reference band in both modes is the whole population, so a selection
is always read against the background distribution rather than in isolation.

A 3-D run gets two extra controls. ``view`` switches between the pairwise
projections and a rotatable 3-D scatter; ``slab`` narrows a projection to an
equal-count section of the out-of-plane axis. Plotly has no lasso in 3-D, so
selection happens in a projection -- and the slab matters there, since looking
down an axis superimposes points that are far apart along it and a lasso would
otherwise sweep them up together.

Everything (data and plotly.js) is inlined into one HTML file, so there is no
server to keep alive and the page keeps working offline.

Usage::

    python examples/exploratory/pr_explorer.py --run runs/sasbdb_pr_l1_frozen
    python examples/exploratory/pr_explorer.py --run runs/sasbdb_pr_umap --no-open
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

# P(r) values are ~1e-3..4e-2; fixed-point ints keep the payload compact and
# still resolve five significant digits.
SCALE = 1_000_000

CSCALE = {
    "mode_pos": "Plasma",
    "dmax": "Viridis",
    "rg_over_dmax": "RdBu",
    "skew": "Cividis",
}
LABELS = {
    "mode_pos": "peak position r/Dmax",
    "dmax": "log10 Dmax",
    "rg_over_dmax": "Rg / Dmax",
    "skew": "P(r) skewness",
}


def color_panels(X: np.ndarray, meta: pd.DataFrame):
    dmax = meta["dmax"].to_numpy(dtype=np.float64)
    rg = meta["rg_pr"].to_numpy(dtype=np.float64)
    bins = np.linspace(0.0, 1.0, X.shape[1])
    w = X / X.sum(axis=1, keepdims=True)
    mean_pos = w @ bins
    return {
        "mode_pos": bins[np.argmax(X, axis=1)],
        "dmax": np.log10(dmax),
        "rg_over_dmax": rg / dmax,
        "skew": w @ (bins**3) - 3 * mean_pos * (w @ (bins**2)) + 2 * mean_pos**3,
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script>__PLOTLYJS__</script>
<style>
 body{margin:0;font:13px -apple-system,Helvetica,Arial,sans-serif;color:#222}
 #bar{display:flex;align-items:center;gap:16px;padding:7px 14px;border-bottom:1px solid #ddd;
      background:#fafafa;flex-wrap:wrap}
 #bar b{font-weight:600}
 #groups{display:flex;align-items:center;gap:8px;padding:6px 14px;border-bottom:1px solid #eee;
         min-height:26px;flex-wrap:wrap;background:#fff}
 .chip{display:flex;align-items:center;gap:6px;border:1px solid #ddd;border-radius:14px;
       padding:2px 8px;background:#fdfdfd}
 .chip .sw{width:11px;height:11px;border-radius:50%;flex:none}
 .chip input{border:none;background:transparent;font:inherit;width:78px;outline:none}
 .chip .n{color:#777;font-size:11px}
 .chip button{border:none;background:transparent;cursor:pointer;padding:0 2px;font-size:13px;
              line-height:1;color:#888}
 .chip button:hover{color:#000}
 .chip.off{opacity:0.4}
 #wrap{display:flex;width:100%}
 #left,#right{width:50%;height:calc(100vh - 88px)}
 label{cursor:pointer}
 button.act{font:inherit;padding:3px 10px;cursor:pointer}
 #count{color:#666}
 #hint{color:#999;font-size:12px}
</style></head><body>
<div id="bar">
  <b>__TITLE__</b>
  <span>right panel:
    <label><input type="radio" name="mode" value="bands" checked> bands</label>
    <label><input type="radio" name="mode" value="curves"> curves</label>
  </span>
  <span>colour: <select id="colour"></select></span>
  <span id="viewwrap" style="display:none">view: <select id="view"></select></span>
  <span id="slabwrap" style="display:none">slab: <select id="slab"></select></span>
  <button class="act" id="add">+ store selection</button>
  <button class="act" id="clearsel">clear lasso</button>
  <button class="act" id="clearall">remove all</button>
  <button class="act" id="dl">download CSV</button>
  <span id="count"></span>
</div>
<div id="groups"><span id="hint">stored selections appear here</span></div>
<div id="wrap"><div id="left"></div><div id="right"></div></div>
<script>
const D = __DATA__;
const NB = D.nb, N = D.codes.length;
const R = Array.from({length: NB}, (_, j) => j / (NB - 1));
const PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                 "#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f"];
const STORE_KEY = "prexp:__TITLE__";

let sel = [];          // current, unstored lasso
let groups = [];       // [{name, colour, idx, visible}]
let mode = "bands", colourKey = "mode_pos", nextId = 1;

/* ---------- 3-D handling ----------
   Plotly has no lasso in 3-D, so a rotatable view can orient but not select.
   Selection therefore happens in a 2-D projection, optionally narrowed to an
   equal-count slab of the out-of-plane axis: that removes the superposition
   which would otherwise let a lasso sweep up points that are far apart along
   the axis you are looking down. */
const DIM = D.z[0].length;
const NSLAB = 6;
let view = "01";       // "ij" axis pair, or "3d"
let slab = -1;         // -1 = whole depth
let slabCache = {};

function pairAxes() { return [+view[0], +view[1]]; }
function outAxis() {
  const [i, j] = pairAxes();
  for (let k = 0; k < DIM; k++) if (k !== i && k !== j) return k;
  return -1;
}
function slabSets(ax) {
  if (slabCache[ax]) return slabCache[ax];
  const order = Array.from({length: N}, (_, i) => i)
                     .sort((a, b) => D.z[a][ax] - D.z[b][ax]);
  const sets = [], per = Math.ceil(N / NSLAB);
  for (let s = 0; s < NSLAB; s++) sets.push(new Set(order.slice(s * per, (s + 1) * per)));
  slabCache[ax] = sets;
  return sets;
}
function inSlab(i) {
  if (slab < 0 || DIM < 3 || view === "3d") return true;
  return slabSets(outAxis())[slab].has(i);
}

function val(i, j) { return D.x[i * NB + j] / D.scale; }
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return "rgba(" + (n >> 16) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
}
function quantile(s, p) {
  const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
}
function bands(idxs) {
  const p5 = [], p50 = [], p95 = [], mu = [];
  const buf = new Float64Array(idxs.length);
  for (let j = 0; j < NB; j++) {
    let acc = 0;
    for (let t = 0; t < idxs.length; t++) { buf[t] = val(idxs[t], j); acc += buf[t]; }
    const s = Array.from(buf).sort((a, b) => a - b);
    p5.push(quantile(s, 0.05)); p50.push(quantile(s, 0.5)); p95.push(quantile(s, 0.95));
    mu.push(acc / idxs.length);
  }
  return {p5, p50, p95, mu};
}
function thin(idxs, cap) {
  if (idxs.length <= cap) return idxs;
  const out = [], step = idxs.length / cap;
  for (let t = 0; t < cap; t++) out.push(idxs[Math.floor(t * step)]);
  return out;
}

/* ---------- persistence ---------- */
function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({groups: groups, nextId: nextId}));
  } catch (e) { /* quota or private mode: selections stay session-only */ }
}
function load() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (s && Array.isArray(s.groups)) { groups = s.groups; nextId = s.nextId || groups.length + 1; }
  } catch (e) { groups = []; }
}

/* ---------- left panel ---------- */
function leftLayout() {
  if (view === "3d") {
    return {
      margin: {l: 0, r: 0, t: 30, b: 0}, showlegend: false,
      scene: {aspectmode: "data",
              xaxis: {title: {text: "dim 1"}}, yaxis: {title: {text: "dim 2"}},
              zaxis: {title: {text: "dim 3"}}},
      title: {text: "drag to rotate — no lasso in 3-D; switch to a pair to select",
              font: {size: 13}}
    };
  }
  const [i, j] = pairAxes();
  const depth = DIM > 2
    ? (slab < 0 ? ", full depth of dim " + (outAxis() + 1)
                : ", slab " + (slab + 1) + "/" + NSLAB + " of dim " + (outAxis() + 1))
    : "";
  return {
    margin: {l: 20, r: 10, t: 30, b: 20}, dragmode: "lasso", hovermode: "closest",
    showlegend: false,
    xaxis: {showticklabels: false, zeroline: false, title: {text: "dim " + (i + 1)}},
    yaxis: {showticklabels: false, zeroline: false, scaleanchor: "x", scaleratio: 1,
            title: {text: "dim " + (j + 1)}},
    title: {text: "lasso a region, then '+ store selection'" + depth, font: {size: 13}}
  };
}

function xyz(idxs, ax) {
  const t = {x: idxs.map(i => D.z[i][ax[0]]), y: idxs.map(i => D.z[i][ax[1]])};
  if (ax.length > 2) t.z = idxs.map(i => D.z[i][ax[2]]);
  return t;
}

function drawLeft() {
  const is3d = view === "3d";
  const ax = is3d ? [0, 1, 2] : pairAxes();
  const kind = is3d ? "scatter3d" : "scattergl";
  const shown = groups.filter(g => g.visible);
  const claimed = new Set();
  for (const g of shown) for (const i of g.idx) claimed.add(i);
  const base = [];
  for (let i = 0; i < N; i++) if (!claimed.has(i) && inSlab(i)) base.push(i);

  const traces = [Object.assign(xyz(base, ax), {
    customdata: base, type: kind, mode: "markers",
    text: base.map(i => D.codes[i]), hovertemplate: "%{text}<extra></extra>",
    marker: {size: is3d ? 2.2 : 4, opacity: shown.length ? 0.28 : 0.9,
             color: base.map(i => D.colors[colourKey][i]),
             colorscale: D.cscale[colourKey], showscale: true, line: {width: 0},
             colorbar: {title: {text: D.labels[colourKey], side: "right"}, thickness: 12}}
  })];
  for (const g of shown) {
    // Stored groups stay whole: a selection made in one slab should not appear
    // to shrink when a different slab is on screen.
    const idx = g.idx.filter(i => inSlab(i));
    traces.push(Object.assign(xyz(idx, ax), {
      customdata: idx, type: kind, mode: "markers",
      text: idx.map(i => D.codes[i] + " — " + g.name),
      hovertemplate: "%{text}<extra></extra>",
      marker: {size: is3d ? 3.4 : 6, color: g.colour, line: {width: 0}}
    }));
  }
  Plotly.react("left", traces, leftLayout(), {responsive: true, displaylogo: false});
  bindSelect();
}

/* ---------- right panel ---------- */
function band(lo, hi, colour, name) {
  return [
    {x: R, y: lo, mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false},
    {x: R, y: hi, mode: "lines", line: {width: 0}, fill: "tonexty", fillcolor: colour,
     name: name, hoverinfo: "skip"}
  ];
}
function seriesFor(idxs, colour, name, traces) {
  if (mode === "curves") {
    const idx = thin(idxs, 400), xs = [], ys = [];
    for (const i of idx) {
      for (let j = 0; j < NB; j++) { xs.push(R[j]); ys.push(val(i, j)); }
      xs.push(null); ys.push(null);
    }
    traces.push({x: xs, y: ys, type: "scattergl", mode: "lines", hoverinfo: "skip",
                 line: {color: rgba(colour, 0.30), width: 1},
                 name: name + " (" + idx.length + (idx.length < idxs.length
                   ? " of " + idxs.length : "") + ")"});
  } else {
    const b = bands(idxs);
    for (const t of band(b.p5, b.p95, rgba(colour, 0.20), name + ": 5-95%")) traces.push(t);
    traces.push({x: R, y: b.p50, mode: "lines", line: {color: colour, width: 2.5},
                 name: name + ": median (n=" + idxs.length + ")"});
    traces.push({x: R, y: b.mu, mode: "lines", hoverinfo: "skip",
                 line: {color: colour, width: 1.4, dash: "dash"}, name: name + ": mean"});
  }
}

function drawRight() {
  let traces = band(D.pop.p5, D.pop.p95, "rgba(150,150,150,0.30)", "all: 5-95%");
  traces.push({x: R, y: D.pop.p50, mode: "lines", line: {color: "#777", width: 1.5},
               name: "all: median"});
  for (const g of groups) if (g.visible) seriesFor(g.idx, g.colour, g.name, traces);
  if (sel.length) seriesFor(sel, "#000000", "lasso", traces);

  Plotly.react("right", traces, {
    margin: {l: 62, r: 10, t: 30, b: 45},
    title: {text: "P(r) — " + (groups.filter(g => g.visible).length) +
                  " stored, " + sel.length + " in lasso", font: {size: 13}},
    xaxis: {title: {text: "r / Dmax"}, range: [0, 1]},
    yaxis: {title: {text: "P(r), unit sum over relative-r bins"}, rangemode: "tozero"},
    legend: {x: 1, y: 1, xanchor: "right", font: {size: 10}}
  }, {responsive: true, displaylogo: false});
}

/* ---------- group chips ---------- */
function drawChips() {
  const host = document.getElementById("groups");
  host.innerHTML = "";
  if (!groups.length) {
    const s = document.createElement("span");
    s.id = "hint"; s.textContent = "stored selections appear here";
    host.appendChild(s); return;
  }
  groups.forEach((g, k) => {
    const c = document.createElement("div");
    c.className = "chip" + (g.visible ? "" : " off");
    const sw = document.createElement("span");
    sw.className = "sw"; sw.style.background = g.colour;
    const nm = document.createElement("input");
    nm.value = g.name;
    nm.onchange = e => { g.name = e.target.value; save(); drawLeft(); drawRight(); };
    const n = document.createElement("span");
    n.className = "n"; n.textContent = "n=" + g.idx.length;
    const eye = document.createElement("button");
    eye.textContent = g.visible ? "◉" : "○"; eye.title = "show / hide";
    eye.onclick = () => { g.visible = !g.visible; save(); refresh(); };
    const del = document.createElement("button");
    del.textContent = "✕"; del.title = "remove";
    del.onclick = () => { groups.splice(k, 1); save(); refresh(); };
    c.append(sw, nm, n, eye, del);
    host.appendChild(c);
  });
}

function setCount() {
  document.getElementById("count").textContent =
    sel.length ? sel.length + " points in lasso" : "lasso a region on the left";
}
function refresh() { drawChips(); drawLeft(); drawRight(); setCount(); }

/* ---------- controls ---------- */
const sc = document.getElementById("colour");
for (const k of Object.keys(D.labels)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = D.labels[k]; sc.appendChild(o);
}
sc.value = colourKey;
sc.onchange = e => { colourKey = e.target.value; drawLeft(); };

if (DIM > 2) {
  document.getElementById("viewwrap").style.display = "";
  document.getElementById("slabwrap").style.display = "";
  const vs = document.getElementById("view");
  for (let i = 0; i < DIM; i++) for (let j = i + 1; j < DIM; j++) {
    const o = document.createElement("option");
    o.value = "" + i + j; o.textContent = "dim " + (i + 1) + " vs " + (j + 1);
    vs.appendChild(o);
  }
  const o3 = document.createElement("option");
  o3.value = "3d"; o3.textContent = "3-D (rotate)";
  vs.appendChild(o3);
  vs.value = view;
  vs.onchange = e => { view = e.target.value; drawSlabOptions(); drawLeft(); };

  const ss = document.getElementById("slab");
  ss.onchange = e => { slab = +e.target.value; drawLeft(); };
  drawSlabOptions();
}
function drawSlabOptions() {
  const ss = document.getElementById("slab");
  if (!ss) return;
  ss.disabled = view === "3d";
  ss.innerHTML = "";
  const all = document.createElement("option");
  all.value = "-1"; all.textContent = "all depth";
  ss.appendChild(all);
  for (let s = 0; s < NSLAB; s++) {
    const o = document.createElement("option");
    o.value = "" + s;
    o.textContent = "slab " + (s + 1) + " / " + NSLAB;
    ss.appendChild(o);
  }
  ss.value = "" + slab;
}
for (const r of document.getElementsByName("mode")) {
  r.onchange = e => { mode = e.target.value; drawRight(); };
}
document.getElementById("add").onclick = () => {
  if (!sel.length) return;
  groups.push({name: "sel " + nextId++, colour: PALETTE[(groups.length) % PALETTE.length],
               idx: sel.slice(), visible: true});
  sel = []; save(); refresh();
};
document.getElementById("clearsel").onclick = () => { sel = []; refresh(); };
document.getElementById("clearall").onclick = () => {
  if (groups.length && !confirm("Remove all " + groups.length + " stored selections?")) return;
  groups = []; nextId = 1; sel = []; save(); refresh();
};
document.getElementById("dl").onclick = () => {
  const rows = ["group,index,sasbdb_code"];
  for (const g of groups) for (const i of g.idx) rows.push(g.name + "," + i + "," + D.codes[i]);
  for (const i of sel) rows.push("lasso," + i + "," + D.codes[i]);
  if (rows.length === 1) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n")], {type: "text/csv"}));
  a.download = "selections.csv"; a.click();
};

// Called after every draw because the first draw is what creates gd.on; the
// guard keeps it to a single registration, since Plotly.react preserves handlers
// across re-plots (including the 2-D <-> 3-D trace-type switch).
let bound = false;
function bindSelect() {
  const gd = document.getElementById("left");
  if (bound || !gd.on) return;
  bound = true;
  gd.on("plotly_selected", ev => {
    sel = ev ? Array.from(new Set(ev.points.map(p => p.customdata))) : [];
    drawRight(); setCount();
  });
  gd.on("plotly_deselect", () => { sel = []; drawRight(); setCount(); });
}

load();
refresh();
</script></body></html>
"""


def build_html(run: Path, Z, X, meta) -> Path:
    import plotly

    cols = color_panels(X, meta)
    pop = np.percentile(X, [5, 50, 95], axis=0)
    data = {
        "z": np.round(Z, 4).tolist(),
        "x": np.rint(X * SCALE).astype(np.int32).ravel().tolist(),
        "nb": int(X.shape[1]),
        "scale": SCALE,
        "codes": meta["sasbdb_code"].astype(str).tolist(),
        "colors": {k: np.round(v, 4).tolist() for k, v in cols.items()},
        "cscale": CSCALE,
        "labels": LABELS,
        "pop": {
            "p5": np.round(pop[0], 8).tolist(),
            "p50": np.round(pop[1], 8).tolist(),
            "p95": np.round(pop[2], 8).tolist(),
        },
    }
    html = (
        PAGE.replace("__PLOTLYJS__", plotly.offline.get_plotlyjs())
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__TITLE__", f"P(r) explorer — {run.name}")
    )
    out = run / "explorer.html"
    out.write_text(html)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=_ROOT / "runs" / "sasbdb_pr_l1_frozen")
    ap.add_argument("--no-open", action="store_true", help="write the file but do not open it")
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    missing = [f for f in ("Z.npy", "X.npy", "meta.csv") if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run} is missing {', '.join(missing)}")

    Z = np.load(run / "Z.npy").astype(np.float64)
    X = np.load(run / "X.npy").astype(np.float64)
    meta = pd.read_csv(run / "meta.csv")

    out = build_html(run, Z, X, meta)
    print(f"{run.name}: {len(Z)} points, {X.shape[1]} bins")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open(out.as_uri())
        print("opened in your browser")


if __name__ == "__main__":
    main()

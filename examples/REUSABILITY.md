# A leanmap is a *model*, not just a picture

Most manifold-learning tools (t-SNE, and UMAP to a lesser extent) give you a
**layout**: a fixed set of 2-D coordinates for the exact points you fed in. You
can look at it, but you cannot cheaply *reuse* it, and it will not tell you when
a new point does not belong.

`leanmap` fits a small parametric encoder and keeps a calibrated notion of "what
the training data covered". Once trained you get three things UMAP's
`transform()` does not give you:

1. **Fast amortized inference** — a single forward pass per point; the neighbor
   graph is thrown away after fitting.
2. **A calibrated out-of-distribution (OOD) score** — landmark **cover
   distance** turned into conformal *p*-values.
3. **Honesty about off-manifold points** — instead of snapping strangers onto
   the chart, leanmap spreads them out *and* flags them.

Everything below is reproduced by one script:

```bash
python examples/reusability.py
# writes plots + numbers to examples/out/
```

It trains **one** leanmap (and one UMAP) on a 2 000-point **Swiss cone with a
hole**, then reuses both on 10 000 fresh in-distribution points and 10 000
uniform ambient (OOD) points.

---

## 1. Reuse is cheap

The encoder is amortized: embedding new points is a batched forward pass, so
throughput is roughly constant per point. UMAP's `transform()` has to run an
approximate-NN query plus an optimization against the stored training graph for
every batch.

| mapper | embed 10 000 new points | throughput |
|--------|------------------------:|-----------:|
| leanmap (CPU) | **~0.1 s** | ~97 000 pts/s |
| UMAP `nn=30` `transform()` | ~10.9 s | ~900 pts/s |

≈ **100× faster** on this toy, and the gap widens as the training set grows
because leanmap's cost does not depend on `N_train` at inference time. The
fitted encoder is a single `.pt` file you can ship and call again later —
`load_plane("reusability_leanmap.pt")` — with no training data attached.

## 2. In-distribution points land where they should

Fresh cone samples (green) that the model never saw during training fall right
back onto the learned ribbon and reproduce the hole. Fidelity holds on
held-out data:

```
geodesic fidelity: spearman = 0.976   stress = 0.123
```

## 3. OOD points are spread out *and* flagged

Feed 10 000 points sampled uniformly over the ambient bounding box (red). Most
of them are nowhere near the cone.

![leanmap: fresh vs uniform](out/reuse_leanmap_green_red.png)

leanmap keeps the green in-distribution points tight on the ribbon while the red
uniform points fan out into a diffuse haze — you can *see* they are different.
More importantly, you don't have to eyeball it. The landmark **cover distance**
(distance to the nearest landmark in the input space) cleanly separates the two
populations:

```
cover distance   fresh (in-dist) median = 1.07
                 uniform (OOD)   median = 2.59
```

Turn that into a conformal *p*-value calibrated on held-out in-distribution data
(`ConformalCalibrator`) and gate at `alpha = 0.05`:

```
reject @ alpha = 0.05:
    fresh cone   (false positives) = 4.8%   ≈ the nominal 5%
    uniform      (true positives)  = 74.8%
```

So the calibration is honest (in-distribution rejection ≈ α) and ~3 of every 4
uniform strangers get caught. Coloring every embedded point by its *p*-value
makes the story obvious — the ribbon is high-support (bright), the surrounding
haze is low-support (dark):

![leanmap: conformal p-value](out/reuse_leanmap_pvalue.png)

## 4. Why this matters: UMAP `transform()` hides OOD points

UMAP places a new point by finding its nearest neighbors **in the training set**
and optimizing its position to match them. That means an OOD point is *always*
dragged onto the manifold next to whatever training points happen to be closest
— it has no way to say "this doesn't belong."

![UMAP: fresh vs uniform](out/reuse_umap_green_red.png)

Here the red uniform points are smeared **all over the same arch** as the green
in-distribution points. Visually and numerically they are indistinguishable, and
`transform()` returns no score to tell them apart. A downstream consumer reading
these coordinates would happily treat pure noise as valid manifold samples.

---

## Takeaways

| property | leanmap | UMAP `transform()` |
|----------|:-------:|:------------------:|
| reusable fitted model (no train data at inference) | ✅ `.pt` file | ⚠️ carries the graph |
| amortized inference cost | ✅ O(1) per point | ❌ grows with `N_train` |
| calibrated OOD score / *p*-values | ✅ cover + conformal | ❌ none |
| OOD points distinguishable | ✅ spread + flagged | ❌ snapped onto chart |
| geometry fidelity on held-out data | ✅ ρ ≈ 0.98 | ✅ (layout only) |

None of this says UMAP is bad at making a *picture* of a dataset — it is
excellent at that. The point is that a leanmap is a **reusable, self-aware
model**: you can save it, call it on new data at scale, and trust it to tell you
when that new data is off the manifold it was trained on.

> Reproduce with `python examples/reusability.py`
> (add `--skip-umap` to skip the slow UMAP comparison).

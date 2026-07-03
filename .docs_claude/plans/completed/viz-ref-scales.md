# Per-scale PCA→RGB viz of baked multi-scale reference boxes

> **Status: completed 2026-07-02** (built + render-verified). Takes **one or more** render
> dirs (`render_dirs` nargs+) and stacks every class as a row, labelled `dataset/class`
> (dataset = render dir's parent name) so duplicate class names across datasets disambiguate;
> each dir keeps its own baked per-scale basis (or, with `--joint-basis`, one refit basis
> per scale pooled over all rows so scale columns are colour-comparable across datasets).
> Verified on a combined 30-row figure over
> `filtered/vis030/{expanded-refseg-v2,shelf-optflow,mixed-persp}/render000` (8+3+19 classes),
> backbone `CleanDiftFinetunedFpn` → `filtered/vis030/ref_scales_<datasets>.png`.
> As-built: `alpha_crop` is imported from `viz_clusters.py`, which imports `faiss` at
> module top, so the invocation needs `--with 'faiss-cpu==1.8.0' --with 'numpy==1.26.0'`
> (mirrors viz_clusters' own documented usage). **Finding (3-way, across all datasets):**
> scale 0 (stride 32) projects the *same* rainbow template on every row regardless of object
> — the coarsest PCs encode shared spatial/positional layout, colour- and content-blind;
> scale 1 (stride 16) varies between genuinely-different objects but stays ~identical among
> the same-template amazon boxes — coarse layout/semantics, still not colour; only scale 2
> (stride 8, 96²) shows per-object colour/appearance (tomato-soup red, mustard yellow, SPAM
> blue…). Confirms the hypothesis — lower strides preserve more colour-correlated info.

## Context

`expanded-refseg-v2` (on disk `/data/user/jeffk/datasets/filtered/vis030/expanded-refseg-v2/render00{0,1,2}`)
carries a `CleanDiftFinetunedFpn` backbone whose per-class leaf is a keyed
multi-scale dict `{scale: (C_k,h_k,w_k)}` — scale `0`=stride32/24²/1280, `1`=stride16/48²/1280,
`2`=stride8/96²/640 — plus a **per-scale shared PCA→RGB basis** in
`principal_components/CleanDiftFinetunedFpn/` (fit over all 8 classes' tokens at that
scale, so colours are comparable across boxes within a scale; not across scales — PCA sign
is arbitrary). No tool renders the *baked reference* catalog per class — `viz_clusters.py`
only clusters the live observation via a descriptor forward pass.

**Goal:** for each of the 8 reference boxes, one image row `[ref image | pca@s0 | pca@s1 | pca@s2]`,
to see whether lower strides preserve more colour-correlated info. Montage only (no metrics).

**Precedent:** no `min-size` helper exists; the sole "resize the feature volume first, then
operate" precedent is `viz_clusters.extract_fpn_volumes` (bilinear `F.interpolate` before
use). Adopted; `--min-hw` defaults to the finest native scale (96, upsample-to-at-least)
since no minimum constant exists to inherit.

## New file — `isaac_datagen/src/isaac_datagen/viz_ref_scales.py`

Reuses `pca_rgb`/`rgba_chw_to_rgb`/`panel_grid`/`save_figure` (vision_core.viz),
`alpha_crop` (viz_clusters), `ObsMaskDescriptorMetadata.deserialize_field` — nothing new.

```python
def scale_pca_rgb(feats, basis, min_hw):                       # baked (C,h,w) -> (H,W,3)
    h, w = feats.shape[-2:]
    s = max(1.0, min_hw / min(h, w))                           # "min h,w": never below native
    up = F.interpolate(feats[None].float(), scale_factor=s,    # UPSAMPLE FEATURES FIRST
                       mode="bilinear", align_corners=False)[0]  # (mirrors viz_clusters)
    return pca_rgb(up, basis)                                  # THEN project → (H,W,3) in [0,1]


def load_rows(render_dir, backbone):                          # one dir → [(dataset/class, ref, leaf, pcs)]
    md = ObsMaskDescriptorMetadata                            # narrowest reads: 3 fields only
    c2r = md.deserialize_field(0, render_dir, "class_to_ref")
    c2d = md.deserialize_field(0, render_dir, "class_to_descriptors")[backbone]
    pcs = md.deserialize_field(0, render_dir, "principal_components")[backbone]
    assert isinstance(next(iter(c2d.values())), dict), "needs a keyed multi-scale FPN backbone"
    ds = render_dir.parent.name                               # dataset name disambiguates dup classes
    return [(f"{ds}/{cls}", c2r[cls], c2d[cls], pcs) for cls in sorted(c2d)]


def ref_scales_figure(rows, min_hw):                          # one row per (dataset, class)
    scales = list(next(iter(rows))[2])                        # ["0","1","2"] coarse→fine
    fig, axes = panel_grid(len(rows) * (1 + len(scales)), cols=1 + len(scales))
    it = iter(axes)
    for label, ref, leaf, pcs in rows:
        a = next(it); a.imshow(rgba_chw_to_rgb(alpha_crop(ref)[0])); a.set_title(label, fontsize=8); a.axis("off")
        for k in scales:
            f = leaf[k]; a = next(it); a.imshow(scale_pca_rgb(f, pcs[k], min_hw))
            a.set_title(f"scale {k}  {f.shape[1]}×{f.shape[2]}  C={f.shape[0]}", fontsize=8); a.axis("off")
    return fig

# joint_basis(rows): refit one fit_pca_basis per scale over ALL rows' pooled tokens (columns
#   comparable across datasets), overriding each row's per-dir basis.
# main(): render_dirs nargs+ (stack all their classes); --backbone required (no default);
#   --joint-basis opt-in (+_joint suffix); out default: <first_dir>/../../ref_scales_<ds1+ds2+…>.png
```

## Run / verify

```bash
cd isaac_datagen && env -u PYTHONPATH uv run --with 'faiss-cpu==1.8.0' --with 'numpy==1.26.0' \
  src/isaac_datagen/viz_ref_scales.py \
  /data/user/jeffk/datasets/filtered/vis030/expanded-refseg-v2/render000 \
  /data/user/jeffk/datasets/filtered/vis030/shelf-optflow/render000 \
  /data/user/jeffk/datasets/filtered/vis030/mixed-persp/render000 \
  --backbone CleanDiftFinetunedFpn
```

A combined 30-row montage lands at `filtered/vis030/ref_scales_expanded-refseg-v2+shelf-optflow+mixed-persp.png`.

## Follow-ups (not done)

- `alpha_crop` is a pure-numpy RGBA-bbox crop trapped behind `viz_clusters`' top-level
  `faiss` import; moving it to `vision_core.viz` would drop the faiss extra from this tool.
- `--min-hw 256` for smoother coarse-scale panels; explicit stride labels in titles.

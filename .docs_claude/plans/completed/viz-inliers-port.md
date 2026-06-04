# Port `viz_refseg` → `isaac_datagen` inlier-label visualizer (`viz_inliers`)

## Context

The old `viz_refseg.py` (now only a `.bak` in `visual_servoing/datagen2_isaacsim/`, never ported —
it was renamed in place and abandoned) visualized the **old flat `ReferenceSegSample`** layout
(`rgb/`, `seg_mask/`, `proposal_coordinates/`, `reference_features/`). That layout no longer exists.

We want a visualizer for the **new normalized layout** whose specific job is a **sanity check that the
phase-3 inlier/outlier labels are correct** — i.e. that proposal points labeled *inlier* really fall
inside their object's instance mask and *outlier* points fall outside. Per frame (= one
`ImageInlierSample`) with N present instances:

1. **One overview panel** — all N instance masks overlaid on `obs`, each a unique color, with a legend.
2. **N per-instance panels** — `obs` with that instance's proposal points scattered, **green = inlier /
   red = outlier** (from `labels`), the instance's mask outline drawn faintly so inside/outside is
   visually obvious, and a **thumbnail of that instance's reference image**.

Bijection holds (verified earlier): each instance id ↔ one name, so `name = id_to_name[uid]`,
`mask = (id_mask == uid)`, `proposals[name]`, `labels[name]`, `name_to_ref[name]` all line up.

## New file: `src/isaac_datagen/viz_inliers.py`

Headless matplotlib (`matplotlib.use("Agg")`). One combined figure per frame at **dpi 300**.

**Reused helpers** — copy from `visual_servoing/datagen2_isaacsim/.viz_refseg.py.bak`:
- `composite_over_white(rgba_hw4) -> rgb_hw3` (lines 25-31) — verbatim.
- `_mask_border(mask, thickness=5) -> bool mask` (lines 54-60) — verbatim (uses `scipy.ndimage.binary_dilation`, confirmed available 1.15.3).
- New tiny adapter `rgba_chw_to_rgb(t)`: `composite_over_white(t.permute(1, 2, 0).numpy())` — for the
  channels-first uint8 RGBA `obs` (4,H,W) and `name_to_ref` (4,h,w) tensors.

**Data flow** (mirrors `RenderDirReferenceSegDataset` joins + `add_inlier_data` keys):
```python
md = ObsMaskMetadata.deserialize(0, render_dir)                 # catalog: id_to_name, name_to_ref
cmap = plt.get_cmap("tab20")
for idx in selected_frames:
    s = ImageInlierSample.deserialize(idx, render_dir)          # obs, id_mask, proposals, labels
    obs_rgb = rgba_chw_to_rgb(s.obs)                            # (H, W, 3) uint8
    idm = s.id_mask                                             # (H, W) int32
    present = sorted({int(i) for i in idm.unique().tolist()} & set(md.id_to_name))
    colors = {uid: cmap(i % 20) for i, uid in enumerate(present)}

    n_panels = 1 + len(present)
    cols = args.cols; rows = -(-n_panels // cols)              # ceil
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.ravel()

    # panel 0: overview — alpha-blend each instance's color over its mask + colored border + legend
    render_overview(axes[0], obs_rgb, idm, present, md.id_to_name, colors)

    # panels 1..N: per-instance scatter colored by label + mask outline + ref thumbnail inset
    for p, uid in enumerate(present, start=1):
        name = md.id_to_name[uid]
        render_instance(axes[p], obs_rgb, (idm == uid).numpy(),
                        s.proposals.get(name), s.labels.get(name),
                        md.name_to_ref.get(name), name, colors[uid])
    for ax in axes[n_panels:]:                                 # hide unused grid cells
        ax.axis("off")
    fig.suptitle(f"{render_dir.name}  frame {idx:04d}  —  {len(present)} instances", fontsize=10)
    fig.savefig(out_dir / f"sample_{idx:04d}.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
```

**`render_overview(ax, obs_rgb, id_mask, present, id_to_name, colors)`**
- Start `overlay = obs_rgb.float()`; for each `uid`: `m = (id_mask == uid)`, blend
  `overlay[m] = (1-α)*overlay[m] + α*color` (α≈0.45), then set `overlay[_mask_border(m)] = darker(color)`.
- `ax.imshow(overlay.uint8())`; build a legend from `mpatches.Patch(color=colors[uid], label=id_to_name[uid])`.

**`render_instance(ax, obs_rgb, mask_hw, coords, labels, ref_rgba, name, color)`**
- `ax.imshow(obs_rgb)`; draw the instance outline: `ax.imshow` an RGBA overlay that is `color` only on
  `_mask_border(mask_hw)` (so inside/outside is visible without hiding the scatter).
- If `coords` is None/empty → `ax.text(... "no proposals")`; else split by `labels.bool()`:
  `ax.scatter(coords[lab,0], coords[lab,1], c="lime", s=6, alpha=0.6, label="inlier")` and the `~lab`
  set `c="red"`. (coords are (x,y) in obs pixels — matches imshow axes directly, no flip.)
- Title `f"{name}  in={int(lab.sum())}/{len(lab)}"`, colored with `color`.
- Ref thumbnail: `inset = ax.inset_axes([0.72, 0.72, 0.26, 0.26]); inset.imshow(rgba_chw_to_rgb(ref_rgba)); inset.axis("off")`.
- `--max-points`: if set and `len(coords) > max_points`, stride-subsample deterministically
  (`coords[:: len//max_points]`) and note it in the title. Default: plot all.

**CLI (argparse, like the old script)** — `src/isaac_datagen/viz_inliers.py:main`:
- positional `render_dir`
- `--out` (default `render_dir.parent / (render_dir.name + "_viz_inliers")`)
- frame selection: `--frames 0,5,10` (explicit CSV) **or** `--max-frames K` (default 8) `--stride S` (default 1)
- `--cols` (default 4), `--dpi` (default 300), `--max-points` (default None = all)
- count frames from `len(list((render_dir / "obs").iterdir()))`, same as the phase passes.

## `pyproject.toml`
- Add console script: `isaac-datagen-viz-inliers = "isaac_datagen.viz_inliers:main"`.
- Add `matplotlib` to `[project] dependencies` (importable transitively today — 3.10.3 — but undeclared;
  `scipy`/`numpy` are already declared). No new heavy deps otherwise.

## Verification
1. **Static / import**: `uv run python -c "import ast; ast.parse(open('src/isaac_datagen/viz_inliers.py').read())"`
   and `uv run python -c "import isaac_datagen.viz_inliers"`.
2. **Isolated end-to-end, no GPU** (reuse the phase-3 verification trick): build a temp dir from the real
   catalog + 2 real frames of `obs/`+`id_mask/`, fabricate a known `proposals/` (a few points inside each
   instance mask + a few outside), run `add_inlier_data` to produce `labels/`, then run `viz_inliers` on it
   with `--max-frames 2 --dpi 150`. Assert one `sample_XXXX.png` per frame is written.
3. **Visual confirmation**: `Read` a produced PNG (the Read tool renders images) and confirm by eye that
   green points sit inside the colored instance outline and red points outside — the actual label sanity check.
4. **Real data (optional, after phase-2+phase-3)**: once `render000` has `proposals/` + `labels/`, run
   `uv run isaac-datagen-viz-inliers src/isaac_datagen/expanded-refseg/render000 --max-frames 4`.

## Notes / non-goals
- Drops the old ref-features **PCA** panel — not part of the inlier sanity check (the catalog still has
  `name_to_descriptors` if a feature panel is wanted later).
- High dpi (300) + 6×4-in panels make each grid cell near-full-resolution, so the dense ~5k-point scatter
  stays legible in the combined figure. A `--separate` full-res-per-instance mode can be added later if needed.
- Requires phase-3 output (`labels/`); it deserializes `ImageInlierSample`, which needs `obs/`, `id_mask/`,
  `proposals/`, `labels/` all present.

## Staged checklist
1. `viz_inliers.py` — reused helpers + `rgba_chw_to_rgb` + `render_overview` + `render_instance` + grid + argparse.
2. `pyproject.toml` — console script + `matplotlib` dep.
3. Verify: import → isolated temp-dir run → `Read` the PNG to eyeball label correctness.

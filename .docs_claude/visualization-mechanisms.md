# Visualization mechanisms — table of contents

Reusable viz primitives for the shared datastructs (`ObsMask`, `ObsMaskMetadata`,
`PreReferenceSegSample`, `ImageInlierSample`) live in **`vision_core.viz`**
(`~/repo/vision_core/src/vision_core/viz.py`). The CLI spot-check tools that compose
them live in `isaac_datagen`. This file is the index; signatures and details are in
the module docstrings.

Conventions: every mask primitive is generic over id space — it takes ONE (H, W)
integer mask plus a dict keyed in THAT space (`iid_mask` + iid-keyed dict, or
`cid_mask` + cid-keyed dict); qualify iid/cid at the call site, never bare "id".
`vision_core.viz` never sets a matplotlib backend — headless CLIs set Agg themselves
before importing it.

## `vision_core.viz` — three altitude levels

### 1. Pure array math (numpy/scipy, no matplotlib)

| Primitive | What it does |
|---|---|
| `composite_over_white(rgba)` | (H, W, 4) uint8 RGBA → (H, W, 3) RGB over white |
| `rgba_chw_to_rgb(t)` | (4, H, W) RGBA tensor (the datastruct convention) → (H, W, 3) RGB |
| `mask_border(mask, thickness)` | bool mask → bool border band |
| `mask_outline_rgba(mask, color)` | transparent RGBA layer with only the border colored — imshow over a base |
| `mask_centroid(mask)` | (cx, cy) of a bool mask, or None if empty |
| `present_ids(id_mask, catalog)` | sorted ids in the mask ∩ catalog keys (the drawable ids) |
| `overlay_id_masks(rgb, id_mask, id_to_color, alpha)` | alpha-blend each id's region in its color + darker border |
| `subsample_labeled_points(coords, labels, max_points)` | deterministic stride subsample + "(showing K/N)" note |

### 2. Axis-level matplotlib helpers

| Helper | What it does |
|---|---|
| `assign_colors(ids, cmap_name)` | stable unique color per id (tab20 cycle) |
| `color_legend(ax, id_to_color, id_to_label, **kw)` | color-patch legend; label formatting is the caller's policy |
| `OUTSIDE_LEGEND_KW` | kwargs that place a legend in the right gutter, outside the image |
| `scatter_labeled(ax, coords, labels, max_points)` | green=True/inlier, red=False/outlier point scatter |
| `add_thumbnail(ax, rgba_chw, loc)` | inset a (4, H, W) reference image in a corner |
| `annotate_points(ax, [(x, y, text)])` | white-on-black text at image coords (e.g. ratios at centroids) |
| `error_panel(ax, label, exc)` | stamp a failed panel instead of aborting the figure |
| `panel_grid(n, cols, w, h, **gridspec_kw)` | ceil-div grid; returns (fig, first n axes); blanks the rest |
| `save_figure(fig, path, dpi)` | tight-bbox savefig + close |

### 3. Datastruct-tied composers (loaded datastructs in, panels/figures out — no I/O)

| Composer | What it does |
|---|---|
| `render_id_masks_panel(ax, obs_rgb, id_mask, …)` | obs + every id's mask in its color + legend (either id space) |
| `proposal_panel(ax, obs_rgb, mask, coords, labels, ref, …)` | obs + green/red proposals + mask outline + ref thumbnail |
| `occlusion_panel(ax, om: ObsMask, md)` | INSTANCE space: per-iid colors + occlusion ratio at each centroid + gutter legend |
| `inlier_figure(sample: ImageInlierSample, md, …)` | CLASS space: overview panel + one proposal_panel per present class → Figure |

Also: `vision_core.datastructs.count_samples(directory, field="obs")` — frame count
from the serialization layout (one file per idx per field).

## CLI spot-check tools (isaac_datagen — thin orchestrators: argparse + file discovery + deserialize → compose → save)

| Tool | Checks | Run |
|---|---|---|
| `isaac-datagen-viz-inliers <render_dir>` | phase-3 labels: inliers inside / outliers outside each class union mask | `[--frames CSV \| --max-frames K --stride S] [--cols] [--dpi] [--max-points]` |
| `isaac-datagen-viz-occlusion <dataset_dir>` | `iid_to_occlusion` sanity: buried boxes ↔ high ratios, over a seeded random sample of all render dirs | `[--n 12] [--seed] [--cols] [--alpha] [--dpi] [--out]` |
| `isaac-datagen-viz-sample <render_dir>` | per class in `sample.proposals`: its proposals scattered over obs (green/red by its labels) with its ref as a corner thumbnail, plus its gt union-mask overlay | same flags as viz-inliers + `[--alpha]` |
| `relabel_classes.py <dataset> --grid-only` | GraspableObject class clusters: 8-col grid of all reference images → `<dataset>/reference_grid.png` | (isaac_datagen; also the interactive relabel loop) |

## Composing a new visualization

```python
import matplotlib; matplotlib.use("Agg")
from vision_core.viz import (present_ids, assign_colors, overlay_id_masks,
                             panel_grid, save_figure, rgba_chw_to_rgb)

obs_rgb = rgba_chw_to_rgb(sample.obs)
cids = present_ids(sample.cid_mask.numpy(), md.cid_to_class)
fig, axes = panel_grid(1, cols=1)
axes[0].imshow(overlay_id_masks(obs_rgb, sample.cid_mask.numpy(), assign_colors(cids)))
save_figure(fig, "check.png", dpi=200)
```

Verification pattern (from the plans): the figures ARE the verification artifact —
render to PNG and eyeball it against the invariant the change introduced (labels ↔
masks, occlusion ↔ visibility, classes ↔ color clusters).

History: extracted from `isaac_datagen/viz_inliers.py` + `viz_occlusion.py` via a
/reusable-parts review (2026-06-04); see
`isaac_datagen/.docs_claude/plans/completed/{viz-inliers-port,obsmask-occlusion-and-viz,cid-mask-dual}.md`.

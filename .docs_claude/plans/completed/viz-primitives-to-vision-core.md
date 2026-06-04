# Extract viz primitives → `vision_core.viz` (reusable-parts refactor)

## Context

`viz_inliers.py` and `viz_occlusion.py` had accumulated reusable, datastruct-tied
visualization primitives inside CLI scripts — `viz_occlusion` even imported the
private `_mask_border` from its sibling script. User directive: primitives tied to
the shared datastructs belong in `vision_core`; rewrite per `/reusable-parts` so
future pipelines (segmentation training, reference matching) can compose them.

## Review (3 independent reusable-parts agents, aggregated)

- **Unanimous PASS, wrong home**: `composite_over_white`, `_mask_border`,
  `rgba_chw_to_rgb` — clean leaves; moved verbatim, `mask_border` de-underscored.
- **Unanimous FAIL**: `visualize_frame` (deserialize + id-discovery + colors + grid
  + tensor unwrap + savefig fused — couldn't render an in-memory sample) and
  `viz_occlusion.main` (manual `md_cache` shared mutable state, try/except recovery
  in the orchestrator, missing per-panel renderer layer).
- **Majority FAIL**: `render_class` (outline construction, max-points decimation
  policy, scatter, thumbnail trapped), `render_panel` (centroid math in a loop
  building three parallel structures).
- **Confirmed duplications**: the alpha-blend overlay loop (byte-identical between
  the two files), present-ids set-intersection, `cmap(i % 20)` color assignment,
  ceil-div panel grid + blank trailing axes, `len(list((dir/"obs").iterdir()))`.

## What was built

**NEW `vision_core/src/vision_core/viz.py`** — three altitude levels (see module
docstring): pure array math (`overlay_id_masks`, `mask_outline_rgba`,
`mask_centroid`, `present_ids`, `subsample_labeled_points`, the moved leaves),
axis-level helpers (`assign_colors`, `color_legend` + `OUTSIDE_LEGEND_KW`,
`scatter_labeled`, `add_thumbnail`, `annotate_points`, `error_panel`, `panel_grid`,
`save_figure`), and datastruct-tied composers (`render_id_masks_panel`,
`proposal_panel`, `occlusion_panel`, `inlier_figure`). Generic over id space —
each primitive takes one mask + that space's dict; iid/cid qualified at call sites.
The module sets no matplotlib backend (CLIs set Agg before importing).

**`vision_core/datastructs.py`** — added `count_samples(directory, field="obs")`
next to `SerializableSample` (it queries the layout that class defines).

**`viz_inliers.py` / `viz_occlusion.py`** — rewritten as thin orchestrators:
argparse + file discovery (`select_frames`, `frame_pairs`) + deserialize →
compose → `save_figure`. The occlusion metadata cache became
`functools.cache(lambda rd: ObsMaskMetadata.deserialize(0, rd))`; per-panel
try/except now delegates to `error_panel`. Console scripts unchanged.

**Docs** — `.docs_claude/visualization-mechanisms.md` TOC written to all of:
isaac_datagen, vision_core, reference_matching, segmentation, segmentation-train.

## Behavior changes (intentional, cosmetic)

- `panel_grid` clamps `cols = min(cols, n_panels)` (occlusion already did; inliers
  figures with few panels are no longer padded to 4 columns).
- Inlier suptitle dropped the "— N classes" suffix (the overview panel title
  already shows N).
- `assign_colors` cycles `cmap.N` instead of hardcoded 20.

## Verification (2026-06-04)

Both CLIs run against `src/isaac_datagen/cid-mask-verify/render900`; output PNGs
read and eyeballed: inliers figure reproduces overview + 10 per-class panels
(green/red scatters, union outlines, ref thumbnails, in=N/M titles); occlusion
figure reproduces per-instance colors, centroid ratios, gutter legends.

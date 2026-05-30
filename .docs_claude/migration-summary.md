# isaac_datagen — migration summary

Extracted from `visual_servoing/datagen2_isaacsim/` into a standalone uv package.
Centered on the one live entry point, `clean_datagen.py` (primary mode
`reference_segmentation()`). Full decision log:
[`plans/completed/extract-isaac-datagen.md`](plans/completed/extract-isaac-datagen.md).

## What landed here (and where it came from)

The 11-module transitive closure of `clean_datagen.py`, copied (byte-verified)
from `visual_servoing/datagen2_isaacsim/` into `src/isaac_datagen/`:

`clean_datagen.py, scene.py, capture.py, pose_planning.py, runtime_config.py,
objects.py, isaac_utils.py, hardwares.py, stereo_writer.py,
reference_seg_writer.py, __init__.py` — plus `resources/` (86 files; a hard
relative dep — `scene.py` loads `resources/workbench_world.usd`).

Originals + 17 legacy/scratch scripts + `resources/` were **hidden as `.bak`**
(29 entries, none deleted) in `visual_servoing/datagen2_isaacsim/`.

## Renamed (imports)

- Intra-package, flat → namespaced: `from scene import …` → `from isaac_datagen.scene import …`
  (same for `capture, objects, isaac_utils, hardwares, pose_planning, runtime_config,
  stereo_writer, reference_seg_writer`).
- Shared core → vision_core: `from datastructs …` → `from vision_core.datastructs …`;
  `from pose_utils …` → `from vision_core.pose_utils …`.
- Proposer/descriptor → reference_matching (in `reference_seg_writer.py`):
  `from segmentation import proposal/descriptor` → `from reference_matching import proposal/descriptor`.

## Removed / left behind

- **Not extracted (left as `.bak` in datagen2_isaacsim):** legacy/one-off
  scripts — `build_object_dataset, export_*, filter_objects, inspect_distributions,
  test_*, view_scene, visualize_seg, viz_refseg, dump_usda, verify_export,
  alpha_injection, usd_utils, notes` + scratch (`seggen`, `referencesegwriter`).
- **Stayed external (not bundled):** `object_dataset_amazon/` (the GraspableObjects)
  is loaded by config path via `GraspableObject.deserialize(path)`, so it remains
  in `visual_servoing/datagen2_isaacsim/`, not copied here. Generated dirs
  (`refseg*`, `canonical-multibox`, `render000_viz`, `object_dataset`) likewise
  stay there.

## Dependencies / entry points

- `isaacsim[all,extscache]==5.1.0` (NVIDIA index), `vision-core`,
  `reference-matching`, plus numpy/scipy/torch/torchvision **unpinned** so
  isaacsim drives them (numpy 1.26, pillow 11.3, torch 2.7).
- `gim` + `lightglue` sources redeclared here (transitive via reference-matching;
  uv.sources don't propagate through path deps).
- Console scripts: `isaac-datagen` → `clean_datagen:reference_segmentation`
  (primary); `isaac-datagen-stereo` → `clean_datagen:main`.

## Still open

`uv sync` + runtime verification (isaacsim is a multi-GB pull); the duplicate
`~/repo/datagen/datagen2_isaacsim*`; `visual-servo-rollout` still imports
`from datagen2_isaacsim…`; `.bak` cleanup once runtime-verified.

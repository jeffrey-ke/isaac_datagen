---
description:
alwaysApply: true
---

# isaac_datagen

Standalone Isaac Sim Replicator data-generation package for visual servoing.

## Quick start

```
uv run clean_datagen.py <config.yaml>
```

Config + OmegaConf dotlist overrides, e.g. `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml idx=0 num_frames=8`. Console scripts: `isaac-datagen` → `reference_segmentation` (default), `isaac-datagen-stereo` → `main`.

## Module index

| Module | Role | Key exports |
|---|---|---|
| `clean_datagen.py` | Entry points; orchestrates config→sim→scene→capture | `reference_segmentation`, `main`, `collect_objects` |
| `runtime_config.py` | `RuntimeConfig` schema + OmegaConf YAML/dotlist loader (`${call:…}` resolver) | `RuntimeConfig`, `load_config` |
| `scene.py` | USD scene build, RTX `boot_sim`, lighting/texture randomizers | `boot_sim`, `build_scene`, `make_replicator`, `SceneHandle` |
| `capture.py` | Replicator capture orchestration (sessions, pose-driven steps) | `capture_with_poses`, `get_target2world`, `make_index` |
| `pose_planning.py` | Pure-numpy target-frame pose sampling | `plan_poses` |
| `objects.py` | Graspable objects, occupancy placement, sample datastructs | `GraspableObject`, `ProtoReferenceSegSample`, `OccupancyGrid` |
| `isaac_utils.py` | USD/Replicator helpers (camera-from-K, transforms, prim search, USDZ export) | `setup_camera`, `load_asset`, `set_transform`, `find_prims`, `export_subtree_usdz`, `export_flattened_usdz` |
| `hardwares.py` | ZED Mini stereo camera rig | `ZedMini` |
| `stereo_writer.py` | Replicator Writer → `StereoSample` | `StereoSampleWriter` |
| `reference_seg_writer.py` | Replicator Writer → `ProtoReferenceSegSample` (precomputes DIFT ref features) | `ProtoReferenceSegWriter` |

External (sibling editable packages + heavy deps): `vision_core` (datastructs `SerializableSample`/`StereoSample`/`ReferenceSegSample`, `pose_utils`), `reference_matching` (DIFT `descriptor`, `proposal`), `isaacsim==5.1.0` + `omni.replicator.core`, `pxr`/USD, torch/torchvision.

## Data flow

```
config.yaml + CLI overrides ─load_config→ RuntimeConfig
  ├ boot_sim → SimulationApp (RTX path tracing)
  └ collect_objects → [GraspableObject]  (external object_dataset_amazon, by config path)
         ↓
build_scene → stage + workbench + lights + object stack (OccupancyGrid) + grasp frames + ZedMini → SceneHandle
         ↓
 reference_segmentation():  grasp pts → get_target2world → plan_poses → world_poses
                            → ProtoReferenceSegWriter → capture_with_poses → per-frame write() → serialize
 main() (stereo):           single grasp pt → make_index → StereoSampleWriter → capture_with_poses → serialize
         ↓
 render{idx:03d}/  (rgb/ ref_rgb/ seg_mask/ reference_features/  OR stereo fields) + runtime/descriptor yaml
```

Capture mechanism: `capture_with_poses` → `capture_session` opens `rep.new_layer()`, attaches the writer to the camera's render products, uses `rep.trigger.on_frame()` to move the camera through the planned world poses and apply randomizers, steps the orchestrator once per frame, then `wait_until_complete()`. The writer's `write()` runs synchronously inside each render step (see `.docs_claude/` notes on render/write scheduling).

## Where to look next

Documentation, plans, style guidance, and investigation notes live in `.docs_claude/`.

- `.docs_claude/plans/active/` -- plans currently in progress
- `.docs_claude/plans/completed/` -- finished plans (see `extract-isaac-datagen.md` for how this repo was extracted standalone)
- `.docs_claude/style-and-beliefs/` -- code style and design principles

## Plans & workflow

Plans are first-class artifacts in `.docs_claude/plans/`.

- **Small change** (one file, obvious fix): no plan needed.
- **Medium change** (new feature, wire up a subsystem): lightweight plan in `plans/active/`.
- **Complex change** (new architecture, pipeline redesign): full execution plan with goal, approach, staged checklist, and decision log in `plans/active/`.

Move completed plans to `plans/completed/`.

**Before planning any new implementation:**
1. Read `plans/active/` -- don't duplicate in-progress work.
2. Read `plans/completed/` -- learn from past decisions and avoid re-solving solved problems.
3. Read relevant docs in `.docs_claude/` -- context that shaped the current design.

## Core beliefs

Before planning any implementation, read `/reusable-parts` and apply its guidelines to the design.

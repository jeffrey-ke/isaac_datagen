---
description:
alwaysApply: true
---

# isaac_datagen

Standalone Isaac Sim Replicator data-generation package for visual servoing.

## Project context: reference-prompted instance segmentation

Four sibling repos — `vision_core`, `reference_matching`, `isaac_datagen`, `segmentation` — implement one
system: **few-shot reference-prompted instance segmentation**. Given a single canonical reference image of
a target object, segment that object in novel observation images; this is the perception front-end for
visual servoing / robotic grasping.

Inference is a three-stage pipeline, and the reference image conditions every stage:

```
reference image (one canonical shot per class)
  ├─→ [reference_matching descriptors] DIFT tokens / M2F·DIFT FPN volumes — condition stages 2 and 3
  ▼
observation image
  │ 1. propose  [reference_matching]  ALIKED/LightGlue or GIM ref↔obs matching → candidate (x,y)
  │             point prompts on the observation (>50% outliers)
  │ 2. verify   [segmentation/verifier — in progress]  point descriptors grid_sampled from FPN volumes
  │             cross-attend a global reference representation → per-point inlier logits. Learned
  │             replacement for RANSAC/MAGSAC: correspondence is an expedient to get anchor-box-like
  │             candidates, so this is anchor-box classification ("does this point lie on an instance
  │             of the reference's class?"), not match verification — no 2-view geometry downstream.
  │ 3. segment  [segmentation]  verified points → SAM prompt_encoder (SAM is brittle to outlier
  │             prompts — hence stage 2); reference features injected into the frozen image encoder
  │             via GLIGEN/Flamingo-style tanh-gated cross-attention (GligenWrapper)
  ▼
instance mask of the target object
```

How the repos fit:

| Repo | Role in the system |
|---|---|
| `vision_core` | Shared library: the serializable sample datastructs that are the dataset contract between repos (`ObsMask`/`ObsMaskMetadata` → `PreReferenceSegSample` → `PreImageInlierSample`; `ReferenceSegSample`; `StereoSample`) plus mask/pose/viz/transform/config utilities. |
| `reference_matching` | Stage-1 proposers plus the descriptor backbones for stages 2 and 3 (`M2FFpn`/`DiftFpn` volumes the verifier samples; `DiftDescriptor` tokens the gligen blocks consume). Library-only editable dep of both pipeline repos. |
| `isaac_datagen` **(this repo)** | Isaac Sim Replicator synthetic data generation: renders the phased datasets that train both learned stages — `PreImageInlierSample` (phase-2 proposals + phase-3 union-mask inlier labels: a point on ANY same-class instance is an inlier) for the verifier, `ReferenceSegSample` for the SAM fine-tune. Hosts the verifier design pseudocode (`verifier`) and its design note (`.docs_claude/multiscale-point-descriptor.md`). |
| `segmentation` | Stage-3 training/eval: `GligenWrapper` installs gated cross-attention on a frozen point-prompted SAM; hermetic Lightning checkpoints. Also hosts the stage-2 verifier implementation (`segmentation/verifier/`). |

Both learned stages (2 and 3) train from the same render dirs and condition on the same per-class
reference descriptors (`ObsMaskMetadata.class_to_descriptors`).

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
| `filters.py` | GraspableObject filter registry (name+args specs) | `filter_objects`, `FilterSpec`, `ClassFilter`, `ShuffleFilter` |
| `objects.py` | Graspable objects, occupancy placement, sample datastructs | `GraspableObject`, `ProtoReferenceSegSample`, `OccupancyGrid` |
| `isaac_utils.py` | USD/Replicator helpers (camera-from-K, transforms, prim search, USDZ export) | `setup_camera`, `load_asset`, `set_transform`, `find_prims`, `export_subtree_usdz`, `export_flattened_usdz` |
| `hardwares.py` | ZED Mini stereo camera rig | `ZedMini` |
| `stereo_writer.py` | Replicator Writer → `StereoSample` | `StereoSampleWriter` |
| `reference_seg_writer.py` | Replicator Writer → `ProtoReferenceSegSample` (precomputes DIFT ref features) | `ProtoReferenceSegWriter` |
| `mesh_convert.py` | Build a `GraspableObject` dataset from arbitrary meshes (+ YCB download): stage candidate renders, then finalize winners | `convert`, `finalize`, `ycb_download` |
| `mesh_blender.py` | Blender worker: mesh → `/World` usdz + 4 side-face ortho reference tiles | (run via `blender --background`) |

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

`mesh_convert.py`'s reference-image render generalizes the original `build_object_dataset.py`
(headless Blender, ortho cam at the −Y face), preserved verbatim at
`visual_servoing/datagen2_isaacsim/.build_object_dataset.py.bak`. `mesh_blender.py` extends it to
all 4 side faces and orients each ortho camera with `cv2opengl(look_at)`.

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

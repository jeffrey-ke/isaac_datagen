---
description:
alwaysApply: true
---

# isaac_datagen

Standalone Isaac Sim Replicator data-generation package for visual servoing.

## Project context: reference-prompted instance segmentation

Four sibling repos — `vision_core`, `reference_matching`, `isaac_datagen`, `segmentation` — implement one
system: **goal-conditioned suction-grasp pose prediction from a single reference image**. The runtime
input is **RGB-D** (a point-map, since camera intrinsics are known), one canonical reference image of the
target object, and a **grasp-goal** specification. Stages 1–3 are a few-shot **reference-prompted instance
segmentation** front-end — given the reference, segment the target in the observation; stage 4 then
segments the point-cloud with that mask and predicts the **6-DoF pose of the desired grasp point** in the
camera frame for a **suction-cup** end-effector. Target objects are box- or can-shaped.

Inference is a four-stage pipeline; the reference image conditions every segmentation stage:

```
reference image (one canonical shot per class)
  ├─→ [reference_matching descriptors] DIFT tokens / M2F·DIFT FPN volumes — condition stages 2 and 3
  ▼
observation RGB-D (point-map; intrinsics known)
  │ 1. propose  [reference_matching]  ALIKED/LightGlue or GIM ref↔obs matching → candidate (x,y)
  │             point prompts on the observation (>50% outliers)
  │ 2. verify   [segmentation/verifier — in progress]  point descriptors grid_sampled from FPN volumes
  │             cross-attend a global reference representation → per-point inlier logits. Learned
  │             replacement for RANSAC/MAGSAC: correspondence is an expedient to get anchor-box-like
  │             candidates, so this is anchor-box classification ("does this point lie on an instance
  │             of the reference's class?"), not match verification — no 2-view geometry downstream.
  │ 3. segment  [segmentation]  verified points → SAM prompt_encoder (SAM is brittle to outlier
  │             prompts — hence stage 2); reference features injected into the frozen image encoder
  │             via GLIGEN/Flamingo-style tanh-gated cross-attention (GligenWrapper) → instance mask
  │ 4. grasp    [in design; trained on isaac_datagen renders]  the mask segments the RGB-D
  │             point-cloud; a grasp-goal spec (the desired grasp — e.g. an approach direction in
  │             spherical coords about the object centroid, or a 2D pixel on the reference) conditions
  │             a per-point heatmap over the object surface → hotspot = suction contact point, its
  │             surface normal = approach direction → full 6-DoF pose
  ▼
6-DoF suction-grasp pose of the goal-specified point, in the camera frame
```

How the repos fit:

| Repo | Role in the system |
|---|---|
| `vision_core` | Shared library: the serializable sample datastructs that are the dataset contract between repos (`ObsMask`/`ObsMaskMetadata` → `PreReferenceSegSample` → `PreImageInlierSample`; `ReferenceSegSample`; `StereoSample`) plus mask/pose/viz/transform/config utilities. |
| `reference_matching` | Stage-1 proposers plus the descriptor backbones for stages 2 and 3 (`M2FFpn`/`DiftFpn` volumes the verifier samples; `DiftDescriptor` tokens the gligen blocks consume). Library-only editable dep of both pipeline repos. |
| `isaac_datagen` **(this repo)** | Isaac Sim Replicator synthetic data generation: renders the phased datasets that train both learned stages — `PreImageInlierSample` (phase-2 proposals + phase-3 union-mask inlier labels: a point on ANY same-class instance is an inlier) for the verifier, `ReferenceSegSample` for the SAM fine-tune. Hosts the verifier design pseudocode (`verifier`) and its design note (`.docs_claude/multiscale-point-descriptor.md`), and the stage-4 grasp-pose research note (`.docs_claude/grasp-pose-stage-research.md`); will render the grasp-pose training data. |
| `segmentation` | Stage-3 training/eval: `GligenWrapper` installs gated cross-attention on a frozen point-prompted SAM; hermetic Lightning checkpoints. Also hosts the stage-2 verifier implementation (`segmentation/verifier/`). |

Both segmentation stages (2 and 3) train from the same render dirs and condition on the same per-class
reference descriptors (`ObsMaskMetadata.class_to_descriptors`). The **stage-4 grasp-pose head**
(suction-cup; goal-conditioned) is in design — it also trains on `isaac_datagen` renders, but conditions
on the grasp-goal spec and the mask-segmented point-cloud rather than the reference descriptors. See the
research note `isaac_datagen/.docs_claude/grasp-pose-stage-research.md`.

## Quick start

```
uv run clean_datagen.py <config.yaml>
```

Config + OmegaConf dotlist overrides, e.g. `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml idx=0 num_frames=8`. Console script: `isaac-datagen` → `reference_segmentation`.

## Module index

| Module | Role | Key exports |
|---|---|---|
| `clean_datagen.py` | Entry point; orchestrates config→sim→scene→capture | `reference_segmentation`, `collect_objects` |
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

External (sibling editable packages + heavy deps): `vision_core` (datastructs `SerializableSample`/`StereoSample`/`ReferenceSegSample`, `pose_utils`; source at `/home/jeffk/repo/vision_core/src/vision_core/`, `datastructs.py` is the dataset contract — the (de)serialization routines `SerializableSample.deserialize`/`deserialize_field`/`serialize` live at `datastructs.py:120`/`:129`/`:99`, dispatching on the per-type `_serializers` table at `:68`; `GraspableObject` (datagen-side, `objects.py:28`) inherits `deserialize` and only extends that table — `GraspableObject.deserialize(idx, dir)` reads the `usd_path/`+`meta/`+`reference_image/`+`grasp_point/` subdirs of `object_dataset_amazon/`), `reference_matching` (DIFT `descriptor`, `proposal`; source at `/home/jeffk/repo/reference_matching/src/reference_matching/`), `isaacsim==5.1.0` + `omni.replicator.core`, `pxr`/USD, torch/torchvision. These editable deps import only inside the project venv — use `uv run` (plain `python3` has no venv).

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
         ↓
 render{idx:03d}/  (rgb/ ref_rgb/ seg_mask/ reference_features/) + runtime/descriptor yaml
```

Capture mechanism: `capture_with_poses` → `capture_session` opens `rep.new_layer()`, attaches the writer to the camera's render products, uses `rep.trigger.on_frame()` to move the camera through the planned world poses and apply randomizers, steps the orchestrator once per frame, then `wait_until_complete()`. The writer's `write()` runs synchronously inside each render step (see `.docs_claude/` notes on render/write scheduling).

## Viewing GraspableObject assets in open3d (`correspondence/`)

Inspect a `GraspableObject`'s `.usdz` mesh + its `grasp_point` SE3 in open3d. **open3d cannot read USD/USDZ** (`read_triangle_mesh` → "unknown file extension"), so a two-stage, two-env pipeline converts to formats it can:

- `correspondence/extract_mesh.py <dataset_dir> <data_dir> [idx]` — **Stage A**, run `uv run --with usd-core python …` (needs `vision_core` + standalone `pxr` from `usd-core`; plain `uv run` has no `pxr` unless a full kit boots). Batch-`GraspableObject.deserialize(idx, dataset_dir)` over every sample (counted from `usd_path/`) → pxr reads each usdz: bakes points to the object frame (`ComputeLocalToWorldTransform`), fan-triangulates n-gons, captures faceVarying `st` UVs aligned to those triangles, and pulls the bound `UsdUVTexture` PNG straight out of the usdz zip → dumps `_<name>.npz` + `<name>_texture.png` (names from `meta["name"]`) into `<data_dir>`.
- **Stage B** (run `uvx --from open3d python …`; env-agnostic npz bridges the two venvs), two builders:
  - `build_obj_axes.py <data_dir> <out_dir>` — **the per-object viewer artifact**: one self-contained **multi-material** `<name>.obj` (+`.mtl`+`<name>_tex.png`) per npz, carrying the textured object mesh (`map_Kd`) *and* baked origin + grasp coordinate axes as flat-color materials (`Kd`). Multi-material is what lets one OBJ show texture *and* colored axes; opens in any viewer.
  - `build_ply.py <npz>` — single-object alternative: `<name>.ply` (grey mesh + frames, vertex-colored) **and** a single-material textured `<name>.obj`.
  - **Do NOT flip UVs.** USD `st`, OBJ `vt`, and open3d `triangle_uvs` all use a **bottom-left** origin — pass `st` through verbatim. A `v → 1−v` flip *looks* harmless on box geometry (it just vertically mirrors within each face) but scrambles **atlas** textures (YCB), placing the label on the wrong region. Verify orientation against a non-box, atlas-textured mesh (e.g. the French's mustard bottle), not a box.
- `correspondence/view.py` — installed on PATH as **`plyview`** (PEP-723 self-contained `uv run --script`; `uv` provisions open3d). `plyview FILE [--frame] [--grasp] [--wire] [--save PNG]`. **`.obj`/`.glb`/`.gltf` load via `read_triangle_model`** so per-part materials render (object texture + colored axes) — `read_triangle_mesh` would flatten them to one untextured material (everything black). Other files use the legacy path; `--frame`/`--grasp` add large *separate* frame geometries (axis len ~0.7×bbox-diag, so they clear the surface — a triad baked at the centroid hides inside the opaque mesh), `--grasp` reads the SE3 from sibling `_<name>.npz`, `--wire` renders edges so an interior frame shows through, `--save` renders offscreen (EGL headless; reuse one `OffscreenRenderer` per process — a second EGL context crashes).

Appearance **is** recoverable — the "geometry only" limit is about a naive pxr point/face dump, not the asset: the usdz bundles the diffuse texture + UVs + a `UsdPreviewSurface`. Textures don't survive a USD *flatten* (UsdShade bindings are dropped — color is lost when re-importing a flattened usdz into Blender), but reading them directly off the bound `UsdUVTexture` does work.

## Where to look next

Documentation, plans, style guidance, and investigation notes live in `.docs_claude/`.

- `.docs_claude/plans/active/` -- plans currently in progress
- `.docs_claude/plans/completed/` -- finished plans (see `extract-isaac-datagen.md` for how this repo was extracted standalone)
- `.docs_claude/style-and-beliefs/` -- code style and design principles
- `.docs_claude/psc-isaac-datagen-footguns.md` -- **read before debugging a stuck PSC/Singularity render**: the operational hazard map (Vulkan ICD, RT-core-only GPU = L40S, container `.venv` rules, sibling/HF-asset sync, Slurm/backfill, optflow config, the finalize/timeout landmine)

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
4. Scan `../alldocs/` -- the workspace aggregate that symlinks **every** sibling repo's `.docs_claude/`
   (`vision_core`, `reference_matching`, `isaac_datagen`, `segmentation`, `UFM-train`, `model`,
   `nnscope`, `benchmark`) into one tree. Earlier implementation plans there often record **design
   decisions to maintain**, **possible bugs / incorrect behavior** to watch for, and **why** a decision
   was made -- i.e. what to look for and where. Changes here ripple across repos (the shared dataset
   contract, the pipeline stages), so read the cross-repo plans before designing.

## Core beliefs

Before planning any implementation, read `/reusable-parts` and apply its guidelines to the design.

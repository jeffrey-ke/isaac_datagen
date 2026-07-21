# `clean_datagen` — design and usage

> **What this document covers.** `clean_datagen` is the single Isaac Sim entry point of the
> `isaac_datagen` package: the program that boots Isaac Sim with RTX path tracing, assembles a
> scene, plans camera poses, renders frames, and writes a self-describing *render directory* that
> everything downstream in the thesis stack consumes. This document is the canonical reference for
> it, at three altitudes: **Part I** — what it is, how it fits the system, the lifecycle of one
> invocation, the output contract, and the architectural commitments; **Part II** — the ten
> subsystems it is assembled from, each with its extension recipe; **Part III** — the operator's
> manual: install, CLI, every config key, the shipped-config catalog, a cookbook, the on-disk
> layout, debugging, and footguns.
>
> **Date reflected: 2026-07-20.** Assembled from the `isaac_datagen/.docs_claude/plans/`
> corpus (`{active,completed}/`) reconciled against the current source tree at
> `isaac_datagen/src/isaac_datagen/`. **Where a plan document and the source disagree, the source
> is stated as truth and the plan is cited as history.** Every `path:line` citation and every
> `plans/…` citation is deliberate; they are the provenance of the claims around them.
>
> All source paths are relative to `isaac_datagen/src/isaac_datagen/` unless noted. All plan
> citations are relative to `isaac_datagen/.docs_claude/plans/{active,completed}/`; the ones marked
> *active* are open work, not settled design.

---

## Table of contents

**[Part I — System overview and architecture](#part-i--system-overview-and-architecture)**

- [I.1 What `clean_datagen` is and the problem it solves](#i1-what-clean_datagen-is-and-the-problem-it-solves)
- [I.2 The two generation modes](#i2-the-two-generation-modes)
- [I.3 Lifecycle of one invocation](#i3-lifecycle-of-one-invocation)
- [I.4 The output contract](#i4-the-output-contract)
- [I.5 Architectural principles the codebase commits to](#i5-architectural-principles-the-codebase-commits-to)
- [I.6 How it got here](#i6-how-it-got-here)

**[Part II — Subsystem design](#part-ii--subsystem-design)**

- [II.1 Configuration](#ii1-configuration)
- [II.2 Object datasets](#ii2-object-datasets)
- [II.3 The filter registry](#ii3-the-filter-registry)
- [II.4 Scene construction](#ii4-scene-construction)
- [II.5 Placement policies](#ii5-placement-policies)
- [II.6 Lighting and domain randomization](#ii6-lighting-and-domain-randomization)
- [II.7 Capture](#ii7-capture)
- [II.8 Writers](#ii8-writers)
- [II.9 Store-USD scenes](#ii9-store-usd-scenes)
- [II.10 Downstream phases](#ii10-downstream-phases)

**[Part III — Usage and operations reference](#part-iii--usage-and-operations-reference)**

- [III.1 Install and environment](#iii1-install-and-environment)
- [III.2 CLI reference](#iii2-cli-reference)
- [III.3 Config reference](#iii3-config-reference)
- [III.4 Shipped config catalog](#iii4-shipped-config-catalog)
- [III.5 Cookbook](#iii5-cookbook)
- [III.6 Output layout](#iii6-output-layout)
- [III.7 Verification and debugging](#iii7-verification-and-debugging)
- [III.8 Footguns and operational notes](#iii8-footguns-and-operational-notes)

**[Appendices](#appendix-a--known-documentation-staleness)**

- [Appendix A — Known documentation staleness](#appendix-a--known-documentation-staleness)
- [Appendix B — Open questions and unverified claims](#appendix-b--open-questions-and-unverified-claims)

---

# Part I — System overview and architecture

## I.1 What `clean_datagen` is and the problem it solves

`clean_datagen` is the single Isaac Sim entry point of the `isaac_datagen` package. One invocation
boots Isaac Sim with RTX path tracing, assembles a scene out of a catalog of serialized objects,
plans a set of camera poses, renders N frames, and writes a self-describing *render directory* to
disk (`clean_datagen.py:57-97` for reference-segmentation mode, `:100-142` for optflow mode).
Everything downstream in the thesis stack consumes render directories; nothing downstream boots
Isaac.

### Where it sits in the four-repo system

```text
   vision_core ────────────────────────────────────────────────────────┐
   (SerializableSample, datastructs: ObsMask / OptFlowSample /          │ imported by
    ObsMaskDescriptorMetadata / OptFlowMetadata / ImageInlier*,         │ all three
    pose_utils, mask_utils, seed_utils, viz)                            │
        ▲                    ▲                        ▲                 │
        │                    │                        │                 │
   reference_matching   isaac_datagen             segmentation ─────────┘
   (DIFT / CleanDIFT     ── clean_datagen ──►     (stage-1 proposer,
    descriptors,            RENDER DIRS            stage-2 verifier,
    proposers, GIM/                                stage-3 GLIGEN +
    LightGlue matchers)                            segmenter training)
```

The data contract between the repos is the *datastruct*, not an API. `isaac_datagen` never imports
`segmentation`; it imports `vision_core` for the serialization schema and `reference_matching` for
the descriptor backbone it bakes into each render dir. That split is not aesthetic — it is a
dependency-resolution fact. `isaacsim==5.1.0` hard-pins `numpy==1.26.0`, `pillow==11.3.0`,
`torch==2.7.0`, while `segmentation` floors `numpy>=2.4`, `pillow>=12.2`, `torch>=2.11`; the two
are unsatisfiable in one environment, so the descriptor/proposer cluster was extracted into
`reference_matching` with *deliberately unpinned* torch/numpy/pillow so it floats to whatever env
loads it (`plans/completed/extract-isaac-datagen.md`, "reference_matching — second extraction").
The packaging consequences of that pin structure are in
[§III.1](#iii1-install-and-environment).

### Where it sits in the dataset pipeline

`clean_datagen` is **phase 1**. The phases are separate console scripts, each its own process,
chained by `run_pipeline.py`:

```text
  phase 1   isaac-datagen                  Isaac Sim   RGB-D + cid/iid masks + reference
            (clean_datagen:main)                       descriptors  ->  render{idx:03d}/
                │
                ├─ validate: validate_render_dir() — cid/iid orphan check, hard stop
                │            (run_pipeline.py:93-102)
                ▼
  phase 2   isaac-datagen-proposals        no Isaac    per-class candidate points, gated by
            (add_proposals:main)                       reprojection coverage -> proposals/
                │
  phase 2.5 isaac-datagen-downsample-proposals (optional) FPS-cap points per class
                │
                ▼
  phase 3   isaac-datagen-inliers          no Isaac    inlier/outlier label per point
            (add_inlier_data:main)                     -> labels/, stats/

  side branch:
            isaac-datagen-unseen           takes a finished phase-1 dir, subsets + R/B-swaps
            (make_unseen:main)             it, re-encodes references, re-runs phases 2+3
                                           into a dir kept OUTSIDE data.paths (0-shot eval)
```

The phase split exists for a hard reason: Isaac Sim only releases GPU memory when its process
exits, and the phase-2 proposer needs on the order of 14 GB (`run_pipeline.py:10-13`,
`plans/completed/pipeline-orchestrator.md`). So phases cannot be one process, and the render dir on
disk is the only handoff. The interface between them is detailed in
[§II.10](#ii10-downstream-phases); running them is [§III.2](#iii2-cli-reference).

Resumability is asymmetric and this asymmetry is load-bearing:

| Phase | Resumable? | Mechanism |
|---|---|---|
| 1 render | **no** | `run_pipeline` skips it entirely if `obs/` already has ≥1 frame; to re-render you delete `render{idx:03d}/` (`run_pipeline.py:83-91`) |
| 2 proposals | yes | membership-set check over existing `proposals_*.pt`, atomic writes (`plans/completed/add-proposals-resumable.md`) |
| 3 inliers | n/a — unconditional rewrite | labels are cheap; relabelling under a new `--eps` must be transparent (`add_inlier_data.py:22-40`) |

Phase 1 being non-resumable is the single most expensive operational fact about the system:
`finalize_metadata()` runs only at the very end (`clean_datagen.py:90`, `:135`), so a timeout
mid-capture leaves `obs/` and masks on disk with no metadata and no `runtime.yaml` — an unusable
directory that must be deleted and re-rendered from scratch
(`.docs_claude/psc-isaac-datagen-footguns.md`; operational consequences in
[§III.8.1](#iii81-the-render-is-not-resumable-and-half-a-render-is-worthless)).

### The problem it solves

Reference-prompted segmentation needs, per observation frame, all of: an RGB observation; a
per-instance and per-class ground-truth mask; a *canonical reference image per class* plus its
precomputed descriptor tokens; per-instance occlusion; and — for correspondence supervision —
metric depth, the camera pose, the reference camera's intrinsics/pose, and the world transform of
every instance. No real-capture rig produces that tuple. Synthetic rendering does, and
`clean_datagen` is the machine that produces it reproducibly from a seed, a config, and a catalog
of objects.

---

## I.2 The two generation modes

`mode` is a mandatory `RuntimeConfig` field with no default, validated to be exactly
`"reference_segmentation"` or `"optflow"` (`runtime_config.py:148`). `main()` dispatches on it:
`optflow` → `optflow_generation()`, anything else → `reference_segmentation()`
(`clean_datagen.py:159-162`). The console script `isaac-datagen` is
`isaac_datagen.clean_datagen:main`, i.e. one script for both modes (`pyproject.toml [project.scripts]`).

### `reference_segmentation`

Collects `GraspableObject`s (`collect_objects`, `clean_datagen.py:27-40`), writes through
`ObsMaskWriter` (`reference_seg_writer.py:108`). Per frame it emits an `ObsMask`: `obs` (RGBA),
`iid_mask` (instance ids), `cid_mask` (class ids), `iid_to_occlusion`
(`vision_core/datastructs.py:317-321`). At finalize it emits one `ObsMaskDescriptorMetadata` per
render dir: `iid_to_name`, `cid_to_class`, `name_to_class`, `class_to_ref`, `class_to_descriptors`,
`principal_components` (`vision_core/datastructs.py:383-389`).

Consumers: phase 2 (the proposer needs `class_to_ref` + `class_to_descriptors`), phase 3 (inlier
labels are computed against `cid_mask == class_to_cid[cls]`, `add_inlier_data.py:27-28`), and the
stage-2 verifier / stage-3 segmenter training in `segmentation`.

### `optflow`

Collects `OptFlowObject`s (`collect_preoptflow`, `clean_datagen.py:43-54`), writes through
`OptFlowWriter` (`optflow_writer.py:19`). Per frame it emits an `OptFlowSample` that **nests a
complete `ObsMask`** plus `observation_depth`, `cam2world` (OpenCV convention), and
`iid_to_visibility` (`vision_core/datastructs.py:413-417`, produced at `optflow_writer.py:58-68`).
Its metadata `OptFlowMetadata` nests the full `ObsMaskDescriptorMetadata` as `obsmaskmeta` and adds
`obs_intrinsics`, `class_to_name`, `class_to_reference{,_depth}`, `class_to_ref_intrinsics`,
`class_to_ref_pose`, `class_to_l2w` (`vision_core/datastructs.py:507-515`).

Consumers: dense-matcher training (RoMa / UFM / DKM / LoFTR) via the correspondence warp, and —
*because* of the nesting — phases 2 and 3 and the seg pipeline, unchanged. That was the explicit
design goal of the nesting refactor: one physical dataset on disk serving both the UFM adapter and
seg-pipeline phases 2/3 without duplication (`plans/completed/optflow-6-nested-obsmask.md`). Note
the backward-compatibility cost: render dirs produced before that refactor used a flat
`observation`/`cid_mask`/`iid_mask` layout and must be regenerated.

The 1-to-many mechanic is what makes optflow *class*-keyed rather than instance-keyed: one
canonical reference per class is warped into every same-class instance through that instance's own
`local2world`, stacked as `class_to_l2w[cls]` of shape `(N,4,4)` (`optflow_writer.py:88-91`,
`plans/completed/optflow-4-class-keyed-one-to-many.md`). This mirrors the segmentation objective
exactly — a point on *any* same-class instance is an inlier. The mechanism is in
[§II.8](#ii8-writers).

### Why one skeleton

Read side by side, `reference_segmentation()` and `optflow_generation()` are the same eleven
statements. They differ in exactly three places: the collector, the writer construction, and
(optflow only) an extra `get_target2world(scene.object_prim_paths)` call to capture per-instance
world transforms for the writer (`clean_datagen.py:113-130`). Everything else — render-dir
creation, `cid_iid_trace.init`, `seed_everything(effective_seed)`, `boot_sim`, `filter_objects`,
`plan_capture`, the dry-run early exit, `make_replicator`, `warmup_render`, `capture_with_poses`,
`finalize_metadata`, the two provenance dumps, `app.close()` — is duplicated verbatim. The two
writers even share their reference-catalog construction and their per-frame ObsMask assembly:
`OptFlowWriter` imports `reference_catalog`, `obsmask_from_data` and `obsmask_metadata` straight
out of `reference_seg_writer` (`optflow_writer.py:14`).

**Current-source caveat on mode symmetry.** The symmetry is real but *incomplete*, and the docs do
not say so. `optflow_generation()` routes scene construction through the registry —
`scene_builders.get(runtime.scene_builder)(runtime, objects)` (`clean_datagen.py:117`) — while
`reference_segmentation()` calls `build_scene(runtime, objects)` directly (`clean_datagen.py:74`),
silently ignoring `runtime.scene_builder`. `RuntimeConfig` validates `scene_builder` non-empty for
*both* modes (`runtime_config.py:124`), so a reference-segmentation config that sets
`scene_builder: build_store_scene` passes validation and then silently builds a plain scene.
Consequently the store scenes (`build_store_scene`, `build_repopulated_store_scene`, registered at
`scene_builders.py:3-4`) are reachable **only from optflow mode**, and all shipped store configs
are optflow configs. That is a real asymmetry in the current source, not a documented policy; it is
tracked in [Appendix B](#appendix-b--open-questions-and-unverified-claims).

---

## I.3 Lifecycle of one invocation

```text
argv:  isaac-datagen configs/expanded-refseg-v2.yaml idx=0 num_frames=100 num_targets=10
                          │              └──────────── dotlist overrides ───────────┘
                          ▼
 [1] main()                       argparse: one positional `config`, everything else passed
     clean_datagen.py:145-162     through parse_known_args as a dotlist. Zero args -> print
                                  TLDR help and exit 0.
                          │
                          ▼
 [2] load_config()                OmegaConf.merge(structured(RuntimeConfig), yaml, dotlist)
     runtime_config.py:185-192    -> to_object -> RuntimeConfig.__post_init__ validation.
                                  `${call:fn args}` resolver lets YAML invoke module-level
                                  functions at load time (e.g. _glob_amazon_textures,
                                  runtime_config.py:167-174).
                                  ALL validation happens HERE, before the ~3-minute sim boot:
                                  num_frames XOR grid_dims; frame window sanity; eps >= 0;
                                  light-range bounds; exposure > 0; mode in {...};
                                  objects_path non-empty; dataset_dir / intrinsics_path /
                                  proposer_config_path / descriptor_config_path all exist
                                  (runtime_config.py:123-153).
                          │
                          ▼
 [3] dispatch on runtime.mode  ──►  reference_segmentation()  |  optflow_generation()
                          │
                          ▼
 [4] render_dir = dataset_dir / f"render{idx:03d}"; mkdir(parents, exist_ok)
     cid_iid_trace.init(render_dir)        append-only log of Isaac semantic tokenization
     seed_everything(runtime.effective_seed)   effective_seed = seed + idx
     (clean_datagen.py:61-65, :104-108)        (runtime_config.py:162-164)
                          │
                          ▼
 [5] boot_sim(runtime, render_dir)        SimulationApp: path tracing (spp, max_bounces),
     scene.py:305-348                     denoiser, exposure (set_exposure / exposure_time /
                                          f_number / film_iso), texture streaming.
                          │
                          ▼
 [6] objects = filter_objects(collect_{objects,preoptflow}(runtime.objects_path),
                              runtime.filter_specs)
     collectors chain multiple dataset dirs by globbing meta/meta_*.yaml and deserializing
     by index; duplicate meta['name'] across datasets raises ValueError
     (clean_datagen.py:27-54). Filters (Shuffle/Meta/Replicate/Regex) run in declared order
     and raise if the pool ever becomes empty (filters.py:26-37).
                          │
                          ▼
 [7] scene = build_scene(...)   /   scene_builders.get(runtime.scene_builder)(...)
     Stage -> assets -> lights -> placement (placers registry) -> mutations -> grasp frames
     -> camera rig. Returns SceneHandle(zed, grasp_points, objects, object_prim_paths)
     (scene.py:18-23, :436-492).  [ mechanism: §II.4 ]
     optflow additionally: l2w = get_target2world(scene.object_prim_paths)
                          │
                          ▼
 [8] _idx, _grasp_points, world_poses = plan_capture(runtime, scene)
     capture.py:26-34    sample grasp targets -> read target2world off the USD stage ->
                         instantiate poser from the registry -> poser(num_frames) gives
                         target-frame SE(3) -> einsum-project to world -> flatten to
                         (num_targets * num_frames, 4, 4).
                         Deterministic given effective_seed; stochastic randomizers are NOT
                         here, they are in the replicator layer. Pose list is immutable
                         from this point on.
                          │
             dry_run? ────┴──► export_debug_bundle(decorate_debug_scene(scene, world_poses),
                                                   render_dir); app.close(); RETURN
                                (clean_datagen.py:78-82, :122-126) — no frames, no metadata.
                          │
                          ▼
 [9] writer = ObsMaskWriter(descriptor_config_path, descriptor_device, scene.objects,
                            render_dir, full_alpha=obs_full_alpha)
     -- or --
     writer = OptFlowWriter(scene.objects, l2w, scene.zed.intrinsics, render_dir,
                            descriptor_config_path, descriptor_device, full_alpha=...)
     Writer __init__ runs reference_catalog(): assigns cid per class (sorted, start=2,
     0=BACKGROUND / 1=UNLABELLED), picks the canonical reference per class as the
     sorted-first member by meta['name'], and precomputes descriptors ONCE
     (reference_seg_writer.py:55-72, :134-137).
                          │
                          ▼
[10] replicator = make_replicator(runtime, len(world_poses), render_dir)
     warmup_render(app, runtime.warmup_frames)          <-- BEFORE capture_session, so that
     scene.py:227-256, :351-353                             warmup app.update() calls cannot
                                                            desync rep.distribution.sequence
                                                            (plans/completed/
                                                             pre-capture-render-warmup.md)
                          │
                          ▼
[11] capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=...)
     capture.py:99-112   rep.new_layer() -> broadcast writer<->render-products -> attach ->
                         bind pose sequence to the rig node -> one orchestrator.step() per
                         pose, per_frame() callback fires immediately before each step
                         (this is where per-frame light jitter is applied by direct USD
                         writes) -> writer.write(data) per frame serializes one sample.
                          │
                          ▼
[12] writer.finalize_metadata(render_dir)
     asserts iid_to_name is 1:1 (reference_seg_writer.py:153-154, optflow_writer.py:97-98),
     fits the PCA basis over all class descriptors, serializes metadata at index 0.
                          │
                          ▼
[13] yaml.safe_dump(asdict(runtime))      -> render_dir/runtime.yaml
     copy of the descriptor config        -> render_dir/descriptor.yaml
     (clean_datagen.py:92-95, :137-140)
                          │
                          ▼
[14] app.close()
```

Two properties of this flow are worth naming explicitly.

**Everything expensive is behind everything cheap.** Config validation, path existence, and mode
dispatch all complete before `boot_sim`. Scene construction, pose planning and asset loading all
complete before the writer is built, and `dry_run` exits right there — which is why dry-run is the
standard way to validate a config without paying for RTX (`clean_datagen.py:78-82`; see
[§III.7.2](#iii72-dry_run-and-the-blender-sanity-render)).

**Planning is separated from rendering.** `plan_capture` is a pure function of
`(runtime, scene-geometry)`; the pose array is fixed before any randomizer runs. Randomization
(lighting, textures) lives in the replicator layer and is applied per frame inside the step loop.
This is what allows frame-synchronized writes: frame *i* of the output corresponds to
`world_poses[i]` by construction, and the lighting schedule is precomputed in Python and logged to
`lighting_log.json` so "applied == logged"
(`plans/completed/lighting-jitter-mechanism.md`; mechanism in
[§II.6](#ii6-lighting-and-domain-randomization)).

---

## I.4 The output contract

One invocation produces exactly one directory, `{dataset_dir}/render{idx:03d}/`. `dataset_dir` must
pre-exist — `__post_init__` asserts it (`runtime_config.py:150`) — but the render dir is created by
the entry point (`clean_datagen.py:61-62`, `:104-105`).

The layout is not hand-authored: `SerializableSample` writes one subdirectory per dataclass field,
with zero-padded per-index filenames, and **nested samples flatten into the same namespace**. The
annotated full listing, with file extensions and operational notes, is
[§III.6](#iii6-output-layout); the summary shape is:

```text
render000/
├── obs/  iid_mask/  cid_mask/  iid_to_occlusion/      per-frame ObsMask fields
├── observation_depth/  cam2world/  iid_to_visibility/ per-frame, optflow only
├── cid_to_class/  name_to_class/  iid_to_name/        metadata, one entry at index 0
│   class_to_ref/  class_to_descriptors/               (nested as `obsmaskmeta` in optflow)
│   principal_components/
├── obs_intrinsics/  class_to_name/  class_to_l2w/     OptFlowMetadata, optflow only
│   class_to_reference{,_depth}/  class_to_ref_{intrinsics,pose}/
├── runtime.yaml  descriptor.yaml                      provenance
├── lighting_log.json  cid_iid_trace.log               logs
└── proposals/  labels/  stats/                        added by phases 2 and 3
```

Contract points that downstream code depends on:

- **`runtime.yaml` is the downstream config.** Phases 2/3 are commonly invoked *against the render
  dir's own `runtime.yaml`*, not against the original config, so they replay the exact capture
  parameters and seed. This is the reason `proposer_device`, `proposer_config_path`,
  `proposer_min_visible_ratio` and `inlier_border_eps` live in `RuntimeConfig` even though phase 1
  never reads them (`runtime_config.py:29-53`).
- **`runtime.yaml` is also the completion marker.** It is written only after `finalize_metadata()`
  succeeds. A render dir with `obs/` but no `runtime.yaml` is an aborted render, not a dataset.
- **cid numbering is `sorted(classes)` enumerated from 2**, with 0=BACKGROUND and 1=UNLABELLED per
  Isaac convention (`reference_seg_writer.py:56-57`). It is deterministic across render dirs *only*
  as a function of the class set: add or remove a class and the numbering shifts, which is why class
  merges require an explicit LUT remap (`plans/completed/optflow-5-cid-iid-masks.md`).
- **Instance ids are session-local.** The same numeric iid in two different render dirs is
  unrelated. The writer asserts `iid_to_name` is 1:1 at finalize; the validator deliberately keys on
  per-frame `iid_to_occlusion.keys()` rather than the cross-frame `iid_to_name`
  (`validate_obsmask.py:26-27`).
- **`iid_to_*` dicts serialize as `.pt`, not JSON**, because JSON stringifies integer keys and would
  break mask indexing (`plans/completed/cid-mask-dual.md`). The same `.pt` serializer covers every
  `dict`-typed metadata field, including `cid_to_class/` and `name_to_class/`.
- **`principal_components` is mandatory** on `ObsMaskDescriptorMetadata`. Deserializing a pre-PCA
  render dir fails loudly; `migrate_pca_basis.py` back-fills legacy dirs
  (`plans/completed/pca-basis-mandatory-field.md`).
- **`class_to_descriptors` is per-backbone** (a `SubfolderDict` keyed by backbone name), so a dir
  rendered with DIFT can later be given a CleanDIFT/FPN backbone by
  `migrate_descriptors_backbone add-backbone` without re-rendering.
- **A render dir is only valid if phase 1 completed.** No `runtime.yaml` ⇒ no metadata ⇒ the
  directory is garbage (see [§I.1](#i1-what-clean_datagen-is-and-the-problem-it-solves)).

`run_pipeline` enforces one integrity property before letting phase 2 spend GPU time: it runs
`validate_render_dir()` and hard-exits on any cid/iid orphan — an instance present in `iid_mask`
whose pixels are all background in `cid_mask` (`run_pipeline.py:93-102`,
`validate_obsmask.py:51-77`). The known root cause of orphans is that Isaac's
`instance_segmentation_fast` tokenizes class semantics on whitespace, so a class named `fish can`
is recorded as `fish` and misses the LUT
(`plans/completed/tuna-fish-can-cid-orphan-root-cause.md`). **Single-token class names are
effectively part of the contract.**

---

## I.5 Architectural principles the codebase commits to

These are stated as *commitments the source honours*, with the plan that introduced each.

### I.5.1 Config-driven `name + args` registries, resolved by module reflection

Every swappable policy is selected by a string naming a class or function in a module, with a
sibling `*_args` dict passed verbatim to its constructor. The lookup is
`getattr(sys.modules[__name__], name)` — registration *is* defining the symbol, there is no
registration step and no factory dict.

| Axis | Key | Args key | Registry |
|---|---|---|---|
| scene construction | `scene_builder` | `scene_builder_args` | `scene_builders.py:7-11` |
| object placement | `placement` | `placement_args` | `placers.py:12-16` |
| camera pose policy | `pose_generation_policy` | `pose_generation_policy_args` | `posers.py:15-19` |
| occluder placement | `occluder_pose_policy` | `occluder_pose_policy_args` | `posers.py:15-19` (same) |
| object filtering | `filter_specs` (ordered list of `{name, args}`) | — | `filters.py:16-23` |
| scene mutations | `mutations` (ordered list of `{name, args}`) | — | `store_mutations.py:23-31` |
| grasp-frame policy | `grasp_frame_policy` | `grasp_frame_policy_args` | `grasp_policies.py:18-38` |
| object orientation | `orientation` (`{name, args}`) | — | `orientations.py` |

The pattern was established by the poser registry, explicitly mirroring `segmentation`'s optimizer
registry (`plans/completed/pose-generation-poser-registry.md`), then replicated for placers
(`plans/completed/object-placer-registry.md`), filters
(`plans/completed/graspable-object-filter-registry.md`), and mutations — which the plan notes is
the sixth instance of the pattern in the codebase (`plans/completed/store-scene-mutations.md`).

Two rules ride along with the pattern:

1. **No silent defaults on shape-determining parameters.** A config that omits a required
   constructor parameter gets a `TypeError` at scene-build time rather than a plausible-looking
   wrong scene (`plans/completed/object-placer-registry.md`). `ShuffleFilter`'s seed is mandatory,
   because a silently unseeded shuffle would destroy reproducibility.
   *Current-source qualification:* the placers do carry defaults for their **jitter magnitudes**
   (`min_y`, `max_y`, `min_gap`, `max_gap`, `epsilon`); only `max_column_height` /`column_height`
   is required (`placers.py:72-73`, `:111-113`). The plan's blanket "no defaults" claim is history
   — see [§II.5](#ii5-placement-policies).
2. **Fail loud at parse, not at build.** `StoreSceneSpec`/`PlainSceneSpec` validate their mutation
   lists at construction; `PlainSceneSpec` rejects any mutation not marked `PLAIN_SAFE`, so a
   store-only mutation in a plain config errors immediately instead of raising `AttributeError`
   ~30 s after boot (`plans/completed/plain-scene-mutations-disable-physics.md`,
   `scene.py:402-423`).

### I.5.2 Separation of placement / pose *policy* from *mechanism*

`organize_objects(policy, prim_paths)` applies whatever `(translation, rotation)` a placer returns;
it contains no layout math (`scene.py:71-76`). `capture.py` never computes a pose; it asks the
poser for `(N,4,4)` in the target frame and projects it to world by einsum against `target2world`
read off the USD stage (`capture.py:26-34`). Adding `LookAtPoser` and `DecenteredLookAtPoser`
alongside `GridFixedPoser` required no change to `capture.py` (`posers.py:22-92`).

The same discipline governs the debug path. `decorate_debug_scene` is a pure mechanism (stage
mutation, no I/O); `export_debug_bundle` is the policy that persists. Both are dry-run only, and the
shared mechanisms — `plan_capture`, `se3_to_pos_euler`, `set_prim_pose`, the ZedMini intrinsics —
are single-sourced and called by both paths, so the dry-run view cannot drift from what the real
capture does (`plans/completed/blender-dry-run-sanity-renderer.md`).

### I.5.3 Mode symmetry

Both modes run the same eleven-step skeleton; both writers share `reference_catalog`,
`obsmask_from_data` and `obsmask_metadata`; `cid_iid_masks` was lifted into `isaac_utils` to be
shared rather than duplicated (`plans/completed/optflow-5-cid-iid-masks.md`); and `OptFlowSample`
nests `ObsMask` so that one dataset serves both consumers
(`plans/completed/optflow-6-nested-obsmask.md`). The `objects_path` unification removed the last
gratuitous asymmetry in the config: `mode` already selects the collector, so having separate
`graspable_objects_path` / `optflow_objects_path` fields only re-encoded that knowledge, and the
asymmetric validation had been letting a silent no-objects render through
(`plans/completed/unify-objects-path.md`; the surviving single field is `runtime_config.py:117`,
asserted non-empty at `:149`).

As noted in [§I.2](#i2-the-two-generation-modes), symmetry is *not* complete:
`reference_segmentation()` bypasses the `scene_builder` registry. That is unfinished, not
intentional.

### I.5.4 Provenance dumps and seeded reproducibility

Every render dir carries `runtime.yaml` (the full `RuntimeConfig` as a dict) and `descriptor.yaml`
(the descriptor config as loaded), written at the end of a successful render
(`clean_datagen.py:92-95`, `:137-140`). With `log_lighting`, it also carries `lighting_log.json`
recording the actual per-frame light schedule (`scene.py:252-255`).

Reproducibility rests on one scalar: `effective_seed = seed + idx` (`runtime_config.py:162-164`),
passed to `seed_everything` before anything stochastic happens (`clean_datagen.py:65`, `:108`).
`seed_everything` seeds Python `random`, the **global** `np.random`, and torch
(`vision_core/seed_utils.py`). A batch sweep over `idx` at fixed `seed` therefore yields independent
but individually reproducible samples; re-running the same `(seed, idx)` reproduces geometry, poses
and randomizer draws. Jitter schedules are drawn from decorrelated substreams
(`np.random.default_rng([effective_seed, k])`) so key-light and dome-light variation are
independent (`plans/completed/lighting-jitter-mechanism.md`).

One honest caveat, restated correctly against the source: occluder placement and occluder scale
both draw from the **globally seeded** `np.random` stream (`scene.py:377`, `:383`, `:387`) rather
than from a dedicated `default_rng([effective_seed, k])` substream. They are therefore reproducible
for a fixed `(seed, idx)`, but they are *order-coupled*: any change upstream that consumes a
different number of global draws shifts occluder layout. The same is true of camera poses, which
are deliberately drawn from the globally seeded stream inside `vision_core/pose_utils` rather than a
private RNG (`plans/completed/store-snacks-finetune-renders.md`). The plan-record phrasing
"occluder placement is unseeded" (`plans/completed/obs-full-alpha-toggle.md`) means *not
decorrelated*, not *not seeded*.

### I.5.5 Fail loud, early, and at the right layer

The recurring move across this codebase is to convert a silent wrong result into a loud early
error: duplicate object names across chained datasets raise before the stage is touched
(`clean_datagen.py:36-39`, because a duplicate name would collide in the USD prim path *and* in
`name_to_class`); `filter_objects` raises if the pool ever empties (`filters.py:29-30`);
`inlier_border_eps` is mandatory with no default so a run can never silently label with an
unintended margin (`runtime_config.py:42`, `plans/completed/inlier-border-eps-margin.md`);
`num_frames` XOR `grid_dims` is asserted (`runtime_config.py:128-129`); the writers assert
`iid_to_name` is 1:1 at finalize; `run_pipeline` refuses to proceed past a cid/iid orphan; and
`StoreSceneSpec.require_tracked_only` was added specifically so that a store config with no
mutations block cannot silently leave another dataset's held-out classes in frame as unlabeled
background (`plans/completed/store-snacks-training-datasets.md`).

---

## I.6 How it got here

**Extraction (`plans/completed/extract-isaac-datagen.md`).** `clean_datagen` was
`visual_servoing/datagen2_isaacsim/clean_datagen.py`. The extraction pulled the 11-module
transitive closure of that one entry point — `clean_datagen, scene, capture, pose_planning,
runtime_config, objects, isaac_utils, hardwares, stereo_writer, reference_seg_writer, __init__`
plus `resources/` — into a standalone uv package, and simultaneously extracted
`proposal.py + descriptor.py + gim_helper.py + dift/` into `reference_matching` to break the
torch/numpy pin conflict described in [§I.1](#i1-what-clean_datagen-is-and-the-problem-it-solves).
The object catalogs stayed external, loaded by config path. The package has been organized around
this one entry point ever since.

**Then, roughly in order of the completed-plan record:**

- *Scene generation became configurable rather than hard-coded.* `OccupancyGrid` (a static full-wall
  policy demanding exactly `prod(pallet_dims)` objects) gave way to the placer registry, then
  `UntilExhaustedStacker` with successive additions of depth jitter, gap jitter, and jagged
  (random-height) columns; `ShelfPlacer` arrived as a class-grouping subclass
  (`object-placer-registry`, `until-exhausted-stacker`, `depth-jitter-stacker`, `gap-jitter-stacker`,
  `jagged-columns-jitter-stacker`, `shelf-placer`).
- *Pose planning became a registry.* `plan_poses` was wrapped as `GridFixedPoser` for
  behaviour preservation, then joined by `LookAtPoser` and `DecenteredLookAtPoser`
  (`pose-generation-poser-registry`).
- *The mask contract went dual.* A single mask became `iid_mask` + `cid_mask`, with occlusion joined
  by prim path rather than by id because the occlusion annotator uses a different id space
  (`cid-mask-dual`, `obsmask-occlusion-and-viz`).
- *Lighting was rebuilt around a DistantLight key + DomeLight fill*, and per-frame jitter was
  re-routed from the (broken) Replicator randomizer graph to direct USD writes in the capture step
  loop (`distant-light-key-light`, `lighting-jitter-mechanism`).
- *The optflow mode was built in six plans* and converged on class-keyed 1-to-many correspondence
  with a nested `ObsMask` (`optflow-1` … `optflow-6`).
- *Store scenes arrived* as a whole second world model: externally authored store USD, product
  extraction to catalogs, a mutation registry, and per-class front-face grasp policies
  (`store-usd-inverse-datagen`, `store-scene-mutations`, `store-front-face-check`,
  `store-perclass-faces-verify-dataset`).
- *The pipeline was wrapped* into `run_pipeline` with validation, resumability and multi-GPU phase-2
  sharding (`pipeline-orchestrator`, `add-proposals-resumable`, `sharded-proposals`).
- *Gating moved from absolute pixel count to reprojection coverage ratio*, which is
  camera-distance invariant; `gate_classes` survives only for backward compatibility and is unused
  (`proposal_gate.py:7-15`, `reproj-coverage-gate-and-ycb-ref-pose-fix`).
- *The CLI grew a real help surface* (`tldr.py`, `cli-tldr-help`) and the config field set was
  cleaned up (`unify-objects-path`).

What remains unsettled is collected once, in
[Appendix B](#appendix-b--open-questions-and-unverified-claims).

---

# Part II — Subsystem design

This part walks the ten subsystems `clean_datagen` is assembled from, one section each:
responsibility, the key types and functions, the extension point, the design decisions and their
rationale, and the plan documents that established them.
[Part I](#part-i--system-overview-and-architecture) covers how these fit together at the
architecture level; [Part III](#part-iii--usage-and-operations-reference) covers invocation, config
catalogs and operational footguns.

## II.1 Configuration

### Responsibility

`runtime_config.py` owns the single configuration object for a render: `RuntimeConfig`. It is a
plain dataclass with ~70 fields covering scene selection, object sourcing, placement, pose
generation, lighting, exposure, path-tracing settings, output layout, and the parameters that
downstream phases 2 and 3 read back out of the serialized `runtime.yaml`. Nothing in the render path
reads YAML directly; everything reads a validated `RuntimeConfig`. The per-field table — every key,
its default, and its meaning — is [§III.3](#iii3-config-reference); this section covers the
machinery.

### Key types and functions

- `RuntimeConfig` (`runtime_config.py:21-164`) — the schema. Required fields (no default) come
  first: `idx`, `mode`, `num_targets`, `scene`, `dataset_dir`, `intrinsics_path`,
  `descriptor_device`, `proposer_device`, `proposer_config_path`, `descriptor_config_path`,
  `placement`, `dome_light`, `dry_run`, `inlier_border_eps`. A config that omits any of these fails
  at `OmegaConf.to_object` time, before Isaac boots.
- `LightJitterSpec` (`runtime_config.py:14-18`) — `{root, pattern, intensity_scale_range}`, the
  per-frame selective-light-jitter descriptor consumed by `scene.register_light_pattern_jitter`.
- `load_config(yaml_path, dotlist)` (`runtime_config.py:185-192`) — the loader.
- `RuntimeConfig.sampling` (`:155-160`) and `RuntimeConfig.effective_seed` (`:162-164`) — derived
  properties.

### The loader and the `${call:...}` resolver

`load_config` merges three layers in a fixed order (`runtime_config.py:188-191`):

```python
schema   = OmegaConf.structured(RuntimeConfig)     # defaults + types
yaml_cfg = OmegaConf.load(yaml_path)               # the config file
cli_cfg  = OmegaConf.from_dotlist(dotlist)         # key=value overrides
merged   = OmegaConf.merge(schema, yaml_cfg, cli_cfg)
```

Dotlist last is what makes `idx=0 num_frames=8` on the command line shadow the YAML. The structured
schema is what gives typed coercion: a YAML list of `{name, args}` dicts becomes a
`list[FilterSpec]` because the annotation says so, and a YAML list of
`{root, pattern, intensity_scale_range}` dicts becomes `list[LightJitterSpec]`. There is no include
or inheritance mechanism in the YAML layer — config reuse is by copy, which is why the config
directory has many near-duplicate files ([§III.4](#iii4-shipped-config-catalog)).

`register_resolvers()` (`:181-182`) installs one custom OmegaConf resolver, `call`, backed by
`_call(name, *args)` (`:177-178`), which is `getattr(sys.modules[__name__], name)(*args)`. It
resolves only against module-level functions of `runtime_config.py` itself. The single shipped
callee is `_glob_amazon_textures()` (`:167-174`), which lists
`<RESOURCE_PATH>/boxes/textures/amazon_texture_*` at load time so a YAML can say
`background_textures: ${call:_glob_amazon_textures}` instead of hardcoding 40 file paths. A typo in
the resolver name surfaces as an OmegaConf resolution error with the `AttributeError` buried inside
it — the failure mode is loud but not friendly; check the resolver first when `load_config` explodes
on a config that looks fine.

### `__post_init__` invariants

Validation is a flat block of asserts (`runtime_config.py:123-153`). They fall into four groups:

1. **Structural** — `scene_builder` must be non-empty; every `LightJitterSpec` must have a non-empty
   root and pattern and `0 < lo <= hi`.
2. **Mutual exclusion** — `(num_frames is None) ^ (grid_dims is None)`. Exactly one sampling mode.
   The `sampling` property then returns whichever is set, and callers never branch.
3. **Range checks** — `start_frame >= 0` and `end_frame > start_frame`; `inlier_border_eps >= 0`;
   `dome_intensity_range` ordered; `distant_offset_jitter >= 0`; `distant_intensity_jitter`
   satisfying `0 <= lo <= hi`; `distant_temperature_jitter` clamped to `1000 <= lo <= hi <= 10000` K
   (a physical Kelvin range, not an arbitrary one); `exposure_time`, `f_number`, `film_iso` all
   `> 0`; `rt_subframes >= 1`.
4. **Existence and enum** — `mode in ("reference_segmentation", "optflow")`; `objects_path`
   non-empty; `dataset_dir`, `intrinsics_path`, `proposer_config_path` and `descriptor_config_path`
   all must exist on disk.

The reasoning throughout is fail-at-load, not fail-at-render. Booting Isaac Sim with path tracing
costs minutes (first boot compiles RTX shaders for roughly three minutes; later boots on the same
node reuse the cache and start in ~15 s — `.docs_claude/psc-isaac-datagen-footguns.md`), so a config
typo must not be discovered after the boot. Note the asymmetry: `dataset_dir` must *pre-exist*
(`runtime_config.py:150`) even though `clean_datagen` immediately does
`render_dir.mkdir(parents=True, exist_ok=True)` (`clean_datagen.py:61-62`). A driver script that
creates `dataset_dir` after constructing the config will fail validation.

`inlier_border_eps` deserves a note: it is mandatory with no default even though phase 1 never uses
it. It is carried purely so that the `runtime.yaml` snapshot phase 3 reads back has an explicit
value — `inlier-border-eps-margin.md` records the decision as fail-loud rather than silently
labelling with an unintended margin.

### `effective_seed`

```python
@property
def effective_seed(self) -> int:
    return self.seed + self.idx
```
(`runtime_config.py:162-164`)

`clean_datagen` calls `seed_everything(runtime.effective_seed)` once, before `boot_sim`
(`clean_datagen.py:65`, `:108`), and `make_replicator` additionally calls Replicator's
`set_global_seed(runtime.effective_seed)` (`scene.py:227-230`). Everything downstream — column
chunking, gap and depth jitter, per-object lateral jitter, grasp-target choice, camera offsets,
occluder placement and scale — draws from the seeded global `np.random`. The consequence is the
intended one: a batch sweep over `idx` with a fixed `seed` yields independent-but-reproducible
renders, and a re-run of `(seed, idx)` reproduces the geometry exactly. K-shot fine-tune pools
deliberately choose a disjoint base seed (1001) from the frozen benchmark (1) so the two never
collide (`store-snacks-finetune-renders.md`).

Some RNG streams are deliberately *not* on the global stream: lighting jitter schedules use
`np.random.default_rng([effective_seed, k])` with `k` = 0 for the distant light, 1 for the dome, and
`(2, stream_index)` for each `light_jitter_patterns` entry (`scene.py:148`, `:164`, `:206-207`); the
store mutation chain uses `[effective_seed, 3]` (`store_mutations.py:34-41`). Decorrelating them
means changing the number of dome frames does not shift the key-light schedule.

### Provenance dump

At the end of a successful render both entry points write `render{idx:03d}/runtime.yaml`
(`yaml.safe_dump(asdict(runtime))`) and `render{idx:03d}/descriptor.yaml` (a re-dump of the loaded
descriptor config) — `clean_datagen.py:92-95` and `:137-140`. Phases 2 and 3 are invoked against
that `runtime.yaml`, not against the original config file, so they replay the exact effective
parameters including any CLI overrides.

### Extension recipe: adding a config field

1. Add the field to `RuntimeConfig` with a type annotation. If it must be supplied by every config,
   put it in the no-default block above `num_frames`; otherwise give it a default and put it in the
   defaults block (dataclass ordering is enforced by Python, not by us).
2. If it is a nested structure, declare a small dataclass (as `LightJitterSpec` does) and annotate
   the field `list[ThatType]` — OmegaConf's structured schema handles the coercion.
3. Add any invariant to `__post_init__` as an `assert` with the offending value in the message.
4. If the field must be computed at load time from the filesystem, add a module-level function to
   `runtime_config.py` and reference it from YAML as `${call:your_function}`.

---

## II.2 Object datasets

### Responsibility

`objects.py` defines the two serializable per-object records that a render consumes, and
`clean_datagen.py` defines the collectors that turn dataset directories into lists of them.

### The two record types

```python
@dataclass
class GraspableObject(SerializableSample):
    usd_path: UsdPath
    meta: dict
    reference_image: PILImage.Image
    grasp_point: np.ndarray
```
(`objects.py:26-31`)

```python
@dataclass
class OptFlowObject(SerializableSample):
    usd_path: UsdPath
    meta: dict
    reference_image: PILImage.Image
    reference_depth: np.ndarray
    ref_intrinsics: np.ndarray
    ref_pose: np.ndarray
    grasp_point: np.ndarray
```
(`objects.py:51-59`)

`OptFlowObject` is `GraspableObject` plus the reference camera: a metric depth map (float32, zero
outside the object), the 3×3 `K` that rendered it, and `ref_pose`, the 4×4 camera-to-object-local
SE3. `ref_pose` is stored in **OpenCV convention (+Z forward)**, never OpenGL, because the
downstream warp math (`get_gt_warp`, `instance_visibility`) is OpenCV-only; the GL pose used to
author the Isaac camera prim is transient and never persisted
(`optflow-1-reference-dataset.md`, "Two poses").

Correction against older plans: `grasp_point` is a **mandatory field on `OptFlowObject`** in current
source (`objects.py:58`). `store-usd-inverse-datagen.md` records this as a deliberate change after
three iterations — the grasp frame is baked once at extraction time and replayed verbatim at
capture, never re-derived, so that the reference face is a property of the dataset rather than a
runtime policy.

### The serialization contract

Both types share one `_serializers` table (`objects.py:32-48`; `OptFlowObject._serializers =
GraspableObject._serializers`, `:60`), layered on `SerializableSample._serializers` from
`vision_core.datastructs`. Three type-keyed entries are added:

| Field type | Extension | Write | Read |
|---|---|---|---|
| `UsdPath` (a `str` subclass, `objects.py:22-23`) | `.usdz` | `shutil.copy` | wrap the destination path |
| `PIL.Image.Image` | `.png` | `v.save(path)` | `PILImage.open(path).copy()` |
| `dict` | `.yaml` | `yaml.dump` | `yaml.safe_load` |

`np.ndarray` falls through to the base table (`.npy`). The on-disk layout is one subdirectory per
field, with 0-indexed numeric filenames:

```text
<dataset>/meta/meta_0000.yaml
<dataset>/usd_path/usd_path_0000.usdz
<dataset>/reference_image/reference_image_0000.png
<dataset>/grasp_point/grasp_point_0000.npy
...
```

Two properties of this layout matter. First, `serialize(idx, dir, only={...})` writes only the named
fields and leaves the rest untouched — *residual* serialization. That is what makes
`relabel_classes.py` able to rewrite only `meta/meta_NNNN.yaml` without re-copying a multi-megabyte
`.usdz` onto itself (which would raise `SameFileError`); see `relabel-graspable-classes.md`. Second,
because the index is the filename suffix, `deserialize(i, path)` is a pure function of the integer,
which is what the collectors exploit.

`meta` carries at minimum `name` (globally unique instance identifier) and `class` (the semantic
label used for cid assignment and for the reference catalog). Store-extracted objects additionally
carry `store_prim`, the product's path relative to the store root (`store_scene.py:52-58`).

### `collect_objects` and `collect_preoptflow`

```python
def collect_objects(paths: list[str | Path]) -> list[GraspableObject]:
    list_of_lists = []
    for p in paths:
        path = Path(p)
        n = len(sorted((path / "meta").glob("meta_*.yaml")))
        list_of_lists.append([GraspableObject.deserialize(i, path) for i in range(n)])
    objects = list(itertools.chain.from_iterable(list_of_lists))
    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate GraspableObject names across datasets: {dupes}")
    return objects
```
(`clean_datagen.py:27-40`; `collect_preoptflow` at `:43-54` is structurally identical for
`OptFlowObject`)

The cardinality is inferred from the count of `meta/meta_*.yaml` files, and indices `0..n-1` are
deserialized. This makes the `meta/` directory authoritative: if a partial asset pull dropped two
`.usdz` files but left their `meta` yaml in place, the count still says *n* and deserialization
fails or the render later emits "no labeled instances" per frame. The recommended integrity check is
per-object `meta` count == `usd_path` count
(`.docs_claude/psc-isaac-datagen-footguns.md`; see [§III.7.1](#iii71-before-booting-isaac-seconds-free)).

`objects_path` is a **list**, and datasets are concatenated in list order. This is what allows a
config to draw from `amazon + kleenex + YCB` without any code change
(`collect-objects-chain-datasets.md`).

The duplicate-`name` guard is the load-bearing part. Names are used as (a) the USD prim path leaf in
`add_wrapped_reference` (`scene.py:60-68`, via `Tf.MakeValidIdentifier(name)`), (b) the key of
`name_to_class` in `reference_catalog` (`reference_seg_writer.py:58`), and (c) the value space of
`iid_to_name`, which both writers assert is 1:1 at `finalize_metadata`
(`reference_seg_writer.py:153-154`, `optflow_writer.py:97-98`). A duplicate name would silently
collide a prim path and corrupt the instance identity graph. `filter_objects` repeats the same check
*after* filtering (`filters.py:33-36`), because `ReplicateFilter` mints new names and could in
principle collide with an existing one.

`objects_path` was previously two fields, `graspable_objects_path` and `optflow_objects_path`, with
asymmetric validation that let a mode-mismatched config render zero objects silently.
`unify-objects-path.md` collapsed them: `mode` already selects the collector, so the split field only
re-encoded that knowledge. Current source has the single field and asserts it non-empty for both
modes (`runtime_config.py:149`).

### Mesh ingestion

New object datasets are produced by `mesh_convert.py`, a two-phase tool (`mesh-convert-ycb.md`):

- **Phase 0** (optional) `ycb` — `ycb_download()` (`mesh_convert.py:374-405`) pulls the YCB
  `google_16k` textured meshes from S3.
- **Phase 1** `stage` — `convert(input_path, stage_path, ...)` (`:88-114`) finds meshes, runs Blender
  as a subprocess to import each mesh, export it as `.usdz`, and render four orthographic side-face
  tiles, then writes a uniquely-labelled candidate directory with `candidate.json` and a 1×4 contact
  sheet. Nothing is serialized as a `GraspableObject` yet.
- **Phase 2** `finalize` — `finalize(stage_path, output_path, winners, ...)` (`:171-191`) takes a
  `{label: face}` (or `{label: {face, class}}`) mapping, picks the winning face's grasp frame and
  reference image, and serializes the `GraspableObject`.

The split exists because rendering is expensive and human face-judging is not reproducible; keeping
the candidates on disk makes `finalize` cheap, resumable and re-runnable.

Three details are worth carrying forward. `find_meshes(input_path, one_per_dir=True)`
(`mesh_convert.py:20-35`) deduplicates: YCB ships four mesh representations per object
(`textured.obj/.dae`, `nontextured.ply/.stl`), so a naive recursive glob yields 312 files for 78
objects; the ranking prefers a name containing "textured" and the `.obj` extension so the choice is
deterministic rather than filesystem-ordering dependent. `face_grasp_frames(bbox_min, bbox_max)`
(`:62-78`) computes four **side-face** SE3 frames with the convention **+X = outward face normal,
+Z = world up, origin = bbox face centre**; ±Z faces are excluded because `look_at` is singular when
the view normal is parallel to world up. And all SE3 math lives in the driver, never in
`mesh_blender.py` — Blender's Python has no torch/scipy/matplotlib, so per-face camera rotations are
precomputed via `cv2opengl(look_at(...))` and handed to the stateless Blender worker through
`cameras.json` (`mesh_convert.py:43-48`).

One standing hazard: USD caches opened layers in a process-global `Sdf.Layer` registry, so after
overwriting a `.usdz` in place, re-opening it in the same process returns the stale layer. Verify
`.usdz` edits in a fresh process (`rotate-graspable-meshes-z.md`).

### Extension recipe: adding a new object dataset

1. Produce the assets: either run `mesh_convert.py stage` / `finalize` on a mesh tree, or write a
   bespoke extractor (`extract_store_objects.py` is the template for pulling objects out of an
   externally authored USD — see [§II.9](#ii9-store-usd-scenes)).
2. Ensure every `meta` carries a `name` globally unique across *all* datasets a config will chain,
   plus a single-token `class` (whitespace in a class name is truncated by Isaac's segmentation
   annotator — see [§II.8](#ii8-writers)).
3. Serialize with `obj.serialize(i, dataset_dir)` for consecutive `i` starting at 0. Sparse or
   1-based indices will not be found by the collectors.
4. Add the directory to `objects_path`. If your objects need a field the existing records do not
   have, add it to the dataclass and, if its type is new, add a `(ext, write, read)` entry to
   `_serializers`.

---

## II.3 The filter registry

### Responsibility

`filters.py` transforms the collected object pool before scene construction: subsetting, capping,
duplicating and shuffling — all from config, with no code change per dataset variant.

### Key types and functions

```python
@dataclass
class FilterSpec:
    name: str
    args: dict = field(default_factory=dict)

def make_filters(specs):
    return [getattr(sys.modules[__name__], spec.name)(**spec.args) for spec in specs]

def filter_objects(objects, specs):
    for f in make_filters(specs):
        if not objects:
            raise ValueError(f"no GraspableObjects left to feed filter {f!r}")
        objects = f(objects)
    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate object names after filtering: {dupes}")
    return objects
```
(`filters.py:16-37`)

The registry is `getattr` over the module itself — the same reflection pattern used by `posers`,
`placers`, `scene_builders`, `grasp_policies`, `orientations` and `store_mutations`. Registration
*is* defining a class in the module; there is no separate registration step and no factory dict.
`graspable-object-filter-registry.md` records this as deliberate: thin orchestration, no new
abstraction, and the config names the policy explicitly so there are no silent defaults.

Filters are applied **in order**, and the emptiness guard fires *before* each filter, so an
over-restrictive filter is reported by name rather than causing a confusing failure two stages
later. The post-filter duplicate-name check is the same invariant as in the collectors
([§II.2](#ii2-object-datasets)) and is the reason `ReplicateFilter` must mint suffixed names.

### The shipped filters

- **`ShuffleFilter(seed)`** (`filters.py:40-46`) — `np.random.RandomState(seed).permutation`. The
  seed is **mandatory, no default**: datagen is fully seeded, and a silently non-deterministic
  shuffle would break reproducibility of a render dir.
- **`ReplicateFilter(count, key="name", value="*")`** (`:49-65`) — duplicates every object whose
  `meta[key]` matches the fnmatch glob, `count` times. The first copy keeps the original name; copies
  `k >= 1` get `f"{name}_dup{k}"`. It uses
  `dataclasses.replace(o, meta={**o.meta, "name": ...})`, so the copies share the same `usd_path`,
  `reference_image` and `grasp_point` object references — cheap, and correct because those are
  read-only during a render. This is the mechanism for filling a wall from a small catalog.
- **`MetaFilter(key, value, max)`** (`:68-83`) — walks in order, keeps **at most `max`** objects
  whose `meta[key]` matches the glob, and passes *non-matching* objects through untouched. That
  asymmetry is the surprising part: `MetaFilter` caps a subset, it does not select one. A `max: 44`
  on a 44-object amazon pool is inert
  (`graspable-object-filter-registry.md`, "Capacity interaction").
- **`RegexFilter(key, value)`** (`:86-96`) — `re.compile(value).search` on `str(meta[key])`, keeping
  only matches. This *is* the selection filter, and it is the primary mechanism for config-only
  dataset variants: a single-class fine-tune pool is a `RegexFilter` on `meta['class']`, not a
  re-extracted catalog (`store-snacks-training-datasets.md`).

Matching semantics differ deliberately between the two: `MetaFilter` uses `fnmatch.fnmatchcase`
(Unix glob, whole-string) and `RegexFilter` uses `re.search` (substring regex). A `RegexFilter` with
`value: "snack017"` matches `snack017_3`; a `MetaFilter` with the same value would not.

### Extension recipe: adding a filter

1. Define a class in `filters.py` with `__init__(self, **config_args)` and
   `__call__(self, objects) -> list`. Give required parameters no defaults so a missing config key
   raises `TypeError` at load rather than behaving unexpectedly.
2. Preserve the invariants the guards check: never return an empty list if you can help it, and if
   you mint objects, give them unique `meta['name']` values.
3. Reference it from config: `filter_specs: [{name: YourFilter, args: {...}}]`. No registration.
4. It is unit-testable without Isaac — `from isaac_datagen.filters import filter_objects, FilterSpec`
   imports nothing from `isaacsim`.

---

## II.4 Scene construction

### Responsibility

`scene.py` turns a `RuntimeConfig` plus a list of objects into a live USD stage: a new stage, an
optional workbench asset, lights, the placed object stack, in-place mutations, grasp frames, shadow
occluders and the camera rig. It returns a `SceneHandle`. `store_scene.py`
([§II.9](#ii9-store-usd-scenes)) does the same job for externally authored store USDs.
`scene_builders.py` is the registry that names them.

### `build_scene` vs the `scene_builders` registry — a live inconsistency

```python
# scene_builders.py (whole file)
import sys

from isaac_datagen.scene import build_scene
from isaac_datagen.store_scene import build_store_scene, build_repopulated_store_scene


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e
```

**Current source truth:** only the optflow entry point goes through the registry.
`optflow_generation` calls `scene_builders.get(runtime.scene_builder)(runtime, objects)`
(`clean_datagen.py:117`), while `reference_segmentation` calls `build_scene(runtime, objects)`
directly (`clean_datagen.py:74`), ignoring `runtime.scene_builder` entirely. `RuntimeConfig`
validates that `scene_builder` is non-empty (`runtime_config.py:124`) regardless of mode, so a
reference-segmentation config that sets `scene_builder: build_store_scene` will pass validation and
then silently build a plain scene. This is an unresolved wart, not a documented design decision —
treat `scene_builder` as an optflow-mode field until it is fixed
([Appendix B](#appendix-b--open-questions-and-unverified-claims)).

A consequence follows: store scenes and repopulated store scenes are reachable only from
`mode: optflow`. All shipped store configs are optflow configs.

### `SceneHandle`

```python
@dataclass(frozen=True)
class SceneHandle:
    zed: ZedMini
    grasp_points: list          # prim paths of GraspPoint xforms
    objects: List[GraspableObject]
    object_prim_paths: List[str] = None
```
(`scene.py:18-23`)

`grasp_points` are *prim paths*, not poses — `plan_capture` resolves them to world transforms at
capture time ([§II.7](#ii7-capture)). `object_prim_paths` point at the **`/geo` child**, not the
wrapper (`scene.py:491`), because `l2w` consumers need the mesh frame: any orientation yaw applied
by an `orientations` policy lives on `geo`, and the optflow writer's `class_to_l2w` must include it.
`objects` is returned rather than passed through because mutations may substitute object records in
place.

### The `build_scene` sequence

`build_scene` (`scene.py:436-492`) runs a fixed order, and the order is load-bearing:

1. `PlainSceneSpec(**runtime.scene_builder_args)` — validate the builder args first, so an unknown
   key raises `TypeError` before the sim does any work.
2. `create_new_stage()`, define `/World` as the default prim.
3. If `runtime.scene != "empty"`, reference `resources/workbench_world.usd` under
   `/World/Workbench`. Assets load before any labelling so the stage context exists.
4. `make_dome_light(...)` with `intensity = dome_fill_intensity if dome_light else 0.0`. Note that
   the dome prim is created unconditionally — `dome_light: false` creates a zero-intensity dome, it
   does not omit the prim. Anything that later looks up `/World/DomeLight` therefore always finds it.
5. `create_stack_of_objects("/World/GeneratedPallets", objects, runtime, orientation=spec.orientation)`.
6. `apply_plain_mutations(...)` — in-place stage edits (see below).
7. Grasp frames for graspable prims only.
8. `set_transform(stack, translation=(0.1, 0.1, 0.045))` — the whole stack is nudged off the origin
   *after* grasp frames are attached, so grasp frames ride along.
9. If `distant_light`, aim it: the key light's `eye` is `centroid_of_grasp_frames +
   distant_light_offset`, and its rotation is `look_at_euler(eye, centroid)` (`scene.py:475-480`).
   The light is aimed at the actual target centroid, so moving the wall does not require retuning the
   light.
10. If `occluders_per_target`, add shadow occluders
    ([§II.6](#ii6-lighting-and-domain-randomization)).
11. Construct the `ZedMini` rig from `np.load(runtime.intrinsics_path)`.

`create_stack_of_objects` (`scene.py:79-93`) is the object-loading core: for each object,
`add_object` creates an `Xform` wrapper at `<stack>/<MakeValidIdentifier(name)>` and references the
`.usdz` under `<wrapper>/geo` (`add_wrapped_reference`, `:60-68`), then labels `geo` via
`label_product`. Optionally an `orientations` policy runs. Then the placer is instantiated from the
registry and `organize_objects` applies its transforms.

`label_product` (`scene.py:36-42`) is a three-step dance that exists because vendor assets ship their
own semantics:

```python
displaced = _override_vendor_class_labels(prim, obj.meta["class"])
remove_labels(prim, include_descendants=True)
add_labels(prim, labels=[obj.meta["class"]], instance_name="class")
add_labels(prim, labels=[obj.meta["name"]], instance_name="instance")
```

The override must come **first**. Vendor `semantic:*:params:semanticData` opinions arrive through the
reference arc, and `RemoveProperty` cannot delete an opinion that lives in a referenced layer — it
can only be *overridden* from the root layer, which `_override_vendor_class_labels`
(`scene.py:45-57`) does by walking `Usd.PrimRange` and `Set`-ing every class-typed semanticData
attribute to our class. Only then does `remove_labels(include_descendants=True)` clear our own
layer's labels, and `add_labels` write the correct pair. Getting this order wrong produces an
all-zero `cid_mask` — the failure documented at length in `store-usd-inverse-datagen.md`.

### Scene mutations

`PlainSceneSpec` (`scene.py:402-423`) is the plain-scene builder-args schema:

```python
@dataclass(frozen=True)
class PlainSceneSpec:
    mutations: list = field(default_factory=list)
    grasp_frames: str = "bbox"     # or "catalog"
    orientation: dict = None       # {name, args?} -> orientations registry
```

`__post_init__` validates each mutation spec's shape and then asserts
`getattr(store_mutations.get(m["name"]), "PLAIN_SAFE", False)`. Only `DisablePhysics` carries
`PLAIN_SAFE = True`; the store-shaped mutations read `spec.product_patterns`, a field
`PlainSceneSpec` does not have, and would raise `AttributeError` ~30 s into a boot. Failing at
config-parse instead is the whole point (`plain-scene-mutations-disable-physics.md`).

`apply_plain_mutations` (`scene.py:425-433`) wraps the placed objects in `CaptureTarget` bindings,
runs the mutation chain, and then asserts that the returned prim-path list is *identical* to the
input:

```python
assert [t.prim_path for t in targets] == list(objects_paths), \
    "plain-scene mutations must not add/remove/reorder targets (in-place stage edits only)"
```

Plain-scene mutations may edit the stage and may swap the bound `obj` record, but they may not change
the target set, because the object list returned to the caller must stay index-aligned with
`objects_paths` and hence with the grasp-frame lookup that follows.

`DisablePhysics` is not cosmetic. Store-extracted `.usdz` files carry `PhysicsRigidBodyAPI` and
`CollisionAPI` baked in from the live shelf; dropped into a plain scene they free-fall during capture
(36/50 frames, −3.14 m drift in the stored `l2w`). Setting `rigidBodyEnabled=False` via a root-layer
override suffices — omni.physx never creates a dynamic actor — and it must run before the grasp
frames' world transforms are read (`plain-scene-mutations-disable-physics.md`).

### Grasp frames

`GRASP_FRAME_SOURCES = {"bbox": _bbox_grasp_frame, "catalog": _catalog_grasp_frame}`
(`scene.py:390-399`) selects where the frame comes from:

- `bbox` → `add_grasp_frame(box_path)` (`scene.py:356-365`), which computes the bbox midpoint with
  **`ComputeUntransformedBound`** and places a `GraspPoint` empty at `(cx, cy - half_y, cz - half_z)`
  with rotation `(0, 0, -90)` — the front-bottom edge of the box. `ComputeUntransformedBound` rather
  than `ComputeLocalBound` is the fix for a real bug: `ComputeLocalBound` bakes the prim's own
  local-to-parent transform into the result, so after placement the measured midpoint is contaminated
  by the placement offset (`until-exhausted-stacker.md`, "Landmine").
- `catalog` → `add_catalog_grasp_frame(f"{path}/geo", obj)` (`store_scene.py:61-65`), which replays
  the `grasp_point` SE3 baked into the object record. This is the path store-extracted objects take
  even in plain scenes.

Grasp frames are created only for prims the placer reports graspable (`scene.py:466-469`), so
mid-column objects have no frame and are never chosen as camera targets.

### Extension recipe: adding a scene builder

1. Write `build_<name>(runtime, objects) -> SceneHandle` in a module of your choosing, and import it
   into `scene_builders.py` so `getattr` can find it.
2. Define a frozen spec dataclass for its `scene_builder_args` and construct it as the first
   statement, so bad args fail before the stage is touched. Validate registry names inside
   `__post_init__` (`StoreSceneSpec` calls `grasp_policies.get` / `store_mutations.get` purely for
   the side effect of raising).
3. Populate all four `SceneHandle` fields. `object_prim_paths` must point at the frame optflow's
   `l2w` needs (the mesh frame), and must stay index-aligned with `objects`.
4. Set `scene_builder: build_<name>` in an **optflow** config. Reference-segmentation configs do not
   consult the registry (see the inconsistency above).

---

## II.5 Placement policies

### Responsibility

`placers.py` decides where each object prim sits. It is the geometry layer between "these objects are
referenced into the stage" and "the stack looks like a pallet wall / a shelf".

### The contract

A placer is a class with three members:

- `__init__(self, prim_paths, **config_args)` — measures and computes the full layout eagerly.
- `__call__(self, prim_path) -> (translation, rotation)` — a lookup, not a computation.
- `graspability(self) -> dict[str, bool]` — precomputed per prim path.

`create_stack_of_objects` instantiates it as
`placers.get(runtime.placement)(prim_paths_added, **runtime.placement_args)` (`scene.py:89`), then
`organize_objects` (`scene.py:71-76`) calls the policy once per prim and applies the transform. The
registry is the same `getattr`-over-module pattern (`placers.py:12-16`).

Computing everything in `__init__` is not incidental. All bbox measurements must happen **before**
`organize_objects` writes placement transforms, because `size_of` / `center_of` read
`local_bbox_range`, which is contaminated once a placement transform exists. Measuring lazily inside
`__call__` would give correct results for the first prim and wrong ones for the rest
(`until-exhausted-stacker.md`).

### Geometry helpers

Four pure helpers carry the layout math (`placers.py:30-65`):

- `size_of(path)` / `center_of(path)` — bbox extent and midpoint as `Vec3` named tuples.
- `centroid_at_point(center, target)` — the translation that puts the bbox centroid at `target`.
  Necessary because a mesh's origin is generally not its centroid.
- `compute_cols_stride(columns, min_gap, max_gap, sizes)` — pure numpy. Column width is the max
  `x`-extent in that column; gaps are drawn `U(min_gap, max_gap)`; columns abut left-to-right and the
  whole run is centred on `x = 0` (`left_edges = -total_w/2 + cumsum(...)`, then `+ col_widths/2`
  for centres).
- `fixed_columns(paths, height)` / `jagged_columns(paths, max_height)` — the two chunking policies.
  `jagged_columns` draws `h ~ randint(1, max_height+1)` per column. Splitting chunking from layout
  math means both are independently testable and swappable.

### `UntilExhaustedStacker`

```python
class UntilExhaustedStacker:
    EPSILON = 0.002
    def __init__(self, prim_paths, max_column_height, min_y=0, max_y=0,
                 min_gap=EPSILON, max_gap=EPSILON, epsilon=EPSILON):
```
(`placers.py:68-80`)

It chunks into jagged columns, then `_seat` (`:82-99`) does the layout: measure all sizes and
centres, compute column `x` positions, draw one `y ~ U(min_y, max_y)` **per column**, and stack each
column from `floor_z = 0` upward, advancing `floor_z += size.z + epsilon` after each object.

Three jitter channels, each with a distinct purpose:

- **Depth jitter** `(min_y, max_y)` — one draw *per column*, shared by the whole stack in that
  column, giving a staggered front-to-back wall rather than a flat plane
  (`depth-jitter-stacker.md`).
- **Gap jitter** `(min_gap, max_gap)` — the horizontal spacing between adjacent columns
  (`gap-jitter-stacker.md`).
- **Per-object jitter** `epsilon` — drives both the vertical inter-object gap and the σ of a Gaussian
  lateral perturbation `np.random.normal(0, epsilon, size=2)` applied to `(x, y)` only. `z` is left
  clean so objects still rest exactly on the floor or on each other
  (`jagged-columns-jitter-stacker.md`).

`graspability()` returns `{p: p in tops}` where `tops = {col[-1] for col in columns if col}`
(`placers.py:104-106`) — only the top of each column is exposed, which is both realistic and the
thing that keeps camera targets from being buried inside the wall. Columns are `deque`s with the top
as the last element.

**Correction against the plan record.** `object-placer-registry.md` states there are no defaults in
placer constructors, so every parameter must be named in config. Current source (`placers.py:72-73`)
gives defaults for `min_y`, `max_y`, `min_gap`, `max_gap` and `epsilon` (`EPSILON = 0.002`); only
`max_column_height` is required and raises `TypeError` if omitted. Validation is limited to
`max_column_height >= 1` and `len(prim_paths) >= 1` (`placers.py:74-77`). The fail-loud principle
survives for the one parameter that determines the layout's shape, not for the jitter magnitudes.

Also note the rename history: `column_height` became `max_column_height` on `UntilExhaustedStacker`
and is **not** backward compatible; a config carrying the old key raises `TypeError`
(`jagged-columns-jitter-stacker.md`). `ShelfPlacer` still takes `column_height`.

### `ShelfPlacer`

```python
class ShelfPlacer(UntilExhaustedStacker):
    def __init__(self, prim_paths, column_height, ...):
        prims = sorted(prim_paths, key=class_label)
        self.columns = fixed_columns(prims, column_height)
        self._seat(min_y, max_y, min_gap, max_gap, epsilon)
```
(`placers.py:109-118`)

It inherits `_seat`, `__call__` and `graspability` unchanged, and differs in exactly two ways: it
sorts prim paths by class label before chunking, and it uses `fixed_columns` rather than
`jagged_columns`. Fixed height is required — random heights would split a class run at an arbitrary
boundary and destroy the same-class-per-column property that is the whole point (`shelf-placer.md`).
Reordering is safe because `__call__` and `graspability` are keyed by prim-path string, not by
position. The class label is read from the stage via `class_label(path)`, not parsed out of the path,
so a geo prim with no class label raises.

### `OccupancyGrid`

`OccupancyGrid` still exists in `objects.py:240` and is used only by
`debug_scripts/debug_occupancy.py`. **It is not in `placers.py` and is not reachable from
`runtime.placement`** — it has been retired from the main path in favour of the stackers. Its design
is worth one sentence of history: it was a static full-wall policy whose grid is all-ones regardless
of actual objects, requiring exactly `prod(pallet_dims)` objects and raising `ValueError` on
under-supply, because a partially filled grid would expose phantom slots that read as graspable but
hold nothing (`fix-occupancy-grid-full-wall.md`). The `pallet_dims` config field remains in
`RuntimeConfig` (`runtime_config.py:75`) as a vestige.

### Extension recipe: adding a placer

1. Define a class in `placers.py` with the three-member contract. If you subclass
   `UntilExhaustedStacker`, you can reuse `_seat` and override only the chunking, as `ShelfPlacer`
   does.
2. Do **all** bbox measurement in `__init__`, before `organize_objects` runs. Use `size_of` /
   `center_of`, which go through `local_bbox_range`.
3. Give required, shape-determining parameters no default so an incomplete `placement_args` raises
   `TypeError` at scene-build time.
4. `graspability()` must mark at least one prim graspable, or `plan_capture` has nothing to target.
5. Set `placement: YourPlacer` and `placement_args: {...}` in config.

---

## II.6 Lighting and domain randomization

### Responsibility

`scene.py` also owns the light rig and the per-frame randomization schedule. Two subsystems live
here: static rig construction (key + fill + occluders) and the per-frame jitter mechanism.

### Key/fill design

The rig is a **DistantLight key + DomeLight fill**:

- `make_distant_light(stage, parent, intensity, angle, rotation)` (`scene.py:293-302`) creates
  `/World/DistantLight` and orients it via `set_transform`. It is aimed, never hand-authored:
  `look_at_euler(eye, target)` (`scene.py:272-275`) is `cv2opengl(look_at(target, eye))` decomposed
  to XYZ Euler degrees — the same aiming convention as `LookAtPoser`, reused so a USD prim that emits
  along −Z points where you say. In `build_scene` the target is the centroid of all grasp frames and
  the eye is that centroid plus `distant_light_offset` (`scene.py:475-480`).
- `make_dome_light(stage, parent, intensity, normalize)` (`scene.py:259-267`) creates
  `/World/DomeLight`.

The rationale (`distant-light-key-light.md`): a DistantLight emits parallel rays from infinity, so
there is no inverse-square falloff darkening the far end of a wall, and it casts crisp directional
shadows on occluders. The dome was demoted from primary to ~10–20 % ambient fill
(`dome_fill_intensity` default 200 against `distant_intensity` default 3000) so shadowed faces do not
crush to black while still having shape. `look_at_euler` inherits `look_at`'s degeneracy: a
near-vertical `distant_light_offset` makes the `cross(z, up)` basis construction singular, so keep
the offset off-vertical.

### Per-frame jitter — the mechanism and why it is what it is

This is the subsystem with the most scar tissue. **Replicator's randomizer-registration route does
not work for these attributes.** `lighting-jitter-mechanism.md` records the diagnosis: nodes built
inside `rep.randomizer.register(fn)` produce *zero layer opinions on any channel* — verified with a
per-frame USD probe — while the same nodes built directly inside a trigger body execute per frame and
land in the root layer. Separately,
`rep.modify.attribute("intensity", rep.distribution.sequence(vals))` does **not** advance per
`orchestrator.step()` the way `rep.modify.pose` does; every frame stays pinned to the first element,
which is how a varying schedule (995 → 24476) produced a constant, dark render while camera poses
varied correctly (`lighting-diagnostic-dark-box-flags.md`).

**Current source truth:** light jitter is done with plain Python closures invoked from the capture
step loop, writing directly to USD under an explicit `Usd.EditContext(stage, stage.GetRootLayer())`.
The plumbing is `ReplicatorWrapper` (`scene.py:120-139`):

```python
class ReplicatorWrapper:
    def __init__(self, rep): self.rep = rep; self._randomizers = []; self._per_frame = []
    def register(self, fn): ...            # the Replicator-graph route, still used for textures
    def apply_randomizers(self): ...
    def register_per_frame(self, fn): self._per_frame.append(fn)
    def per_frame(self, i):
        for fn in self._per_frame: fn(i)
```

`capture_session` calls `per_frame(i)` immediately before each `orchestrator.step()`
(`capture.py:81-84`), and `capture_with_poses` threads `replicator.per_frame` in
(`capture.py:108`). This supersedes the `rep.distribution.sequence` dome-jitter route described in
the older plans — those are history.

The three per-frame registrars:

- `register_dome_jitter(replicator, prim_path, runtime, num_frames)` (`scene.py:142-154`) —
  precomputes `U(lo, hi)` intensities with `default_rng([effective_seed, 1])`, closes over them, and
  sets `GetIntensityAttr()` per frame.
- `register_distant_jitter(...)` (`scene.py:157-183`) — precomputes rotations via
  `sample_offset_eulers(offset, distant_offset_jitter, n, rng)` (`scene.py:278-290`, which perturbs
  the eye position uniformly per axis and re-aims), plus optional intensity and colour-temperature
  schedules from the same `default_rng([effective_seed, 0])` stream. Temperature jitter also flips
  `EnableColorTemperatureAttr` on.
- `register_light_pattern_jitter(replicator, spec, runtime, num_frames, stream)`
  (`scene.py:195-214`) — for externally authored scenes: globs `spec.root`/`spec.pattern` for
  `UsdLux` prims, snapshots each light's base intensity, asserts the pattern matched something, and
  per frame multiplies each base by a shared factor. Factors are drawn **log-uniformly**:
  `exp(U(ln lo, ln hi))`. Light is perceptually multiplicative, so a uniform draw over `[0.25, 8.0]`
  would spend half its mass on washed-out extremes; the shipped range is exactly that 32×
  (`store-usd-inverse-datagen.md`).

`make_replicator(runtime, num_frames, render_dir)` (`scene.py:227-256`) wires all of the above, seeds
Replicator's own RNG with `set_global_seed(effective_seed)`, and — if `log_lighting` — writes
`render_dir/lighting_log.json` containing `num_frames`, the seed, and the full per-light schedule.
The log is written from the same precomputed arrays the closures read, so *logged == applied*, which
is what makes `isaac-datagen-measure-luminance --with-lighting` able to join per-frame foreground
luminance against the intended intensity ([§III.7.5](#iii75-luminance--dark-frame-audit)).

Magnitude matters more than it looks. A ±0.75 m DistantLight wobble is only ≈±10–13° of direction
change, spanning a cosine factor of 0.62–0.76 — about ±10 % luminance, essentially invisible after
the ACES shoulder. Configs that want visible variation use `distant_offset_jitter: 2.0` (a ±30° cone)
and `distant_intensity_jitter: [500, 4000]` (`lighting-jitter-mechanism.md`).

One route still uses the Replicator graph: `register_background_jitter` (`scene.py:186-192`) swaps
the dome's `texture:file` via `rep.distribution.choice`, and `register_box_texture_jitter`
(`:217-224`) does the same for box shader `file` inputs. Both are `choice` on a texture asset,
invoked through `apply_randomizers()` inside the `rep.trigger.on_frame()` body
(`capture.py:110-112`) — the direct-in-trigger form that does work.

### Shadow occluders

`add_shadow_occluders(stage, parent, grasp_frames, runtime)` (`scene.py:371-388`) places
`occluders_per_target` primitive shapes (cycling `Cube/Cone/Cylinder/Sphere`) per grasp target. Each
is posed with `set_prim_pose(path, t2w @ poser(1)[0])` — one sample from the configured
`occluder_pose_policy` in target-frame coordinates, transformed to world by that target's
`target2world`. They are **static for the render**: placement happens entirely in `build_scene` and
never changes per frame. Per-frame shadow variation comes for free from camera motion and light
jitter (`per-target-shadow-occluders.md`).

Invisibility is via a primvar:

```python
UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
    "hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)
```
(`scene.py:385-386`)

`hideForCamera` is the token Replicator's asset cache honours. Crucially the occluder gets **no
semantic label**, so even if it did render it would contribute nothing to `cid_mask` / `iid_mask` —
only its shadow reaches the output.

Two open issues, both unresolved. Occluders that *penetrate* a box render as opaque black blobs; a
controlled test (identical box, camera and seed, varying only the x-offset between penetrating and
floating) reproduced the symptom only in the penetrating case, so the remedy is an air gap or
out-of-frustum placement, not a `hideForCamera` fix. And whether `hideForCamera` is honoured at all
under path tracing is itself unconfirmed (`per-target-shadow-occluders.md`).

On seeding: `occluder_scale`, when `None`, draws `np.random.uniform(0.04, 0.2)` (`scene.py:383`) and
the occluder poser draws (`scene.py:377`, `:387`) from the **global** `np.random` stream — which
`seed_everything(effective_seed)` has already seeded. Occluder layout is therefore reproducible for a
fixed `(seed, idx)` but has no decorrelated substream, so it shifts whenever an upstream consumer of
the global stream changes. Its variance across configs is high enough that the same config can put
occluders usefully peripheral or uselessly centred (`obs-full-alpha-toggle.md`, notes).

### `obs_full_alpha`

`composite_rgba(rgb, seg, valid_ids, full_alpha=False)` (`reference_seg_writer.py:29-32`) emits
`alpha = 255` everywhere when `full_alpha` is true, and otherwise
`alpha_from_instance_seg(seg, valid_ids)` — 255 only on labelled-instance pixels. Default `False` is
the training contract (foreground-only observations); `True` is for inspection renders where you need
to see the shadows and background. It changes only the alpha channel, never the RGB and never the
masks (`obs-full-alpha-toggle.md`). Both entry points pass `full_alpha=runtime.obs_full_alpha`
(`clean_datagen.py:85`, `:130`), so the config field controls it in both modes; the "hardcoded
`False`" in `optflow-6-nested-obsmask.md` refers to the intended production value, not the code.

### The render-darkness flakes and the warmup remedy

Two distinct bugs, both documented in `plans/active/render-darkness-investigation.md` (an **active**,
not completed, plan):

**Bug 1 — dark boxes, root-caused and fixed.** The ACES tonemap operating point was set by a stale
carb default `exposureTime = 0.02` (a daylight shutter), which pushed a dome-lit indoor scene into
the ACES toe and crushed to uint8 zero — a near-binary cliff with no midtone. `boot_sim` now applies
exposure explicitly (`scene.py:333-336`) when `set_exposure`, and the default `exposure_time = 1.0`
(`runtime_config.py:107`) clears the toe (foreground mean ≈178 vs ≈0.0). `boot_sim` prints a
`[TONEMAP]` line with the resolved settings (`scene.py:338-346`) — that line is the evidence the fix
is in effect.

**Bug 2 — intermittent all-black processes, unresolved.** Roughly 60 % of processes render the entire
wall pure black for *every* frame. The outcome is per-process all-or-nothing, decided at renderer
init before frame 0, and immutable for the process lifetime. An HDR probe shows the dome reporting the
correct intensity while HDR RGB is ≈0 for both dome (IBL) and distant (analytic) lights, which points
at a path-tracer light-initialization race rather than anything light-type specific. Things that did
**not** fix it: material warmup via `orchestrator.step`, `app.update` warmup,
`resetPtAccumOnlyWhenExternalFrameCounterChanges`, removing the distant light, and `multi_gpu=False`.
The operational remedy is detect-and-retry
([§III.8.2](#iii82-the-all-black-render-coin-flip-unsolved)).

**The warmup remedy is for a third, milder symptom**: the *first captured frame* flaking between lit
and black due to unsettled RTX shader / HDRI-streaming / PT-accumulation state.
`warmup_render(app, n_frames)` (`scene.py:351-353`) is literally `for _ in range(n): app.update()`,
matching Isaac's own camera-sensor test pattern. It is called from both entry points **before**
`capture_with_poses` (`clean_datagen.py:87`, `:132`) and never inside `capture_session`, and that
placement is mandatory: `rep.distribution.sequence` advances per `orchestrator.step`, so any extra
orchestrator step inside the session would desync the camera pose schedule. `app.update()` does not
step the orchestrator, which is exactly why it is the safe warmup primitive. Default
`warmup_frames = 32` (`runtime_config.py:113`); `pre-capture-render-warmup.md` records 16 as the
original default with 8–24 as the typical range, so the shipped value is conservative — treat the
plan's 16 as history. Set to 0 to A/B against pre-warmup behaviour. Note that warmup settles global
RTX state but does not exercise the exact PT-accumulation-on-render-product path, and it does not
touch Bug 2 at all.

### Extension recipe: adding a randomizer

1. Write `register_<thing>_jitter(replicator, ..., runtime, num_frames)` in `scene.py`. Precompute
   the whole schedule in Python from `np.random.default_rng([runtime.effective_seed, k])` with a
   fresh `k` so you do not correlate with existing channels (0, 1, 2 and 3 are taken).
2. Close over the schedule and register the closure with `replicator.register_per_frame(fn)`, where
   `fn(i)` writes under `with Usd.EditContext(stage, stage.GetRootLayer())` using plain schema-API
   setters. Do **not** reach for `rep.randomizer.register` or `rep.modify.attribute` +
   `rep.distribution.sequence` for scalar attributes — both are known not to advance.
3. Return the schedule and have `make_replicator` fold it into the `log` dict so `lighting_log.json`
   records what was actually applied.
4. Any sequence you *do* hand to Replicator must be `len(world_poses)` long, i.e.
   `num_targets × num_frames`, not `num_frames` — `plan_capture` flattens `(B, N, 4, 4)` to
   `(B*N, 4, 4)` and there is one step per element. `make_replicator` receives `len(world_poses)`
   from both entry points (`clean_datagen.py:86`, `:131`), so the correct length is already in hand.

---

## II.7 Capture

### Responsibility

`capture.py` separates **pose planning** (deterministic, a function of scene geometry) from **pose
execution** (stochastic, inside Replicator, with randomizers and writers attached).

### `plan_capture`

```python
def plan_capture(runtime, scene):
    idx = (np.arange(len(scene.grasp_points)) if runtime.num_targets is None
           else np.random.choice(len(scene.grasp_points), size=runtime.num_targets))
    grasp_points = [scene.grasp_points[i] for i in idx]
    target2worlds = get_target2world(grasp_points)
    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)
    world_poses = np.einsum('bij,njk->bnik', target2worlds, target_frame_poses).reshape(-1, 4, 4)
    return idx, grasp_points, world_poses
```
(`capture.py:26-34`)

Four things happen in order. Targets are chosen: `num_targets: null` means **every grasp frame,
once** (`np.arange`), which is the verification-dataset mode; an integer means
`np.random.choice(..., size=n)` **with replacement** — repeated targets are possible and are not
guarded against (`fix-occupancy-grid-full-wall.md` notes `replace=False` would be safe while
`num_targets < graspable count`, but the source has not changed). `get_target2world`
(`capture.py:13-23`) reads each grasp prim's `ComputeLocalToWorldTransform` and transposes (USD is
row-vector convention, numpy here is column-vector), stacking to `(B, 4, 4)`. The poser is built from
the registry and called **once** with `num_frames`, producing `(N, 4, 4)` poses *in the target
frame*. Then the outer product: every target × every pose, flattened to `(B*N, 4, 4)`.

That flattening is the single most consequential fact in this subsystem. **The number of rendered
frames is `num_targets × num_frames`, not `num_frames`**, and any per-frame schedule must be sized to
match. Size walltime off that number too.

`runtime.num_frames` is used directly here, so a config in `grid_dims` mode (`num_frames is None`)
would pass `None` to the poser. `GridFixedPoser(random=False)` slices `[:num_frames]`, which with
`None` is a full slice — but `random=True` would call `plan_poses(..., None)`. This path is
effectively untested; treat `grid_dims` as reachable only through
`pose_generation_policy_args.grid_dims` with `random: false`.

### The poser registry

`posers.get(name)` (`posers.py:15-19`) is the same reflection registry. A poser is a stateful
callable: `__init__(**config_args)`, `__call__(num_frames) -> (N, 4, 4)`.

- **`GridFixedPoser(xrange, yrange, zrange, target_to_ego_ypr, grid_dims=None, random=True)`**
  (`posers.py:22-37`) — wraps `plan_poses`, which samples offsets in the halo box and applies one
  *fixed* YPR rotation to all of them. `random=False` requires `grid_dims` and asserts so. This is
  the behaviour-preserving default that predates the registry.
- **`LookAtPoser(xrange, yrange, zrange, offset_sampler=None)`** (`posers.py:40-49`) — samples halo
  offsets, then `cv2opengl(look_at(zeros(3), off))` per offset. Because `look_at` already returns a
  camera-to-target SE3 whose translation is the offset, orientation varies naturally per frame; no
  YPR argument is needed, which is why `LookAtPoser` configs are shorter.
- **`DecenteredLookAtPoser(xrange, yrange, zrange, intrinsics_path, resolution, object_radius,
  margin_deg=1.0, max_roll_deg=15.0, offset_sampler=None)`** (`posers.py:52-92`) — `LookAtPoser` plus
  in-plane variation. Its design constraint is A/B comparability: offsets are drawn in **one**
  `generate_random_offsets` call *before* any decentering draw, so under a fixed seed the camera
  *positions* are bit-identical to `LookAtPoser`'s. Per pose it computes the object's angular radius
  `arcsin(object_radius / |off|) + margin`, erodes the frame rect by that cone (`erode_frame_rect`),
  samples a target pixel uniformly in the eroded rect, and composes a point-at rotation with a random
  roll about the target ray (roll is visibility-invariant by construction). An
  `assert cone_in_frustum(...)` guards the erosion math. When the eroded rect collapses — a close-up
  where the object cannot fit off-centre — it returns the centred pose. That is a *defined policy*,
  not an error path, and it accounts for roughly 10 % of frames at `object_radius = 0.25` over the
  shipped halo box. `intrinsics_path` has no default and `np.load` raises if wrong.
- **`FixedOffsetPoser(offset, ypr=(0,0,0))`** (`posers.py:95-103`) — repeats a single pose
  `num_frames` times. Built for controlled experiments (it is the poser used in the occluder
  penetration A/B).

### The ZedMini rig

`ZedMini(name, parent_path, intrinsics, width, height)` (`hardwares.py:7-63`) creates a parent
`Xform` at `<parent>/<name>` and two child cameras at `LEFT_CAM_OFFSET = (-0.0315, 0, 0)` and
`RIGHT_CAM_OFFSET = (+0.0315, 0, 0)` — a hardcoded 63 mm baseline. Both share the intrinsics loaded
from `intrinsics_path`. It exposes `rps -> (left_rp, right_rp)`, `intrinsics`, and `left2rig`, a
translation-only SE3 (`hardwares.py:52-55`); if real stereo extrinsics carry rotation between the
eyes, that is not modelled here.

The rig is posed as a unit: `capture_with_poses` gets `rep.get.prim_at_path(camera.prim_path)` — the
*parent*, not a camera — and moves that (`capture.py:100-101`). So `world_poses` describe the rig
centre, and the left eye that writers actually record from sits 31.5 mm off it. Any check that
compares a stored `cam2world` against the planned pose must account for that offset
(`store-perclass-faces-verify-dataset.md`).

`hardwares.py` also defines `Gripper` (`:65-101`), a two-`OrbbecGemini2` rig at 27.64 cm separation
with 18° pitch each and a 180° Z rotation, exposing `get_all_render_products()`. It is not used by
either `clean_datagen` entry point.

### Replicator layering and the write contract

```python
@contextmanager
def capture_session(writers, cameras, n_frames, replicator, rt_subframes=20, per_frame=None):
    rep = replicator.rep
    pairs = _broadcast_pairs(writers, cameras)
    with rep.new_layer():
        attach_writers(pairs)
        yield rep
    for i in range(n_frames):
        if per_frame is not None:
            per_frame(i)
        rep.orchestrator.step(rt_subframes=rt_subframes)
    rep.orchestrator.wait_until_complete()
```
(`capture.py:74-85`)

Read the scoping carefully: writer attachment **and the caller's `with` body** both execute inside
`rep.new_layer()`; the stepping loop runs after that layer context has closed. So the caller's
declarative graph setup — `rep.trigger.on_frame()`, `move_prims`, `apply_randomizers` — is recorded
into the new layer, and only then does the orchestrator replay it `n_frames` times. `per_frame(i)`
fires immediately before each step, outside the layer, doing its direct USD writes
([§II.6](#ii6-lighting-and-domain-randomization)). `wait_until_complete()` at the end is what makes
the write contract *synchronous*: by the time `capture_with_poses` returns, every `writer.write(data)`
has run, which is why `finalize_metadata` can be called on the next line (`clean_datagen.py:90`).

`capture_with_poses` (`capture.py:99-112`) is the whole caller:

```python
rig_node = rep.get.prim_at_path(camera.prim_path)
with capture_session(..., n_frames=len(world_poses), per_frame=replicator.per_frame) as rep:
    with rep.trigger.on_frame():
        move_prims([rig_node], [world_poses], replicator)
        replicator.apply_randomizers()
```

`move_prims` (`capture.py:88-96`) converts each pose with `se3_to_pos_euler` and binds the two
resulting lists with
`rep.modify.pose(position=rep.distribution.sequence(...), rotation=rep.distribution.sequence(...))`.
`sequence` advances one element per `orchestrator.step`, so `len(world_poses) == n_frames` is a hard
requirement — a mismatch silently exhausts the sequence early or leaves poses unused.

`se3_to_pos_euler` (`capture.py:37-39`) is
`(translation, R.from_matrix(...).as_euler('xyz', degrees=True))`. It is the single SE3 decomposition
used by `move_prims`, `set_prim_pose` and the dry-run debug exporter, which is deliberate: the
dry-run Blender preview must not drift from the real capture, so both call the same mechanism
(`blender-dry-run-sanity-renderer.md`).

Writer-to-camera routing is `_broadcast_pairs` (`capture.py:62-66`): with one writer and multiple
cameras, **all** render products are unioned into a single `attach` call; otherwise `broadcast`
(`:50-59`) pairs 1:N, N:1 or N:N and raises on anything else. In practice both writers override
`attach` to keep only `rps[0]` — the left eye (`reference_seg_writer.py:139-142`,
`optflow_writer.py:44-46`) — so the union never materializes as two-eye output.

### Extension recipe: adding a poser

1. Define a class in `posers.py` with `__init__(self, xrange, yrange, zrange, **your_args)` and
   `__call__(self, num_frames) -> np.ndarray` of shape `(num_frames, 4, 4)`.
2. Return poses in the **target frame**, not world — `plan_capture` composes `target2world`. Return
   them in the convention the USD prim expects; the existing posers all end with `cv2opengl(...)`
   because `look_at` produces OpenCV and the camera prim is OpenGL.
3. If you draw random numbers, draw them from the global `np.random` (seeded by `seed_everything`)
   unless you have a reason to decorrelate; if you want A/B comparability with an existing poser,
   draw the shared quantities first and in the same call shape, as `DecenteredLookAtPoser` does.
4. Config: `pose_generation_policy: YourPoser` with `pose_generation_policy_args` — commonly
   `xrange: ${xrange}` etc. via OmegaConf interpolation of the top-level fields.
5. Validate without booting Isaac: `load_config(cfg, []).pose_generation_policy_args` prints the
   interpolated args; `dry_run=true` exports the planned camera poses to `scene.usdz` for visual
   inspection ([§III.7.2](#iii72-dry_run-and-the-blender-sanity-render)).

---

## II.8 Writers

### Responsibility

A writer is an `omni.replicator.core.Writer` subclass whose `write(data)` is called once per rendered
frame with the annotator payload, and which owns the on-disk shape of a render dir. The two writers
share most of their machinery; they differ in what a sample contains.

### Shared machinery

Both are constructed with `data_structure = "renderProduct"` and a list of `AnnotatorRegistry`
annotators, and both override `attach` to keep only the first render product (the left eye).

**Annotators.** `ObsMaskWriter` requests `rgb`, `instance_segmentation_fast` (with `colorize=False`),
and `occlusion` (`reference_seg_writer.py:118-125`). `OptFlowWriter` requests those three plus
`distance_to_image_plane` and `camera_params` (`optflow_writer.py:21-27`). The extra two are exactly
the difference between "2D masks" and "3D geometry".

**`reference_catalog(object_specs, descriptor_config_path, descriptor_device)`**
(`reference_seg_writer.py:55-72`) runs **once, at writer construction**, before any frame is
rendered. It:

1. Assigns class IDs: `classes = sorted({obj.meta["class"]})`, then
   `class_to_cid = {cls: cid for cid, cls in enumerate(classes, start=2)}`. Start 2 because Isaac
   reserves 0 = BACKGROUND and 1 = UNLABELLED. The sorted rule makes cids deterministic *given the
   class set* — add or remove a class and the numbering shifts, which is why a class merge requires a
   LUT remap of existing `cid_mask` files (`optflow-relabel-residual-migrate.md`).
2. Builds `name_to_class` for every object.
3. Picks one canonical reference RGBA per class: iterate objects **sorted by `meta['name']`** and
   `setdefault` — the first member by name wins. If two same-class objects have different reference
   images, only the first is ever used.
4. Loads the descriptor backbone, runs each canonical reference through it under
   `torch.inference_mode()`, and `del`s the model. Precomputing here rather than per frame is a large
   saving (one forward per *class*, not per frame per instance) and frees VRAM before the render loop
   starts.

It returns `(class_to_cid, name_to_class, class_to_ref, class_to_descriptors, backbone)`.

**`obsmask_from_data(data, rp_key, class_to_cid, *, canon, full_alpha)`**
(`reference_seg_writer.py:75-104`) is the per-frame assembly both writers call:

```python
iid_mask, cid_mask, frame_iid_to_name = cid_iid_masks(seg_hw, labels, class_to_cid)
if not frame_iid_to_name:
    raise ValueError("write() called with no labeled instances — expected ≥1")
```

`cid_iid_masks` (`isaac_utils.py:10-22`) is the mask machinery:

- `iid_mask` is the raw `instance_segmentation_fast` image cast to `int32`.
- `frame_iid_to_name` maps annotator id → the `instance` semantic (our object name).
- `cid_mask` is built by a LUT: an array of `uint8` sized
  `max(seg.max(), max(frame_iid_to_cid)) + 1`, filled with `lut[iid] = cid` for every id whose
  `class` semantic **is a key of `class_to_cid`**, then `lut[seg_hw]`. Any id whose class is
  unrecognized maps to 0.

That last clause is the source of the *cid orphan* failure mode: an instance is present in `iid_mask`
but its `cid_mask` pixels are all background. The root cause found in
`tuna-fish-can-cid-orphan-root-cause.md` is that Isaac's `instance_segmentation_fast` **tokenizes
class semantics on whitespace**, so a class named `fish can` arrives as `fish`, misses the LUT, and
orphans. The store builder now asserts single-token classes at label time (`store_scene.py:85-86`);
the plain builder does not, so `validate_obsmask.py` remains the safety net. `validate_render_dir`
(`validate_obsmask.py:69-77`) loads every sample's masks and `iid_to_occlusion` and reports
`CidOrphan` rows; it correctly keys on the **per-frame** graspable iids from
`iid_to_occlusion.keys()`, not the cross-frame `iid_to_name`, because annotator ids are
session-local. `cid_iid_trace.py` writes an append-only `cid_iid_trace.log` in the render dir for
post-hoc debugging of exactly these tokenization quirks; both entry points initialize it before
booting the sim (`clean_datagen.py:63-64`, `:106-107`).

**Occlusion.** `_occlusion_by_iid(occ, iid_to_labels, instance_mappings, present_iids)`
(`reference_seg_writer.py:35-52`) exists because the `occlusion` annotator and
`instance_segmentation_fast` use **different id spaces** — occlusion is keyed by leaf-prim
`instanceId`. The join goes through prim paths: build `leaf_id -> ratio`, use
`helpers.get_instance_mappings()` to average leaf ratios into `prim_path -> ratio`, then use the
segmentation annotator's `idToLabels` to map each present iid to its path. Unmatched iids get `NaN`.
Two caveats: `NaN` occlusion ratios from the annotator are dropped before averaging, and the
occlusion metric counts object–object and self occlusion only — an object merely cut off by the frame
edge reads as unoccluded, because the annotator's denominator is the in-frustum render
(`obsmask-occlusion-and-viz.md`).

**`IidCanonicalizer`** (`isaac_utils.py:25-53`) is the last step of `obsmask_from_data`. A single
physical object can be split by the annotator into sibling component ids. The canonicalizer keeps
`name -> first-seen id` for the lifetime of the render and collapses every sibling's pixels onto that
canonical id. It has a hard guard: if an annotator id is ever seen with a *different* name than
before, it raises
`ValueError(f"annotator id {iid} renamed {seen!r} -> {name!r} mid-render")` — never remap, fail loud.
Occlusion rows are folded with a precedence rule (the canonical id's own row wins; a sibling row only
fills a gap), and the resulting values are noted as visualization-only. One canonicalizer instance
per writer, sharing lifetime with `iid_to_name`.

Both writers assert the resulting map is injective at finalize:

```python
assert len(set(self.iid_to_name.values())) == len(self.iid_to_name), \
    "writer contract violated: iid_to_name not 1:1"
```
(`reference_seg_writer.py:153-154`, `optflow_writer.py:97-98`)

### `ObsMaskWriter`

`ObsMaskWriter(descriptor_config_path, descriptor_device, object_specs, render_dir, full_alpha=False)`
(`reference_seg_writer.py:108-158`).

Per frame it accumulates and serializes one `ObsMask`:

- `obs` — RGBA `(4, H, W)` `tv_tensors.Image` from `composite_rgba`.
- `iid_mask` — `int32` instance ids, canonicalized.
- `cid_mask` — `uint8` class ids.
- `iid_to_occlusion` — `dict[int, float]`.

and updates the render-scoped `iid_to_name`. Serialization is
`obsmask.serialize(frame_id, render_dir)`, so each field gets its own subdirectory. `iid_to_occlusion`
and `iid_to_name` are saved as `.pt` (torch), not JSON, specifically because JSON stringifies integer
dict keys and these keys index masks (`cid-mask-dual.md`).

The dual-mask design is the core decision (`cid-mask-dual.md`): instance keying is needed because
occlusion is intrinsically per-instance; class keying is needed because proposals and training
operate on the *union* of all members of a class — "a point on any same-class instance is an inlier"
— and because it deduplicates redundant descriptor forwards.

`finalize_metadata(directory)` (`:152-158`) writes one `ObsMaskDescriptorMetadata` per render dir (at
index 0) via `obsmask_metadata` → `vision_core.datastructs.build_obsmask_metadata`. **The serialized
field set is `iid_to_name`, `cid_to_class`, `name_to_class`, `class_to_ref`, `class_to_descriptors`,
`principal_components`** (`vision_core/datastructs.py:383-389`). Note two things the builder does
(`vision_core/datastructs.py:398-409`): it *inverts* the writer's `class_to_cid` into the stored
`cid_to_class`, and the backbone name is not a field — it is the **key** of the
`SubfolderDict`s wrapping `class_to_descriptors` and `principal_components`, which is what makes a
render dir able to hold several backbones side by side.

The PCA basis is a *mandatory* field: it is fit once per render dir on the concatenated tokens of all
classes' descriptors, storing `{mean, components, scale}`, so that any downstream visualization
projects features to RGB deterministically and comparably across classes. Without a shared stored
basis the same feature renders a different colour in two tools
(`pca-basis-mandatory-field.md`). Because it is mandatory,
`ObsMaskDescriptorMetadata.deserialize` fails on pre-PCA render dirs; `migrate_pca_basis.py` backfills
them idempotently.

### `OptFlowWriter`

`OptFlowWriter(objects, local2worlds, obs_intrinsics, render_dir, descriptor_config_path,
descriptor_device, full_alpha=False)` (`optflow_writer.py:19-100`).

Per frame it builds an `OptFlowSample` that **nests a complete `ObsMask`**:

```python
sample = OptFlowSample(
    obsmask=obsmask,
    observation_depth=depth,                                  # full frame, unmasked
    cam2world=cam2world.astype(np.float32),                   # OpenCV convention
    iid_to_visibility={},
)
sample.iid_to_visibility = instance_visibility(
    sample, self._optflow_metadata(), ref_cache=self._ref_cache)
```
(`optflow_writer.py:58-68`)

Four decisions are encoded here:

- **Nesting rather than duplicating.** One physical dataset serves both the UFM optical-flow adapter
  (deserialize `OptFlowSample`) and reference-seg phases 2/3 (deserialize `ObsMask`). The masks are
  load-bearing for UFM's per-instance flow isolation, and phases 2/3 need full descriptor metadata
  (`optflow-6-nested-obsmask.md`). This nesting broke backward compatibility with the older flat
  layout; pre-nesting render dirs must be regenerated.
- **`cam2world` is `inv(camera_params_to_world2cam(rp["camera_params"]))`** (`optflow_writer.py:56`)
  — read from the annotator and inverted through the stereo writer's helper, never hand-derived from
  the USD/OpenGL `world_poses`. Hand inversion without the GL→CV conversion produces mirrored,
  scattered warps (`optflow-2-writer-capture.md`, "Convention rule").
- **Depth is stored full-frame and unmasked**, background included. UFM's covisibility filter operates
  on `get_gt_warp`'s probability, and its occlusion/consistency check needs background context;
  instance isolation comes from `iid_mask`, not from masking depth
  (`optflow-2-writer-capture.md`).
- **`instance_visibility` is computed at write time**, with a `_ref_cache` carried across frames to
  amortize the reference-geometry transforms. This is the same reprojection-coverage machinery phase
  2's gate uses ([§II.10](#ii10-downstream-phases)).

`_optflow_metadata()` (`:70-94`) is built lazily and memoized, and it is where the **class-keyed
one-to-many** model lives. Objects are grouped by class; the first member is the class
representative; and:

```python
class_to_l2w = {c: torch.from_numpy(np.stack([L for _, L in members])).float()
                for c, members in by_class.items()}
```

So each class carries one reference (RGB, depth, `K`, `ref_pose`) and an `(N, 4, 4)` stack of instance
world transforms. The reference warps into all `N` instances via
`T_ref→obs = inv(cam2world) @ class_to_l2w[cls] @ ref_pose`, batched by einsum — RoMa's `get_gt_warp`
derives its batch from `depth1.shape[0]`, so the reference depth and `K` are `.expand`ed to `N`.
Moving from instance-keyed to class-keyed catalogs cut the reference set by roughly 6× and matches
what reference-prompted segmentation actually needs
(`optflow-4-class-keyed-one-to-many.md`).

`OptFlowMetadata` nests the full `ObsMaskDescriptorMetadata` as `obsmaskmeta` (with the descriptors
and PCA basis), and additionally carries `obs_intrinsics`, `class_to_name`, `class_to_reference`,
`class_to_reference_depth`, `class_to_ref_intrinsics`, `class_to_ref_pose` and `class_to_l2w`
(`vision_core/datastructs.py:507-515`). Note
`self._md.obsmaskmeta.iid_to_name = dict(self.iid_to_name)` on every `_optflow_metadata()` call
(`:93`) — the memoized object is refreshed with the accumulated name map so the final serialization
sees all frames' instances.

`class_to_reference` (top-level, RGB) duplicates `obsmaskmeta.class_to_ref` (RGBA) and is **being
deprecated** in favour of slicing `[:3]` or compositing the RGBA
(`plans/active/class-to-reference-rgba-dedup.md`). Current source still populates both, and both are
present in live render dirs; new consumers should read `md.obsmaskmeta.class_to_ref`.

A note against an older plan: the `OptFlowSample.visualize` keyword that was originally declared as
the reserved word `class` has been renamed. The current signature is
`visualize(self, md, *, cls_name=None, points=None, n_points=12, rel=0.05, title=None)`
(`vision_core/datastructs.py:418`), so the import-time failure recorded in
`optflow-4-class-keyed-one-to-many.md` is history.

### Extension recipe: adding a writer

1. Subclass `omni.replicator.core.Writer`. Set `self.data_structure = "renderProduct"` and
   `self.annotators = [AnnotatorRegistry.get_annotator(...)]` for exactly the channels you need —
   each annotator costs GBuffer memory per render product.
2. Override `attach(self, *rps)` to record `self._rp_key = rps[0].path.rsplit("/", 1)[-1]` and call
   `super().attach([rps[0]])` unless you genuinely want both stereo eyes.
3. Do all per-render precomputation in `__init__` (reference catalog, descriptor forwards) and free
   the model afterwards. Create one `IidCanonicalizer` and thread it into every `obsmask_from_data`
   call.
4. In `write(data)`, reuse `obsmask_from_data` for anything mask-shaped rather than re-deriving the
   masks; serialize with `sample.serialize(self._frame_id, self._render_dir)` and increment.
5. Implement `finalize_metadata(directory=None)` writing exactly one metadata sample at index 0, and
   assert the `iid_to_name` 1:1 contract there.
6. Wire it into the relevant entry point in `clean_datagen.py`. There is currently **no writer
   registry** — writer selection is the `if runtime.mode == "optflow"` branch in `main()`
   (`clean_datagen.py:159-162`), so a third writer means a third mode and a third entry function.
   That is the least extensible seam in the system.

---

## II.9 Store-USD scenes

### Responsibility

The store path inverts the usual direction. Instead of authoring a scene from a catalog of objects, it
takes an externally authored store USD (shelves, gondolas, lighting, hundreds of product prims) and
*extracts* a catalog out of it, then captures inside the original scene.
`extract_store_objects.py`, `store_scene.py` and `store_mutations.py` implement it. The invocations
are in [§III.5.3](#iii53-optflow-reference-catalogs-stages-ab-before-any-capture) and
[§III.5.5](#iii55-store-datasets).

### Stage A — extraction

`extract_store_objects.py` boots Isaac, loads the store, globs product prims by `product_patterns`,
and for each one runs `extract_one` (`:39-56`), which exports the product's subtree as a
self-contained `.usdz`, reads its untransformed bbox, applies a grasp policy, and serializes a
`GraspableObject` whose `meta` includes `store_prim`.

Three conventions are baked in here.

**SKU parsing.** `parse_sku` matches
`model_(?P<name>(?P<cls>[a-z][a-z0-9_]*?\d{3})(?:_\d+)?)`, so `model_sauces001_6` yields
`name = sauces001_6`, `class = sauces001` (`extract_store_objects.py:22-26`). One reference image per
class rather than per facing cuts exports by roughly 6×.

**The frame-consistency invariant.** `xformOps` live on the `model_*` parent; the `v_0` child is the
op-free modeling frame. Extraction exports `model_*/v_0` with its own xformOps neutralized, so the
`.usdz` frame equals the `v_0` local frame equals the modeling frame. At capture, `l2w` is read at
exactly that node, which picks up all ancestors' placement through `ComputeLocalToWorldTransform`.
The invariant is `l2w = get_target2world([v_0_path]) @ ref_pose`, and it holds identically for
store-native products and for inserted replacements. Exporting `model_*` instead would double-count
the placement transform (`store-usd-inverse-datagen.md`). One known generalization gap: `extract_one`
hardcodes `{model_path}/v_0`, and `model_drink101` has `v_69323` instead, so that SKU cannot currently
be extracted (`store-front-face-check.md`).

**Grasp policies.** `grasp_policies.py` is another `getattr` registry with two entries:

```python
class FixedFaceGrasp:
    def __init__(self, face): assert face in FACE_NORMALS  # side faces only, ±Z is singular
    def __call__(self, lo, hi, cls): return face_grasp_frames(lo, hi)[self.face]

class PerClassFaceGrasp:
    def __init__(self, faces): ...   # {class: face}, non-empty, all side faces
    def __call__(self, lo, hi, cls):
        assert cls in self.faces, f"PerClassFaceGrasp: no face for class {cls!r} ..."
        return face_grasp_frames(lo, hi)[self.faces[cls]]
```
(`grasp_policies.py:18-38`)

`PerClassFaceGrasp` exists because the global "front is −Y" assumption was measured and found wrong.
`debug_scripts/check_front_face.py` renders every SKU from all four side faces and picks the front by
whole-frame BT.709 luminance; −Y was correct for **29 of 63 products (46 %)**. The darkness that
distinguishes a back face from a front face is shelf *occlusion*, not dim lighting — an 8× lighting
control produced an identical front-face distribution, with occluded faces moving only ~3 → 8 luma
while true fronts sat at 90–150. Twenty of the 63 SKUs remain ambiguous (top-two margin < 15 luma) and
needed human confirmation (`store-front-face-check.md`). The resulting hand-curated table is baked
into the catalog at Stage A; Stages B and C replay `grasp_point` untouched, which is why capture-side
configs need no face table (`store-perclass-faces-verify-dataset.md`).

### Stage B — reference render

`graspableobj_to_optflow_obj.py` renders each `GraspableObject` in isolation from its grasp-anchored
viewpoint and emits an `OptFlowObject`.
`ref_pose_from_grasp(grasp_point, lo, hi, K, width, height, margin=1.1)` (`:19-28`) anchors the camera
at the **bbox centroid** `(lo+hi)/2` with the view direction taken from the grasp normal, and fits the
FOV to the full extents about that centroid. Anchoring at the grasp point instead (which is a
face-centre, off-centre for e.g. an amazon box) crops the top of the reference
(`optflow-centroid-ref-and-visualize.md`). Depth outside the object is zeroed; `ref_pose` is stored
OpenCV.

### Stage C — capture

`build_store_scene(runtime, objects)` (`store_scene.py:103-109`) is four lines: build and validate
`StoreSceneSpec`, load the store with lights, bind each catalog object to its live prim via
`resolve_product_prim`, apply the mutation chain, and finalize.

`load_store` references the store USD's `/root` default prim under `/World/Store`
(`store_scene.py:41-48`). Referencing `/root` **splices its children directly**, so products live at
`/World/Store/model_*` with no `/root/` segment, and `store_prim` paths stay relative to the store
root and survive a different mount point.

`_finalize_store_scene` (`store_scene.py:68-95`) is the shared tail for both store builders. It does
three things in order: the leak guard, labelling, and camera construction.

The **leak guard** matters more than it looks:

```python
tracked = {t.obj.meta["class"] for t in targets}
for glob in spec.require_tracked_only:
    leaked = sorted(p.GetName() for p in store_mutations.active_products(store, [glob])
                    if store_mutations.parse_sku(p.GetName())[1] not in tracked)
    assert not leaked, ...
```

A store config with no mutations leaves *every* product physically present but unlabelled, so another
dataset's held-out classes appear as unlabelled background — silent contamination that label-set gates
cannot see. `require_tracked_only` plus a `RemoveUntrackedProducts` mutation is the fix
(`store-snacks-training-datasets.md`).

It also asserts `" " not in t.obj.meta["class"]`, the whitespace-tokenization guard from
[§II.8](#ii8-writers), and uses `add_catalog_grasp_frame` unconditionally — store targets always
replay the baked grasp.

### Scene mutations

`store_mutations.py` is the sixth reflection registry in the codebase. Its data model:

- `CaptureTarget(obj, prim_path, scale=1.0)` (frozen, `:16-20`) — a *binding*, not a placement. The
  `obj.meta` and `prim_path` must stay aligned through the whole chain, because the writers'
  `objects[i] ↔ object_prim_paths[i]` contract depends on it.
- `ProductSite(name, path, lo, hi, l2w, grasp)` (frozen, `:80-88`) — a **snapshot** of a shelf slot's
  geometry, taken by `measure_site` *before* any deactivation. Deactivating a prim prunes it from
  `GetChildren` and makes its bbox unreadable, so the measurement must precede the edit.
- `Site(store_prim, grasp, cls)` (`:90-95`) — a curated site read from an `OptFlowObject` catalog by
  `load_sites` (`:97-112`), which fails loud if any object lacks `store_prim`.

`apply_mutations(root, spec, targets, effective_seed)` (`:34-41`) chains the instantiated mutations,
each of which is `__call__(store, spec, targets, rng) -> targets`, and asserts the result is non-empty
with unique names. The RNG stream is `[effective_seed, 3]` — 0, 1 and 2 are taken by the distant, dome
and pattern light jitters respectively.

Removal is `SetActive(False)` on the `model_*` root, and the choice is forced: store USDs are
assembled via references, and `RemoveProperty` cannot delete an opinion that lives in a referenced
arc. `SetActive(False)` is USD's canonical pruning mechanism and has the useful side effect of
removing the prim from `GetChildren`, so later mutations skip it automatically. `hideForCamera` was
rejected here as unreliable under path tracing (`store-scene-mutations.md`).

The shipped mutations (`store_mutations.py:211-307`):

| Mutation | Scope | `PLAIN_SAFE` |
|---|---|---|
| `RemoveClass(pattern)` | deactivates every product whose SKU class matches an fnmatch glob | no |
| `RemovePrims(names)` | deactivates exact prim names; fails loud on a name that matches nothing | no |
| `RemoveUntrackedProducts()` | keep-list *complement* — deactivates every active product whose class has no `CaptureTarget` | no |
| `ReplaceClass(pattern, catalog, source_class='*')` | deactivates matches and inserts sampled replacements at the measured site poses | no |
| `DisablePhysics(pattern)` | sets `rigidBodyEnabled=False` on matching prims, two-level fail-loud | **yes** |

`RemoveUntrackedProducts` is complement-based because the curated front-face table lists the 42
classes to *keep*; enumerating the ~20 to drop would be a second list to keep in sync. `RemovePrims`
exists for per-instance store-authoring quirks that are not class-shaped: 12 label-inward facings in
`snack012`/`snack032` whose local +Y grasp face points into the gondola (the camera at 0.6 m along
that normal lands inside the shelf and photographs flour boxes — physically unphotographable
in-store), and 2 coincident-duplicate prims occupying the same world position to within 0.00 mm, where
the instance segmenter labels the shared surface as one instance and the twin orphans
(`store-perclass-faces-verify-dataset.md`).

Replacement placement is pure math. `replacement_pose(site, lo_r, hi_r, grasp_r, scale)` (`:128-134`)
builds a rotation aligning the replacement's +X grasp face to the site's grasp face and a translation
putting the replacement's bbox bottom-centre on the site's shelf-contact point.
`_orthonormal_rotation` (`:71-77`) de-scales `l2w`'s rotation block column-wise (shelf authoring can
carry non-uniform scale) and asserts orthonormality to 1e-4, positive determinant, and uprightness
(`R[2,2] > 0.99`). `fit_scale(site, ext_r, grasp_r, threshold)` (`:137-147`) computes a uniform shrink
that brings the worst-fitting axis to `threshold ×` the site extent, never enlarging, so a replacement
cannot clip through its neighbours.

### The repopulation builder

`build_repopulated_store_scene` (`store_scene.py:112-141`) is the third builder: it reads a curated
site catalog, zips the (already collected, replicated and shuffled) object queue against sites **in
order**, inserts each object anchored to its site's curated grasp, deactivates every remaining
product, and freezes physics on the inserts. It asserts `spec.site_catalog` and `spec.fit_threshold`
are set, asserts `spec.mutations` is *empty* (sites drive placement, so a mutation chain would fight
it), and asserts `len(objects) <= len(sites)` with a message pointing at the `ReplicateFilter` count
as the knob. `freeze_physics` (`store_mutations.py:195-208`) is mandatory here for the same free-fall
reason as `DisablePhysics` in plain scenes.

### Open issues

**Arm-B is unresolved.** Catalogs whose `.usdz` files were scrubbed of vendor semantics by
`debug_scripts/pull_flatten_usd.py` lose their class labels at capture (32 orphans), while unscrubbed
catalogs and YCB swap-ins are clean. Suspects include `PhysicsRigidBodyAPI`, `instanceable` flags, and
annotator prim-ancestry union; the diagnosis needs an in-kit probe on a scrubbed-swap wrapper
(`store-scene-mutations.md`). Until it is closed, prefer unscrubbed catalogs.

**Index spaces do not correspond.** `OptFlowWriter` has no per-frame → catalog-object tie: `obs_NNNN`
indexes capture frames and `reference_image_NNNN` indexes the catalog. They are unrelated, and any
code that assumes otherwise is wrong (`store-perclass-faces-verify-dataset.md`).

### Extension recipe: adding a mutation

1. Define a class in `store_mutations.py` with `__init__(self, **args)` and
   `__call__(self, store, spec, targets, rng) -> list[CaptureTarget]`.
2. Decide portability: if it reads only the walk root and never `spec.product_patterns` or other
   store-shaped fields, set `PLAIN_SAFE = True` so `PlainSceneSpec` accepts it. Otherwise leave it off
   and it is store-only, enforced at config-parse.
3. Remove by `SetActive(False)`, never by property deletion. Measure any site geometry you need
   *before* deactivating.
4. Keep the `obj ↔ prim_path` binding correct: use `_drop_under` to prune targets under a deactivated
   subtree, and preserve list order — plain scenes assert order is unchanged, and the optflow writer's
   `objects[i] ↔ l2w[i]` zip depends on it.
5. Fail loud on a pattern or name that matches nothing; a silently no-op mutation is the failure mode
   this registry exists to prevent.
6. Reference from `scene_builder_args.mutations: [{name: YourMutation, args: {...}}]`.

---

## II.10 Downstream phases

`clean_datagen` is phase 1 of three. This section covers the *interface* — what phase 1 must write for
phases 2 and 3 to work, and what they read. [§III.2](#iii2-cli-reference) covers running them.

### The handoff

A completed render dir contains, at minimum:

- `obs/`, `iid_mask/`, `cid_mask/`, `iid_to_occlusion/` — per-frame `ObsMask` fields (plus
  `observation_depth/`, `cam2world/`, `iid_to_visibility/` in optflow mode).
- The metadata sample at index 0: `cid_to_class`, `name_to_class`, `iid_to_name`, `class_to_ref`,
  `class_to_descriptors`, `principal_components` (and the optflow catalogs).
- `runtime.yaml` — the `asdict(RuntimeConfig)` snapshot.
- `descriptor.yaml` — the loaded descriptor config.
- `lighting_log.json` if `log_lighting`.
- `cid_iid_trace.log`, if anything in the writer path called `cid_iid_trace.log()`.

Phases 2 and 3 are invoked against `runtime.yaml`, not against the original config, so they inherit
every CLI override the render actually used.

### Why phases are separate processes

`run_pipeline.py` runs each phase as a **subprocess**, and the reason is memory: Isaac Sim releases its
GPU allocation only when its process exits, and the phase-2 proposer needs about 14 GB
(`run_pipeline.py:10-13`, `pipeline-orchestrator.md`). This is not stylistic decomposition; the phases
cannot share an address space.

### Phase 2 — proposals

`add_proposals.py` loads a proposer, and for each frame:

1. Seeds per frame with `seed_everything(effective_seed + idx)` so a frame's proposals are identical
   whether the frame ran whole or as part of a shard (`add_proposals.py:51`).
2. Gates which classes are worth proposing on:
   `gate_classes_reproj(sample, md, min_visible_ratio, tau_d, tau_r, ref_cache)`
   (`proposal_gate.py:18-30`) returns `{class: max_instance_ratio}`, where the ratio is the fraction of
   the class's *reference texture* that reprojects visibly into this observation. This is the same
   `instance_visibility` machinery `OptFlowWriter` runs at write time, which is why the optflow writer
   stores `iid_to_visibility` — the gate is reading geometry phase 1 already established. The ratio
   gate replaced an absolute-pixel-count gate (`proposer_min_visible_px`, still present in
   `RuntimeConfig` but deprecated) because a ratio is camera-distance invariant
   (`reproj-coverage-gate-and-ycb-ref-pose-fix.md`).
3. Writes `proposals/proposals_NNNN.pt` atomically.

Resume is a membership-set check over existing `proposals_NNNN.pt` files, not a prefix counter, so it
composes with arbitrarily interleaved shard completion (`add-proposals-resumable.md`). Multi-GPU
sharding fans contiguous `[start_frame, end_frame)` windows out to `proposer_devices`
(`run_pipeline.py:49-64`).

Note one current-source constraint: `add_proposals.main()` unconditionally does
`OptFlowMetadata.deserialize(0, render_dir)` (`add_proposals.py:21`), i.e. it expects the
optflow-shaped metadata. See [Appendix B](#appendix-b--open-questions-and-unverified-claims).

### Phase 3 — inlier labels

`add_inlier_data.py` labels each proposal point inlier or outlier by testing whether it lies at least
`inlier_border_eps` pixels *inside* its class's union mask, via `coords_in_mask(mask, coords, eps)` and
`cv2.distanceTransform`. The margin exists because grazing points on a silhouette straddle the
background. `--eps` is mandatory on the CLI. Phase 3 has **no resume**: it unconditionally rewrites all
`labels_*.pt` and `stats/stats_0000.json` on every run, because labelling is cheap and re-labelling
with a new eps must be transparent (`inlier-border-eps-margin.md`).

The distance transform only sees in-image zeros, so an object cut off by the frame edge is treated as
interior there. That is intentional — frame truncation is not occlusion.

### What phase 1 must get right for the downstream to work

- **`cid_mask` coverage.** Every instance in `iid_mask` must have non-background `cid_mask` pixels, or
  its class is invisible to the gate. `run_pipeline` validates for orphans before loading the proposer
  (`run_pipeline.py:93-102`), and `isaac-datagen-validate-obsmask` does it standalone. Single-token
  class names are the precondition ([§II.8](#ii8-writers)).
- **Descriptor consistency.** The `class_to_descriptors` phase 1 baked must come from the same backbone
  phase 2 matches against, which is why `descriptor.yaml` is copied into the render dir.
- **Phase 1 is not resumable.** `run_pipeline` skips the render if `obs/` exists with ≥1 frame; to
  re-render you must delete the whole `render{idx:03d}/`, not individual files, because a partial
  re-render under existing `proposals/` silently desyncs them (`run_pipeline.py:83-91`). A render
  killed mid-capture leaves observations on disk but no `runtime.yaml` and no metadata, and is
  unusable — `finalize_metadata` runs only at the very end.
- **Frame windows are a sharding tool only.** Phase 3 labels *all* frames, so a phase-2 run over a
  partial window leaves phase 3 failing on the missing proposals (`sharded-proposals.md`).

An optional fourth consumer, `make_unseen.py`, builds a zero-shot evaluation variant by renumbering a
frame range, swapping the R and B channels of both the observations and the class reference catalog,
recomputing descriptors for the shifted domain, and re-running phases 2 and 3 — no re-render required
([§III.5.7](#iii57-unseen-eval)).

### Extension recipe: adding a downstream phase

1. Read the render dir, not the original config:
   `load_config(render_dir / "runtime.yaml", overrides)`. Any parameter your phase needs must be a
   `RuntimeConfig` field so it is in the snapshot.
2. Serialize your outputs as a new field subdirectory of the existing samples using residual
   serialization (`sample.serialize(i, dir, only={"your_field"})`), the pattern phases 2 and 3 use to
   add `proposals/` and `labels/` to render dirs they did not create.
3. Write atomically (`tempfile.mkstemp` + `os.replace`) and make resume a membership check over final
   filenames, so a hard kill leaves at worst a stray dotfile and never a half-written sample.
4. Add a console script and chain it in `run_pipeline.py`, after the geometry validation step.

---

# Part III — Usage and operations reference

This part is the operator's manual: how to install it, what to type, what every knob means, what
lands on disk, and what will bite you. For *what the pieces are* see
[Part I](#part-i--system-overview-and-architecture); for *why the subsystems are shaped the way they
are* see [Part II](#part-ii--subsystem-design). Nothing here re-derives those.

## III.1 Install and environment

### III.1.1 The package and its siblings

`isaac_datagen` is a `uv`-managed package (`uv_build` backend, src layout) that depends on three
editable siblings and one very large binary dependency:

| Dependency | How it is wired | Why it matters to you |
|---|---|---|
| `isaacsim[all,extscache]==5.1.0` | PyPI + `https://pypi.nvidia.com` extra index (`[[tool.uv.index]] name = "nvidia"`, non-explicit so it acts as a fallback) | Pins `numpy==1.26.0`, `pillow==11.3.0`, drives `torch==2.7.0`. Every other pin in the tree floats to these. |
| `vision-core` | `{ path = "../vision_core", editable = true }` | Owns `SerializableSample` and the on-disk dataset contract (`ObsMask`, `OptFlowSample`, `PreReferenceSegSample`, …). Imported *from source*. |
| `reference-matching` | `{ path = "../reference_matching", editable = true }` | Descriptor backbones (DIFT/CleanDIFT/FPN) and the stage-1 proposers. Imported from source. Deliberately unpinned on numpy/torch/pillow so it floats to Isaac's versions. |
| `gim`, `mask2former`, `multiscaledeformableattention`, `lightglue`, `detectron2`, `romatch` | Transitive via `reference-matching`, but **re-declared** in `isaac_datagen/pyproject.toml` `[tool.uv.sources]` | `uv.sources` from a path dep do **not** propagate to the consuming project; without the re-declaration resolution silently picks the wrong artifact. |

Two pins in `pyproject.toml` exist purely to survive ABI breakage and must move in lockstep with
`torch`:

- `constraint-dependencies = ["detectron2==0.6+fd27788pt2.7.0cu128"]` — the prebuilt detectron2 wheel
  encodes its `pt<torch><cuda>` tag in the version string. Without the exact constraint, `uv` takes
  the lexically highest variant (e.g. `pt2.9.1cu130`) and imports die on `libcudart.so.13` /
  undefined torch symbols.
- `no-build-isolation-package = ["multiscaledeformableattention"]` — the MSDeformAttn CUDA op's
  `setup.py` imports `torch`, so it must build against the env's torch, not an isolated one.

`romatch` is an **optional extra** (`--extra viz`); it is fetched from git, not from a workspace
submodule, because `refseg-workspace` has no RoMa submodule. `OptFlowSample.visualize()` and
`debug_scripts/viz_optflow*.py` lazily import it.

The architectural reason for this split — the unsatisfiable `isaacsim` vs `segmentation` pin sets — is
in [§I.1](#i1-what-clean_datagen-is-and-the-problem-it-solves).

### III.1.2 Why plain `python3` fails

The three editable siblings and `isaacsim` are installed **only into the project venv**. There is no
`pip install -e` into a system interpreter anywhere in this stack. `python3 clean_datagen.py`
therefore fails at the first `from isaac_datagen.scene import …` / `import isaacsim`. Always
`uv run …` or use a console script installed into `.venv/bin` (`isaac-datagen`,
`isaac-datagen-pipeline`, …), which is the same thing with the shebang baked in. This is stated
explicitly in the repo's `CLAUDE.md` "External" note: *"these editable deps import only inside the
project venv — use `uv run` (plain `python3` has no venv)."*

### III.1.3 Required launch cwd

**Launch cwd must be `isaac_datagen/src/isaac_datagen/`.** Every shipped config resolves its paths
relative to the process cwd, not to the config file's directory:

- `intrinsics_path: zed_K.npy` → `src/isaac_datagen/zed_K.npy`
- `objects_path: [../../assets/optflow_objects/amazon-v2/]` → `isaac_datagen/assets/…`
- `descriptor_config_path: ../../../reference_matching/src/reference_matching/configs/descriptor.yaml`

`RuntimeConfig.__post_init__` asserts each of these exists (`runtime_config.py:150-153`), so launching
from the repo root fails immediately with a `missing:` assertion rather than mysteriously later. That
is the good case; the bad case is a config with an absolute `dataset_dir` and relative asset paths,
which gets *further* before dying.

Two scripts deliberately want a **different** cwd: `migrate_descriptors_backbone` and the
`make_unseen` shell-out to it want the repo root, because the descriptor config's relative paths are
anchored there ([§III.8.11](#iii811-miscellaneous-sharp-edges)).

### III.1.4 Environment defaults set at import time

`clean_datagen.py` sets two environment variables **at module import**, before anything else is
imported (`clean_datagen.py:3-5`):

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
```

- `OMNI_KIT_ACCEPT_EULA=YES` — without it, Kit prompts interactively on boot and **blocks forever
  under `nohup`/`sbatch`** (`store-usd-inverse-datagen.md`, EULA note). Because it is `setdefault`, an
  explicit env var wins.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation when the descriptor
  backbone and Isaac's own torch allocations share a device.

Scripts that boot Isaac **without** going through `clean_datagen.py` (e.g.
`extract_store_objects.py`, `graspableobj_to_optflow_obj.py`, `debug_scripts/check_front_face.py`) do
not necessarily inherit this, which is why the store-pipeline recipes in
[§III.5](#iii5-cookbook) all carry an explicit `OMNI_KIT_ACCEPT_EULA=YES` prefix.

RTX/Kit settings are *not* environment variables — they are carb settings applied inside `boot_sim`
(`scene.py:305-348`): `/rtx/renderMode=PathTracing`, `/rtx/pathtracing/totalSpp`,
`/rtx/pathtracing/maxBounces`, `/rtx/denoiser/enabled=True`, texture-streaming budget,
`/rtx/post/tonemap/{exposureTime,fNumber,filmIso}` when `set_exposure`, and
`/omni/replicator/backends/disk/root_dir` pointed at the render dir. `boot_sim` prints a `[TONEMAP]`
line with the resolved tonemap settings (`scene.py:338-346`) — that line is your evidence that the
exposure fix ([§III.8.3](#iii83-exposure-the-other-dark-render-bug)) is actually in effect.

---

## III.2 CLI reference

All of these are console scripts declared in `pyproject.toml [project.scripts]`. Phase-1 scripts boot
Isaac Sim; everything else does not (and therefore does not need a GPU with RT cores).

### III.2.1 Pipeline phases

| Command | Entry point | Phase | Arguments |
|---|---|---|---|
| `isaac-datagen <config.yaml> [k=v …]` | `clean_datagen:main` | 1 (Isaac) | positional config; **all** remaining `key=value` args are passed to OmegaConf as a dotlist via `parse_known_args`. Zero args → prints help (`TLDR` epilog) and exits 0. Dispatches on `runtime.mode`: `optflow` → `optflow_generation()`, anything else → `reference_segmentation()`. |
| `isaac-datagen-pipeline <config.yaml> [k=v …]` | `run_pipeline:main` | 1→2→3 | Same arg surface. Runs render (skipped if `obs/` already has frames), then cid/iid validation, then an interactive y/N gate (auto-skipped when stdin is not a tty), then proposals (sharded if `proposer_devices` has >1 entry), then inliers with `--eps runtime.inlier_border_eps`. |
| `isaac-datagen-proposals <config.yaml> [k=v …]` | `add_proposals:main` | 2 | **Takes a config, not a render dir.** Resolves `render_dir = dataset_dir/render{idx:03d}` itself. Honours `start_frame` / `end_frame` / `proposer_device` / `proposer_min_visible_ratio` / `proposer_tau_d` / `proposer_tau_r`. Resumable: skips frames whose `proposals/proposals_NNNN.pt` already exists. |
| `isaac-datagen-downsample-proposals <render_dir> [--max-points K] [--dry-run]` | `downsample_proposals:main` | 2.5 | FPS-caps each class's proposals to `K` (default 256). `--dry-run` counts and prints without writing. Rewrites residually (`serialize(..., only={"proposals"})`). |
| `isaac-datagen-inliers <render_dir> --eps <float>` | `add_inlier_data:main` | 3 | `--eps` is **required** (argparse `required=True`). Rewrites *all* `labels_*.pt` and `stats/stats_0000.json` unconditionally — no resume, no skip. |
| `isaac-datagen-unseen <config.yaml> <source_render_dir> (--start S --end E \| --split-manifest J [--split val] [--limit N]) [k=v …]` | `make_unseen:main` | eval-set build | Builds a channel-swapped 0-shot eval dir. Refuses to overwrite: exits if the destination `render{idx:03d}` already exists. |

> **Conflict resolved against current source.** `tldr.py` documents phase 2 as
> `isaac-datagen-proposals <render_dir> [key=value ...]` (`tldr.py:8`). The actual usage string in
> `add_proposals.py:16` is `isaac-datagen-proposals <config.yaml> [key=value ...]`, and `main()` calls
> `load_config(sys.argv[1], sys.argv[2:])` (`add_proposals.py:18`). **Pass a config.** `run_pipeline`
> does exactly that (`_run("isaac-datagen-proposals", *sys.argv[1:], *extra)`). The help text is
> stale — see [Appendix A](#appendix-a--known-documentation-staleness).

`isaac-datagen-unseen` is more capable than a start/end window: besides `--start/--end` it accepts
`--split-manifest <json> [--split val] [--limit N]` (`make_unseen.py:43-49`), which selects exactly
the frames a frozen train/val split assigned to that render dir
(`make_unseen.py:_select_val_frames`, keyed `f"{src.parent.name}/{src.name}"`). It then:

1. Renumbers the selected frames to `0..N-1`, R↔B-flipping `s.obsmask.obs` (`SwapRedBlue`).
2. Flips `class_to_ref` in the catalog and **blanks** `class_to_descriptors` and
   `principal_components` (`_flip_catalog`) so they must be recomputed.
3. Copies `runtime.yaml`, `descriptor.yaml`, `lighting_log.json` if present.
4. Shells out to `python -m isaac_datagen.migrate_descriptors_backbone add-backbone …` **with
   `cwd=REPO_ROOT`** (`make_unseen.py:15`, `:87-89`), then `add_proposals` and `add_inlier_data`.

### III.2.2 Validation, debug and visualization

| Command | Entry point | Isaac? | Arguments |
|---|---|---|---|
| `isaac-datagen-validate-obsmask <render_dir>` | `validate_obsmask:main` | no | Loads only `iid_mask/`, `cid_mask/`, `iid_to_occlusion/`; prints one row per cid/iid orphan to stderr; `exit 1` if any. |
| `isaac-datagen-debug-render <config.yaml> [k=v …]` | `debug_render:main` | phase-1 boot only | Runs `isaac-datagen … dry_run=true`, checks `render_dir/debug/scene.usdz` exists, then `blender --background --python blender_render.py -- <debug_dir>`. Fails loudly if `blender` is on neither `PATH` nor `/usr/local/bin/blender`. |
| `isaac-datagen-measure-luminance <render_dir> [--pixel-threshold 8.0] [--frame-threshold 0.5] [--metric dark_frac\|fg_mean] [--csv OUT] [--with-lighting]` | `measure_luminance:main` | no | Per-frame foreground BT.709 luminance; flags dark frames; `--with-lighting` joins to `lighting_log.json`. Also invoked automatically by `run_pipeline` before its y/N gate. |
| `isaac-datagen-viz-inliers <render_dir> [--out] [--frames a,b,c] [--max-frames 8] [--stride 1] [--cols 4] [--dpi 300] [--max-points N] [--class C]` | `viz_inliers:main` | no | Inlier/outlier label overlay. |
| `isaac-datagen-viz-sample <render_dir> [--out] [--frames] [--max-frames 8] [--stride] [--cols 4] [--dpi 200] [--max-points] [--alpha 0.45]` | `viz_sample:main` | no | Per-class refs + all-proposals + gt-mask panels. |
| `isaac-datagen-viz-gate <roots…> [--min-visible-ratio 0.30] [--out-root DIR] [--limit N]` | `viz_gate:main` | no | Per-frame proposer gate decision; takes **multiple** dataset roots and/or render dirs. |
| `isaac-datagen-viz-occlusion <dataset_dir> [--n 12] [--seed 0] [--cols 4] [--alpha 0.45] [--dpi 200] [--out]` | `viz_occlusion:main` | no | Random-sample spot check of `iid_to_occlusion`. Takes the **dataset dir** (the one holding `render000/ render001/ …`), not a render dir. |
| `isaac-datagen-sweep-label-eps <render_dir> [--idx 0] [--eps 0 1 2 3 5 8] [--out] [--cols 4] [--dpi 100] [--max-points] [--no-composite]` | `sweep_label_eps:main` | no | Sweeps `coords_in_mask` border-eps on one frame to pick `inlier_border_eps`. |
| `isaac-datagen-ingest30-configs …` | `ingest30_configs:main` | no | Emits per-dataset render configs from the ingest30 root manifest. |
| `meta <init\|arm\|…> …` | `ingest30_drive:main` | orchestrator | The ingest30 experiment driver (init an artifact root, run arms `gligen`/`closed`/`retrained`, score labels). Out of scope for this document beyond noting that it *generates* configs of the shape described in [§III.3](#iii3-config-reference). |

### III.2.3 Not console scripts (run with `uv run python …`)

| Script | Role |
|---|---|
| `extract_store_objects.py <config.yaml> <out_dir> [k=v …]` | Stage A of the store pipeline: export each `model_*/v_0` subtree as a self-contained usdz + bbox-derived grasp frame ([§II.9](#ii9-store-usd-scenes)). |
| `graspableobj_to_optflow_obj.py <config.yaml> <in_dir> <out_dir>` | Stage B: render one isolated centroid-anchored reference RGB-D per object → `OptFlowObject` catalog. |
| `mesh_convert.py {ycb\|stage\|finalize} …` | Mesh → `GraspableObject` dataset (download / render candidates / apply `winners.yaml`). |
| `relabel_classes.py <dataset_dir> [--grid-only]` | Interactive class relabel with residual re-serialization. |
| `migrate_descriptors_backbone.py add-backbone <dataset_root> <descriptor_cfg> --device …` | Add a second descriptor backbone to existing render dirs. **Run from the repo root, not `src/isaac_datagen/`** ([§III.8.11](#iii811-miscellaneous-sharp-edges)). |
| `migrate_pca_basis.py <dataset_root> [--dry-run]` | Backfill the now-mandatory `principal_components/` on legacy dirs (`pca-basis-mandatory-field.md`). |
| `debug_scripts/{viz_optflow,viz_optflow_objects,check_front_face,debug_scene,debug_occupancy,backfill_grasp_point,pull_flatten_usd,verify_centroid_ref}.py` | Ad-hoc debugging; see [§III.7.6](#iii76-visualization-tools). |

---

## III.3 Config reference

`RuntimeConfig` (`runtime_config.py:21-164`) is a plain dataclass turned into an OmegaConf structured
config. `load_config` merges **schema < YAML < CLI dotlist** in that order, so a CLI `key=value`
always wins. The loader machinery and the validation rationale are
[§II.1](#ii1-configuration).

Fields with **no default** are mandatory: the merge produces `MISSING` and `OmegaConf.to_object`
raises. These are: `idx`, `mode`, `num_targets`, `scene`, `dataset_dir`, `intrinsics_path`,
`descriptor_device`, `proposer_device`, `proposer_config_path`, `descriptor_config_path`, `placement`,
`dome_light`, `dry_run`, `inlier_border_eps`.

### III.3.1 Identity, output, seeding

| Key | Default | Meaning | Example |
|---|---|---|---|
| `idx` | — | Render index; output goes to `dataset_dir/render{idx:03d}/`, and `effective_seed = seed + idx` | `0` |
| `mode` | — | `reference_segmentation` \| `optflow`. Case-sensitive; anything else fails `__post_init__` | `optflow` |
| `dataset_dir` | — | Parent dir; **must pre-exist** (`assert Path(self.dataset_dir).exists()`) | `/data/user/jeffk/datasets/expanded-refseg-v2` |
| `seed` | `1` | Base RNG seed; `seed_everything(effective_seed)` is called before `boot_sim` | `1001` |
| `num_frames` | `None` | Camera poses per target. Exactly one of `num_frames` / `grid_dims` must be set (XOR assert) | `100` |
| `grid_dims` | `None` | `(nx, ny, nz)` grid sampling instead of `num_frames` | `[5,5,4]` |
| `num_targets` | — | Grasp targets sampled per render. `null` = use every grasp frame once | `10` or `null` |
| `width` / `height` | `1920` / `1080` | Render resolution | |
| `start_frame` / `end_frame` | `0` / `None` | Phase-2 frame window `[start, end)`. **Sharding only** — see [§III.8.6](#iii86-phase-23-semantics) | |

Observation count is `num_frames × num_targets`, not `num_frames`
(`.docs_claude/psc-isaac-datagen-footguns.md`; mechanism in [§II.7](#ii7-capture)). Size walltime off
that.

### III.3.2 Objects, filtering, placement

| Key | Default | Meaning |
|---|---|---|
| `objects_path` | `[]` | List of catalog dirs; each must have `meta/meta_*.yaml`. Asserted non-empty for **both** modes (`unify-objects-path.md` replaced the old asymmetric `graspable_objects_path` / `optflow_objects_path` split). Duplicate `meta["name"]` across catalogs raises `ValueError` in `collect_objects` / `collect_preoptflow`. |
| `filter_specs` | `[]` | Ordered `{name, args}` list applied by `filter_objects`; registry is `filters.py` (`ShuffleFilter(seed)`, `RegexFilter(key, value)`, `MetaFilter(key, value, max)`, `ReplicateFilter(count, key, value)`). Raises if the pool ever becomes empty. Semantics: [§II.3](#ii3-the-filter-registry). |
| `placement` | — | Placer registry class name (`placers.py`): `UntilExhaustedStacker` \| `ShelfPlacer` |
| `placement_args` | `{}` | Ctor kwargs. `UntilExhaustedStacker(max_column_height, min_y, max_y, min_gap, max_gap, epsilon)`; `ShelfPlacer(column_height, …)`. Only the column-height parameter is required — the jitter parameters carry defaults (`placers.py:72-73`); omitting the column height raises `TypeError` at scene-build time. |
| `scene` | — | Scene name; `empty` skips the workbench asset load |
| `scene_builder` | `build_scene` | `scene_builders.py` registry key: `build_scene` \| `build_store_scene` \| `build_repopulated_store_scene`. **Consulted only in `mode: optflow`** — see [§II.4](#ii4-scene-construction) |
| `scene_builder_args` | `{}` | Forwarded to the builder. `build_scene` → `PlainSceneSpec(mutations, grasp_frames, orientation)`; `build_store_scene` → `StoreSceneSpec(store_usd, product_patterns, grasp_frame_policy, grasp_frame_policy_args, mutations, require_tracked_only, site_catalog, fit_threshold)` |
| `pallet_dims` | `None` | Legacy `OccupancyGrid` dims; retired from the main path |

Note: store configs still set `placement: UntilExhaustedStacker` with the inline comment *"required by
RuntimeConfig; unused in store mode"* — it is a mandatory schema field that the store builder ignores.

### III.3.3 Camera posing

| Key | Default | Meaning |
|---|---|---|
| `pose_generation_policy` | `GridFixedPoser` | `posers.py` registry: `GridFixedPoser` \| `LookAtPoser` \| `DecenteredLookAtPoser` \| `FixedOffsetPoser`. Lookup is `getattr(sys.modules[__name__], name)` — only module-level class names work. |
| `pose_generation_policy_args` | `{}` | Ctor kwargs, typically OmegaConf-interpolated from the top-level ranges: `{xrange: ${xrange}, yrange: ${yrange}, zrange: ${zrange}}`. `GridFixedPoser` additionally wants `target_to_ego_ypr` and `random`; `random=False` requires `grid_dims`. `DecenteredLookAtPoser` requires `intrinsics_path`, `resolution`, `object_radius` (+ `margin_deg`, `max_roll_deg`). |
| `xrange` / `yrange` / `zrange` | `(0.550,0.550)` / `(-0.22,0.22)` / `(0.01,0.01)` | Halo-box offsets in the target frame |
| `target_to_baseline_ypr_desired` | `(90, 0, 90)` | Fixed YPR for `GridFixedPoser` |
| `intrinsics_path` | — | `.npy` K matrix, must exist. Ships as `zed_K.npy` |
| `warmup_frames` | `32` | `app.update()` calls before capture, to settle RTX/PT/denoiser state (`runtime_config.py:113`). `pre-capture-render-warmup.md` documents 16; **32 is the current default** and the plan is history. |
| `rt_subframes` | `20` | Replicator accumulation subframes per orchestrator step (`>= 1` asserted) |

### III.3.4 Lighting and exposure

| Key | Default | Meaning |
|---|---|---|
| `distant_light` | `True` | The key light (parallel rays, no inverse-square falloff) |
| `distant_intensity` | `3000.0` | Key intensity |
| `distant_angle` | `0.53` | Angular diameter; raise for softer penumbra |
| `distant_light_offset` | `(1.0, -3.0, 3.0)` | Sun position relative to the grasp-target centroid; direction only. Keep it non-vertical — `look_at` builds `x` via `cross(z, [0,0,1])` and goes singular on a near-vertical offset (`distant-light-key-light.md`) |
| `jitter_distant` | `False` | Enable per-frame key jitter |
| `distant_offset_jitter` | `0.75` | Position jitter half-width, metres; asserted `>= 0` |
| `distant_intensity_jitter` | `None` | `[lo, hi]`, asserted `0 <= lo <= hi` |
| `distant_temperature_jitter` | `None` | `[lo, hi]` Kelvin, asserted `1000 <= lo <= hi <= 10000` |
| `dome_light` | — | Dome fill (mandatory field). `false` still creates a `/World/DomeLight` prim, at intensity 0 |
| `dome_fill_intensity` | `200.0` | ~10-20 % of key |
| `jitter_dome` | `False` | Per-frame dome intensity jitter — **see the caveat in [§III.8.4](#iii84-per-frame-lighting-jitter-is-partly-broken)** |
| `dome_intensity_range` | `(500.0, 1000.0)` | Asserted `lo <= hi` |
| `dome_normalize` | `False` | Solid-angle normalization (unused in the current code path) |
| `light_jitter_patterns` | `[]` | `LightJitterSpec{root, pattern, intensity_scale_range}` for store fixtures; each asserted `root and pattern and 0 < lo <= hi`. Factors are drawn **log-uniformly** (`store-usd-inverse-datagen.md`) |
| `log_lighting` | `True` | Write `lighting_log.json` |
| `set_exposure` | `True` | Apply the photographic tonemap settings |
| `exposure_time` | `1.0` | Asserted `> 0`. **Do not leave this at Kit's stale 0.02** — that is the root cause of dark-box Bug 1 ([§III.8.3](#iii83-exposure-the-other-dark-render-bug)) |
| `f_number` | `5.0` | Asserted `> 0` |
| `film_iso` | `100.0` | Asserted `> 0` |
| `background_textures` / `texture_paths` | `()` / `()` | Background randomization sources; `background_textures` is commonly `${call:_glob_amazon_textures}` |

### III.3.5 Occluders

| Key | Default | Meaning |
|---|---|---|
| `occluders_per_target` | `0` | Invisible (`primvars:hideForCamera=True`) shadow-casting primitives per grasp target |
| `occluder_scale` | `None` | Half-width; `None` → random in `[0.04, 0.2]` from the seeded global stream |
| `occluder_pose_policy` | `GridFixedPoser` | Poser registry key for occluder placement |
| `occluder_pose_policy_args` | `{}` | Target-frame ranges + rotation |

### III.3.6 Renderer

| Key | Default | Meaning |
|---|---|---|
| `path_tracing_spp` | `256` | `/rtx/pathtracing/totalSpp` |
| `path_tracing_max_bounces` | `12` | `/rtx/pathtracing/maxBounces` |
| `enable_texture_streaming` | `False` | `/rtx-transient/resourcemanager/enableTextureStreaming` |
| `texture_streaming_budget` | `0.6` | GB budget when streaming is on |
| `debug_material_type` | `-1` | `/rtx/debugMaterialType`; `-1` = off |

`ingest30_configs.py:_COMMON` ships a validated fast preset —
`rt_subframes=10, path_tracing_spp=96, path_tracing_max_bounces=6` — annotated in source as *"render
speed (validated ~identical to spp256/rt20/bounces12)"*. That is the empirical evidence for turning
quality down.

### III.3.7 Writer / downstream

| Key | Default | Meaning |
|---|---|---|
| `descriptor_config_path` | — | Descriptor backbone YAML; must exist; copied verbatim to `render_dir/descriptor.yaml` |
| `descriptor_device` | — | Device for the one-time reference-descriptor precompute in the writer |
| `proposer_config_path` | — | Proposer YAML; must exist. Not used during render — validated and carried into `runtime.yaml` for phase 2 |
| `proposer_device` | — | Phase-2 device. `cpu` is fine for the grid proposer (no NN) |
| `proposer_devices` | `None` | Tuple of devices; `run_pipeline` shards phase 2 across them |
| `proposer_min_visible_ratio` | `0.30` | Reprojection-coverage gate threshold |
| `proposer_tau_d` / `proposer_tau_r` | `0.001` / `0.005` | `instance_visibility` reprojection tolerances |
| `proposer_min_visible_px` | `60000` | **Deprecated** pixel-area gate (`gate_classes`), kept for back-compat; the pipeline uses `gate_classes_reproj` |
| `proposer_max_occlusion` | `1.0` | Legacy occlusion cap |
| `inlier_border_eps` | — | Phase-3 border margin, px; asserted `>= 0`. Mandatory by design (`inlier-border-eps-margin.md`) |
| `obs_full_alpha` | `False` | `True` → alpha=255 over the whole frame (inspection). Default keeps alpha = instance foreground |
| `dry_run` | — | Skip capture; export the debug bundle instead |

### III.3.8 Dotlist overrides

Anything after the positional config is collected by `parse_known_args` and handed to
`OmegaConf.from_dotlist`. Rules that actually bite:

- `key=value` — no leading dashes, no spaces around `=`.
- Nested keys use dots: `placement_args.max_column_height=5`,
  `scene_builder_args.product_patterns=[model_cereal*]`.
- Lists use YAML-ish brackets and **must be shell-quoted**: `'proposer_devices=[cuda:0,cuda:1]'`,
  `'distant_light_offset=[1.0,-3.0,3.0]'`.
- Booleans are `true`/`false` (OmegaConf parses them); `dry_run=true`.
- `null` clears an optional: `num_targets=null`.
- The dotlist is merged **last**, so it shadows the YAML unconditionally.

### III.3.9 The `${call:…}` resolver

`register_resolvers()` registers one custom OmegaConf resolver, `call`, bound to `_call`, which does
`getattr(sys.modules["isaac_datagen.runtime_config"], name)(*args)` (`runtime_config.py:177-182`). The
only function currently reachable through it is `_glob_amazon_textures()` (`:167-174`), which lists
`<RESOURCE_PATH>/boxes/textures/amazon_texture_*` at config-load time. A typo in the function name
produces an `AttributeError` from deep inside OmegaConf's resolution, which reads as a cryptic load
failure — check the resolver first when `load_config` explodes on a config that looks fine.

---

## III.4 Shipped config catalog

`src/isaac_datagen/configs/` — **31 files**. "mode on CLI?" = yes means the file omits `mode:`, which
is a mandatory no-default field, so the run fails at load without `mode=…`.

### Plain (non-store) scenes

| File | Renders | Poser | `mode` on CLI? |
|---|---|---|---|
| `amazon.yaml` | Amazon-only optflow catalog (`assets/optflow_objects/amazon/`), `UntilExhaustedStacker` | `LookAtPoser` | **yes** — `mode=optflow` |
| `mixed.yaml` | Heterogeneous graspable catalog: amazon + kleenex + ycb, `dataset_dir: datasets/mixed` | `GridFixedPoser` | **yes** |
| `staggered.yaml` | `mixed.yaml` + column y-depth / x-gap stagger | `GridFixedPoser` | **yes** |
| `shelf.yaml` | Multi-source optflow catalogs (kleenex + amazon + ycb) with a class-filter chain and occluders | `LookAtPoser` | **yes** |
| `random3_smoke.yaml` | Small smoke test, random 3-object subset, `datasets/random3-smoke` | `GridFixedPoser` | no (`reference_segmentation`) |
| `tuna_only_smoke.yaml` | Single-class smoke test from `ycb_graspable`, `datasets/tuna-only-smoke` | `GridFixedPoser` | no (`reference_segmentation`) |
| `expanded-refseg.yaml` | Optflow re-render over the amazon optflow catalog, `/data/user/jeffk/datasets/expanded-refseg` | `LookAtPoser` | no (`optflow` baked) |
| `expanded-refseg-v2.yaml` | Curated 10-class `amazon-v2` + per-frame light/exposure jitter | `LookAtPoser` | no |
| `jagged-expanded-refseg-v2.yaml` | Jagged column-height ablation of v2 | `LookAtPoser` | no |
| `jagged2-expanded-refseg-v2.yaml` | Same as jagged, distinct RNG stream (`seed=100`) | `LookAtPoser` | no |
| `blues-expanded-refseg-v2.yaml` | Jagged v2 restricted to the 4 blue-family classes via `RegexFilter` on class | `LookAtPoser` | no |

### Empty-world k-shot pools (`scene_builder: build_scene`)

| File | Renders | `mode` on CLI? |
|---|---|---|
| `emptyworld-optflow-snacks-kshot-snack031-1inst.yaml` | One `snack031` instance from `store001-optflow-objects-keep`, `LookAtPoser` halo, `datasets/snack031-1inst` | no |
| `emptyworld-optflow-snacks-kshot-snack033-1inst.yaml` | Same shape, `snack033` | no |
| `emptyworld-optflow-snacks-kshot-snack034-1inst.yaml` | Same shape, `snack034` | no |
| `emptyworld-optflow-snacks-kshot-snack035-1inst.yaml` | Same shape, `snack035` | no |
| `emptyworld-optflow-snacks-kshot-snack031-1inst-decentered.yaml` | Delta of the snack031 pool: `DecenteredLookAtPoser` + a new `dataset_dir` (`datasets/snack031-1inst-decentered`) | no |

### Store scenes (`scene_builder: build_store_scene`)

| File | Renders | `mode` on CLI? |
|---|---|---|
| `store001-optflow.yaml` | Base in-store render off `datasets/store001-optflow-objects`, `RegexFilter` class subset | no |
| `store001-optflow-keep.yaml` | All 42 curated classes (`filter_specs: []`) off `store001-optflow-objects-keep` | no |
| `store001-optflow-remove.yaml` | Store render exercising the removal mutations | no |
| `store001-optflow-replace.yaml` | Store render exercising `ReplaceClass` swap-ins | no |
| `store001-optflow-verify.yaml` | Per-class front-face verification set: `filter_specs: []`, `num_targets: null`, one fixed pose per target | no |
| `store001-optflow-snacks-set1.yaml` | Snack training set 1 (snack017/020/023/027) | no |
| `store001-optflow-snacks-set2.yaml` | Snack training set 2 (held-out complement) | no |
| `store001-optflow-snacks-set1-plus-snack03{1,3,4,5}.yaml` (4 files) | Set 1 plus one fine-tune class each; `objects_path` points at `../../assets/optflow_objects/store001-optflow-objects-keep` — the in-file comment warns set1's own `datasets/…` value is **stale, do not copy it** | no |
| `store001-optflow-snacks-kshot-snack03{1,3,4,5}-1inst.yaml` (4 files) | In-store single-instance k-shot pools, `RegexFilter` on `meta["class"]` | no |

`tldr.py` documents only the 11 plain configs; all 20 store/emptyworld configs are absent from the
help text — [Appendix A](#appendix-a--known-documentation-staleness).

---

## III.5 Cookbook

Every invocation below is recovered verbatim (or near-verbatim, with paths normalized) from the
as-built plans in `.docs_claude/`. Unless noted, **cwd is `isaac_datagen/src/isaac_datagen/`** and
`dataset_dir` has already been `mkdir -p`'d.

### III.5.1 Smoke tests

```bash
# 1 observation, catches config/path errors before spending a full render.
CUDA_VISIBLE_DEVICES=1 uv run isaac-datagen configs/expanded-refseg-v2.yaml \
    idx=0 num_frames=1 num_targets=1 descriptor_device=cuda:0
```
→ one-frame `render000/`; delete it after verifying. (`expanded-refseg-optflow-regen.md:118`)

```bash
# Config validation with no sim at all (~1 s).
uv run python -c "from isaac_datagen.runtime_config import load_config; \
  c = load_config('configs/expanded-refseg-v2.yaml', []); \
  print(c.pose_generation_policy, c.pose_generation_policy_args)"
```
→ exercises every path assertion and the poser interpolation.
(`pose-generation-poser-registry.md:216`)

```bash
# Import-only smoke for the optflow datastructs.
uv run python -c "import isaac_datagen.objects, isaac_datagen.optflow_writer"
```
(`optflow-4-class-keyed-one-to-many.md`)

```bash
# Scene-construction check, no RTX render.
uv run isaac-datagen configs/mixed.yaml mode=reference_segmentation idx=0 \
    num_frames=1 num_targets=1 dry_run=true
```
→ writes `render000/debug/scene.usdz` + `dryrun.npz`/`dryrun.json`; no frames.
(`collect-objects-chain-datasets.md:101`; the plan used `randomized.yaml`, which no longer exists —
see [Appendix A](#appendix-a--known-documentation-staleness))

### III.5.2 refseg / expanded-refseg datasets

```bash
# One 1000-observation optflow dir (100 frames x 10 targets).
cd isaac_datagen/src/isaac_datagen
isaac-datagen configs/expanded-refseg-v2.yaml idx=0 num_frames=100 num_targets=10
```
→ `/data/user/jeffk/datasets/expanded-refseg-v2/render000` with DIFT descriptors and the grid proposer
baked into `runtime.yaml`. (`expanded-refseg-v2-regen.md:7-8`)

```bash
# Three independent dirs (effective_seed = seed+0, seed+1, seed+2).
for IDX in 0 1 2; do
  isaac-datagen configs/expanded-refseg-v2.yaml idx=$IDX num_frames=100 num_targets=10 \
      descriptor_device=cuda:0
done
```
→ `render000..002`, 1000 obs each, independent placement/pose/light streams.
(`expanded-refseg-v2-regen.md:65-70`)

```bash
# Render + validate + proposals + inliers in one resumable command.
isaac-datagen-pipeline configs/jagged2-expanded-refseg-v2.yaml idx=0
```
(`jagged2-expanded-refseg-v2-gen.md:35`)

```bash
# Multi-GPU phase 2 (phase 1 stays single-process).
isaac-datagen-pipeline configs/expanded-refseg-v2.yaml idx=1 'proposer_devices=[cuda:0,cuda:1]'
```
(`sharded-proposals.md`)

```bash
# Phase 2 / 3 by hand against an existing render dir's own runtime.yaml snapshot.
isaac-datagen-proposals $RD/runtime.yaml dataset_dir=$DS idx=$IDX \
    intrinsics_path=$PWD/zed_K.npy \
    proposer_config_path=$RM/grid_proposal.yaml \
    proposer_device=cpu proposer_min_visible_ratio=0.3
isaac-datagen-inliers $RD --eps 0.0
```
→ 576 grid anchors per gated class into `proposals/`, then inlier labels into `labels/` and
`stats/stats_0000.json`. (`expanded-refseg-optflow-regen.md:153-156`)

```bash
# Post-render: add a second descriptor backbone. NOTE the cwd and the PYTHONPATH scrub.
cd isaac_datagen
env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone add-backbone \
    /data/user/jeffk/datasets/expanded-refseg-v2 \
    ../reference_matching/src/reference_matching/configs/cleandift_finetuned.yaml --device cuda:0
```
(`expanded-refseg-optflow-regen.md:131-133`; `store-snacks-finetune-renders.md:20-24` for the
`env -u PYTHONPATH` and repo-root cwd)

### III.5.3 optflow reference catalogs (Stages A/B before any capture)

```bash
# Stage A: extract store products to GraspableObjects with baked grasp frames.
OMNI_KIT_ACCEPT_EULA=YES uv run python extract_store_objects.py \
    configs/store001-optflow.yaml datasets/store001-objects-cereal \
    'scene_builder_args.product_patterns=[model_cereal*]'

# Stage B: render one isolated centroid-anchored reference RGB-D per object.
OMNI_KIT_ACCEPT_EULA=YES uv run python graspableobj_to_optflow_obj.py \
    configs/store001-optflow.yaml datasets/store001-objects-cereal \
    datasets/store001-optflow-objects
```
(`store-usd-inverse-datagen.md:282-287`)

```bash
# Mesh ingestion (YCB): download, stage candidates, finalize the human's face picks.
uv run src/isaac_datagen/mesh_convert.py ycb      <download_dir>
uv run src/isaac_datagen/mesh_convert.py stage    <input_dir> <stage_dir>
uv run src/isaac_datagen/mesh_convert.py finalize <stage_dir> <output_dir> <winners.yaml>
```
(`mesh-convert-ycb.md`)

### III.5.4 optflow capture datasets

```bash
# Debug optflow capture into a scratch render dir.
uv run isaac-datagen <config.yaml> mode=optflow \
    'objects_path=[datasets/ycb_preoptflow]' \
    num_frames=2 num_targets=1 dataset_dir=datasets/debug idx=950
```
→ `datasets/debug/render950` with `OptFlowSample` + `OptFlowMetadata` (nested `ObsMask`).
(`optflow-2-writer-capture.md`, `optflow-5-cid-iid-masks.md`)

### III.5.5 Store datasets

```bash
# Stage C: capture in-store, 3 random targets x 10 poses.
OMNI_KIT_ACCEPT_EULA=YES uv run isaac-datagen configs/store001-optflow.yaml \
    idx=0 num_targets=3 num_frames=10
```
(`store-usd-inverse-datagen.md:289-290`)

```bash
# Snack training set 1: 1000 frames, full store as background.
mkdir -p datasets/store001-optflow-snacks-set1
isaac-datagen configs/store001-optflow-snacks-set1.yaml idx=0
```
→ `cid_to_class` must come out exactly `{snack017, snack020, snack023, snack027}`; anything else means
untracked products leaked (see [§III.8.7](#iii87-store-scene-traps)).
(`store-snacks-training-datasets.md:137-139`)

```bash
# Per-class front-face verification set: every tracked object once, one fixed 0.6 m pose each.
isaac-datagen configs/store001-optflow-verify.yaml idx=0
```
→ 269 frames with per-category vis sheets. (`store-perclass-faces-verify-dataset.md:578`)

### III.5.6 k-shot / fine-tune pools

```bash
# In-store single-instance pool (200 LookAtPoser views of one snack031 facing).
isaac-datagen configs/store001-optflow-snacks-kshot-snack031-1inst.yaml idx=0

# Empty-world equivalent, with DisablePhysics on the store-extracted usdz.
isaac-datagen configs/emptyworld-optflow-snacks-kshot-snack031-1inst.yaml idx=0
```
(`emptyworld-1inst-regeneration.md:63`; `store-snacks-kshot-surround-renders.md`)

The k-shot pools deliberately use `seed: 1001` (and rehearsal sets `2001`) so their RNG streams cannot
collide with the frozen benchmark's `seed: 1` (`store-snacks-finetune-renders.md:58-66`).

### III.5.7 Unseen eval

```bash
# Frames [0,100) of a source dir -> R/B-flipped, renumbered 0..99, phases 2+3 re-run.
isaac-datagen-unseen configs/expanded-refseg-v2.yaml /path/to/source_render \
    --start 0 --end 100 dataset_dir=/tmp/unseen idx=0

# Or select exactly the frames a frozen split assigned to this render dir.
isaac-datagen-unseen configs/expanded-refseg-v2.yaml /path/to/source_render \
    --split-manifest splits/refseg_split.json --split val --limit 100 \
    dataset_dir=/tmp/unseen idx=0
```
(`channel-swap-unseen-eval-and-callback.md`)

> **Conflict resolved against current source.** `channel-swap-unseen-eval-and-callback.md` shows a
> positional `<start> <end>` form. `make_unseen.py:45-46` declares `--start` / `--end` as **flags**
> (`p.add_argument("--start", type=int)`), so the flag form above is the working invocation today. The
> positional form in the plan is history.

### III.5.8 HPC (PSC Bridges-2, Singularity)

```bash
sbatch render_amazon_l40s.sbatch                                   # the durable path
```

```bash
# Manual container invocation (ICD binds are baked into the .def in the sbatch path).
singularity exec --nv -B $REFSEG_WS:/ws -B $OCEAN:/ocean containers/isaac_datagen.sif \
  bash -c 'cd /ws/isaac_datagen/src/isaac_datagen && uv run isaac-datagen configs/expanded-refseg-v2.yaml idx=0'
```
(`psc-isaac-singularity.md:32-42`; note `bash -c`, **never** `bash -lc` —
[§III.8.10](#iii810-hpc--singularity-psc-bridges-2))

```bash
# In-container sanity checks, cheap, no GPU needed for (1)/(2).
uv sync --locked
uv run --no-sync python -c "import isaac_datagen, vision_core, reference_matching"
uv run python -c "import isaacsim; print(isaacsim.__version__)"      # expects 5.1.0
```
(`.docs_claude/psc-isaac-datagen-footguns.md` §6, `psc-isaac-singularity.md:50`)

---

## III.6 Output layout

Everything lands under `dataset_dir/render{idx:03d}/`. `SerializableSample` writes **one subdirectory
per field**, with zero-padded 4-digit indices; the extension is chosen by the per-type serializer
table in `vision_core/datastructs.py`. The contract-level meaning of these fields is
[§I.4](#i4-the-output-contract).

The listing below is a real optflow render dir (`datasets/snack031-1inst/render000/`). A
`reference_segmentation` render dir is the same minus the optflow-only entries, which are marked.

```text
render000/
├── obs/                        obs_0000.png …           RGBA observation (alpha = instance foreground
│                                                        unless obs_full_alpha=true)
├── iid_mask/                   iid_mask_0000.npy …      int32 instance-id mask (session-local ids)
├── cid_mask/                   cid_mask_0000.npy …      uint8 class-id mask (0=BACKGROUND, 1=UNLABELLED,
│                                                        classes start at 2)
├── iid_to_occlusion/           …_0000.pt                dict[int,float] per-instance occlusion ratio
│                                                        (torch.save — JSON would stringify int keys)
├── observation_depth/          …_0000.npy               [optflow] full unmasked distance_to_image_plane
├── cam2world/                  …_0000.npy               [optflow] 4x4 SE3, OpenCV convention (+Z fwd)
├── iid_to_visibility/          …_0000.pt                [optflow] per-instance reprojection visibility
│
│   ── per-render-dir catalog, written once at index 0 by finalize_metadata() ──
├── cid_to_class/               cid_to_class_0000.pt
├── name_to_class/              name_to_class_0000.pt
├── iid_to_name/                iid_to_name_0000.pt
├── class_to_ref/               class_to_ref_0000.pt     canonical RGBA reference per class
├── class_to_descriptors/       class_to_descriptors_0000.pt + one subfolder PER BACKBONE
│     ├── DiftDescriptor/                                 precomputed reference descriptors
│     └── CleanDiftFinetunedFpn/
├── principal_components/       …_0000.pt                shared PCA->RGB basis (MANDATORY field),
│                                                        also keyed per backbone
├── obs_intrinsics/             …_0000.npy               [optflow] observation K
├── class_to_name/              …                        [optflow]
├── class_to_reference/         …                        [optflow] RGB refs — DEPRECATED, see below
├── class_to_reference_depth/   …                        [optflow]
├── class_to_ref_intrinsics/    …                        [optflow]
├── class_to_ref_pose/          …                        [optflow] reference camera pose, OpenCV
├── class_to_l2w/               …                        [optflow] (N,4,4) instance placements per class
│
│   ── phase 2 / 3 ──
├── proposals/                  proposals_0000.pt        dict[class -> (N,2) xy]
├── labels/                     labels_0000.pt           dict[class -> (N,) bool inlier]
├── stats/                      stats_0000.json          {n_inliers, n_total, eps}
│
│   ── provenance / logs ──
├── runtime.yaml                asdict(RuntimeConfig) dump — the completion marker
├── descriptor.yaml             verbatim copy of descriptor_config_path
├── lighting_log.json           per-frame dome/distant schedule (log_lighting=true)
├── cid_iid_trace.log           append-only trace (only present if anything called cid_iid_trace.log())
└── debug/                      dry_run only: scene.usdz, dryrun.npz, dryrun.json, poses/*.png, orbit.gif
```

Notes that matter operationally:

- **`runtime.yaml` is the completion marker.** It is written *after* `finalize_metadata()`
  (`clean_datagen.py:90-95`, `:135-140`). A render dir with `obs/` but no `runtime.yaml` is an aborted
  render, not a dataset.
- **Every `dict`-typed field serializes as `.pt`**, including `cid_to_class/` and `name_to_class/` —
  `torch.save`, not JSON, because JSON stringifies integer dict keys and these keys index masks
  (`cid-mask-dual.md`).
- **`class_to_reference` (RGB) is deprecated** in favour of `obsmaskmeta.class_to_ref` (RGBA); new
  code should read `md.obsmaskmeta.class_to_ref[cls][:3]`
  (`plans/active/class-to-reference-rgba-dedup.md` — the field is still written, and both are present
  in live dirs).
- `principal_components/` is a mandatory field of `ObsMaskDescriptorMetadata`
  (`vision_core/datastructs.py:383-389`). Legacy dirs without it fail deserialization; run
  `migrate_pca_basis.py` (`pca-basis-mandatory-field.md`).
- `obs_*` (capture-frame index) and `reference_image_*` (catalog index) live in **different index
  spaces**. There is no per-frame → catalog-object tie in `OptFlowWriter`
  (`store-perclass-faces-verify-dataset.md:424-450`).
- Atomic writes can leave stray `.tmp` dotfiles after a `SIGKILL`. They are harmless and never inflate
  frame counts, because every count globs exact final names (`obs_*.png`, `proposals_*.pt`) —
  `add-proposals-resumable.md`.

---

## III.7 Verification and debugging

Ordered roughly by cost.

### III.7.1 Before booting Isaac (seconds, free)

| Check | Command |
|---|---|
| Config parses, every path exists, poser args interpolate | `uv run python -c "from isaac_datagen.runtime_config import load_config; print(load_config('configs/X.yaml', []))"` |
| Sibling imports resolve (catches detached-HEAD siblings) | `uv run --no-sync python -c "import isaac_datagen, vision_core, reference_matching"` |
| Filter chain does what you think | construct `FilterSpec`s and call `filter_objects` in a unit test — no sim required (`graspable-object-filter-registry.md`) |
| Asset catalog integrity | per object dataset, `ls meta \| wc -l` must equal `ls usd_path \| wc -l` (`.docs_claude/psc-isaac-datagen-footguns.md` §3) |

### III.7.2 `dry_run` and the Blender sanity render

```bash
isaac-datagen configs/X.yaml idx=0 dry_run=true          # exports the bundle, no frames
isaac-datagen-debug-render configs/X.yaml idx=0          # the above, then Blender renders it
```

`dry_run=true` takes the branch in both entry points that calls
`export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)` and then `app.close()`
(`clean_datagen.py:78-82`, `:122-126`) — no writer, no capture, no metadata. `decorate_debug_scene`
bakes the planned left-camera prims plus RGB axis gizmos into the live stage; `export_debug_bundle`
writes `render_dir/debug/scene.usdz` + `dryrun.npz`/`dryrun.json` (`debug_export.py:51-52`).

Two properties make this trustworthy: the real capture path **never imports `debug_export`**, so
dry-run decorations cannot drift into production; and both paths share the same mechanisms
(`plan_capture`, `se3_to_pos_euler`, `set_prim_pose`, `ZedMini` intrinsics), so what you see in
Blender is what the camera will do (`blender-dry-run-sanity-renderer.md:47-48, 73-94`; the
architectural statement of this is [§I.5.2](#i52-separation-of-placement--pose-policy-from-mechanism)).

Known dry-run-only fixup: Blender's USDZ importer **skips untyped prims**, which severs the transform
chain and collapses every object to the origin. `_retype_untyped_for_blender(stage)` types every
untyped prim as `Xform` before export. Semantically neutral, dry-run only
(`blender-dry-run-sanity-renderer.md:113-120`).

### III.7.3 `validate_obsmask` — the cid/iid orphan gate

```bash
isaac-datagen-validate-obsmask <render_dir>     # exit 1 + one row per orphan
```

An *orphan* is an instance present in `iid_mask` and in `iid_to_occlusion` whose pixels are all
`cid < 2` in `cid_mask` — i.e. the instance rendered but its class never made it into the LUT. The
validator loads only three fields (`iid_mask`, `cid_mask`, `iid_to_occlusion`) and deliberately uses
**per-frame** graspable iids from `iid_to_occlusion.keys()`, because iids are session-local — the same
integer on frame 7 and frame 50 need not be the same object (`obsmask-cid-iid-validator.md`).

`run_pipeline` runs this automatically after phase 1, prints the first 20 orphans, and exits before
spending proposer time (`run_pipeline.py:93-102`).

Known root cause worth memorizing: **Isaac's `instance_segmentation_fast` tokenizes class semantics on
whitespace.** A class named `fish can` arrives in `idToSemantics` as `fish`, the LUT lookup misses,
and every instance of it becomes an orphan (`tuna-fish-can-cid-orphan-root-cause.md`). Use
single-token class names. The mask machinery is [§II.8](#ii8-writers).

### III.7.4 `cid_iid_trace`

`cid_iid_trace.init(render_dir)` is called unconditionally by both entry points
(`clean_datagen.py:63-64`, `:106-107`), setting the module-global path to
`render_dir/cid_iid_trace.log`. `log(msg)` then appends; if `init` never ran it is a no-op.
`is_tuna(name, cls)` is a leftover predicate from the fish-can investigation
(`cid == "fish can" or "tuna" in name.lower()`). The log file only appears if something in the writer
path actually called `log()` — an absent file is normal.

### III.7.5 Luminance / dark-frame audit

```bash
isaac-datagen-measure-luminance <render_dir> --pixel-threshold 8 --frame-threshold 0.5 \
    --csv /tmp/lum.csv --with-lighting
```

Per-frame foreground (alpha>0) BT.709 luminance; `--metric dark_frac|fg_mean`; `--with-lighting` joins
to `lighting_log.json` so you can tell "the light schedule was dim" from "the renderer produced
nothing". `run_pipeline` calls it (without flags) before its interactive gate. This is the instrument
for both lighting bugs in [§III.8.2](#iii82-the-all-black-render-coin-flip-unsolved) and
[§III.8.3](#iii83-exposure-the-other-dark-render-bug).

### III.7.6 Visualization tools

| Tool | Reads | Shows |
|---|---|---|
| `isaac-datagen-viz-gate <roots…>` | render dirs or dataset roots | per-frame proposer gate decision (pass/drop per class) |
| `isaac-datagen-viz-sample <render_dir>` | `PreImageInlierSample` | per-class refs + all proposals + gt masks |
| `isaac-datagen-viz-inliers <render_dir>` | `PreImageInlierSample` | inlier/outlier labels over the obs |
| `isaac-datagen-viz-occlusion <dataset_dir>` | `ObsMask` | random-sample occlusion-ratio spot check |
| `isaac-datagen-sweep-label-eps <render_dir> --eps 0 1 2 3 5 8` | one frame | how `inlier_border_eps` moves the label boundary |
| `debug_scripts/viz_optflow.py <render_dir>` | `OptFlowSample` + `OptFlowMetadata` | 1-to-many correspondence fan-out panel; needs `uv run --extra viz` (lazy `romatch`) |
| `debug_scripts/viz_optflow_objects.py <catalog>` | `OptFlowObject` | reference framing + mesh/pose round-trip; needs `uv run --with usd-core` for the mesh panel (`pxr` is absent from the plain venv) |
| `debug_scripts/debug_scene.py <config.yaml> [k=v]` | live sim | grasp-point world coords, per-choice USDZ exports, `grasp_debug.txt` |
| `debug_scripts/debug_occupancy.py <config.yaml>` | pure numpy | legacy `OccupancyGrid` reconstruction / full-wall contract |
| `debug_scripts/check_front_face.py <config.yaml> <out> --all-faces [k=v]` | live store | 4-face renders per SKU + `front_face_luma.csv` |

> **Resolved, no longer a defect.** `OptFlowSample.visualize()` was originally declared with a
> keyword-only parameter named `class` (a Python reserved word) and would not import. The signature in
> `vision_core/datastructs.py:418` is now
> `visualize(self, md, *, cls_name=None, points=None, n_points=12, rel=0.05, title=None)`, so current
> source is fine — `optflow-4-class-keyed-one-to-many.md`'s gotcha is history.

---

## III.8 Footguns and operational notes

Blunt, in the order they will cost you the most.

### III.8.1 The render is not resumable, and half a render is worthless

Phase 1 accumulates writer state in memory and calls `finalize_metadata()` **only at the very end**. A
`TIMEOUT`, OOM, or `Ctrl-C` mid-capture leaves `obs/` and the masks on disk with **no `runtime.yaml`
and no catalog** → the dir cannot be deserialized and cannot be resumed. You must delete the render
dir and re-run from scratch with more walltime (`.docs_claude/psc-isaac-datagen-footguns.md` §5).

Corollary: `run_pipeline` skips phase 1 whenever `obs/` has ≥1 frame (`run_pipeline.py:83-91`). It
will happily run proposals against a *truncated* render. Verify `runtime.yaml` exists before trusting
a skip.

Corollary 2: re-rendering *under* an existing `proposals/` silently desyncs them. Delete the whole
`render{idx:03d}/`, never individual `obs/` files.

### III.8.2 The all-black-render coin flip (unsolved)

Roughly **60 % of processes render the entire scene pure black for every frame**. The outcome is
per-process all-or-nothing, decided at renderer init *before frame 0*, and immutable for the process
lifetime. It is independent of exposure, materials, light type, `rt_subframes`, and `multi_gpu`. An
HDR probe shows `dome_I=1000` (correct) with HDR RGB≈0 — both the dome (IBL) and the distant
(analytic) light contribute ~0 radiance, so it is a path-tracer light-init race, not a light-type
problem.

Things that were tried and **did not** work: material warmup via `orchestrator.step` (crashed — offset
poses), `app.update` warmup (no-op headless),
`resetPtAccumOnlyWhenExternalFrameCounterChanges`, un-ablating the distant light, `multi_gpu=False`
(`plans/active/render-darkness-investigation.md:108-129`).

**Operational answer: detect and retry.** Run `isaac-datagen-measure-luminance` on the fresh dir
(`run_pipeline` does this for you before its y/N gate, `run_pipeline.py:29-45`); if it is black, delete
and re-run. Expect ~2.5 attempts per good render. This is unsettled, not fixed.

### III.8.3 Exposure: the *other* dark-render bug

Distinct from [§III.8.2](#iii82-the-all-black-render-coin-flip-unsolved) and **fully fixed**: Kit's
stale carb default `exposureTime=0.02` (a daylight shutter) pushes a dome-lit scene into the ACES toe
and crushes uint8 to 0 — near-binary, no midtone. `set_exposure=True` + `exposure_time=1.0` clears it
(`fg_mean` goes 0.0 → ~178) (`plans/active/render-darkness-investigation.md:40-62`). Both are now the
schema defaults (`runtime_config.py:106-107`). If you override `set_exposure=false`, you are
re-opting into the bug. Check the `[TONEMAP]` boot line to confirm.

### III.8.4 Per-frame lighting jitter is partly broken

- `rep.randomizer.register(...)` wrapping modify nodes **never executes** — the registered route
  produces zero layer opinions on any channel, silently. The working mechanism is the
  `capture_session` `per_frame(i)` callback doing direct USD writes under
  `Usd.EditContext(stage, stage.GetRootLayer())` (`lighting-jitter-mechanism.md:10-41, 55-80`;
  mechanism in [§II.6](#ii6-lighting-and-domain-randomization)).
- `rep.modify.attribute('intensity', rep.distribution.sequence(vals))` **does not advance per
  `orchestrator.step`** the way `rep.modify.pose` does. Every frame is stuck on the first element.
  Constant dome (`lo == hi`) is reliable; per-frame dome jitter via that route is not
  (`lighting-diagnostic-dark-box-flags.md:266-280`). Verify with `--with-lighting` that logged
  intensities actually correspond to measured luminance.
- Sequence lengths must be `len(world_poses)` = `num_targets × num_frames`, **not** `num_frames`.
  `plan_capture` flattens `(B,N,4,4) → (B*N,4,4)` and the orchestrator steps once per element; a wrong
  length silently skews the frame↔light join (`lighting-diagnostic-dark-box-flags.md:54-57`).
- Magnitude: ±0.75 m of distant-light wobble is only ~±10 % luminance, because parallel rays have no
  falloff. Shipping configs use `distant_offset_jitter: 2.0` (±30° cone) and
  `distant_intensity_jitter: [500, 4000]` (8×) to be visible at all
  (`lighting-jitter-mechanism.md:91-114`).

### III.8.5 Config traps

- **`amazon.yaml`, `mixed.yaml`, `shelf.yaml`, `staggered.yaml` omit `mode:`.** It is a mandatory
  no-default field. Pass `mode=optflow` (amazon/shelf — their `objects_path` points at
  `optflow_objects/`) or `mode=reference_segmentation` (mixed/staggered — `graspable_objects/`).
- **`dataset_dir` must pre-exist.** `mkdir -p` it. A script that creates it *after* calling
  `reference_segmentation()` fails validation before it ever gets there.
- **Exactly one of `num_frames` / `grid_dims`.** Both or neither → assertion at load.
- **`objects_path` must be non-empty.** The old split `graspable_objects_path` /
  `optflow_objects_path` fields allowed a silent no-objects render; the unified field closed that
  (`unify-objects-path.md`).
- **Duplicate `meta["name"]` across catalogs raises `ValueError`.** Names key USD prim paths and the
  `name_to_class` map; a collision would corrupt both.
- **A missing required placer arg raises `TypeError` at scene-build time** — after boot, ~30 s in.
  Only the column-height parameter is required; the jitter parameters default
  (`placers.py:72-73`).
- **`inlier_border_eps` has no default** and is asserted `>= 0`. Fail-loud by design.
- **`scene_builder` is silently ignored in `mode: reference_segmentation`.** It is still validated
  non-empty. See [§II.4](#ii4-scene-construction).
- **`seed` + `idx` fully determine the run.** Repeating a `(seed, idx)` pair reproduces geometry, poses
  and jitter exactly. For independent samples vary `idx` (or `seed`), not both blindly.

### III.8.6 Phase-2/3 semantics

- **`start_frame`/`end_frame` is a sharding tool only.** Phase 3 labels *all* frames unconditionally.
  A partial-window phase-2 run leaves phase 3 crashing on missing proposals (`sharded-proposals.md`,
  known caveat).
- **Phase 3 has no resume and always rewrites.** Re-running with a different `--eps` relabels
  everything; no cleanup needed.
- **`ref_cache` is per-render-dir.** Do not carry one across dirs — the instances differ.
- Both GPU shards' tqdm bars interleave on one tty. Cosmetic, accepted (`sharded-proposals.md`).
- `run_pipeline` is **not mode-aware**: it always does render → validate → proposals → inliers. That
  is fine for optflow because the optflow render dir nests an `ObsMask`. Its interactive y/N gate
  auto-skips when stdin is not a tty, so batch jobs are unaffected
  (`.docs_claude/psc-isaac-datagen-footguns.md` §5).

### III.8.7 Store-scene traps

- **A store config with no `mutations` block leaves every non-target product physically present but
  unlabeled** → the other set's held-out classes leak into the background as unlabeled contamination.
  Add `mutations: [{name: RemoveUntrackedProducts}]`, and set `require_tracked_only` so
  `build_store_scene` asserts (`store-snacks-training-datasets.md:18-27`).
- **Store-extracted usdz carry `PhysicsRigidBodyAPI` + `CollisionAPI` from the live shelf.** In a plain
  scene they free-fall during capture (36/50 frames, −3.14 m drift in the stored `l2w`) unless a
  `DisablePhysics` mutation runs *before* the grasp-frame pose read
  (`plain-scene-mutations-disable-physics.md:14-19`).
- **Vendor semantics fight ours.** Order is critical: override → `remove_labels(include_descendants)`
  → add. Get it wrong and `cid_mask` comes out all zeros. `RemoveProperty` cannot delete opinions from
  a referenced arc; a root-layer `EditContext` override can
  (`store-usd-inverse-datagen.md:195-217`).
- **Front face is per-SKU, not global.** The global `-Y` assumption is correct for only 29/63 = 46 % of
  products; 20/63 are ambiguous (top-2 luminance margin < 15) and need a human eyeball. Darkness is
  shelf occlusion, not dim lighting — confirmed by an 8× lighting control that left the distribution
  unchanged (`store-front-face-check.md:56-99`).
- **`model_drink101` has `v_69323`, not `v_0`.** `extract_store_objects.extract_one` hardcodes
  `{model_path}/v_0` and will skip or crash on it (`store-front-face-check.md:109-111`). Open.
- **Arm-B is unresolved:** scrubbed usdz catalogs (after `pull_flatten_usd.py` removes vendor
  semantics) lose class labels at capture (32 orphans) while unscrubbed and YCB swap-ins are clean.
  Suspects: `PhysicsRigidBodyAPI`, instanceable flags, annotator prim-ancestry union. Needs an in-kit
  probe (`store-scene-mutations.md:152-193`). Open.
- 12 store-authored label-inward facings (`snack012_3..10`, `snack032_4..7`) and 2 coincident-duplicate
  prims (`detergent001_3`, `snack025_2`) are physically unphotographable or degenerate; deactivate them
  with `RemovePrims` (`store-perclass-faces-verify-dataset.md:467-491, 605-637`).

### III.8.8 Occluders

- `primvars:hideForCamera=True` under PathTracing is **unconfirmed**. Fallback if occluders show up:
  place them out of frustum (`per-target-shadow-occluders.md:53-56`).
- **Penetrating geometry renders as opaque black blobs** (z-fighting, not a `hideForCamera` failure) —
  proved by a controlled test varying only the x-offset (`per-target-shadow-occluders.md:8-32`). Leave
  an air gap.
- Occluder placement and scale draw from the **global** `np.random` stream (`scene.py:377`, `:383`,
  `:387`), which `seed_everything` has seeded — so they are reproducible for a fixed `(seed, idx)`, but
  they have no decorrelated substream and shift whenever an upstream global-stream consumer changes.
  The plan record's phrase "unseeded" means *not decorrelated*
  (`obs-full-alpha-toggle.md:60-67`); threading a dedicated seeded RNG into the posers is still the
  proposed fix. Open.

### III.8.9 Cameras, conventions, and stereo

- `ZedMini.left2rig` is **translation-only** (identity rotation). Rotational stereo extrinsics, if any,
  must come from calibration, not this property (`hardwares.py:52-55`).
- `cam2world` is stored in **OpenCV** convention, derived from
  `camera_params_to_world2cam(rp['camera_params'])`, never hand-inverted from the OpenGL/USD
  `world_poses`. Hand inversion without the GL2CV matrix produces mirrored/scattered warps
  (`optflow-2-writer-capture.md`).
- `OptFlowWriter.attach()` keeps only the **left** render product. ZED is stereo; the right RP is
  dropped.
- The stored `cam2world` is the **left eye**, 31.5 mm (baseline/2) ahead of the rig pose. Any "expected
  camera pose" check must account for that offset
  (`store-perclass-faces-verify-dataset.md:451-454`).
- `OptFlowSample.observation` reads back **RGBA** (`tv_tensors.Image` forces `RGB_ALPHA`); downstream
  must slice `[:3]`.

### III.8.10 HPC / Singularity (PSC Bridges-2)

All of this is from `.docs_claude/psc-isaac-datagen-footguns.md` and
`plans/completed/psc-isaac-singularity.md`.

| Symptom | Root cause | Fix |
|---|---|---|
| `VkResult: ERROR_INCOMPATIBLE_DRIVER`, `Vulkan 1.1 is not supported`, `GPU Foundation is not initialized` | `--nv` injects CUDA libs but **not** the NVIDIA Vulkan/EGL ICD JSON (apptainer#2210, nvidia-container-toolkit#16/#1392) | Bind `/usr/share/vulkan/icd.d/nvidia_icd.json` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, set `VK_ICD_FILENAMES`. Durable: bake both (with relative `library_path`) into the `.def` |
| `Skipping unsupported non-RTX GPU` / `No device could be created` | Isaac Sim 5.1 requires hardware RT cores; A100/H100/V100 have none | **L40S only** on Bridges-2: `--gpus=l40s-48:1`. Do not burn allocation on other GPUs even when idle |
| `nvidia-smi` fine but rendering dead | CUDA and Vulkan fail independently | `vulkaninfo --summary` under `--nv`. `--/rtx/verifyDriverVersion/enabled=false` fixes neither the missing-ICD nor the non-RTX case (isaac-sim#357) |
| In-container `uv run` dies on `failed to remove .venv/lib: Permission denied` | Host-side project-mode `uv run`/`uv sync` "repaired" the container-owned venv for the host ABI (host glibc 2.28 / uv 0.11.21 / py 3.11.9 vs container glibc 2.35 / uv 0.11.23 / py 3.11.0rc1) | Never plain `uv run`/`uv sync` for this project on the host — use `uv run --no-project --with <pkg>`, a PEP-723 `uv run --script`, or `cd /tmp`. Recover with in-container `rm -rf .venv && uv sync --locked` (the Lustre `rm` alone took ~6 min) |
| Host `uv` shadows container `uv`; `module`/`conda`/`fzf` errors | `bash -lc` sources the bind-mounted host `~/.bash_profile` and prepends `~/.local/bin` | **`bash -c`, never `bash -lc`**, inside `singularity exec` |
| `uv sync --locked`: "lockfile needs to be updated" | Sibling metadata changed (e.g. a `vision_core` rename) silently staled `uv.lock` | `uv lock` **in-container** and commit. Never `uv lock` on the native host — glibc 2.28 cannot resolve `manylinux_2_35` Isaac wheels |
| GPU job billed for a multi-GB install | `uv sync` ran inside the GPU job | Pre-warm `.venv` on a CPU node (the container runs there too); the GPU job's `uv sync --locked` then no-ops |
| `ImportError` on a symbol/config that exists in git | An editable sibling is on a stale/detached HEAD; the venv and lock are fine, the *source* is wrong | Preflight `git switch master && git pull` on each sibling |
| `OptFlowWriter: write() called with no labeled instances` spam, frames lost | Partial HuggingFace `snapshot_download` silently dropped usdz files (2/44 to an xet hiccup) | `aspull` (= `art pull asset`) re-pulls missing files. Verify: per-object `meta` count == `usd_path` count |
| Job reports `COMPLETED 0:0` with zero output | A failed `singularity exec`/`uv run` did not propagate its exit code | `set -e`, or `rc=$?; …; exit $rc` at the end of the sbatch |
| Trivial job rejected | Allocation `cis260205p` is GPU-only, no CPU RM | Even a logging job needs `--gpus=…:1`. Whole-node `GPU` needs a multiple of 8; `GPU-shared`/`GPU-small` take 1..4, billed per GPU, 24 cores auto-assigned per L40S GPU |
| `squeue --start` shows a multi-day ETA | Pessimistic backfill estimate; L40S is scarce (3 nodes) | Keep walltime tight (30-60 min). Short jobs backfill ahead of 4 h/2-day jobs and land in minutes. Budget ~2 s/obs on L40S; 900 obs ≈ 30-35 min + boot |
| First render of a session very slow | RTX shader compile (~3 min) | Later boots on the same node reuse the cache via mounted `$HOME` and start in ~15 s |

Benign boot noise — **do not chase**: `GLFW initialization failed` / `failed to open default display`
(headless), `carb.audio … eDeviceLost` (no sound card), `NGX isn't enabled` (no DLSS), USD
`OrthogonalizeBasis did not converge`, diffusers `safety_checker=None`, `accelerate was not found`,
and the DIFT model load's stderr, which `omni.kit` mis-tags `[Error]`.

Two more operational habits from the same doc: **Kit eats unflushed stdout on `fastShutdown`** —
always `print(..., flush=True)` in-kit, or redirect the whole run to a file before grepping (SIGPIPE
risk). And **the interactive shell's cwd resets between tool calls** — use absolute paths or `cd` into
`src/isaac_datagen` in every command.

### III.8.11 Miscellaneous sharp edges

- **`make_unseen` must be able to reach the repo root.** It shells out to
  `migrate_descriptors_backbone` with `cwd=REPO_ROOT` because the descriptor config's relative paths
  are anchored there, while phases 2/3 stay in `src/isaac_datagen` (`make_unseen.py:15`, `:87-89`).
- **`migrate_descriptors_backbone` wants `env -u PYTHONPATH`** and repo-root cwd, to avoid import
  shadowing (`store-snacks-finetune-renders.md:20-24`).
- **The USD layer registry caches opened layers in-process.** After overwriting a `.usdz` in place,
  re-opening it in the same process returns the stale layer. Verify usdz edits in a **fresh process**
  (`rotate-graspable-meshes-z.md`).
- **`ComputeLocalBound` bakes in the prim's local-to-parent transform.** Bbox measurement for grasp
  frames and placers must use `ComputeUntransformedBound`, and must happen at placer `__init__` time,
  before `organize_objects` moves anything (`until-exhausted-stacker.md:11-50`).
- **`coords_in_mask` does not treat the image frame as a border.** Points on frame-truncated objects
  still count as interior; `distanceTransform` only sees in-image zeros. Intentional
  (`inlier-border-eps-margin.md`).
- **Occlusion excludes frame-edge truncation.** `iid_to_occlusion` covers object–object and self
  occlusion only, so filtering on it will not catch an object merely cut off by the frame
  (`obsmask-occlusion-and-viz.md`).
- **`cid` numbering is `sorted(classes).index(cls) + 2`.** It is stable only for a fixed class set;
  adding or renaming a class shifts the cid space and requires a LUT remap of existing `cid_mask` data
  (`optflow-5-cid-iid-masks.md`).
- **The canonical reference per class is the first member by sorted `meta["name"]`.** If two instances
  of a class carry different `reference_image`s, the sorted-first one is used for all of them
  (`optflow-4-class-keyed-one-to-many.md`).
- **`num_targets` sampling uses `np.random.choice` with replacement** (`capture.py:27-28`), so the same
  grasp target can be drawn twice in one render.
- **`GraspableObject.grasp_point` does not constrain scene placement** in the reference-seg pipeline —
  live grasp frames come from placement. It defines the reference viewpoint for mesh_convert / optflow
  (`mesh-convert-ycb.md`).
- **A 1-frame smoke test does not exercise `finalize_metadata()` timing or the empty-frame path.** It
  catches config and path errors cheaply; it does not validate walltime
  (`.docs_claude/psc-isaac-datagen-footguns.md` §6).
- **`obs_full_alpha=true` makes every alpha-as-foreground consumer treat the whole frame as
  foreground.** Use it for inspection only; read `iid_mask`/`cid_mask` for true foreground
  (`obs-full-alpha-toggle.md:27-30`).
- **Nesting `ObsMask` inside `OptFlowSample` broke backward compatibility** with pre-`optflow-6` render
  dirs (old flat `observation`/`cid_mask`/`iid_mask` layout). Those dirs must be regenerated
  (`optflow-6-nested-obsmask.md`).
- **`optflow_render` / `graspableobj_to_optflow_obj` load a full `RuntimeConfig`** and therefore
  validate `dataset_dir`, `proposer_config_path` and `descriptor_config_path` even though they use
  none of them. Acknowledged wart (`optflow-1-reference-dataset.md`).
- **`look_at` is singular on ±Z faces** (camera normal parallel to world up). Reference rendering
  supports side faces only; the same limitation applies to a near-vertical `distant_light_offset`.

---

# Appendix A — Known documentation staleness

Collected in one place so it is not re-discovered three times. None of these are code defects; they
are docs and help text that current source has outrun.

| Where | What it says | Current source |
|---|---|---|
| `isaac_datagen/CLAUDE.md:64`, `:67` ("Quick start") | `uv run clean_datagen.py <config.yaml>`, with the example `src/isaac_datagen/configs/randomized.yaml` | The entry point is the console script `isaac-datagen` (`pyproject.toml [project.scripts]`), and **`configs/randomized.yaml` does not exist** — the 31 shipped configs are listed in [§III.4](#iii4-shipped-config-catalog) |
| Most completed plans under `plans/completed/` | verification commands using `randomized.yaml` | Same — read those command blocks as historical; substitute `mixed.yaml` or `expanded-refseg-v2.yaml` |
| `tldr.py:8` (`_PHASES`) | `isaac-datagen-proposals <render_dir> [key=value ...]` | `add_proposals.py:16-18` takes a **config**, and resolves the render dir from `dataset_dir` + `idx` itself |
| `tldr.py` (`_CONFIGS`, lines 44-56) | documents 11 configs | 31 configs ship; all 20 store/emptyworld configs are absent from the help text |
| `pre-capture-render-warmup.md` | `warmup_frames` default 16 (8–24 typical) | `runtime_config.py:113` declares `warmup_frames: int = 32` |
| `object-placer-registry.md` | "no defaults in placer constructors" | `placers.py:72-73`, `:111-113` default `min_y`, `max_y`, `min_gap`, `max_gap`, `epsilon`; only the column-height parameter is required |
| `channel-swap-unseen-eval-and-callback.md` | `isaac-datagen-unseen <config> <src> <start> <end>` (positional) | `make_unseen.py:45-46` declares `--start` / `--end` as flags, plus a `--split-manifest` form |
| `optflow-4-class-keyed-one-to-many.md` | `OptFlowSample.visualize()` has a `class=` keyword that breaks import | Renamed; `vision_core/datastructs.py:418` is `visualize(self, md, *, cls_name=None, …)` |
| `optflow-6-nested-obsmask.md` | `full_alpha` "hardcoded `False`" in the optflow writer | Both entry points pass `full_alpha=runtime.obs_full_alpha` (`clean_datagen.py:85`, `:130`); the plan meant the intended production value |
| `obs-full-alpha-toggle.md` | occluder placement is "unseeded" | It draws from the *globally seeded* `np.random` (`scene.py:377-387` after `seed_everything`); what is missing is a decorrelated substream |

---

# Appendix B — Open questions and unverified claims

One deduplicated list. "Status" is as of 2026-07-20.

| # | Item | Status | Where it bites | Source |
|---|---|---|---|---|
| 1 | **Intermittent all-black renders.** ~60 % of processes render every frame pure black; per-process all-or-nothing, decided at renderer init before frame 0, immutable for the process lifetime. Independent of exposure, materials, light type, `rt_subframes`, `multi_gpu`. HDR probe: correct dome intensity, ~0 radiance from both dome and distant. Warmup, PT-accum resets and material warmup all failed. | **Open, unfixed.** Only mitigation is detect-and-retry (~2.5 attempts per good render) | Every phase-1 render; wasted GPU hours; `run_pipeline` gates on the luminance summary before spending phase-2 time | `plans/active/render-darkness-investigation.md:108-129`; `run_pipeline.py:29-45`; [§III.8.2](#iii82-the-all-black-render-coin-flip-unsolved) |
| 2 | **Replicator dome-attribute jitter is non-functional.** `rep.modify.attribute("intensity", rep.distribution.sequence(...))` does not advance per `orchestrator.step()` the way `rep.modify.pose` does; every frame stays on the first element. `rep.randomizer.register(fn)` produces zero layer opinions on any channel. | **Open upstream; worked around.** The direct-USD `per_frame` callback is the supported mechanism | Any config setting `jitter_dome` and expecting per-frame variation; any new randomizer written the "obvious" Replicator way | `lighting-jitter-mechanism.md:10-41, 55-80`; `plans/active/lighting-diagnostic-dark-box-flags.md:266-280`; [§II.6](#ii6-lighting-and-domain-randomization) |
| 3 | **`reference_segmentation()` ignores `scene_builder`.** `clean_datagen.py:74` calls `build_scene` directly; `:117` uses the registry. `RuntimeConfig` validates the field non-empty for both modes (`runtime_config.py:124`). | **Open; undocumented, unclear if intentional.** Store scenes are optflow-only in current source | A refseg config naming a store builder passes validation and silently renders a plain scene | `clean_datagen.py:74` vs `:117`; [§I.2](#i2-the-two-generation-modes), [§II.4](#ii4-scene-construction) |
| 4 | **`OptFlowMetadata.class_to_reference` (RGB) duplicates `obsmaskmeta.class_to_ref` (RGBA).** Both are still written today — confirmed in live render dirs. | **Mid-deprecation** (plan is *active*) | Duplicate storage; consumers reading the wrong one get RGB where RGBA was meant | `plans/active/class-to-reference-rgba-dedup.md`; [§II.8](#ii8-writers), [§III.6](#iii6-output-layout) |
| 5 | **Occluder placement has no decorrelated RNG substream.** Placement and scale draw from the seeded *global* `np.random`, not `default_rng([effective_seed, k])`. | **Open.** Reproducible per `(seed, idx)` but order-coupled; proposed fix (thread a seeded RNG into the posers) not implemented | Occluder layout shifts if any upstream global-stream consumer changes; high cross-config variance (peripheral vs uselessly centred) | `scene.py:377`, `:383`, `:387`; `obs-full-alpha-toggle.md:60-67`; [§I.5.4](#i54-provenance-dumps-and-seeded-reproducibility) |
| 6 | **`primvars:hideForCamera=True` unconfirmed under path tracing.** Never verified that Replicator's asset cache honours it in PT mode. | **Unverified** | Occluders may be visible in the render rather than shadow-only; fallback is out-of-frustum placement | `per-target-shadow-occluders.md:53-56`; [§II.6](#ii6-lighting-and-domain-randomization) |
| 7 | **Penetrating occluders render as opaque black blobs.** Cause reproduced (controlled A/B varying only x-offset); it is geometry interpenetration, not a `hideForCamera` failure. | **Open, no code fix.** Remedy is placement discipline (air gap) | Any config with `occluders_per_target > 0` and a tight pose policy | `per-target-shadow-occluders.md:8-32`; [§III.8.8](#iii88-occluders) |
| 8 | **Arm-B: scrubbed store `.usdz` catalogs lose class labels at capture** (32 orphans) while unscrubbed catalogs and YCB swap-ins are clean. Suspects: `PhysicsRigidBodyAPI`, `instanceable` flags, annotator prim-ancestry union. | **Open; no in-kit probe run.** Prefer unscrubbed catalogs | Any catalog produced through `debug_scripts/pull_flatten_usd.py` | `store-scene-mutations.md:152-193`; [§II.9](#ii9-store-usd-scenes) |
| 9 | **Phase 1 is not resumable.** Writer state lives in memory; metadata is emitted only at `finalize_metadata`. No checkpoint-resume design has been attempted. | **Open by design-debt** | A timeout/OOM/Ctrl-C mid-capture produces an unusable dir that must be deleted and re-rendered | `clean_datagen.py:90`, `:135`; `run_pipeline.py:83-91`; [§III.8.1](#iii81-the-render-is-not-resumable-and-half-a-render-is-worthless) |
| 10 | **Frame windows are a phase-2-only concept.** `start_frame`/`end_frame` shard the proposer; phase 3 always labels *all* frames. | **Known caveat, unchanged** | A partial-window phase-2 run leaves phase 3 failing on missing proposals | `sharded-proposals.md`; [§III.8.6](#iii86-phase-23-semantics) |
| 11 | **`grid_dims` mode through `plan_capture` passes `runtime.num_frames` (which is `None`) to the poser.** Only `GridFixedPoser(random=False)` tolerates it, and only accidentally (`[:None]` is a full slice). | **Effectively untested** | Any config using `grid_dims` with a poser other than `GridFixedPoser(random=False)` | `capture.py:32`; `posers.py:31-37`; [§II.7](#ii7-capture) |
| 12 | **`num_targets` sampling uses `np.random.choice` with replacement**, so a grasp target can be drawn twice. `replace=False` would be safe while `num_targets < graspable count`. | **Flagged, not changed** | Duplicate targets silently reduce scene coverage | `capture.py:27-28`; `fix-occupancy-grid-full-wall.md` |
| 13 | **`extract_one` hardcodes `{model_path}/v_0`.** `model_drink101` uses `v_69323`. | **Open** | That SKU cannot be extracted from the store USD | `extract_store_objects.py:39-56`; `store-front-face-check.md:109-111` |
| 14 | **There is no writer registry.** Mode dispatch is a two-branch `if` in `main()`; a third sample type means a third mode and a third entry function. | **Known design limit** | The least extensible seam in the system | `clean_datagen.py:159-162`; [§II.8](#ii8-writers) |
| 15 | **`optflow_render` / `graspableobj_to_optflow_obj` load a full `RuntimeConfig`** and therefore validate `dataset_dir`, `proposer_config_path` and `descriptor_config_path` despite using none of them. | **Acknowledged wart** | Stage-B runs fail on unrelated missing paths | `optflow-1-reference-dataset.md`; [§III.8.11](#iii811-miscellaneous-sharp-edges) |
| 16 | **Reference rendering supports side faces only.** `look_at` is singular when the view normal is parallel to world up, so ±Z faces are excluded — and the same degeneracy applies to a near-vertical `distant_light_offset`. | **By construction, not a bug** — but it is a real capability limit | Objects whose informative face is the top or bottom cannot be referenced | `mesh_convert.py:62-78`; `distant-light-key-light.md` |
| 17 | **`add_proposals.main()` unconditionally deserializes `OptFlowMetadata`** (`add_proposals.py:21`), i.e. phase 2 expects optflow-shaped metadata, while `run_pipeline` is not mode-aware and runs phase 2 after every render. | **Newly observed against current source; not covered by any plan.** Needs verification against a `mode: reference_segmentation` render dir | `isaac-datagen-pipeline` on a refseg config would fail at phase 2 if the refseg metadata lacks the optflow fields | `add_proposals.py:21`; `run_pipeline.py:107-115`; [§II.10](#ii10-downstream-phases) |
| 18 | **Occlusion excludes frame-edge truncation**, and `coords_in_mask` does not treat the image border as a boundary. | **Intentional, documented** — listed here because it is repeatedly mistaken for a bug | Filtering on `iid_to_occlusion` will not catch objects merely cut off by the frame | `obsmask-occlusion-and-viz.md`; `inlier-border-eps-margin.md` |

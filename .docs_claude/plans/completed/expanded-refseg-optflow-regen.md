# Regenerate `expanded-refseg` in optflow mode + Stage-B proposals under the reproj-coverage gate

> **Status (2026-06-25): DONE.** Rendered locally on 2× RTX 4090. `render000` (1000 obs) + `render001`
> (1100 obs) re-rendered in `mode: optflow` (full geometry, `runtime.yaml` finalized); `render002` kept as-is.
> `CleanDiftFinetunedDescriptor` re-added to render000/001 (Stage 0.5). All 3 dirs regenerated under the
> reproj gate + grid proposer @ eps=0.0 (inliers 8.0% / 7.4% / 7.5%); `verified_proposals/`+`verification/`
> dropped. Gate + inlier viz spot-checked (`gate_viz_expanded/`). Execution tracker (now reflecting this):
> `plans/active/stage-b-regen-progress.md`. **Remaining (out of scope here):** fresh `freeze_split` over all
> 12 render dirs → verifier retrain → re-bake `verified_proposals/`.

## Context

We replaced the proposer's visible-pixel floor with a **reprojection-coverage gate** (fraction of a
class's reference texture visible in the observation; design `~/.claude/plans/jaunty-skipping-widget.md`,
write-up `isaac_datagen/.docs_claude/plans/completed/reproj-coverage-gate-and-ycb-ref-pose-fix.md`). The
gate requires `OptFlowMetadata`/`OptFlowSample` geometry (obs depth, `cam2world`, per-instance
`class_to_l2w`, reference depth/pose/K). `mixed-persp` and `shelf-optflow` already have it and are
regenerated (tracker `…/plans/active/stage-b-regen-progress.md`). **`expanded-refseg` is the last dataset
and was HELD** because two of its render dirs are legacy *reference-seg* renders with no optflow geometry —
the gate cannot run on them. This plan re-renders those in `mode: optflow`, then regenerates proposals +
inlier labels for the whole dataset under the new gate using **grid proposals**, leaving the data ready for
the downstream split re-freeze + verifier retrain.

### Current on-disk state (verified)

| Render dir | obs (frames×targets) | Mode | Geometry | Action |
|---|---|---|---|---|
| `render000` | 1000 (100×10) | reference-seg | **missing** | **Stage 0 re-render** + Stage B |
| `render001` | 1100 (25×44) | reference-seg | **missing** (yaml also missing `inlier_border_eps`) | **Stage 0 re-render** + Stage B |
| `render002` | 300 (100×3) | **optflow (complete)** | present | **Stage B only** (keep geometry) |
| `render000_viz_clusters_cleandift` | — | viz artifact | — | ignore (leave in place) |

Key facts that shape the plan:
- **expanded-refseg is amazon-only** (render000/001 `graspable_objects_path=…object_dataset_amazon`,
  render002 `objects_path=…/optflow_objects/amazon/`). The **amazon OptFlowObject catalog already exists**
  (`isaac_datagen/assets/optflow_objects/amazon/`, 44 objects, complete) → **no offline
  `graspableobj_to_optflow_obj.py` render needed**, and **the ycb_can reference-pose fix is irrelevant here**.
- **Render locally on the 2× RTX 4090** (RT cores → Isaac Sim 5.1 works) → the entire PSC/Singularity/
  Vulkan-ICD/asset-sync footgun chain does **not** apply.
- **OmegaConf ignores stale yaml keys** (`runtime_config.load_config`), so legacy keys won't error — but the
  **`pallet_dims` OccupancyGrid placer no longer exists in code**, so the old scene layout can't be
  reproduced. We adopt render002's proven optflow recipe (decision below).

## Decisions (from the user)
1. **Scene recipe for render000/001:** mirror render002's optflow recipe, **scaled to preserve each dir's
   obs count** (carry forward `num_frames`/`num_targets` from the old yamls).
2. **Render host:** local 2× RTX 4090.
3. **render002:** keep its optflow geometry; **Stage B only** (no GPU re-render).

---

## Stage 0 — re-render `render000` + `render001` in optflow mode (local)

Build one input config `src/isaac_datagen/configs/expanded-refseg.yaml` modeled on render002's proven
runtime, then drive each dir with per-dir CLI overrides. **Launch cwd must be `src/isaac_datagen/`** (config
paths are relative to it). The render is **not resumable** and `finalize_metadata()` runs only at the end —
**clear the dir first** and **confirm `runtime.yaml` exists** afterward to know the dataset is complete.

**New config `configs/expanded-refseg.yaml`** (values taken verbatim from the working `render002/runtime.yaml`;
drops the dead keys, points the baked runtime.yaml at the new grid proposer + reproj gate so it is honest):

```yaml
mode: optflow                      # MANDATORY, no default — must be present
scene: empty
seed: 1                            # effective_seed = seed + idx → distinct RNG per dir
dataset_dir: /data/user/jeffk/datasets/expanded-refseg   # MUST pre-exist
intrinsics_path: zed_K.npy
descriptor_device: cuda:1            # GPU — Phase 1 precomputes DIFT reference features
proposer_device: cpu                 # baked so the dir's runtime.yaml matches Stage B (grid proposer = reference-free, no NN)
# placement / poses — render002's recipe (replaces the removed pallet OccupancyGrid)
placement: UntilExhaustedStacker
placement_args: {column_height: 3}
pose_generation_policy: LookAtPoser
pose_generation_policy_args: {xrange: ${xrange}, yrange: ${yrange}, zrange: ${zrange}}
xrange: [0.25, 0.85]
yrange: [-0.42, 0.42]
zrange: [-0.40, 0.40]
# amazon-only catalog (relative to src/isaac_datagen/)
objects_path:
  - ../../assets/optflow_objects/amazon/
filter_specs:
  - {name: ReplicateFilter, args: {key: name, value: 'amazon_*', count: 5}}
  - {name: ShuffleFilter, args: {seed: ${idx}}}
# lighting / exposure — render002 values
dome_light: true
distant_light: true
distant_intensity: 2000.0
distant_angle: 0.53
distant_light_offset: [1.0, -3.0, 3.0]
dome_fill_intensity: 200.0
set_exposure: true
exposure_time: 1.0
f_number: 5.0
film_iso: 100.0
rt_subframes: 20
warmup_frames: 32
dry_run: false
obs_full_alpha: false
# proposer/gate — bake the NEW gate into runtime.yaml (Stage B re-overrides on CLI anyway)
proposer_config_path: ../../../reference_matching/src/reference_matching/configs/grid_proposal.yaml
descriptor_config_path: ../../../reference_matching/src/reference_matching/configs/descriptor.yaml
proposer_min_visible_ratio: 0.30
inlier_border_eps: 0.0             # amazon → 0.0 (matches render002 + mixed-persp amazon)
```

**Run (render-only Phase 1; Stage B is run separately below for all dirs):**
```bash
cd .../isaac_datagen/src/isaac_datagen
DS=/data/user/jeffk/datasets/expanded-refseg
# render000 — 1000 obs
rm -rf "$DS/render000"
isaac-datagen configs/expanded-refseg.yaml idx=0 num_frames=100 num_targets=10
# render001 — 1100 obs
rm -rf "$DS/render001"
isaac-datagen configs/expanded-refseg.yaml idx=1 num_frames=25 num_targets=44
```
- Obs count = `num_frames × num_targets`; the carried-over pairs reproduce 1000 / 1100.
- **Smoke first:** run each with `num_frames=1` to catch config/path errors cheaply (≈0 GPU), then scale.
- `num_targets=44` relies on `UntilExhaustedStacker` fitting 44 grasp targets from the 44×5 replicated
  amazon instances — verify the smoke render actually places that many; lower if the placer caps out.

### Stage 0.5 — re-add the CleanDIFT backbone (no re-render)

The render bakes only `DiftDescriptor` into `class_to_descriptors/` (matching mixed-persp/shelf/render002).
But **the verifier trains on `CleanDiftFinetunedDescriptor`** (`segmentation/verifier/configs/
verifier_training.yaml`: `descriptor: CleanDiftFinetunedDescriptor`, provider `CleanDiftFinetunedFpn`), which
is added **post-hoc** — every regenerated dir carries both `DiftDescriptor/` and `CleanDiftFinetunedDescriptor/`
subfolders. Wiping render000/001 drops their cleandift backbone, so re-add it (re-encodes the stored
`class_to_ref` images, minutes, no Isaac Sim — this is also what produced `render000_viz_clusters_cleandift`):
```bash
cd .../isaac_datagen && env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone \
  add-backbone /data/user/jeffk/datasets/expanded-refseg \
  ../reference_matching/src/reference_matching/configs/cleandift_finetuned.yaml --device cuda
```
Use the **single-scale** `cleandift_finetuned.yaml` (CleanDiftFinetunedDescriptor) — an FPN config would fail
the `.squeeze(0)`. The command walks the whole dataset root; it harmlessly re-encodes render002's identical
backbone too (idempotent). The fresh render000/001 are already written in the per-backbone SubfolderDict
layout, so no `relocate` pass is needed.

## Stage B — regenerate proposals + inliers under the reproj gate (grid proposals), all 3 dirs

Identical to the proven mixed-persp/shelf recipe from the tracker. **Grid proposals** = the 32×18=576-point
anchor grid (`reference_matching/configs/grid_proposal.yaml`); the gate is `gate_classes_reproj` (auto-used by
`add_proposals.py` now). render002 needs **only this stage** (its geometry is kept).

```bash
RM=.../reference_matching/src/reference_matching/configs
ID=.../isaac_datagen/src/isaac_datagen
DS=/data/user/jeffk/datasets/expanded-refseg
for N in 000 001 002; do
  RD="$DS/render$N"; IDX=$((10#$N))
  rm -rf "$RD"/{proposals,verified_proposals,verification}
  isaac-datagen-proposals "$RD/runtime.yaml" dataset_dir="$DS" idx=$IDX \
    intrinsics_path=$ID/zed_K.npy proposer_config_path=$RM/grid_proposal.yaml \
    descriptor_config_path=$RM/descriptor.yaml proposer_device=cpu proposer_min_visible_ratio=0.3
  isaac-datagen-inliers "$RD" --eps 0.0
done
```
- Base config is each dir's **on-disk `runtime.yaml`** (faithful per-dir settings; `idx` already resolves the
  dir). The overrides are all path/device/threshold — no behavior change; abs `dataset_dir`/`intrinsics_path`/
  `descriptor_config_path` because snapshot relatives don't resolve from cwd. With `proposer_device: cpu`
  already baked in (above), the `proposer_device=cpu` override here is **redundant/defensive**.
- **`--eps 0.0` for all three** (amazon convention; render002 + mixed-persp amazon use 0.0). This is a
  deliberate standardization — old render000 used 2.0 (a reference-seg-era value). Tune via
  `isaac-datagen-sweep-label-eps` if a spot-check wants a margin.
- `verified_proposals/`/`verification/` are intentionally dropped (need the retrained verifier downstream).

---

## What you might want to KEEP, but the re-render changes (handled here)

- **Per-dir scale (`num_frames`/`num_targets`)** — the program would otherwise take config defaults; we
  **preserve them** by carrying each dir's old values on the CLI (1000 / 1100 obs).
- **amazon-only class identity** — preserved by `objects_path: …/optflow_objects/amazon/` only.
- **Scene RNG content** — a re-render produces *fresh* placements/poses; the old 2400 specific frames are
  gone. Acceptable: the train/val split is being re-frozen over the new content (below).
- **The pallet OccupancyGrid layout (`pallet_dims=[11,1,4]`)** — gone from the code; **cannot** be kept.
  We accept the render002 stacked-column composition instead.
- **Descriptor backbones** — the re-render writes only the `DiftDescriptor/` backbone. The verifier's actual
  reference tokens are `CleanDiftFinetunedDescriptor`, which render000/001 carry today and the wipe drops →
  **Stage 0.5 re-adds it** via `add-backbone` (no re-render). render002 keeps both untouched.

## What you want to CHANGE, but the program does NOT touch (do manually)

- **Dead yaml keys** — `graspable_objects_path` (a pre-extraction `visual_servoing/…` path), `pallet_dims`,
  `proposer_max_occlusion: 0.1`. OmegaConf ignores them, but we **omit them** from the new config rather than
  carry them; `dataset_dir` is corrected to the absolute `/data/user/jeffk/datasets/expanded-refseg`.
- **render001's missing `inlier_border_eps`** — the program defaults it; we **set it explicitly to 0.0**.
- **Train/val split re-freeze** — `isaac-datagen` does not touch it. After all regen, re-run
  `segmentation.verifier.freeze_split` over the regenerated data → `datasets/splits/refseg_split.json`,
  then `dspush splits`. The re-render reuses `expanded-refseg/renderNNN/frame` keys over fresh content, so a
  stale manifest would silently collide — re-freezing dissolves it.
- **Verifier retrain + `verified_proposals/` re-bake** (`segmentation.verifier.process`) — downstream of this
  plan, after the fresh freeze; out of scope here but it is why `verified_proposals/` is dropped.
- **`isaac-datagen-viz-inliers` `classes=` bug** — if you viz-check inliers, call `vision_core.viz.inlier_figure`
  directly (the CLI passes a removed `classes=` kwarg → `TypeError`); noted in the tracker.

---

## Verification

1. **Render completeness:** after each re-render, assert `runtime.yaml` exists in the dir (the not-resumable
   finalize landmine) and the optflow subdirs are present: `observation_depth/ cam2world/ obs_intrinsics/
   class_to_l2w/ class_to_ref_pose/ class_to_reference_depth/ class_to_ref_intrinsics/ class_to_reference/
   class_to_name/`. (Field-presence check used during the gate research.)
2. **Obs count:** `ls render00N/obs | wc -l` == `num_frames × num_targets` (1000 / 1100 / 300).
2b. **Both backbones present:** after Stage 0.5, `class_to_descriptors/` in render000/001 lists both
   `DiftDescriptor/` and `CleanDiftFinetunedDescriptor/` (matching render002 + the siblings).
3. **Gate viz (the deliverable):** `isaac-datagen-viz-gate /data/user/jeffk/datasets/expanded-refseg` and
   spot-check the ratio cut — fully-visible-but-small (far-camera) amazon boxes show high ratio (green),
   genuinely occluded/off-frame instances show low (red).
4. **Inlier spot-check:** run `inlier_figure` directly on a couple of frames per dir (avoid the broken CLI).
5. **Split integrity (after the downstream re-freeze):** every regenerated `(root/render_dir, frame, class)`
   verifier key resolves in the new manifest with 0 hash-fallback, and every `verified_proposals` key resolves
   (`verified ⊆ proposals`).

## Files / artifacts touched
- **New:** `isaac_datagen/src/isaac_datagen/configs/expanded-refseg.yaml` (the optflow input config above).
- **Datasets (gitignored, local):** `expanded-refseg/render000`+`render001` re-rendered in optflow mode;
  `CleanDiftFinetunedDescriptor/` backbone re-added to those two (Stage 0.5); `proposals/`+`labels/`
  regenerated for all three dirs; stale `verified_proposals/`+`verification/` deleted.
- **Downstream (separate steps):** `datasets/splits/refseg_split.json` re-frozen + `dspush splits`; verifier
  retrain + `verified_proposals/` re-bake.
- **Docs:** update `…/plans/active/stage-b-regen-progress.md` (expanded-refseg no longer HELD).

# Regenerate `expanded-refseg-v2` — curated 8-instance amazon subset + depth/gap jitter (AS-BUILT)

## Context

A fresh version of the `expanded-refseg` (amazon-only optflow) dataset that (a) keeps only a curated
**8-instance subset** of the amazon boxes (one box per color family) and (b) adopts the new
**y-depth-jitter** and **x-gap-jitter** placement. Output is a new dataset `expanded-refseg-v2`; it
will **replace** v1 in the M2F/gligen/verifier configs + the shared split (Step 4, pending). v1 is
left on disk untouched.

> **Approach note:** the subset is expressed as an explicit **pre-serialized instance catalog**
> (`assets/optflow_objects/amazon-v2/`, the 8 chosen objects full-serialized to a standalone dir),
> not the originally-planned config-side `RegexFilter` on classes — the user picked specific
> instances by name. The render points `objects_path` at that dir, so no filter is needed.

## Status: dataset generated (Steps 0–3 DONE); Step 4 (training rewire) PENDING.

## Step 0 — Pick the roster (annotated reference grids)

Rendered an annotated montage of all 44 `optflow_objects/amazon` reference images (titled by
class+name, sorted by class) for selection. User chose **8 instances**, one per color family:

| name | class | name | class |
|---|---|---|---|
| amazon_6 | amazon_blue | amazon_40 | amazon_green |
| amazon_21 | amazon_brown | amazon_3 | amazon_orange |
| amazon_18 | amazon_burgundy | amazon_14 | amazon_pink |
| amazon_11 | amazon_cyan | amazon_0 | amazon_red |

Grid built by a throwaway montage over `reference_image/` + `meta/` (reused
`relabel_classes.write_grid`'s idea; `relabel_classes.py` itself is `GraspableObject`-only and was
NOT modified).

## Step 1 — Filter → serialize `amazon-v2`

Deserialized all 44 `OptFlowObject`s (Isaac-free), kept the 8 by `meta["name"]`, **full-serialized**
to `assets/optflow_objects/amazon-v2/` with fresh 0..7 indices (copies usdz/png/npy; no
`SameFileError` since dst≠src). Integrity: `meta` count == `usd_path` count == 8.

## Step 1b — `expanded-refseg-v2.yaml`

Copied `configs/expanded-refseg.yaml` → `configs/expanded-refseg-v2.yaml` (no `base:`/include in
`load_config`), changing only:
- `dataset_dir: /data/user/jeffk/datasets/expanded-refseg-v2`
- `objects_path: ../../assets/optflow_objects/amazon-v2/` (no `RegexFilter`)
- `placement_args`: `min_y: 0.0, max_y: 0.10` (depth) + `min_gap: 0.0, max_gap: 0.50` (x-gap)

`ReplicateFilter(count:5)` + `ShuffleFilter` and the proposer/gate/lighting/eps carry over verbatim.

## Pre-req — committed the jitter placers (working-tree WIP)

- **depth-jitter** (`min_y`/`max_y`): submodule `c9c7fa0`, meta pin `56464e1`.
- **gap-jitter** (`min_gap`/`max_gap`): submodule `898013f`, meta pin `a1a4396`.

Both committed on the `isaac_datagen` `psc-isaac-singularity` branch (NOT pushed). The
`expanded-refseg-v2.yaml` config + `amazon-v2` assets remain uncommitted/un-pushed.

## Step 2 — Render (DONE)

GPU 0 was busy (user's `segmentation` training, ~20 GiB), so everything ran on **GPU 1** via
`CUDA_VISIBLE_DEVICES=1` + `descriptor_device=cuda:0`. Smoke render (4 frames) validated the recipe
first (subset classes only, replicas, both jitters visible via `ObsMask.visualize(md)`), then the
full render — 3 dirs, 1000 obs each (3000 total):

```bash
cd isaac_datagen/src/isaac_datagen
for IDX in 0 1 2; do
  CUDA_VISIBLE_DEVICES=1 uv run isaac-datagen configs/expanded-refseg-v2.yaml \
    idx=$IDX num_frames=100 num_targets=10 descriptor_device=cuda:0
done
```

## Step 3 — Proposals + inliers + CleanDIFT (DONE)

Grid proposer (`GridProposal` 32×18 = 576 pts/class) + reproj-coverage gate, inliers at eps 0.0, then
CleanDIFT re-add (a fresh render bakes only `DiftDescriptor`):

```bash
for IDX in 0 1 2; do
  RD=/data/user/jeffk/datasets/expanded-refseg-v2/render00$IDX
  CUDA_VISIBLE_DEVICES=1 uv run isaac-datagen-proposals "$RD/runtime.yaml" \
    dataset_dir=/data/user/jeffk/datasets/expanded-refseg-v2 idx=$IDX \
    intrinsics_path=$PWD/zed_K.npy proposer_config_path=$RM/grid_proposal.yaml \
    descriptor_config_path=$RM/descriptor.yaml proposer_device=cpu proposer_min_visible_ratio=0.3
  uv run isaac-datagen-inliers "$RD" --eps 0.0
done
cd isaac_datagen && CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH uv run python -m \
  isaac_datagen.migrate_descriptors_backbone add-backbone \
  /data/user/jeffk/datasets/expanded-refseg-v2 \
  ../reference_matching/src/reference_matching/configs/cleandift_finetuned.yaml --device cuda:0
```

**Result (all 3 dirs):** obs=proposals=labels=1000; `stats`; both `class_to_descriptors/{DiftDescriptor,
CleanDiftFinetunedDescriptor}`. Inlier rates @eps0: render000 **8.8%**, render001 **7.9%**, render002
**7.1%** — consistent with the expanded-refseg amazon convention (~7–8%).

## Step 4 — Replace v1 → v2 in configs + re-freeze split (PENDING)

Swap the literal `datasets/expanded-refseg` → `datasets/expanded-refseg-v2` in the `paths:` lists
(drops v1, adds v2 — does NOT delete v1 from disk), then re-freeze. The shared `refseg_split.json` is
keyed by `<root_basename>/<render_dir>`, so the swap changes the keys → re-freeze required.

- `segmentation/src/segmentation/configs/mask2former_training.yaml:25-28` (M2F, primary)
- `segmentation/src/segmentation/configs/gligen_training.yaml:25-28` (stage-3 sibling, same split)
- `segmentation/src/segmentation/verifier/configs/*.yaml` + `verifier/verifier-singlescale-shallow.yaml`
  (sweep for shared-split consistency)
- `segmentation/tests/make_fixtures.py:24` (repoint if it should track v2)

```bash
cd segmentation && uv run python -m segmentation.verifier.freeze_split \
    src/segmentation/configs/mask2former_training.yaml \
    /data/user/jeffk/datasets/splits/refseg_split.json
```

## Open items (user's call)

- **`aspush amazon-v2`** — needed only to render v2 on another machine/PSC (local render needs nothing).
- **Commit `expanded-refseg-v2.yaml`** — its values are captured in each render dir's `runtime.yaml`
  regardless; committing the config makes the recipe tracked (but it references the un-pushed
  `amazon-v2` assets).

## Verification (done for Steps 2–3)

`ObsMask.visualize(md)` spot-checks: 8 classes only, replicas, depth+50cm-gap stagger visible;
`cid_to_class` = the 8 classes; `isaac-datagen` wrote `runtime.yaml` per dir; descriptor subfolders +
inlier stats present. Step-4 verification (pending): re-frozen split keys read `expanded-refseg-v2/…`,
M2F datamodule discovers v2 dirs and partitions without `KeyError`, v1 absent.

## Critical files

- `isaac_datagen/src/isaac_datagen/placers.py` + `configs/staggered.yaml` — depth+gap jitter (committed).
- `isaac_datagen/assets/optflow_objects/amazon-v2/` — NEW 8-instance catalog (un-pushed).
- `isaac_datagen/src/isaac_datagen/configs/expanded-refseg-v2.yaml` — NEW render config (uncommitted).
- Reuse: `clean_datagen.collect_preoptflow`, `migrate_descriptors_backbone add-backbone`,
  `isaac-datagen-proposals`/`-inliers`, `segmentation/verifier/freeze_split.py`,
  `vision_core.datastructs.{ObsMask,OptFlowMetadata}.visualize/deserialize`.
- Step-4 config swaps: `segmentation/.../configs/{mask2former,gligen}_training.yaml`,
  `verifier/configs/*.yaml`, `tests/make_fixtures.py`.

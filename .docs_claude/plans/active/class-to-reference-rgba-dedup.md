# Retire `OptFlowMetadata.class_to_reference` in favor of `class_to_ref` (RGBA)

**Status: active, not yet implemented (planned 2026-07-01).**

## Context

`class_to_ref` (RGBA, on `ObsMaskDescriptorMetadata`, nested at
`OptFlowMetadata.obsmaskmeta.class_to_ref`) and `class_to_reference` (RGB, top-level
field on `OptFlowMetadata`, added in [[optflow-6-nested-obsmask]]'s writer) are two
independently-computed "representative reference image per class" catalogs that live
in the same render dir. They differ only in *how* the representative object is chosen
— `reference_catalog()` (`isaac_datagen/reference_seg_writer.py:88-115`, feeds
`class_to_ref`) sorts by object name; `optflow_writer._optflow_metadata()`
(`isaac_datagen/optflow_writer.py:99-102`, feeds `class_to_reference`) uses raw
object/pose list order — but empirically every instance of a class renders the same
canonical `reference_image` regardless of which member is picked, so
`torch.equal(class_to_ref[cls][:3], class_to_reference[cls])` was `True` for all 30
classes checked across 3 filtered/vis030 render dirs (expanded-refseg-v2,
mixed-persp, shelf-optflow), with zero class-name mismatches between the two dicts
(verified via a throwaway viz + numeric-equality script, both run against real render
dirs). This is redundant computation and a duplicated on-disk field — tech debt.
Two research agents traced every producer and consumer of both fields across all 8
submodules; this plan retires `class_to_reference` and repoints its consumers at
`class_to_ref`, keeping the RGBA field as the single source of truth.

Key asymmetry that shapes the plan: `class_to_reference` never travels alone — it's
part of a 5-field geometry group (`class_to_reference`, `class_to_reference_depth`,
`class_to_ref_intrinsics`, `class_to_ref_pose`, `class_to_l2w`) built together in
`_optflow_metadata()` and consumed together (via `vision_core.pose_utils`) for 3D
warp/reprojection math (visibility gating, UFM ground-truth flow, TOTG point
transfer). Only the RGB image itself (`class_to_reference`) has an RGBA duplicate
(`class_to_ref`); the depth/intrinsics/pose/l2w siblings have no counterpart anywhere
else and are **not** touched by this plan — only the image field is deduplicated.

## Producer changes

- `isaac_datagen/src/isaac_datagen/optflow_writer.py:99-120` (`_optflow_metadata()`):
  delete the `class_to_reference={...}` entry (currently lines 109-112) from the
  `OptFlowMetadata(...)` construction. Everything else in that call
  (`class_to_reference_depth`, `class_to_ref_intrinsics`, `class_to_ref_pose`,
  `class_to_l2w`, `class_to_name`, `obsmaskmeta=...`) is unchanged — `obsmaskmeta`
  already carries `class_to_ref` via the existing `reference_catalog()` /
  `obsmask_metadata()` call at lines 64/105, so no new computation is needed.
- `vision_core/src/vision_core/datastructs.py:540-565` (`OptFlowMetadata` dataclass):
  remove the `class_to_reference: dict` field (line 558) and its entry in the
  `_serializers` dict (line 565). This is the dataset-contract change; every consumer
  below must be repointed before/alongside this lands, since removing the field means
  `md.class_to_reference` becomes an `AttributeError`.

No migration needed for already-serialized render dirs: `SerializableSample`
deserializes by iterating the dataclass's declared fields, not the directory listing,
so old `class_to_reference/*.pt` files on disk simply become orphaned and inert —
safe to leave, no backward-compat shim required.

## Consumer changes

Three real call sites read `.class_to_reference` for its pixel data; all three
repoint to `md.obsmaskmeta.class_to_ref[cls][:3]` (RGBA → RGB slice, no alpha
compositing — these are training/geometry inputs, not display panels):

- `vision_core/src/vision_core/datastructs.py:515` (`OptFlowSample.visualize()`,
  debug panel): this one *is* a display context, so use
  `vision_core.viz.rgba_chw_to_rgb(md.obsmaskmeta.class_to_ref[cls])` (composites
  over white) instead of a raw `[:3]` slice, matching the convention already used by
  every other RGBA viz call site (`viz.py:280`, `viz.py:629`,
  `datastructs.py:373`, `datastructs.py:738`).
- `UFM-train/uniflowmatch/datasets/optflow_isaac.py:122-125` (`OptFlow2UFM.__getitem__`,
  builds `MaskedUFM.ref_rgb`): change
  `ref_rgb=md.class_to_reference[c]` → `ref_rgb=md.obsmaskmeta.class_to_ref[c][:3]`.
  This is the actual reference image fed to the UFM optical-flow model — a real
  training input, not a viz-only field, so this is the highest-stakes call site.
- `benchmark/src/totg_benchmark/convert.py:155-158` (`ref_work()`, builds
  `TOTGInput.ref_rgb`): change `md.class_to_reference[c]` →
  `md.obsmaskmeta.class_to_ref[c][:3]` in the same spot depth/intrinsics/pose are
  already read from their (unchanged) sibling fields.

`vision_core/src/vision_core/pose_utils.py` (`_ref_points`, `instance_visibility`)
only reads the depth/intrinsics/pose/l2w siblings, never `class_to_reference` itself
— **no change needed there**. `isaac_datagen/make_unseen.py`'s `_flip_catalog()` only
flips `class_to_ref` (RGBA) already — **no change needed there** either; it already
treats `class_to_ref` as canonical.

## Rollout order (cross-repo pin dependency)

This repo group pins `vision_core`/`isaac_datagen` as editable/git deps from
consuming repos (per root `CLAUDE.md`'s "Bump pin" commit convention). Land in this
order:
1. `vision_core`: remove the field (datastructs.py) + fix `OptFlowSample.visualize()`.
2. `isaac_datagen`: stop writing `class_to_reference` in `optflow_writer.py`; bump its
   `vision_core` pin.
3. `UFM-train` and `benchmark`: repoint their one call site each; bump their
   `vision_core` pin (both only need the field-removal commit, not the writer change,
   since they only *read* render dirs).
4. Bump the `vision_core`/`isaac_datagen` pins in any other repo that imports
   `OptFlowMetadata` by name for type-checking only (`isaac_datagen/objects.py`,
   `proposal_gate.py`, `clean_datagen.py`, `migrate_descriptors_backbone.py` per the
   research pass — these only reference the type, not the removed field, so they need
   a pin bump but no code change).

## Verification

- `segmentation/tests/test_optflow_visibility_contract.py` and
  `segmentation/src/segmentation/precompute_visibility.py` don't touch
  `class_to_reference` (only the depth/pose/intrinsics siblings via
  `instance_visibility`) — run the contract test as a regression check that the
  untouched geometry path still works after the field removal.
- `UFM-train/scripts/smoke_overfit_optflow.py` — existing smoke script exercising
  `OptFlow2UFM`; run it against one of the 3 already-verified render dirs
  (e.g. `filtered/vis030/shelf-optflow/render000`) to confirm `ref_rgb` still loads
  and has the expected `(3,H,W)` shape/dtype after the repoint.
- `benchmark/src/totg_benchmark/convert.py` has no existing test — after repointing,
  manually run `optflow_metadata_to_totgsamples()` (or its CLI entrypoint) against the
  same render dir and spot-check `TOTGInput.ref_rgb` shape/values against the
  pre-change output (can reuse the byte-identity check already established: values
  should be identical to what `class_to_reference` used to produce).
- Re-run `isaac_datagen`'s writer on one render (or a small test scene) end-to-end and
  confirm the produced `OptFlowMetadata` no longer serializes a `class_to_reference/`
  subdir, and that `md.obsmaskmeta.class_to_ref` is present and covers the same class
  set as before (no coverage regression vs. the old field, matching the "0 mismatches"
  result already observed across all 3 datasets).

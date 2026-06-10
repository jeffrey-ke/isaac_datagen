# `UntilExhaustedStacker` placement policy for heterogeneous objects

> **Status: completed 2026-06-08.** Verified end-to-end on `randomized.yaml` (amazon +
> `occupancy_grid`) and `mixed.yaml` (51 heterogeneous objects + `until_exhausted_stacker`):
> both render with zero empty-frame errors and write real `obs`/`iid_mask`. As-built
> deviations from the original design: depth alignment is **centroid y=0** (`T_y = -cy`), not
> front-coplanar; and the `add_grasp_frame` origin≠center correction had to use
> `ComputeUntransformedBound`, **not** `ComputeLocalBound` — see the Landmine below, which
> cost the most debugging time in this work.

## ⚠️ Landmine: `ComputeLocalBound` bakes in the prim's placement transform

**This is the important thing to remember from this plan.**

The plan added an origin≠center correction to `add_grasp_frame` so heterogeneous (off-center)
assets get their grasp frame at the true bbox front-bottom edge. The first implementation
computed the offset with `local_bbox_range(box_prim).GetMidpoint()`
(= `UsdGeom.BBoxCache.ComputeLocalBound(prim).GetRange().GetMidpoint()`).

**The bug:** `ComputeLocalBound` **includes the prim's own local-to-parent transform**.
`GetSize()` is unaffected (extent is translation-invariant — which is why
`bounding_half_extents` always worked), but `GetMidpoint()` is contaminated by *wherever the
prim currently sits*. `add_grasp_frame` runs in `build_scene` **after** `organize_objects`
has already slotted each wrapper, so the midpoint came back ≈ the slot position instead of a
small geometric offset. `set_transform` then placed the grasp frame at that offset *relative
to the already-placed wrapper* — roughly **doubling** it and flinging the grasp frame off the
object (observed: grasp `z=0.85` for a box centered at `z=0.478`).

**The symptom:** camera poses are computed relative to the grasp frame (`clean_datagen.py`
`get_target2world` → `plan_poses` → `world_poses`), so the camera aimed at empty space.
Every rendered frame had zero labeled instances and `ObsMaskWriter.write()`
(`reference_seg_writer.py`) raised `"write() called with no labeled instances — expected ≥1"`
on **every** frame.

**Why it was hard to localize:**
- It fired for **both** placers and the original `occupancy_grid` — because `add_grasp_frame`
  is on the common path, not the placer. (This correctly ruled out the placement work.)
- `debug_scene.py` showed the stacker geometry looking **correct** — because placement
  measures bboxes at policy *construction*, BEFORE `organize_objects` (clean midpoints), and
  grasp frames are invisible empty Xforms that don't show in the USDZ export. Only the
  capture path exercised the post-placement (contaminated) measurement, and `debug_scene`
  never runs the writer.
- The same render dir had a successful run from earlier the same day (pre-edit), confirming
  a regression introduced by this change.

**The ultimate fix** (`scene.py:add_grasp_frame`): measure with
`bb.ComputeUntransformedBound(box_prim).ComputeAlignedRange().GetMidpoint()`, which ignores
the prim's own placement transform and yields a purely geometric offset (and reduces to the
old `(0, -half[1], -half[2])` behavior for centered boxes). `local_bbox_range` is no longer
imported in `scene.py`.

**Diagnosis method that worked** (after two wrong guesses): instrument the writer to dump the
first empty frame's RGB + `idToSemantics`/`idToLabels` to a file (Isaac floods stdout, so
persist like `debug_scene.py`), plus a one-shot print of grasp/camera/object world coords
before capture. The dump showed `unique seg ids = [0]` (pure background → framing, not
labeling) and the grasp `z` flung above the box — pinning it in one run.

(Also recorded in auto-memory: `usd-computelocalbound-includes-placement`.)

## Context

`isaac_datagen` only had `OccupancyGrid` — a uniform full-wall pallet that sizes every cell
off the *first* object's bbox (`scene.py`) and requires exactly `prod(pallet_dims)` objects.
That breaks for a heterogeneous object dataset (each object its own class), the target for
reference-prompted instance segmentation. This plan added a second policy that measures each
loaded prim's bbox and packs objects column-by-column.

## What was built

- **`objects.py UntilExhaustedStacker`** — stateful callable placer. Constructed with the
  loaded `prim_paths` + `column_height`; measures each prim's bbox size+center (clean, before
  `organize_objects` places them), precomputes a centered layout, `__call__(path)` is a
  lookup. Columns of ≤ `column_height` stacked base-to-base, centroids on the column
  center-line and on y=0, column width = widest member, columns abut left→right centered on
  x=0. No physics (overhang allowed); last column may be partial. Origin≠center corrected via
  the local-frame midpoint (clean here because measured pre-placement).
- **Unified placer contract** — every policy exposes `graspability() -> dict[str, bool]` in
  addition to `__call__`. `UntilExhaustedStacker`: only each column's top is graspable.
  `OccupancyGrid`: records `path→(i,j,k)` during `__call__` and folds the former inline
  `is_front and is_top` computation (`scene.py`) into the method. Refactor proven equivalent
  (11/44 column tops, unchanged).
- **`runtime_config.py`** — `placement` (`"occupancy_grid"` default) + `column_height` (5),
  with `__post_init__` asserts.
- **`scene.py create_stack_of_objects(parent_path, objects, runtime)`** — selects the policy,
  makes the full-wall capacity check per-policy (occupancy == capacity; stacker ≥ 1), and
  sets `is_graspable = policy.graspability()`.
- **`scene.py add_grasp_frame`** — the origin≠center correction (see Landmine for the fix).
- Masking unchanged: instance + semantic labels are applied to **every** placed object at
  `add_object` time, independent of graspability.

Config note: `mixed.yaml` must set `placement: until_exhausted_stacker` (the default is the
uniform `occupancy_grid`; a missing `placement` silently runs the wrong policy and, because
`len(objects) ≥ capacity`, doesn't trip the full-wall guard).

## Verification

- Pure-numpy: `OccupancyGrid.graspability()` byte-equal to the old inline result;
  `UntilExhaustedStacker` layout invariants (centroid y=0, base-to-base, abutting
  non-overlapping columns, centered wall, top-only graspability) on synthetic off-center
  bboxes.
- `debug_scripts/debug_occupancy.py` — 11 graspable tops, unchanged.
- Sim, reduced (`num_frames=1 num_targets=1 path_tracing_spp=8`):
  `clean_datagen.py configs/randomized.yaml` and `configs/mixed.yaml` both exit 0, zero
  `no labeled instances` errors, fresh `obs`/`iid_mask` written. Grasp `z` corrected from
  0.85 (flung) to box-level after the `ComputeUntransformedBound` fix.

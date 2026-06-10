## Keywords / Tags
- datagen
- plan-completed
- blender
- usdz
- dry-run
- debug-tooling
- grasp-frame
- camera-pose
- rendering

# Plan: Blender-based dry-run sanity renderer (complementary to `debug_scene.py`)

**Status:** Complete (2026-06-08)
**Entry point:** `isaac-datagen-debug-render <config.yaml> [overrides]` (run from `src/isaac_datagen/`)
**New modules:** `debug_export.py`, `blender_render.py`, `debug_render.py`
**Touched:** `capture.py`, `hardwares.py`, `runtime_config.py`, `clean_datagen.py`, `pyproject.toml`

> The body below is the plan **as designed/approved**. Reality diverged in a few places (orbit is a
> GIF not an mp4; gizmos cover ALL grasp frames; lighting/exposure tuned; plus three bugs found only
> by running it). See **"Implementation outcome & fixes"** at the end for the authoritative final state.

## Context

`debug_scene.py` rebuilds the reference-seg scene faithfully and exports `scene.usdz`, but the
only way to *look* at that geometry is to open it in a USD/Isaac viewer, and there is no way to
preview **the actual frames the renderer would produce** without running the full RTX capture.

We want an optional, CLI-toggled debug path that:
1. **Orbits a camera around the scene centroid** of the exported USDZ (lit, since the export is
   dark by default) — to inspect the geometry in a lightweight renderer (Blender) instead of Isaac.
2. **Renders the exact planned camera poses** with the dataset's intrinsics — a true sanity check
   that the pose-planning math (`plan_poses` → `einsum` → `world_poses`) aims the camera where we
   think it does.
3. **Visualizes the grasp frames as literal RGB Cartesian axes** at their world poses, so a single
   render answers "are the grasp frames where they should be, and is the camera looking at them?"

Precedent: the original `build_object_dataset.py` (preserved verbatim in
`visual_servoing/datagen2_isaacsim/.docs_claude/plans/completed/build-object-dataset.md`) already
headless-rendered each box USDZ in **Blender 4.2** (`bpy.ops.wm.usd_import`, a `SUN` lamp,
`BLENDER_EEVEE_NEXT`, ortho camera looking down world +Y at the −Y face). Blender 4.2.2 LTS is still
installed at `/usr/local/bin/blender`. We adapt that known-good recipe to the whole-scene USDZ with a
perspective camera.

**Hard constraint (user):** the dry run and the real run **must not drift apart**, and the real
data-generation path must be **provably unaffected**. Both are satisfied by (a) shared, single-source
mechanisms both paths call, and (b) scoping all scene mutation (baked cameras + axis gizmos) to the
**dry-run-only** export helper — the real run never executes that code.

### Why baking cameras into the USDZ is safe (the VRAM question, resolved)

- VRAM is consumed by **render products** (per-product GBuffers/annotator tensors), *not* by `Camera`
  prims. `setup_render_product` / Replicator create exactly **two** render products
  (`zed.left_rp`, `zed.right_rp`); `ObsMaskWriter.attach` even uses only `rps[0]`.
- A baked `Camera` prim with **no render product** is inert scene-graph metadata.
- Decisive: baking is **dry-run-only**. The real `reference_segmentation()` render path never calls
  the baking helper, so there are never extra prims *or* render products at real-capture time.

Sources:
- [Rendering 20+ cameras without running out of memory — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/2022-2-1-rendering-20-cameras-without-running-out-of-memory/266877)
- [Isaac Sim Performance Optimization Handbook](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/sim_performance_optimization_handbook.html)

## Decisions (from review)

| Fork | Decision |
|---|---|
| Where the Isaac-side export lives | `dry_run` config flag on the **real** `reference_segmentation()`, which **delegates** the built scene + planned poses to a dedicated Isaac-side helper module. Shared mechanisms guarantee no drift. |
| How poses reach Blender | **Bake `Camera` prims into the exported USDZ**, *dry-run only* (convention-free; reuses `setup_camera`). Blender imports scene + cameras through one importer transform and renders from each. |
| Render modes | **Orbit/turntable** + **Pose-sanity (left camera)**. (No side-by-side.) |
| Extra | Bake **RGB axis gizmos at every grasp frame** into the USDZ. |
| Pose-setting | One shared SE3→prim-pose mechanism (path-string addressed) reused by `move_prims` and the dry run, so the transform calls can't diverge. |
| K + camera offset | Owned by the `ZedMini` object (property/classvar), not re-loaded from disk or re-derived in the helper. |
| Code shape | `debug_export.py` split into a scene-decoration **mechanism** and an export **policy**. |

## Implementation (as designed)

### 1. Shared mechanisms (single source of truth — no drift) — `capture.py`, `hardwares.py`
- **`plan_capture(runtime, scene, rng)`** — the scene/pose-planning prefix lifted out of
  `clean_datagen.py` so both the real run and the dry run call identical code. Returns
  `(idx, grasp_points, world_poses)`.
- **`se3_to_pos_euler(pose)`** — THE single SE3 decomposition. **`set_prim_pose(prim_path, pose)`** —
  prim-path-addressed applier via `set_transform`. `move_prims` refactored onto `se3_to_pos_euler`.
  Consistency is structural: `set_transform` and `rep.modify.pose` author the same USD ops
  (`xformOp:translate` + `xformOp:rotateXYZ`), so the same euler triple ⇒ identical world transform.
- **`ZedMini`**: `intrinsics` property + `LEFT_CAM_OFFSET` classvar / `left2rig` property; `__init__`
  places the cameras from the offset constants (one definition for rig geometry + debug offset).

### 2. `dry_run` flag — `runtime_config.py`
`dry_run: bool = False` on `RuntimeConfig` (dotlist `dry_run=true`).

### 3. Mechanism + policy — `debug_export.py`
`decorate_debug_scene(...)` bakes left-camera prims (at the planned poses) + RGB axis gizmos into the
live stage and returns the decorated scene (no I/O). `export_debug_bundle(info, render_dir)` persists
it (`scene.usdz` via `export_subtree_usdz`, `dryrun.npz`/`dryrun.json`). Dry-run only.

### 4. Blender renderer — `blender_render.py` (no Isaac deps)
`blender --background --python blender_render.py -- <debug_dir>`: import USDZ (geometry + baked
cameras), light, pose-sanity render per baked camera (intrinsics from the dumped K), orbit turntable.

### 5. Driver — `debug_render.py` (console script `isaac-datagen-debug-render`)
Chains `isaac-datagen ... dry_run=true` then Blender (subprocess-per-phase like `run_pipeline.py`, so
Isaac frees the GPU on exit).

---

## Implementation outcome & fixes (2026-06-08)

Built and **verified end-to-end on real hardware** (RTX 4090). Demo bundle:
`src/isaac_datagen/debug/render097/debug/` (`scene.usdz`, `dryrun.{npz,json}`, `poses/*.png`,
`orbit.gif`). The real render path is byte-for-byte unchanged (only the extracted `plan_capture`
call differs; `move_prims`' decomposition is relocated, not changed).

### Bugs found only by running it
1. **Boxes collapsed to the origin in Blender.** Blender's USD importer **skips untyped prims** when
   building its object hierarchy, severing the box transform chain at the untyped `geo`
   reference-wrapper child (the [referenced-prim-transform bug] fix puts the reference on an untyped
   child). All 44 boxes imported at (0,0,0). The placement is *correct in the USDZ* (verified via pure
   USD: `amazon_0` world z=0.106, `amazon_1` z=0.23, …). **Fix:** `_retype_untyped_for_blender(stage)`
   types every untyped prim `Xform` before export (semantically neutral, dry-run only). → 44 boxes at
   44 distinct positions on import.
2. **UsdShade material bindings don't survive `export_subtree_usdz`'s reference-flatten** (bindings
   are relationships, dropped across the single-reference arc), so the colored axes imported white.
   **Fix:** color the axes **Blender-side** by cylinder prim name (`x`/`y`/`z`) with emission;
   `displayColor` kept on the cylinders as a fallback for native USD viewers.
3. **Glare / over-bright renders.** Blender's default **AgX** view transform has a long highlight
   latitude, so the near-white box cardboard kept glaring at `exposure -1`/`-2.5`. **Fix:** push
   `scene.view_settings.exposure` to **`-5.0`** (default); `--exposure`/`--sun-energy`/`--ambient`
   are CLI-tunable. Confirmed the box materials are *not* emissive — it was purely exposure vs. AgX.

### Divergences from the plan
- **Orbit is a GIF, not an mp4.** `render_orbit` renders frames to a temp folder, muxes them into a
  single `orbit.gif` via ffmpeg's built-in (GPL-free) GIF encoder (two-stage palette filter), then
  removes the frame folder. Reason: the workstation's conda ffmpeg is `--disable-gpl` (no `libx264`),
  and the user asked for a GIF over a PNG folder. **Pose-sanity stays a folder of individual PNGs**
  (those are per-pose frames you inspect one at a time). The driver no longer does any encoding.
- **Gizmos cover ALL candidate grasp frames, not just sampled targets.** `decorate_debug_scene`
  signature changed `(scene, grasp_points, world_poses)` → `(scene, world_poses)`; it now bakes an
  axis gizmo on every `scene.grasp_points`. The baked cameras still mark which targets the capture
  actually uses. For `pallet_dims=[11,1,4]` that's **11** grasp frames (the top row — `is_top AND
  is_front`; a 1-deep wall only exposes its top layer). Note: a pose-sanity close-up only frames ~5
  columns, so count all 11 in `orbit.gif`.
- **Gizmo size** bumped to `length=0.10, radius=0.006` (from 0.08/0.004) for visibility.
- **Lighting defaults:** `sun=1.0, ambient=0.25, exposure=-5.0` (plan had a single sun + ambient 1.0).
- **`clean_datagen.py`** unpacks `_idx, _grasp_points, world_poses = plan_capture(...)` (only
  `world_poses` is consumed now that gizmos use `scene.grasp_points`).

### Confirmed working / non-issues
- **Camera baking is convention-free** as designed: Blender's USD importer applies its up-axis
  conversion (`Blender_y=−USD_z`, `Blender_z=USD_y`) **uniformly** to scene + baked cameras + gizmos,
  so a baked left-camera lands exactly at its planned pose (verified) and the relative view is faithful.
- **Real path provably unaffected** (dry-run-only baking; the real branch never imports `debug_export`).
- **`num_targets` oversampling is visible:** with the real config (`num_targets=44`, ~11 grasp
  points, `rng.choice` with replacement) you get 44 baked cameras collapsing onto ~11 boxes — exactly
  the oversampling `debug_scene.py` was built to detect.

### Follow-ups (not done)
- Case study written into the `separate-mechanism-policy` and `reusable-parts` skills (the
  decorate/export mechanism-vs-policy split + the single-source extractions).
- Gotchas saved to auto-memory (`blender-usdz-import-gotchas`).
- `debug_scene.py` still has its own hand-mirrored scene rebuild; could be pointed at `plan_capture`
  to remove that second drift source (out of scope here).

[referenced-prim-transform bug]: ../../../../visual_servoing/datagen2_isaacsim/.docs_claude/isaacsim_referenced_prim_transform_bug.md

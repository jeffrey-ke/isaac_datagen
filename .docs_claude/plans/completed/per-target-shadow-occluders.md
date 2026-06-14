# Plan: per-target invisible shadow occluders (DR)

> **STATUS (2026-06-13): IMPLEMENTED, render-test pending lighting.** Code landed in `runtime_config.py`,
> `configs/randomized.yaml`, `scene.py` (`add_shadow_occluders` + `build_scene` call). Statically verified
> (syntax, schema, config resolve, occluder poser `poser(1)→(1,4,4)`). Test 2 (end-to-end render) is blocked
> until the key light is restored and the ~60% black-render bug is resolved — see "Update these" below.

## Context
Randomized **cast shadows of shapes** for sim2real robustness. No native shadow randomizer exists —
compose it from per-target invisible 3D occluders whose path-traced shadows fall on the boxes.
- **Placement is the whole feature** → lives entirely in `build_scene`. For *every* grasp target, create
  occluder(s) and position each **once**: one occluder-Poser sample in the target frame, mapped to world
  via that target's `target2world` (`get_target2world`, capture.py:14), authored with `set_prim_pose`
  (capture.py:67). Occluders are **static** for the render — NO per-frame pose sequencing.
- **Per-frame shadow variation is free**: the camera moves through `num_frames` viewpoints per target and
  the key light jitters per frame (`register_distant_jitter`), so a static occluder's shadow already
  changes frame-to-frame. Variety across renders comes from the (unseeded, like the camera) occluder Poser.
- Invisible to camera via `primvars:hideForCamera=True` (confirmed token, used by Replicator at
  `asset_cache.py:132`); no semantic label → shadow lands in `obs`, occluder never enters RGB/masks.

**ASSUMPTION (per user): lighting is fixed** — a directional (`DistantLight`) / local (`SphereLight`) key
light is present in `build_scene`, so occluders cast crisp shadows and renders aren't black. (The working
tree is dome-only for the dark-box debug — see "Update these" below; a pure dome casts no crisp shadows,
so the feature is meaningless until the key light is restored.)

Tested via the existing `ObsMask.visualize(md)`.

## Changes (before → after)

### `runtime_config.py` — occluder config (after `pose_generation_policy_args`, line 100; `field` already imported)
```python
    pose_generation_policy_args: dict = field(default_factory=dict)
+   occluders_per_target: int = 0                       # invisible shadow occluders per target (0 = off)
+   occluder_pose_policy: str = "GridFixedPoser"        # reuses posers.py registry
+   occluder_pose_policy_args: dict = field(default_factory=dict)
```

### `configs/randomized.yaml` — occluder block (dedicated Poser; TARGET-FRAME ranges place occluder toward the key light)
```yaml
+occluders_per_target: 2
+occluder_pose_policy: GridFixedPoser
+occluder_pose_policy_args:          # offset between key light and box — TUNE on first lit render
+  xrange: [0.10, 0.40]
+  yrange: [-0.15, 0.15]
+  zrange: [-0.15, 0.15]
+  target_to_ego_ypr: [90, 0, 90]
+  random: true
```

### `scene.py` — occluder builder (new) + one `build_scene` call. No SceneHandle change, no capture/clean_datagen changes.
```python
SHADOW_SHAPES = ("Cube", "Cone", "Cylinder", "Sphere")

def add_shadow_occluders(stage, parent, grasp_frames, runtime, rng):
    """Place runtime.occluders_per_target invisible occluders per grasp target. Each casts a path-traced
    shadow on its box but is hidden from the camera (primvars:hideForCamera=True) and carries NO semantic
    label, so only its shadow reaches obs. Positioned ONCE: an occluder-Poser sample in the target frame
    @ that target's target2world. Call AFTER the stack is positioned so target2world is final."""
    from pxr import Sdf, UsdGeom
    from isaacsim.core.utils.prims import create_prim
    from isaac_datagen.capture import get_target2world, set_prim_pose
    poser = posers.get(runtime.occluder_pose_policy)(**runtime.occluder_pose_policy_args)
    target2worlds = get_target2world(grasp_frames)                       # (M,4,4)
    create_prim(f"{parent}/ShadowOccluders", "Xform")
    for ti, t2w in enumerate(target2worlds):
        for k in range(runtime.occluders_per_target):
            path = f"{parent}/ShadowOccluders/t{ti:03d}_occ{k}"
            s = float(rng.uniform(0.04, 0.2))
            create_prim(path, SHADOW_SHAPES[(ti + k) % len(SHADOW_SHAPES)], scale=(s, s, s))
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
                "hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)     # doNotCastShadows left unset
            set_prim_pose(path, t2w @ poser(1)[0])                       # target-frame sample → world
```
`scene.py` already imports `posers`? No — add `from isaac_datagen import posers` (top). `set_prim_pose`
authors translate+rotateXYZ only, so the `create_prim(scale=…)` scale persists.
```python
# build_scene, right after the stack is positioned:
    set_transform(get_current_stage().GetPrimAtPath(stack_path), translation=(0.1, 0.1, 0.045))
+   if runtime.occluders_per_target:
+       add_shadow_occluders(stage, "/World", grasp_frames_paths, runtime, rng)
```

## Files
- `runtime_config.py` — 3 fields.
- `configs/randomized.yaml` — occluder block.
- `scene.py` — `posers` import, `SHADOW_SHAPES`, `add_shadow_occluders`, one `build_scene` call.

## Verify
1. **Blender (placement):** `clean_datagen.py configs/randomized.yaml idx=0 dry_run=true occluders_per_target=2`
   → open `scene.usdz`; occluders are placed in `build_scene` so they appear in the export. Confirm each sits
   between the key light and its box. (Blender ignores `hideForCamera`, so they're visible there — placement
   check only.)
2. **Render (real test):** `uv run clean_datagen.py configs/randomized.yaml num_targets=2 num_frames=4 occluders_per_target=2`,
   then load frames with `ObsMask.visualize(md)`. Must show: shadows on the boxes in `obs`, **no** visible
   occluder shape (→ `hideForCamera` honored under PathTracing — the one renderer unknown), shadow regions
   absent from the instance/class mask overlays.

## ⚠️ Update these — assertions that lighting is NOT fixed (flagged for you)
This plan assumes a key light exists; these currently say the opposite — reconcile when the production
lighting recipe lands:
- **`scene.py:334-335`** — `make_sphere_light` / `make_distant_light` **commented out** in `build_scene`
  (the key light the shadows need). Restore a directional/local light.
- **`scene.py:171-192` `make_replicator`** — docstring "Dome-only lighting randomizer for the dark-box
  debug"; `register_distant_jitter` dormant (the intended shadow-direction source).
- **`runtime_config.py:105-119`** — "Lighting diagnostics … debug scene is dome-only", `jitter_dome=False`,
  `dome_normalize=False`.
- **`.docs_claude/plans/active/render-darkness-investigation.md`** — Bug 2 (~60% all-black renders) marked
  unresolved; if truly fixed, update its STATUS, else Test 2 here needs detect-and-retry.
- **`.docs_claude/plans/active/lighting-diagnostic-dark-box-flags.md`** — documents the sphere+distant ablation.

## Risks
- **Occluder Poser ranges** need empirical tuning so the occluder sits between the key light and the box —
  the shadow must land *on* the box. Tune on the first lit render.
- **`hideForCamera` under PathTracing** — settled by Test 2; fallback is drop the flag + place occluders
  outside the camera frustum.
- **Ordering** — `add_shadow_occluders` must run after the stack `set_transform`, or `target2world` is stale.

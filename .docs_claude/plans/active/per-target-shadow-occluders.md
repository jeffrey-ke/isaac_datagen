# Plan: per-target invisible shadow occluders (DR)

## Context
Randomized **cast shadows of shapes** for sim2real robustness. No native shadow randomizer exists —
compose it: per-target invisible 3D occluders whose path-traced shadows land on the boxes. Two pieces,
each reusing existing mechanism:
- **Placement** is prim-building → lives in `build_scene`: for *every* grasp target, add occluder(s).
- **Pose jitter** reuses the **Poser API** + the camera's target-frame→world path: each occluder's
  world pose = `its target2world @ occluder-Poser(frame)`, applied as a precomputed sequence through
  `move_prims` (NOT a `rep.distribution` randomizer, NOT following the camera). Occluders stay pinned
  to their own target; jitter is sampled in the target frame.
- Invisible to camera via `primvars:hideForCamera=True` (confirmed token, used by Replicator at
  `asset_cache.py:132`); no semantic label → shadow lands in `obs`, caster never enters RGB/masks.

Tested via the existing `ObsMask.visualize(md)` (added this session).

## Changes (before → after)

### `runtime_config.py` — occluder config (mirrors `pose_generation_policy`, lines 97-100)
```python
# after pose_generation_policy_args
    pose_generation_policy_args: dict = field(default_factory=dict)
+   occluders_per_target: int = 0                       # invisible shadow occluders per target (0 = off)
+   occluder_pose_policy: str = "GridFixedPoser"        # reuses posers.py registry
+   occluder_pose_policy_args: dict = field(default_factory=dict)
```

### `configs/randomized.yaml` — occluder block (dedicated Poser, ranges place occluder toward the light)
```yaml
+occluders_per_target: 2
+occluder_pose_policy: GridFixedPoser
+occluder_pose_policy_args:          # TARGET-FRAME box between light and box — TUNE on first render
+  xrange: [0.10, 0.40]
+  yrange: [-0.15, 0.15]
+  zrange: [-0.15, 0.15]
+  target_to_ego_ypr: [90, 0, 90]
+  random: true
```

### `scene.py` — `SceneHandle` carries the occluders
```python
# before
@dataclass(frozen=True)
class SceneHandle:
    zed: ZedMini
    grasp_points: list
    objects: List[GraspableObject]
# after  (import: from dataclasses import dataclass, field)
    objects: List[GraspableObject]
+   occluders: list = field(default_factory=list)   # (occluder_path, target_grasp_path) pairs
```

### `scene.py` — occluder builder + `build_scene` wiring (after grasp frames, ~line 319)
```python
SHADOW_SHAPES = ("Cube", "Cone", "Cylinder", "Sphere")

def add_shadow_occluders(stage, parent, grasp_frames, per_target, rng):
    """For each grasp target add `per_target` invisible occluders: primvars:hideForCamera=True
    (invisible to camera, still casts a path-traced shadow), NO semantic label, random fixed scale.
    Returns (occluder_path, target_grasp_path) pairs; per-frame world pose is set later from the
    target's target2world @ occluder Poser."""
    from pxr import Sdf, UsdGeom
    from isaacsim.core.utils.prims import create_prim
    create_prim(f"{parent}/ShadowOccluders", "Xform")
    pairs = []
    for ti, gf in enumerate(grasp_frames):
        for k in range(per_target):
            path = f"{parent}/ShadowOccluders/t{ti:03d}_occ{k}"
            s = float(rng.uniform(0.04, 0.2))
            create_prim(path, SHADOW_SHAPES[(ti + k) % len(SHADOW_SHAPES)], scale=(s, s, s))
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
                "hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)   # doNotCastShadows left unset
            pairs.append((path, gf))
    return pairs
```
```python
# build_scene, after: grasp_frames_paths = [add_grasp_frame(p) for p in graspable_paths]
+   occluders = (add_shadow_occluders(stage, "/World", grasp_frames_paths,
+                runtime.occluders_per_target, rng) if runtime.occluders_per_target else [])
    ...
-   return SceneHandle(zed=zed, objects=objects, grasp_points=grasp_frames_paths)
+   return SceneHandle(zed=zed, objects=objects, grasp_points=grasp_frames_paths, occluders=occluders)
```

### `capture.py` — `plan_occluders` (new) + `capture_with_poses` extra prims
```python
def plan_occluders(runtime, scene):
    """World pose sequence per occluder: its target2world @ occluder-Poser(frame). Jitter is sampled
    in the TARGET frame then mapped to world — same path as the camera (get_target2world + move_prims).
    Each occluder is pinned to its own target; it does NOT follow the camera. Returns (paths, seqs)."""
    if not scene.occluders:
        return [], []
    occ_paths = [op for op, _ in scene.occluders]
    target2worlds = get_target2world([tp for _, tp in scene.occluders])        # (M,4,4)
    poser = posers.get(runtime.occluder_pose_policy)(**runtime.occluder_pose_policy_args)
    n_frames = runtime.num_targets * runtime.num_frames                        # == len(camera world_poses)
    seqs = [t2w @ poser(n_frames) for t2w in target2worlds]                    # each (n_frames,4,4)
    return occ_paths, seqs
```
```python
# capture_with_poses — before
def capture_with_poses(world_poses, writer, camera, replicator):
    ...
        with rep.trigger.on_frame():
            move_prims([rig_node], [world_poses], replicator)
            replicator.apply_randomizers()
# after  (move_prims already loops over multiple prims/sequences)
def capture_with_poses(world_poses, writer, camera, replicator, extra_prims=(), extra_pose_seqs=()):
    ...
    extra_nodes = [rep.get.prim_at_path(p) for p in extra_prims]
        with rep.trigger.on_frame():
            move_prims([rig_node, *extra_nodes], [world_poses, *extra_pose_seqs], replicator)
            replicator.apply_randomizers()
```

### `clean_datagen.py` — `reference_segmentation` wiring
```python
# before
    _idx, _grasp_points, world_poses = plan_capture(runtime, scene, rng)
    ...
    capture_with_poses(world_poses, writer, scene.zed, replicator)
# after  (import plan_occluders alongside plan_capture)
    _idx, _grasp_points, world_poses = plan_capture(runtime, scene, rng)
    occ_paths, occ_seqs = plan_occluders(runtime, scene)
    ...
    capture_with_poses(world_poses, writer, scene.zed, replicator,
                       extra_prims=occ_paths, extra_pose_seqs=occ_seqs)
```

## Files
- `runtime_config.py` — 3 fields. `configs/randomized.yaml` — occluder block.
- `scene.py` — `SHADOW_SHAPES`, `add_shadow_occluders`, `SceneHandle.occluders`, `build_scene` wiring.
- `capture.py` — `plan_occluders`, `capture_with_poses(extra_prims, extra_pose_seqs)`.
- `clean_datagen.py` — `reference_segmentation` threads occluders into capture.

## Verify
1. **Blender (placement):** `clean_datagen.py configs/randomized.yaml idx=0 dry_run=true occluders_per_target=2`
   → open `scene.usdz`; confirm occluders sit between lights and boxes. (Blender ignores `hideForCamera`,
   so they're visible there — placement check only.)
2. **Render (real test):** `uv run clean_datagen.py configs/randomized.yaml num_targets=2 num_frames=4 occluders_per_target=2`,
   then load frames with `ObsMask.visualize(md)` (optionally a new `viz_obsmask.py` CLI, 3-pane sibling of
   `viz_occlusion.py`). Must show: shadows on the boxes in `obs`, **no** visible occluder shape
   (→ `hideForCamera` honored under PathTracing — the one unknown), shadow regions absent from the
   instance/class mask overlays.

## Risks
- **`hideForCamera` under PathTracing** — settled by step 2. Fallback if occluders show: drop `hideForCamera`
  and shift the occluder Poser ranges out of the camera frustum.
- **Occluder Poser ranges** need empirical tuning so the occluder sits between the (overhead) light and the
  box — the shadow must land *on* the box. Tune on the first render.
- **Count/perf** — occluders = (#grasp targets × `occluders_per_target`), each a (num_targets·num_frames)
  pose sequence. Keep `occluders_per_target` small; raise only if more shadow density is needed.

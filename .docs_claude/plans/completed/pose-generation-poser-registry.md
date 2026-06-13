# Pose-generation poser registry

## Context

Camera-pose generation in `plan_capture` is hardcoded to `plan_poses` (fixed-rotation grid/random
sampling). We want to **separate the policy of which poses to generate from the mechanism that
generates them**: introduce named, stateful poser callables selected by config (registry style,
mirroring the optimizer registry in `segmentation/optim.py` — `name` string + `args` dict).

Two posers ship:
- **`GridFixedPoser`** — returns exactly what `plan_poses` returns today (one fixed orientation for
  every camera; position varies over the halo box). Behavior-preserving.
- **`LookAtPoser`** — same random position sampling, but each camera *looks at* the target origin, so
  orientation varies per frame (uses `vision_core.pose_utils.look_at`, which already returns a
  camera2target SE3 with translation = the camera offset).

Both entry points route pose generation through the registry. The `reference_segmentation` path
already uses the modular `plan_capture()` + `capture_with_poses()` split (extracted in commit
`1313cc0`). The stereo `make_index()` is the *old* fused monolith that same refactor superseded — it
welds planning + writer construction + capture into one function and even re-derives `target2world`
that `main()` already computed. **It is deprecated: delete it and inline its decomposed body into
`main()`** (poser → `world_poses` → `StereoSampleWriter` → `capture_with_poses`), so the stereo path
reads like `reference_segmentation()`.

Top-level `xrange/yrange/zrange` stay as the single source of truth (also read by `scene.py` light
placement); configs feed them to the poser via OmegaConf interpolation.

## Reused, not rebuilt

- `vision_core.pose_utils.look_at(at_coord, from_coord)` → `(4,4)` camera2target SE3, translation = `from_coord`. Exactly what `LookAtPoser` needs.
- `vision_core.pose_utils.generate_random_offsets(xrange, yrange, zrange, num)` → `(N,3)`.
- `isaac_datagen.pose_planning.plan_poses(...)` — kept; `GridFixedPoser` wraps it.
- `get(name)` registry shape copied from `segmentation/segmenter.py:247` (`getattr(sys.modules[__name__], name)`).

---

## 1. `posers.py` (new — replaces the pseudocode stub)

```python
"""Camera-pose generation policy registry.

Stateful callables that, given a frame count, return (N, 4, 4) camera2target SE3
poses in the grasp-target frame. Selected by name from a config block
(`pose_generation_policy` + `pose_generation_policy_args`), mirroring the
optimizer registry in segmentation/optim.py.
"""
from __future__ import annotations

import sys
import numpy as np

from vision_core.pose_utils import generate_random_offsets, look_at
from isaac_datagen.pose_planning import plan_poses


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


class GridFixedPoser:
    """Fixed-rotation poser: exactly the poses plan_poses used to return. Every
    camera shares one orientation (target_to_ego_ypr); only position varies over
    the halo box. random=True samples num_frames positions; random=False lays a
    fixed grid (grid_dims) and takes the first num_frames."""

    def __init__(self, xrange, yrange, zrange, target_to_ego_ypr,
                 grid_dims=None, random: bool = True):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange
        self.target_to_ego_ypr = target_to_ego_ypr
        self.grid_dims = grid_dims
        self.random = random

    def __call__(self, num_frames: int) -> np.ndarray:
        if self.random:
            return plan_poses(self.target_to_ego_ypr, self.xrange, self.yrange,
                              self.zrange, num_frames)
        assert self.grid_dims is not None, "GridFixedPoser(random=False) needs grid_dims"
        return plan_poses(self.target_to_ego_ypr, self.xrange, self.yrange,
                          self.zrange, tuple(self.grid_dims))[:num_frames]


class LookAtPoser:
    """Look-at poser: each camera sits at a random halo-box offset and is oriented
    to face the target origin (look_at returns a camera2target SE3, translation =
    the offset). Orientation varies per pose, unlike GridFixedPoser's fixed ypr."""

    def __init__(self, xrange, yrange, zrange):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange

    def __call__(self, num_frames: int) -> np.ndarray:
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange, num_frames)
        return np.array([look_at(np.zeros(3), off) for off in offsets])
```

> **Run-blocking detail filled in:** the pseudocode omitted `grid_dims` from `GridFixedPoser.__init__`;
> it's required for the `random=False` branch, so it's added (unused when `random=True`).

---

## 2. `runtime_config.py` — add the registry fields

```python
# BEFORE
from dataclasses import dataclass
...
    target_to_baseline_ypr_desired: tuple[float, float, float] = (90, 0, 90)

# AFTER
from dataclasses import dataclass, field
...
    target_to_baseline_ypr_desired: tuple[float, float, float] = (90, 0, 90)

    # Pose-generation policy registry (posers.py): name a poser class, pass its
    # ctor kwargs verbatim. Mirrors segmentation OptimConfig (name + args).
    pose_generation_policy: str = "GridFixedPoser"
    pose_generation_policy_args: dict = field(default_factory=dict)
```

`num_frames` / `grid_dims` / `sampling` / `__post_init__` are **unchanged** (`sampling` no longer
read by the migrated paths but kept harmless).

---

## 3. `capture.py` — registry in `plan_capture`, **delete `make_index`**

```python
# BEFORE (imports)
from isaac_datagen.pose_planning import plan_poses
# AFTER  (plan_poses no longer used here — it now lives behind GridFixedPoser)
from isaac_datagen import posers
```

```python
# plan_capture — BEFORE
    target_frame_poses = plan_poses(                                     # (N, 4, 4)
        runtime.target_to_baseline_ypr_desired,
        runtime.xrange, runtime.yrange, runtime.zrange, runtime.sampling,
    )
# plan_capture — AFTER
    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)                       # (N, 4, 4)
```

**Delete the entire `make_index` function** (capture.py:159–176). Its responsibilities are now the
modular building blocks — `poser(num_frames)` (planning) + `StereoSampleWriter` (writer) +
`capture_with_poses` (capture) — composed directly in `main()` below. This is the same decomposition
`reference_segmentation()` already received.

---

## 4. `clean_datagen.py` — inline the decomposed stereo flow into `main()`

```python
# imports — drop make_index, add posers
from isaac_datagen.capture import get_target2world, capture_with_poses, plan_capture
from isaac_datagen import posers
```

```python
# main() — BEFORE
    target2world = get_target2world(grasp_point)
    replicator = make_replicator(runtime, target2world)

    make_index(
        runtime.target_to_baseline_ypr_desired,
        runtime.xrange, runtime.yrange, runtime.zrange,
        runtime.sampling, grasp_point, scene.zed, replicator, render_dir,
    )
# main() — AFTER  (mirrors reference_segmentation(): plan → writer → capture)
    from isaac_datagen.stereo_writer import StereoSampleWriter

    target2world = get_target2world(grasp_point)            # reused below, no longer re-derived
    replicator = make_replicator(runtime, target2world)

    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)          # (N, 4, 4)
    world_poses = target2world @ target_frame_poses
    offsets = [pose[:3, 3].tolist() for pose in target_frame_poses]

    stereo_writer = StereoSampleWriter(output_dir=str(render_dir),
                                       offsets=offsets, target2world=target2world)
    capture_with_poses(world_poses, stereo_writer, scene.zed, replicator)
```

(The single `target2world` is now computed once and shared with both `make_replicator` and the
writer — `make_index` redundantly recomputed it.)

---

## 5. Configs — `randomized.yaml`, `mixed.yaml`

`pose_generation_policy_args` defaults to `{}`, so each config must supply the block. Ranges are
interpolated from the existing top-level fields (no duplication; `scene.py` still owns them):

```yaml
# append after the xrange/yrange/zrange block
pose_generation_policy: GridFixedPoser
pose_generation_policy_args:
  xrange: ${xrange}
  yrange: ${yrange}
  zrange: ${zrange}
  target_to_ego_ypr: ${target_to_baseline_ypr_desired}
  random: true
```

To switch a config to look-at framing: `pose_generation_policy: LookAtPoser` with args
`{xrange: ${xrange}, yrange: ${yrange}, zrange: ${zrange}}`.

---

## Verification

1. **Config load** (fast, no sim): `uv run python -c "from isaac_datagen.runtime_config import load_config; c=load_config('src/isaac_datagen/configs/randomized.yaml', []); print(c.pose_generation_policy, c.pose_generation_policy_args)"` — confirms interpolation resolves to concrete ranges.
2. **Poser parity** (unit): assert `GridFixedPoser(**args)(N)` equals the old `plan_poses(ypr, xr, yr, zr, N)` under a fixed `np.random.seed` — behavior preserved.
3. **LookAtPoser shape/orientation**: `LookAtPoser(xr, yr, zr)(N)` → `(N,4,4)`; spot-check that each pose's translation is the sampled offset and `z_axis ≈ normalize(origin − offset)` (camera faces target). Confirm no NaN (current ranges keep x ≥ 0.35, so the look direction is never parallel to world up).
4. **Dry-run render** (visual, end-to-end): `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml idx=0 dry_run=true` for both posers → exports `scene.usdz` + baked debug cameras; open in Blender to confirm GridFixedPoser cameras match the pre-refactor layout and LookAtPoser cameras all point at the grasp target. If LookAtPoser orientation is flipped vs the renderer's camera convention, the fix is a fixed axis flip (`vision_core.pose_utils.cv2opengl`) inside `LookAtPoser.__call__`.

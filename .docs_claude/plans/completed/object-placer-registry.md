# Plan: object-placement policy registry

## Context

`create_stack_of_objects` (scene.py) currently hard-codes an `if/elif` on `runtime.placement`
(`"occupancy_grid"` vs `"until_exhausted_stacker"`): each branch validates the object count, adds
the objects, and constructs a placement policy. The user started rewriting it (see scene.py:71–80,
now non-parseable pseudocode) toward a **name+args registry**, exactly like `posers.py` /
`filters.py` / segmentation's optimizer registry: one `add_object` loop shared by all policies, then
`policy = placers.get(runtime.placement)(prim_paths_added, **runtime.placement_args)`.

Two facts from exploration shape the plan:
- **`OccupancyGrid` is retired as a placement policy but the class stays.** It's still imported by the
  (callerless, dead) `LoadedPallet` and by `debug_scripts/debug_occupancy.py`. Decision: remove it
  only from the placement path, leave the class in `objects.py`.
- **`clean_datagen.py:144` is a broken half-edit:** `build_scene(runtime, objects, runtime.policy)`
  passes a 3rd arg `build_scene` doesn't accept and reads a nonexistent `runtime.policy`. Revert it.

Outcome: adding a new placement policy = define a class in `placers.py` (contract:
`__init__(prim_paths, **kwargs)`, `__call__(prim_path) -> (translation, rotation)`,
`graspability() -> dict[path, bool]`) and name it in config. No edits to `scene.py`'s dispatch.
(Originally the placer classes lived in `objects.py` and `placers.py` imported them; they were later
**moved into `placers.py`** so the registry's home also defines its entries — see §1b. Registration
is now just defining the class there.)

## Decisions (confirmed)

- Keep the `OccupancyGrid` class; drop it only from `create_stack_of_objects`/the registry.
- Key the registry by **class name** (`placement: UntilExhaustedStacker`), mirroring `posers.py`.
- Registry module named `placers.py` (parallel to `posers.py`); the user's pseudocode wrote
  `policies.get(...)` — `placers` is the same idea, renamed for consistency.

## The placer contract (already satisfied by `UntilExhaustedStacker`)

`UntilExhaustedStacker` (now in `placers.py`; originally `objects.py:379`) already satisfies the
contract as-is — **no ctor change**:
`__init__(self, prim_paths, column_height)`, `__call__(self, prim_path) -> (translation, rotation)`,
`graspability() -> dict`.

Per your note: **no default arg.** `column_height` stays required. Every config names it explicitly
in `placement_args`, so (a) the value is always visible in config, and (b) a config that forgets it
fails loudly — `placers.get(...)(prim_paths_added, **{})` → `TypeError: missing required positional
argument 'column_height'` at scene-build time — rather than silently stacking at a hidden 5. This
removes the old default that lived on `RuntimeConfig.column_height = 5` (deleted in §4) rather than
relocating it into the ctor.

## Changes

### 1. New file `src/isaac_datagen/placers.py` (mirrors `posers.py:1-22`)

```python
"""Object-placement policy registry.

Stateful callables that lay out the placed-object prim paths on the stage. Selected by
name from a config block (`placement` + `placement_args`), mirroring posers.py and the
optimizer registry in segmentation/optim.py.

Placer contract:
  __init__(self, prim_paths, **kwargs)                  # measure bboxes, precompute layout
  __call__(self, prim_path) -> (translation, rotation)  # per-prim, in the stack frame
  graspability(self) -> dict[str, bool]                 # per-prim graspable flag
"""
from __future__ import annotations

import sys
from collections import deque

from isaac_datagen.isaac_utils import local_bbox_range


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e

# ... followed by the UntilExhaustedStacker and ShelfPlacer class definitions (see §1b).
```

> Final state shown. As first landed, `placers.py` instead held
> `from isaac_datagen.objects import UntilExhaustedStacker  # noqa: F401 — registry entry` and defined
> no classes; §1b moved the classes in.

### 1b. Follow-up — placer classes moved into `placers.py`

After the registry landed (and `ShelfPlacer` was added in the sibling `shelf-placer.md`), the
registered placer classes were moved out of `objects.py` into `placers.py`, so the registry's home
also *defines* its entries (no import-to-register step a rebase can silently drop — which is exactly
what happened to `ShelfPlacer`). Net:

- `placers.py` now **defines** `UntilExhaustedStacker` and `ShelfPlacer` (verbatim move) and imports
  `collections.deque` + `isaac_utils.local_bbox_range` directly; the `from isaac_datagen.objects
  import …` line is gone.
- `objects.py` drops those two classes and the now-unused `deque` / `local_bbox_range` imports.
- `OccupancyGrid` and `LoadedPallet` **stay** in `objects.py` (retired-from-registry / warehouse
  domain — unchanged from the original decision).
- Dependency now flows one way: `scene.py → placers.py → isaac_utils`; `placers.py` no longer imports
  `objects.py`.

### 2. `scene.py` — registry dispatch + drop dead helper

- **Import line 14** — drop the now-indirect classes, add the registry:
```python
# before
from isaac_datagen.objects import OccupancyGrid, UntilExhaustedStacker, GraspableObject
# after
from isaac_datagen.objects import GraspableObject
from isaac_datagen import placers
```
- **Delete `bbox_size_of` (lines 30-34)** — only fed the removed `OccupancyGrid` branch (grep: no
  other callers; `add_grasp_frame` still uses `bounding_half_extents` directly, so keep that import).
- **`create_stack_of_objects` (lines 66-84)** — collapse the if/elif to the shared loop + lookup:
```python
def create_stack_of_objects(parent_path, objects: List[GraspableObject], runtime):
    from isaacsim.core.utils.prims import create_prim
    stack_prim = create_prim(f"{parent_path}/stack", "Xform")
    stack_path = stack_prim.GetPath().pathString

    prim_paths_added = [add_object(at_parent=stack_path, obj=o) for o in objects]
    policy = placers.get(runtime.placement)(prim_paths_added, **runtime.placement_args)

    organize_objects(policy=policy, prim_paths=prim_paths_added)
    is_graspable = policy.graspability()
    return stack_path, prim_paths_added, is_graspable
```
The per-policy count guards go away: `OccupancyGrid`'s full-wall check is gone with the policy, and
`UntilExhaustedStacker.__init__` already raises on `len(prim_paths) < 1`.

### 3. `clean_datagen.py:144` — revert the broken edit

```python
# before
scene = build_scene(runtime, objects, runtime.policy)
# after
scene = build_scene(runtime, objects)
```
(Matches line 91; `build_scene(runtime, objects)` reads `runtime.placement`/`placement_args`
internally via `create_stack_of_objects`.)

### 4. `runtime_config.py` — add `placement_args`, retire policy-specific fields

- Add beside `placement` (mirrors `pose_generation_policy_args`):
```python
placement: str                                       # registry class name, e.g. UntilExhaustedStacker
placement_args: dict = field(default_factory=dict)   # **kwargs into that placer's ctor
```
- **Remove top-level `column_height` field (line 68)** — now lives in `placement_args`.
- **Make `pallet_dims` optional** (line 44): `pallet_dims: tuple[int, int, int] | None = None`
  — no longer required by the live path now that `OccupancyGrid` is off it; still read by debug
  scripts.
- **`__post_init__` (lines 195-197)** — drop both asserts:
  `assert self.placement in (...)` (registry raises `KeyError` on miss, like `posers`) and
  `assert self.column_height >= 1` (field removed; `UntilExhaustedStacker` validates its own).
- Update the `placement` comment block (lines 39-43) to describe the registry instead of the two
  hard-coded names.

### 5. Configs — rename to class name, move `column_height` into `placement_args`, drop `pallet_dims`

`configs/mixed.yaml:6-10`:
```yaml
# before
# Heterogeneous dataset: use the until-exhausted column stacker, not the
# uniform full-wall OccupancyGrid (which sizes every cell off object[0]).
placement: until_exhausted_stacker
column_height: 5
pallet_dims: [11,1,4] # ignored by until_exhausted_stacker (still a required field)
# after
# Heterogeneous dataset: the until-exhausted column stacker (placers.py registry).
placement: UntilExhaustedStacker
placement_args:
  column_height: 5
```
`configs/randomized.yaml:6-7`:
```yaml
# before
placement: until_exhausted_stacker
pallet_dims: [11,1,4]
# after
placement: UntilExhaustedStacker
placement_args:
  column_height: 5   # explicit — no hidden ctor default (was relying on RuntimeConfig's old default)
```

### 6. `debug_scripts/debug_scene.py:65` — guard the now-optional field

```python
# before
lines.append(f"pallet_dims          = {runtime.pallet_dims}  (capacity = {int(np.prod(runtime.pallet_dims))})")
# after
cap = int(np.prod(runtime.pallet_dims)) if runtime.pallet_dims else "n/a"
lines.append(f"pallet_dims          = {runtime.pallet_dims}  (capacity = {cap})")
```

## Reuse / alignment

- Pure mirror of the existing **`posers.py:18-22` `get(name)`** registry (and `filters.py`,
  `segmentation/optim.py`) — thin orchestration, no new abstraction, per `/reusable-parts`.
- `OccupancyGrid`, `LoadedPallet`, `debug_occupancy.py` untouched (kept per decision).
- `temp` (untracked scratch holding the old branches) left as-is.

## Verification

1. **Config loads:** `uv run clean_datagen.py src/isaac_datagen/configs/mixed.yaml dry_run=true idx=0`
   and the same for `randomized.yaml` — exercises `load_config` (placement_args merge, optional
   `pallet_dims`) and `build_scene → create_stack_of_objects → placers.get(...)` through the dry-run
   path (scene built + USDZ exported, no RTX capture).
2. **Registry miss is clean:** `... placement=Nonsense` should raise `KeyError('Nonsense')` from
   `placers.get`, not an `AttributeError` or a silent wrong policy.
2b. **No silent default:** `... 'placement_args={}'` (empty) should raise `TypeError: missing required
   positional argument 'column_height'` at scene-build — confirming the foot-gun is gone, not relocated.
3. **Stacking unchanged:** a real short run
   `uv run clean_datagen.py src/isaac_datagen/configs/mixed.yaml num_frames=2` produces the same
   column layout / graspable set as before the refactor (behavior is identical — only the dispatch
   moved).
4. **debug_scene** still prints without `pallet_dims` set:
   `uv run debug_scripts/debug_scene.py src/isaac_datagen/configs/randomized.yaml`.

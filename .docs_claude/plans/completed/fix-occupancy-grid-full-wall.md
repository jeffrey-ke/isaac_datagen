## Keywords / Tags
- datagen
- isaac-sim
- occupancy-grid
- grasp-points
- reference-segmentation
- bugfix
- usdz-export
- debug-tooling
- pose-halo
- plan-completed

# Plan: fix grasp-point collapse — OccupancyGrid is a static full-wall policy

**Completed 2026-06-02.**

## Symptom

Running `reference_segmentation()` with `num_targets: 5` produced a dataset where
all rendered frames were random poses around **one** box, not 5 distinct targets.
Separately, the render showed a narrow 2-column tower with "one odd box on top"
even though `pallet_dims: [11,1,4]` (44 slots) implies a wide 11×4 wall.

## Root cause

Both symptoms are the same bug: **`OccupancyGrid`'s occupancy was decoupled from
the objects actually placed.**

- `OccupancyGrid.__init__` does `self.grid = np.ones(grid_dims)` (`objects.py:160`)
  — it marks **all 44** slots occupied regardless of object count.
- `reference_segmentation()` passed only **7** objects (`collect_objects(...)[4:11]`),
  and `create_stack_of_objects` placed `objects[:capacity]` → 7 boxes into the first
  7 `sequence` slots (`np.nonzero` walks `k` fastest → column i=0 k=0..3, then
  column i=1 k=0..2: a 2-wide tower, not a wall).
- `is_graspable = is_front AND is_top` (`scene.py`) reads the all-ones grid:
  - `is_front` is always true (pallet is 1 deep, `j==0`), so it never limits.
  - `is_top` is true only at `k=3` (the grid believes every column is full to the
    top, so lower cells always have a phantom cell "above").
- Of the 7 real boxes, only `(0,0,3)` (the green box, `amazon_15`) sits at `k=3`
  → **exactly 1** grasp point. `rng.choice(1, size=5)` → `[0,0,0,0,0]` → all 5
  targets are the same box. Physically-exposed boxes like the top of column i=1
  (`(1,0,2)`) were "buried" under phantom slots that held no real box.

## Decision

`OccupancyGrid` as used by `create_stack_of_objects` **is, by design, a static
full-wall policy** (grid = all-ones, nothing ever removed). The correct fix is to
honor that identity: enforce that the wall is actually full, and supply enough
objects to fill it — *not* to make the grid track partial placement. (A varied
"partially-picked pallet" was considered and rejected for now; `LoadedPallet`
already implements that start-full-then-remove model if needed later.)

## Files & changes

### 1. `src/isaac_datagen/scene.py` — enforce the full-wall contract
`create_stack_of_objects` now raises if under-supplied, so the all-ones grid can
never silently describe phantom boxes again:
```python
capacity = dims[0] * dims[1] * dims[2]
if len(objects) < capacity:
    raise ValueError(
        f"create_stack_of_objects: full-wall pallet {tuple(dims)} needs {capacity} "
        f"objects, got {len(objects)}. Supply more objects or shrink pallet_dims."
    )
```
A comment documents *why* the full grid is the policy's identity.

### 2. `src/isaac_datagen/clean_datagen.py` — stop under-supplying
`reference_segmentation()`: `collect_objects(...)[4:11]` → `collect_objects(...)`.
The dataset has exactly 44 objects, which exactly fills `[11,1,4]`. (`main()` was
already passing all objects, so only this path was affected.)

### 3. `src/isaac_datagen/isaac_utils.py` — USDZ export helpers (new)
Ported the proven "reference-and-flatten" idiom from
`visual_servoing/datagen2_isaacsim/.docs_claude/usdz_export_investigation.md`:
- `export_subtree_usdz(stage, subtree_path, output_dir, base_name)`
- `export_flattened_usdz(stage, output_dir, base_name)` (exports the defaultPrim)

Build a fresh `Usd.Stage.CreateInMemory()`, *reference* the live subtree into it,
`SetDefaultPrim`, `Export` to a temp `.usdc` (flattens the arc), then
`UsdUtils.CreateNewUsdzPackage`. Never `Usd.Stage.Open(file)` inside Isaac Sim —
it returns the live sim stage. The `@N/foo.ext@` resolve warnings from packaging
are a red herring; the textures are still bundled (often under `0/`, `1/`).

### 4. Debug tooling (new, kept in-repo)
- `src/isaac_datagen/debug_scene.py` — boots the sim, rebuilds the exact
  reference-seg scene, prints grasp-point count + `rng.choice` indices + each
  grasp point's world translation, exports `/World` to `scene.usdz`, and extracts
  each distinct selected grasp box (the `GraspPoint`'s parent prim) to its own
  `selected_<name>.usdz`. Writes diagnostics to `<render_dir>/grasp_debug.txt`
  (Isaac floods stdout). Run from `src/isaac_datagen/`:
  `uv run debug_scene.py configs/randomized.yaml proposer_device=cuda:1`.
- `src/isaac_datagen/debug_occupancy.py` — **pure numpy, no sim**. Reconstructs
  the grid state and prints a side-view (real box / phantom / graspable), a
  per-box `is_front`/`is_top`/`graspable` table mapped to real box names, and a
  full-wall sanity check. Run: `uv run debug_occupancy.py configs/randomized.yaml`.

### 5. `src/isaac_datagen/configs/randomized.yaml` — surface pose-halo ranges
`xrange`/`yrange`/`zrange` (previously hidden `RuntimeConfig` defaults at
`runtime_config.py:61-63`) are now explicit in the config so the pose halo is
tunable. The halo was near-degenerate (x,z fixed; one-sided y), which is why
renders fanned along one axis instead of orbiting — independent of the wall bug.
`yrange` widened to symmetric `[-0.22, 0.22]` for a fuller halo.

### 6. `CLAUDE.md`
`isaac_utils.py` module-index row updated to list the new USDZ-export exports.

## Verification

`uv run debug_occupancy.py configs/randomized.yaml` (no sim) and a full
`debug_scene.py` sim run both confirm:

| | Before (`[4:11]`, 7 objs) | After (full wall, 44 objs) |
|---|---|---|
| slots filled | 7/44 | 44/44 |
| `len(grasp_points)` | 1 | 11 (entire top row) |
| `rng.choice(size=5)` | `[0,0,0,0,0]` → 1 unique | `[5,8,9,5,0]` → 4 unique |
| `scene.usdz` | 73 MB (2-wide tower) | 136 MB (11×4 wall) |

The 11 grasp points span the top row x = −1.035 … +1.235 (one per column).

## Notes / non-goals / follow-ups

- **`rng.choice` still uses `replace=True`** (`clean_datagen.py:81`): 5 draws from
  11 can repeat (this run got 4 unique). For guaranteed-distinct targets, set
  `replace=False` (safe while `num_targets < top-row count`; errors otherwise).
  Not changed — flagged for the user.
- **Pose-halo bias is separate** from this fix: the grasp frame sits at the box's
  bottom-front edge (`add_grasp_frame`, `scene.py`), and the ranges are narrow.
  Tune via the now-explicit `xrange`/`yrange`/`zrange`.
- A varied partially-picked pallet (multiple graspable heights, real occlusion)
  is out of scope; it needs depth > 1 (e.g. `[3,3,4]`) and the
  start-full-remove-picks model already in `LoadedPallet`.

# Depth-jittering placer — complete the WIP `UntilExhaustedStacker`

## Context

`UntilExhaustedStacker` (`isaac_datagen/src/isaac_datagen/placers.py`) packs placed objects into
columns and lays them as a **flat wall on y=0** — every column's centroid sits at `y = -cy`
(depth 0). We want columns spread across **varying depths (world-y)** so the stacks are staggered
front-to-back instead of coplanar.

The working tree already holds the user's design toward this (lines 31-100). This plan **completes
that design as written** — keeping its structure — and only fixes the bugs that stop it importing
/ running. The design:

- `min_y`/`max_y` live on the **base** `UntilExhaustedStacker.__init__`, defaulting to `0,0` so
  existing configs (`shelf.yaml`, `mixed.yaml`) and `ShelfPlacer` keep the flat-wall behavior with
  no change. Depth jitter is opt-in purely by supplying `min_y`/`max_y` in `placement_args` at
  runtime — **no new placer class**.
- A precomputed `self.columns_xy` list carries **per-column `(col_x, col_y)`**; `col_y` is one
  uniform-random draw in `[min_y, max_y]` per column (per-column depth, shared by the stack).

Bugs in the current WIP that must be fixed (do not change the design, just make it correct):
1. `centers { ... }` — missing `=`.
2. `size_of(p)` / `center_of(p)` don't exist yet (the user wants them created); and `size[p]` /
   `center[p]` later disagree with the `sizes` / `centers` dict names.
3. `self.columns` is read by `col_widths` **before** it is assigned — assignment must move up.
4. `col_x = left_edge + col_widths[col_idx]` does **not** advance left→right; columns would all
   pack near the left edge and overlap. Needs a running cumulative x (as the committed code did).
5. `random(min=min_y, max=min_y)` — not a real call, and `max=min_y` is a typo for `max=max_y`.
6. No `numpy` import.

## Approach

### File: `src/isaac_datagen/placers.py`

**1. Import numpy + add the `size_of` / `center_of` helpers** the WIP referenced. Keep them as
module-level functions (the placer world deals in prim *paths*; these resolve a path → bbox on the
current stage, wrapping the existing `local_bbox_range`). Repo convention for ranged randomness is
`np.random.uniform`, seeded globally by `seed_everything(runtime.effective_seed)` before the placer
is built.

```python
from __future__ import annotations

import sys
from collections import deque

import numpy as np                              # NEW

from isaac_datagen.isaac_utils import local_bbox_range, class_label


def _stage():
    from isaacsim.core.utils.stage import get_current_stage
    return get_current_stage()


def size_of(prim_path):                         # NEW — (sx, sy, sz) local bbox extent
    sz = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetSize()
    return (sz[0], sz[1], sz[2])


def center_of(prim_path):                       # NEW — (cx, cy, cz) local bbox midpoint
    mid = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetMidpoint()
    return (mid[0], mid[1], mid[2])
```

**2. Rewrite `UntilExhaustedStacker.__init__` keeping the user's `columns_xy` structure, but build
it elegantly** — the y's are just N random numbers (one vectorized draw), the x's are a cumulative
"range" of column widths centered on x=0:

```python
def __init__(self, prim_paths, column_height, min_y=0, max_y=0):
    if column_height < 1:
        raise ValueError(f"column_height must be >= 1, got {column_height}")
    if len(prim_paths) < 1:
        raise ValueError("UntilExhaustedStacker needs >= 1 object")

    # Columns of prim paths (deques so the "top" is unambiguous: last pushed).
    self.columns = [                                          # MOVED UP (was read before assign)
        deque(prim_paths[s:s + column_height])
        for s in range(0, len(prim_paths), column_height)
    ]

    # Measure size + center per prim once (helpers resolve prim_path -> bbox on the stage).
    sizes = {p: size_of(p) for p in prim_paths}              # FIX: real helper, `=`
    centers = {p: center_of(p) for p in prim_paths}

    # Per-column footprint width = widest member's x-extent.
    col_widths = np.array([max(sizes[p][0] for p in col) for col in self.columns])
    total_w = col_widths.sum() + (len(self.columns) - 1) * self.EPSILON

    # x: column left edges march left->right (a "range" whose step is each column's own
    #    width + gap), centered on x=0; column center = left edge + half width.
    # y: one uniform-random depth per column (min_y==max_y==0 -> flat wall, the default).
    left_edges = -total_w / 2.0 + np.concatenate(
        [[0.0], np.cumsum(col_widths[:-1] + self.EPSILON)]
    )
    col_xs = (left_edges + col_widths / 2.0).tolist()
    col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
    self.columns_xy = list(zip(col_xs, col_ys))

    # set_transform places the prim ORIGIN; subtract the bbox midpoint per axis to seat the
    # centroid on (col_x, col_y) and the bbox base at floor_z.
    self._placements = {}  # prim_path -> (translation, rotation)
    for col, (col_x, col_y) in zip(self.columns, self.columns_xy):
        floor_z = 0.0
        for p in col:  # bottom -> top
            sx, sy, sz = sizes[p]
            cx, cy, cz = centers[p]
            translation = (col_x - cx, col_y - cy, floor_z - cz + sz / 2.0)
            self._placements[p] = (translation, (0.0, 0.0, 0.0))
            floor_z += sz + self.EPSILON
```

`__call__` and `graspability` are unchanged from the committed version. (Note `.tolist()` casts the
numpy scalars back to python floats so the translation tuples match the rest of the codebase.)

**3. Delete the `Kindofrandomplacer` pseudo-stub** (lines 97-101 of the working tree). No new
placer class is added — depth jitter is reached by selecting `UntilExhaustedStacker` and passing
`min_y`/`max_y` in `placement_args`. `ShelfPlacer` is left exactly as committed (it forwards only
`column_height`, so it stays flat; if a jittered *class-grouped* layout is ever wanted, that's a
later one-line change to forward `min_y`/`max_y` through `ShelfPlacer.__init__`).

No change to `scene.py`, `runtime_config.py`, or `objects.py`: dispatch already flows through
`placers.get(runtime.placement)(prim_paths, **runtime.placement_args)` (`scene.py:67`) and
`organize_objects` applies `(translation, rotation)` via `set_transform` (`scene.py:53-58`), which
already writes the y-translation.

### File: `src/isaac_datagen/configs/staggered.yaml` (NEW, optional but recommended)

Mirror `mixed.yaml`, swapping only the placement block so the variant is runnable:

```yaml
# base: <same base/include as mixed.yaml>
placement: UntilExhaustedStacker
placement_args:
  column_height: 5
  min_y: 0.0
  max_y: 0.10        # 10 cm of front-to-back stagger (USD meters)
```

## Critical files

- `isaac_datagen/src/isaac_datagen/placers.py` — add `size_of`/`center_of` helpers + numpy import;
  fix the base ctor (`columns_xy` precompute with cumulative x + vectorized random y, dict-name
  fixes, ordering); delete the `Kindofrandomplacer` stub.
- `isaac_datagen/src/isaac_datagen/configs/staggered.yaml` — new example config (optional).

## Reuse / grounding

- `local_bbox_range(prim).GetSize()/.GetMidpoint()` (`isaac_utils.py:278`) — per-prim size/center;
  the new `size_of`/`center_of` path-based helpers wrap it.
- `set_transform(prim, translation, rotation)` (`isaac_utils.py:116`) — applies placement;
  y-translation already supported.
- `np.random.uniform(size=N)` — matches existing ranged randomness (`scene.py:359`); seeded by
  `seed_everything` (`clean_datagen.py`) + Replicator `set_global_seed` (`scene.py:160`).

## Verification

1. **Import / module loads** (no sim):
   ```
   cd isaac_datagen && uv run python -c \
     "import isaac_datagen.placers as p; print(p.get('UntilExhaustedStacker'), p.size_of, p.center_of)"
   ```
2. **Backward compatible**: a `UntilExhaustedStacker` run with no `min_y/max_y` (e.g. `shelf.yaml`)
   still yields all-equal column y (`translation[1] == -cy`). With `min_y/max_y` set in
   `placement_args`, per-column y are distinct and within `[min_y, max_y]`.
3. **End-to-end render** (Isaac Sim):
   ```
   cd isaac_datagen && uv run clean_datagen.py \
     src/isaac_datagen/configs/staggered.yaml idx=0 num_frames=4
   ```
   Inspect serialized poses / a captured frame: stacks staggered front-to-back, stacking and
   graspability (column tops) intact, no inter-column overlap at the chosen `max_y`.

## Plans-TOC maintenance

If a plan doc is added under `.docs_claude/plans/` for this change, update
`alldocs/PLANS_TOC.md` (topic 2: Masks, Labels & Dataset Generation) per CLAUDE.md.

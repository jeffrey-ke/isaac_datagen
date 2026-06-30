# Gap-jittering placer — randomize spacing between columns in `UntilExhaustedStacker`

## Context

`UntilExhaustedStacker` (`isaac_datagen/src/isaac_datagen/placers.py`) packs placed objects into
columns laid out left-to-right along x. Depth jitter (`min_y`/`max_y`, see
`depth-jitter-stacker.md`) staggers each column to a random world-y depth so the wall isn't
coplanar. This plan adds the same treatment for the **horizontal spacing between columns**: the gap
between adjacent columns was the fixed `EPSILON = 0.002` (2mm) — just enough to avoid intersection,
not a meaningful visual variation.

Scope: only the horizontal x-gap between columns is jittered. The vertical in-column stacking gap
(also `EPSILON`) is unchanged. New args default to reproducing the exact current fixed-`EPSILON`
spacing, so existing configs (`mixed.yaml`, `shelf.yaml`, the `min_y/max_y`-only use of
`staggered.yaml`) are unaffected unless they opt in.

## Approach

### File: `src/isaac_datagen/placers.py`

`UntilExhaustedStacker.__init__` gained `min_gap`/`max_gap` (default `EPSILON, EPSILON`): one
random x-gap per column boundary (`len(self.columns) - 1` draws), replacing the fixed `EPSILON`
term in both the total-width sum and the cumulative left-edge walk:

```python
def __init__(self, prim_paths, column_height, min_y=0, max_y=0,
             min_gap=EPSILON, max_gap=EPSILON):
    ...
    col_widths = np.array([max(sizes[p][0] for p in col) for col in self.columns])
    gaps = np.random.uniform(min_gap, max_gap, size=len(self.columns) - 1)
    total_w = col_widths.sum() + gaps.sum()

    left_edges = -total_w / 2.0 + np.concatenate(
        [[0.0], np.cumsum(col_widths[:-1] + gaps)]
    )
    col_xs = (left_edges + col_widths / 2.0).tolist()
    col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
    self.columns_xy = list(zip(col_xs, col_ys))
```

With a single column, `gaps` is an empty array (`size=0`) and `gaps.sum() == 0.0` — same degenerate
case the old `(len(self.columns) - 1) * EPSILON == 0` already handled.

`ShelfPlacer` is unchanged (still forwards only `column_height`, same as it already didn't forward
`min_y`/`max_y`).

### File: `src/isaac_datagen/configs/staggered.yaml`

Added `min_gap`/`max_gap` alongside the existing `min_y`/`max_y`, so the one example config
demonstrates both axes of staggering.

## Critical files

- `isaac_datagen/src/isaac_datagen/placers.py` — `min_gap`/`max_gap` on
  `UntilExhaustedStacker.__init__`.
- `isaac_datagen/src/isaac_datagen/configs/staggered.yaml` — example config.

## Verification

1. Import check: `inspect.signature(UntilExhaustedStacker.__init__)` shows
   `min_gap=0.002, max_gap=0.002`.
2. Backward compatible: with mocked `size_of`/`center_of` and a fixed RNG seed, a call with no
   `min_gap`/`max_gap` produces `columns_xy` identical to the pre-change code.
3. Jittered: with `min_gap=0.01, max_gap=0.05`, the measured gap between adjacent columns
   (`col1_x - w1/2 - (col0_x + w0/2)`) lands inside `[min_gap, max_gap]` and columns don't overlap.
4. End-to-end render: `uv run clean_datagen.py src/isaac_datagen/configs/staggered.yaml idx=0
   num_frames=4` — columns staggered in both depth and horizontal spacing, stacking/graspability
   intact.

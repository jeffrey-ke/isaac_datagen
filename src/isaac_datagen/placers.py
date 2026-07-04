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
from typing import NamedTuple

import numpy as np

from isaac_datagen.isaac_utils import local_bbox_range, class_label


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


def _stage():
    from isaacsim.core.utils.stage import get_current_stage
    return get_current_stage()


class Vec3(NamedTuple):
    """A named 3-vector so bbox math reads as intent (`.z`) instead of `sz[2]`."""
    x: float
    y: float
    z: float


def size_of(prim_path):
    """Vec3 local bbox extent of the prim at prim_path."""
    sz = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetSize()
    return Vec3(sz[0], sz[1], sz[2])


def center_of(prim_path):
    """Vec3 local bbox midpoint of the prim at prim_path."""
    mid = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetMidpoint()
    return Vec3(mid[0], mid[1], mid[2])


def centroid_at_point(center, target):
    """Origin translation that lands the bbox centroid on `target` (no rotation).

    set_transform places the prim ORIGIN; the centroid lands at origin + center, so the
    origin must go to target - center. The caller decides `target` (e.g. lifts it half a
    height so the base rests on a floor) -- this stays pure geometry.
    """
    return tuple(t - c for t, c in zip(target, center))


def compute_cols_stride(columns, min_gap, max_gap, sizes):
    """Per-column x-centers: footprints packed left->right with random gaps, centered on 0.

    Per-column footprint width = widest member's x-extent; one uniform-random gap per
    column boundary (min_gap==max_gap reproduces constant spacing). Pure numpy -- no
    stage, so it is unit-testable on made-up widths.
    """
    col_widths = np.array([max(sizes[p].x for p in col) for col in columns])
    gaps = np.random.uniform(min_gap, max_gap, size=len(columns) - 1)
    total_w = col_widths.sum() + gaps.sum()

    left_edges = -total_w / 2.0 + np.concatenate(
        [[0.0], np.cumsum(col_widths[:-1] + gaps)]
    )
    return (left_edges + col_widths / 2.0).tolist()


def fixed_columns(prim_paths, height):
    """Chunk prim_paths into deques of exactly `height` (the last may be short)."""
    return [deque(prim_paths[s:s + height]) for s in range(0, len(prim_paths), height)]


def jagged_columns(prim_paths, max_height):
    """Greedily chunk prim_paths into deques of random height in [1, max_height]."""
    cols, i = [], 0
    while i < len(prim_paths):
        h = int(np.random.randint(1, max_height + 1))   # +1 -> max_height inclusive
        cols.append(deque(prim_paths[i:i + h]))
        i += h
    return cols


class UntilExhaustedStacker:

    EPSILON = 0.002

    def __init__(self, prim_paths, max_column_height, min_y=0, max_y=0,
                 min_gap=EPSILON, max_gap=EPSILON, epsilon=EPSILON):
        if max_column_height < 1:
            raise ValueError(f"max_column_height must be >= 1, got {max_column_height}")
        if len(prim_paths) < 1:
            raise ValueError("UntilExhaustedStacker needs >= 1 object")

        # Columns of random height in [1, max_column_height] (deques so the "top" is
        # unambiguous: last pushed).
        self.columns = jagged_columns(prim_paths, max_column_height)
        self._seat(min_y, max_y, min_gap, max_gap, epsilon)

    def _seat(self, min_y, max_y, min_gap, max_gap, epsilon):
        """Measure -> layout -> stack over self.columns, filling self.columns_xy/_placements.

        epsilon drives both the inter-object vertical gap and the (x, y) placement jitter.
        """
        # Measure (impure): the only lines that touch the stage. `sizes` + `centers` are
        # the bridge -- everything below is pure arithmetic on these two dicts.
        prims = [p for col in self.columns for p in col]  # columns = single source of truth
        sizes = {p: size_of(p) for p in prims}
        centers = {p: center_of(p) for p in prims}

        # Layout (pure): col_xs (packed footprint centers) and col_ys (random per-column
        # depth; min_y==max_y==0 -> flat wall) are independent -- zip is the only join.
        col_xs = compute_cols_stride(self.columns, min_gap, max_gap, sizes)
        col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
        self.columns_xy = list(zip(col_xs, col_ys))

        # Stack (pure): the only logic here is policy -- base-on-floor (the half-height
        # lift baked into the target), the (x, y) jitter, and the stacking cursor. Geometry
        # lives in centroid_at_point.
        self._placements = {}  # prim_path -> (translation, rotation)
        for col, xy in zip(self.columns, self.columns_xy):
            floor_z = 0.0
            for p in col:  # bottom -> top
                target = (*xy, floor_z + sizes[p].z / 2.0)
                bx, by, bz = centroid_at_point(centers[p], target)
                jx, jy = np.random.normal(0, epsilon, size=2)  # x,y jitter only; z stays clean
                self._placements[p] = ((bx + jx, by + jy, bz), (0.0, 0.0, 0.0))
                floor_z += sizes[p].z + epsilon                 # gap now configurable

    def __call__(self, prim_path):
        return self._placements[prim_path]

    def graspability(self):
        """Per-prim-path graspability: only the top object of each column."""
        tops = {col[-1] for col in self.columns if col}
        return {p: (p in tops) for p in self._placements}


class ShelfPlacer(UntilExhaustedStacker):
    """UntilExhaustedStacker that groups same-class objects into the same columns.

    Sorts prim_paths by their semantic "class" label (read off the loaded stage -- the
    wrapper path encodes only the instance name) so each run of `column_height` adjacent
    paths is one class: a shelf of cans next to a shelf of boxes. Layout/graspability are
    inherited unchanged. Stable sort preserves within-class order; a class count not a
    multiple of `column_height` yields one mixed boundary column, as with plain chunking.

    Uses a FIXED `column_height` (not the base's jagged heights) on purpose: random heights
    would split a class run across columns and mix classes at arbitrary boundaries.
    """

    def __init__(self, prim_paths, column_height, min_y=0, max_y=0,
                 min_gap=UntilExhaustedStacker.EPSILON, max_gap=UntilExhaustedStacker.EPSILON,
                 epsilon=UntilExhaustedStacker.EPSILON):
        if column_height < 1:
            raise ValueError(f"column_height must be >= 1, got {column_height}")
        prims = sorted(prim_paths, key=class_label)
        self.columns = fixed_columns(prims, column_height)  # fixed -> class grouping preserved
        self._seat(min_y, max_y, min_gap, max_gap, epsilon)

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


def size_of(prim_path):
    """(sx, sy, sz) local bbox extent of the prim at prim_path."""
    sz = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetSize()
    return (sz[0], sz[1], sz[2])


def center_of(prim_path):
    """(cx, cy, cz) local bbox midpoint of the prim at prim_path."""
    mid = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetMidpoint()
    return (mid[0], mid[1], mid[2])


class UntilExhaustedStacker:

    EPSILON = 0.002

    def __init__(self, prim_paths, column_height, min_y=0, max_y=0,
                 min_gap=EPSILON, max_gap=EPSILON):
        if column_height < 1:
            raise ValueError(f"column_height must be >= 1, got {column_height}")
        if len(prim_paths) < 1:
            raise ValueError("UntilExhaustedStacker needs >= 1 object")

        # Columns of prim paths (deques so the "top" is unambiguous: last pushed).
        self.columns = [
            deque(prim_paths[s:s + column_height])
            for s in range(0, len(prim_paths), column_height)
        ]

        # Measure size + center per prim once, from the loaded stage prims.
        sizes = {p: size_of(p) for p in prim_paths}
        centers = {p: center_of(p) for p in prim_paths}

        # Per-column footprint width = widest member's x-extent.
        col_widths = np.array([max(sizes[p][0] for p in col) for col in self.columns])
        # One random x-gap per column boundary (N-1 gaps for N columns); min_gap==max_gap==
        # EPSILON (the default) reproduces the old constant spacing exactly.
        gaps = np.random.uniform(min_gap, max_gap, size=len(self.columns) - 1)
        total_w = col_widths.sum() + gaps.sum()

        # x: column left edges march left->right (a "range" whose step is each column's
        #    own width + a possibly-jittered gap), centered on x=0; column center = left
        #    edge + half width.
        # y: one uniform-random depth per column (min_y==max_y==0 -> flat wall, default).
        left_edges = -total_w / 2.0 + np.concatenate(
            [[0.0], np.cumsum(col_widths[:-1] + gaps)]
        )
        col_xs = (left_edges + col_widths / 2.0).tolist()
        col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
        self.columns_xy = list(zip(col_xs, col_ys))

        self._placements = {}  # prim_path -> (translation, rotation)
        for col, (col_x, col_y) in zip(self.columns, self.columns_xy):
            floor_z = 0.0
            for p in col:  # bottom -> top
                sx, sy, sz = sizes[p]
                cx, cy, cz = centers[p]
                # set_transform places the prim ORIGIN; the bbox center lands at
                # origin + (cx,cy,cz), so subtract the midpoint per axis to seat
                # the centroid on (col_x, col_y) and the bbox base at floor_z.
                translation = (col_x - cx, col_y - cy, floor_z - cz + sz / 2.0)
                self._placements[p] = (translation, (0.0, 0.0, 0.0))
                floor_z += sz + self.EPSILON

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
    """

    def __init__(self, prim_paths, column_height):
        super().__init__(sorted(prim_paths, key=class_label), column_height)

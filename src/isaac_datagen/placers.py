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
    x: float
    y: float
    z: float


def size_of(prim_path):
    sz = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetSize()
    return Vec3(sz[0], sz[1], sz[2])


def center_of(prim_path):
    mid = local_bbox_range(_stage().GetPrimAtPath(prim_path)).GetMidpoint()
    return Vec3(mid[0], mid[1], mid[2])


def centroid_at_point(center, target):
    return tuple(t - c for t, c in zip(target, center))


def compute_cols_stride(columns, min_gap, max_gap, sizes):
    col_widths = np.array([max(sizes[p].x for p in col) for col in columns])
    gaps = np.random.uniform(min_gap, max_gap, size=len(columns) - 1)
    total_w = col_widths.sum() + gaps.sum()

    left_edges = -total_w / 2.0 + np.concatenate(
        [[0.0], np.cumsum(col_widths[:-1] + gaps)]
    )
    return (left_edges + col_widths / 2.0).tolist()


def fixed_columns(prim_paths, height):
    return [deque(prim_paths[s:s + height]) for s in range(0, len(prim_paths), height)]


def jagged_columns(prim_paths, max_height):
    cols, i = [], 0
    while i < len(prim_paths):
        h = int(np.random.randint(1, max_height + 1))
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

        self.columns = jagged_columns(prim_paths, max_column_height)
        self._seat(min_y, max_y, min_gap, max_gap, epsilon)

    def _seat(self, min_y, max_y, min_gap, max_gap, epsilon):
        prims = [p for col in self.columns for p in col]
        sizes = {p: size_of(p) for p in prims}
        centers = {p: center_of(p) for p in prims}

        col_xs = compute_cols_stride(self.columns, min_gap, max_gap, sizes)
        col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
        self.columns_xy = list(zip(col_xs, col_ys))

        self._placements = {}
        for col, xy in zip(self.columns, self.columns_xy):
            floor_z = 0.0
            for p in col:
                target = (*xy, floor_z + sizes[p].z / 2.0)
                bx, by, bz = centroid_at_point(centers[p], target)
                jx, jy = np.random.normal(0, epsilon, size=2)
                self._placements[p] = ((bx + jx, by + jy, bz), (0.0, 0.0, 0.0))
                floor_z += sizes[p].z + epsilon

    def __call__(self, prim_path):
        return self._placements[prim_path]

    def graspability(self):
        tops = {col[-1] for col in self.columns if col}
        return {p: (p in tops) for p in self._placements}


class ShelfPlacer(UntilExhaustedStacker):

    def __init__(self, prim_paths, column_height, min_y=0, max_y=0,
                 min_gap=UntilExhaustedStacker.EPSILON, max_gap=UntilExhaustedStacker.EPSILON,
                 epsilon=UntilExhaustedStacker.EPSILON):
        if column_height < 1:
            raise ValueError(f"column_height must be >= 1, got {column_height}")
        prims = sorted(prim_paths, key=class_label)
        self.columns = fixed_columns(prims, column_height)
        self._seat(min_y, max_y, min_gap, max_gap, epsilon)

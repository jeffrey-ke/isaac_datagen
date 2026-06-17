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


class UntilExhaustedStacker:

    EPSILON = 0.002

    def __init__(self, prim_paths, column_height):
        if column_height < 1:
            raise ValueError(f"column_height must be >= 1, got {column_height}")
        if len(prim_paths) < 1:
            raise ValueError("UntilExhaustedStacker needs >= 1 object")

        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()

        # Columns of prim paths (deques so the "top" is unambiguous: last pushed).
        self.columns = [
            deque(prim_paths[s:s + column_height])
            for s in range(0, len(prim_paths), column_height)
        ]

        # Measure size + center per prim once, from the loaded stage prims.
        size, center = {}, {}
        for p in prim_paths:
            rng = local_bbox_range(stage.GetPrimAtPath(p))
            sz, mid = rng.GetSize(), rng.GetMidpoint()
            size[p] = (sz[0], sz[1], sz[2])
            center[p] = (mid[0], mid[1], mid[2])

        # Column footprint width = widest member's x-extent; center the wall on x=0.
        col_widths = [max(size[p][0] for p in col) for col in self.columns]
        total_w = sum(col_widths) + (len(self.columns) - 1) * self.EPSILON
        left_edge = -total_w / 2.0

        self._placements = {}  # prim_path -> (translation, rotation)
        for col, col_w in zip(self.columns, col_widths):
            col_x = left_edge + col_w / 2.0
            floor_z = 0.0
            for p in col:  # bottom -> top
                sx, sy, sz = size[p]
                cx, cy, cz = center[p]
                # set_transform places the prim ORIGIN; the bbox center lands at
                # origin + (cx,cy,cz), so subtract the midpoint per axis to seat
                # the centroid on (col_x, 0) and the bbox base at floor_z.
                translation = (col_x - cx, -cy, floor_z - cz + sz / 2.0)
                self._placements[p] = (translation, (0.0, 0.0, 0.0))
                floor_z += sz + self.EPSILON
            left_edge += col_w + self.EPSILON

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
        from isaacsim.core.utils.semantics import get_labels
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()

        def class_label(prim_path):
            geo = stage.GetPrimAtPath(f"{prim_path}/geo")  # add_object labels geo as "class"
            labels = get_labels(geo)
            if not labels.get("class"):
                raise ValueError(f"ShelfPlacer: no 'class' label on {prim_path}/geo")
            return labels["class"][0]

        super().__init__(sorted(prim_paths, key=class_label), column_height)

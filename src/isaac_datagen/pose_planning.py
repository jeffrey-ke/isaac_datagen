"""Target-frame pose sampling. Pure numpy, no Isaac Sim deps."""

from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_core.pose_utils import generate_random_offsets, generate_grid_offsets, offset_to_4x4, add_rotation


def plan_poses(target_to_baseline_ypr_desired, xrange, yrange, zrange,
               sampling: int | tuple[int, int, int]):
    yaw, pitch, roll = target_to_baseline_ypr_desired
    if isinstance(sampling, tuple):
        offsets = generate_grid_offsets(xrange, yrange, zrange, *sampling)
    else:
        offsets = generate_random_offsets(xrange, yrange, zrange, sampling)
    return np.array([
        add_rotation(offset_to_4x4(offset), z=yaw, y=pitch, x=roll)
        for offset in offsets
    ])

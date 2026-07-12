from __future__ import annotations

import sys

import numpy as np

from vision_core.pose_utils import generate_random_offsets, look_at, cv2opengl, offset_to_4x4, add_rotation
from isaac_datagen.pose_planning import plan_poses


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


class GridFixedPoser:

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

    def __init__(self, xrange, yrange, zrange):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange

    def __call__(self, num_frames: int) -> np.ndarray:
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange, num_frames)
        return np.array([cv2opengl(look_at(np.zeros(3), off)) for off in offsets])


class FixedOffsetPoser:

    def __init__(self, offset, ypr=(0.0, 0.0, 0.0)):
        self.offset = np.asarray(offset, dtype=float)
        self.ypr = ypr

    def __call__(self, num_frames: int) -> np.ndarray:
        yaw, pitch, roll = self.ypr
        pose = add_rotation(offset_to_4x4(self.offset), z=yaw, y=pitch, x=roll)
        return np.repeat(pose[None], num_frames, axis=0)

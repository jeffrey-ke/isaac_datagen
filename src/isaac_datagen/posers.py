"""Camera-pose generation policy registry.

Stateful callables that, given a frame count, return (N, 4, 4) camera2target SE3
poses in the grasp-target frame. Selected by name from a config block
(`pose_generation_policy` + `pose_generation_policy_args`), mirroring the
optimizer registry in segmentation/optim.py.
"""
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
    """Fixed-rotation poser: exactly the poses plan_poses used to return. Every
    camera shares one orientation (target_to_ego_ypr); only position varies over
    the halo box. random=True samples num_frames positions; random=False lays a
    fixed grid (grid_dims) and takes the first num_frames."""

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
    """Look-at poser: each camera sits at a random halo-box offset and is oriented
    to face the target origin (look_at returns a camera2target SE3, translation =
    the offset). Orientation varies per pose, unlike GridFixedPoser's fixed ypr."""

    def __init__(self, xrange, yrange, zrange):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange

    def __call__(self, num_frames: int) -> np.ndarray:
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange, num_frames)
        # look_at orients +Z toward the target (OpenCV); USD/renderer cameras look down
        # -Z, so convert to the -Z-forward (OpenGL/USD) convention — cv2opengl negates the
        # Y and Z columns, flipping the look direction while keeping a proper rotation
        # (no mirrored image). Without this every camera faces 180deg away from the target.
        return np.array([cv2opengl(look_at(np.zeros(3), off)) for off in offsets])


class FixedOffsetPoser:
    """Deterministic single-offset poser: every call returns ONE fixed pose
    (translation = offset, optional ypr), replicated num_frames times. For
    controlled occluder placement (E4): add_shadow_occluders applies it as
    t2w @ poser(1)[0], so the offset is a TARGET-frame point — +x is toward the
    camera and the box sits at x<=0, making depth-along-x the penetrate/float knob
    (offset=[0,0,0] straddles the box; offset=[0.2,0,0] floats in front)."""

    def __init__(self, offset, ypr=(0.0, 0.0, 0.0)):
        self.offset = np.asarray(offset, dtype=float)
        self.ypr = ypr

    def __call__(self, num_frames: int) -> np.ndarray:
        yaw, pitch, roll = self.ypr
        pose = add_rotation(offset_to_4x4(self.offset), z=yaw, y=pitch, x=roll)
        return np.repeat(pose[None], num_frames, axis=0)

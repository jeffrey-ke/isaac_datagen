from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from vision_core.pose_utils import (
    generate_random_offsets, look_at, cv2opengl, offset_to_4x4, add_rotation,
    frustum_normals, cone_in_frustum, pixel_direction, erode_frame_rect, resolve_offset_sampler,
)
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

    def __init__(self, xrange, yrange, zrange, offset_sampler=None):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange
        self.sampler = resolve_offset_sampler(offset_sampler)

    def __call__(self, num_frames: int) -> np.ndarray:
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange,
                                          num_frames, sampler=self.sampler)
        return np.array([cv2opengl(look_at(np.zeros(3), off)) for off in offsets])


class DecenteredLookAtPoser:                                # NEW
    """LookAtPoser halo + decentering: the target's grasp origin lands at a pixel sampled
    uniformly over the frame rect eroded so the whole object_radius sphere stays visible,
    plus roll about the target ray (visibility-invariant by construction).

    Offsets are drawn in ONE generate_random_offsets call before any decentering draw, so
    under the same seed camera POSITIONS are identical to LookAtPoser's — clean A/B vs the
    existing pools. Close-ups whose eroded rect collapses stay centered (defined policy,
    matches baseline behavior; ~10% of frames at object_radius 0.25 over the pool halo box).
    """

    def __init__(self, xrange, yrange, zrange, intrinsics_path, resolution,
                 object_radius, margin_deg=1.0, max_roll_deg=15.0, offset_sampler=None):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange
        self.K = np.load(intrinsics_path)                   # fail-loud: no default intrinsics
        self.resolution = tuple(resolution)
        self.object_radius = float(object_radius)
        self.margin = np.radians(margin_deg)
        self.max_roll = np.radians(max_roll_deg)
        self.normals = frustum_normals(self.K, self.resolution)
        self.sampler = resolve_offset_sampler(offset_sampler)

    def __call__(self, num_frames: int) -> np.ndarray:
        # ONE call, BEFORE any decentering draw: identical position stream to LookAtPoser
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange,
                                          num_frames, sampler=self.sampler)
        return np.array([self._decentered(off) for off in offsets])

    def _decentered(self, off: np.ndarray) -> np.ndarray:
        pose = look_at(np.zeros(3), off)                     # CV cam2world, z at the target
        ang_r = np.arcsin(min(self.object_radius / np.linalg.norm(off), 1.0)) + self.margin
        rect = erode_frame_rect(self.K, self.resolution, ang_r)
        if rect is None:                                     # close-up: object can't fit off-center
            return cv2opengl(pose)                           # defined policy: centered, as today
        uv = (np.random.uniform(rect[0], rect[1]), np.random.uniform(rect[2], rect[3]))
        d = pixel_direction(self.K, uv)
        assert cone_in_frustum(d, ang_r, self.normals), f"eroded rect violated at {uv}"
        r_point, _ = R.align_vectors([[0.0, 0.0, 1.0]], [d])         # maps d -> optical axis
        r_roll = R.from_rotvec(np.random.uniform(-self.max_roll, self.max_roll) * d)
        pose[:3, :3] = pose[:3, :3] @ (r_point * r_roll).as_matrix() # target ray -> pixel uv
        return cv2opengl(pose)


class FixedOffsetPoser:

    def __init__(self, offset, ypr=(0.0, 0.0, 0.0)):
        self.offset = np.asarray(offset, dtype=float)
        self.ypr = ypr

    def __call__(self, num_frames: int) -> np.ndarray:
        yaw, pitch, roll = self.ypr
        pose = add_rotation(offset_to_4x4(self.offset), z=yaw, y=pitch, x=roll)
        return np.repeat(pose[None], num_frames, axis=0)

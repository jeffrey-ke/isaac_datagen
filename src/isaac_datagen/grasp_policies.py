"""Grasp-frame policy registry (posers.py / placers.py get(name) idiom).

policy(**args)(lo, hi) -> (4, 4) SE3 grasp frame in OBJECT-LOCAL (usdz) frame,
+X = outward face normal, +Z = up (mesh_convert.face_grasp_frames convention).
Policies MUST return side faces only: ref_pose_from_grasp's look_at(up=[0,0,1])
is singular for ±Z normals — fail loud, never fall back.
Selected by an explicit config key (StoreSceneSpec.grasp_frame_policy); no defaults.
"""
from __future__ import annotations

import sys

import numpy as np

from isaac_datagen.mesh_convert import FACE_NORMALS, face_grasp_frames


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


class FixedFaceGrasp:
    """Fixed-local-face policy: the grasp frame of one named side face of the
    local bbox, via mesh_convert.face_grasp_frames. face ∈ {-Y, +Y, -X, +X}."""

    def __init__(self, face: str):
        assert face in FACE_NORMALS, \
            f"face must be one of {sorted(FACE_NORMALS)} (side faces only, ±Z is singular): {face!r}"
        self.face = face

    def __call__(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return face_grasp_frames(lo, hi)[self.face]

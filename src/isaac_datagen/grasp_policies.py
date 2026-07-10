"""Grasp-frame policy registry (posers.py / placers.py get(name) idiom).

policy(**args)(lo, hi, cls) -> (4, 4) SE3 grasp frame in OBJECT-LOCAL (usdz) frame,
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

    def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
        return face_grasp_frames(lo, hi)[self.face]      # cls ignored: one fixed face for all


class PerClassFaceGrasp:
    """Per-SKU-class front-face policy: the grasp frame of each class's hand-curated
    aisle-facing side face (from the store front-face check). faces = {class: face},
    face in FACE_NORMALS. Fail-loud on a class not in the table — never guess a face;
    the caller must only pass classes it covers (product_patterns must match the keys)."""
    def __init__(self, faces: dict):
        assert faces, "PerClassFaceGrasp needs a non-empty {class: face} table"
        bad = {c: f for c, f in faces.items() if f not in FACE_NORMALS}
        assert not bad, f"faces must be side faces {sorted(FACE_NORMALS)} (±Z singular): {bad}"
        self.faces = dict(faces)

    def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
        assert cls in self.faces, \
            f"PerClassFaceGrasp: no face for class {cls!r} (keys={sorted(self.faces)})"
        return face_grasp_frames(lo, hi)[self.faces[cls]]

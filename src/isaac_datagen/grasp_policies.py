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

    def __init__(self, face: str):
        assert face in FACE_NORMALS, \
            f"face must be one of {sorted(FACE_NORMALS)} (side faces only, ±Z is singular): {face!r}"
        self.face = face

    def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
        return face_grasp_frames(lo, hi)[self.face]


class PerClassFaceGrasp:
    def __init__(self, faces: dict):
        assert faces, "PerClassFaceGrasp needs a non-empty {class: face} table"
        bad = {c: f for c, f in faces.items() if f not in FACE_NORMALS}
        assert not bad, f"faces must be side faces {sorted(FACE_NORMALS)} (±Z singular): {bad}"
        self.faces = dict(faces)

    def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
        assert cls in self.faces, \
            f"PerClassFaceGrasp: no face for class {cls!r} (keys={sorted(self.faces)})"
        return face_grasp_frames(lo, hi)[self.faces[cls]]

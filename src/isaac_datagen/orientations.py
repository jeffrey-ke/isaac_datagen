from __future__ import annotations

import sys

import numpy as np


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


def yaw_to_azimuth(grasp_point: np.ndarray, azimuth_deg: float, label: str = "") -> float:
    f = np.asarray(grasp_point)[:3, 0]
    assert abs(f[2]) < 1e-3, f"{label}grasp +X not horizontal: {f}"
    return float(azimuth_deg) - float(np.degrees(np.arctan2(f[1], f[0])))


class AlignGraspFronts:
    """Yaw each staged object's geo prim so grasp +X hits azimuth_deg (world)."""

    def __init__(self, azimuth_deg: float):
        self.azimuth_deg = float(azimuth_deg)

    def __call__(self, prim_paths, objects):
        from isaacsim.core.utils.stage import get_current_stage
        from isaac_datagen.isaac_utils import set_transform
        stage = get_current_stage()
        for path, obj in zip(prim_paths, objects, strict=True):
            geo = stage.GetPrimAtPath(f"{path}/geo")
            assert geo.IsValid(), f"no geo child under {path}"
            yaw = yaw_to_azimuth(obj.grasp_point, self.azimuth_deg,
                                 label=f"{obj.meta['name']}: ")
            set_transform(geo, rotation=(0.0, 0.0, yaw))

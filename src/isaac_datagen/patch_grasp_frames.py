"""Patch the 6 Z-rotated YCB grasp frames so grasp +X tracks the same physical face.

``rotate-graspable-meshes-z.md`` rotated these meshes about Z but left ``grasp_point``
untouched, so grasp +X now points at a different physical face (a corner for the -120 deg
mustard). Re-sync each by ``grasp_point <- R_z(angle) @ grasp_point`` with the same angle the
mesh was rotated by — both axes and origin rotate about the mesh Z, tracking the same face.

Idempotent: the pristine ``grasp_point/`` field is backed up once to ``grasp_point.orig.bak/``,
and every run restores from that backup before applying R_z, so re-runs never compound.

    uv run src/isaac_datagen/patch_grasp_frames.py <ycb_dataset_dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from isaac_datagen.objects import GraspableObject

# object name (meta["name"]) -> mesh Z-rotation that was applied, in degrees.
TARGETS = {
    "ycb_003_cracker_box": -90.0,
    "ycb_004_sugar_box": -90.0,
    "ycb_005_tomato_soup_can": 180.0,
    "ycb_006_mustard_bottle": -120.0,
    "ycb_007_tuna_fish_can": -90.0,
    "ycb_010_potted_meat_can": 180.0,
}


def rot_z(deg: float) -> np.ndarray:
    """4x4 rotation about +Z by ``deg`` degrees."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0, 0.0],
                     [s,  c, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def main() -> None:
    dataset = Path(sys.argv[1])
    gp_dir = dataset / "grasp_point"
    bak_dir = dataset / "grasp_point.orig.bak"

    # One-time pristine backup of the whole grasp_point/ field dir.
    if not bak_dir.exists():
        shutil.copytree(gp_dir, bak_dir)
        print(f"backed up pristine grasp frames -> {bak_dir}")

    n = len(sorted((dataset / "meta").glob("meta_*.yaml")))
    found: set[str] = set()
    for idx in range(n):
        name = GraspableObject.deserialize_field(idx, dataset, "meta")["name"]
        if name not in TARGETS:
            continue
        found.add(name)
        angle = TARGETS[name]

        # Restore the pristine grasp_point for this idx so R_z never compounds across runs.
        shutil.copy(bak_dir / f"grasp_point_{idx:04d}.npy", gp_dir / f"grasp_point_{idx:04d}.npy")

        obj = GraspableObject.deserialize(idx, dataset)            # grasp_point now pristine
        old = obj.grasp_point.copy()
        obj.grasp_point = (rot_z(angle) @ old).astype(np.float32)
        obj.serialize(idx, dataset, only={"grasp_point"})

        # Verify: rotation stays orthonormal and +X is exactly the rotated old +X.
        R = obj.grasp_point[:3, :3]
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4), f"{name}: det(R)={np.linalg.det(R)}"
        assert np.allclose(obj.grasp_point[:3, 0], rot_z(angle)[:3, :3] @ old[:3, 0], atol=1e-4)
        print(f"  [{idx:04d}] {name}: R_z({angle:+.0f})  +X {old[:3, 0].round(3)} -> {obj.grasp_point[:3, 0].round(3)}")

    missing = set(TARGETS) - found
    if missing:
        raise SystemExit(f"targets not found in {dataset}: {sorted(missing)}")
    print(f"patched {len(found)}/{len(TARGETS)} grasp frames in {dataset}")


if __name__ == "__main__":
    main()

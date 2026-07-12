from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from isaac_datagen.objects import OptFlowObject, UsdPath


def usdz_world_bbox(usdz_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usdz_path))
    prim = stage.GetPrimAtPath("/World")
    assert prim.IsValid(), f"no /World prim in {usdz_path}"
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(rng.GetMin()), np.array(rng.GetMax())


def grasp_from_ref_pose(ref_pose: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = -ref_pose[:3, 2].astype(np.float64)
    assert abs(x[2]) < 0.5, f"reference view too vertical for a side-face frame: {x}"
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    se3 = np.eye(4)
    se3[:3, :3] = np.column_stack([x, y, z])
    se3[:3, 3] = c + x * np.abs(half @ x)
    return se3.astype(np.float32)


def backfill_dir(cat: Path) -> None:
    n = len(sorted((cat / "meta").glob("meta_*.yaml")))
    print(f"{cat}: {n} objects")
    for idx in range(n):
        ref_pose = np.load(cat / "ref_pose" / f"ref_pose_{idx:04d}.npy")
        usdz = cat / "usd_path" / f"usd_path_{idx:04d}.usdz"
        lo, hi = usdz_world_bbox(usdz)
        grasp = grasp_from_ref_pose(ref_pose, lo, hi)
        align = float(np.dot(grasp[:3, 0], -ref_pose[:3, 2]))
        assert align > 0.95, f"{cat} idx {idx}: derived +X misaligned with ref view: {align:.4f}"
        stub = OptFlowObject(
            usd_path=UsdPath(str(usdz)), meta={}, reference_image=PILImage.new("RGB", (1, 1)),
            reference_depth=np.zeros((1, 1), np.float32), ref_intrinsics=np.eye(3, dtype=np.float32),
            ref_pose=ref_pose, grasp_point=grasp,
        )
        stub.serialize(idx, cat, only={"grasp_point"})
        print(f"  [{idx:04d}] +X={grasp[:3, 0].round(3).tolist()} align={align:.4f}")
    obj = OptFlowObject.deserialize(0, cat)
    assert obj.grasp_point.shape == (4, 4)
    print(f"  deserialize smoke test OK ({obj.meta.get('name')})")


def main() -> None:
    assert len(sys.argv) > 1, "usage: backfill_grasp_point.py <catalog_dir> [...]"
    for cat in map(Path, sys.argv[1:]):
        assert (cat / "ref_pose").is_dir(), f"not an OptFlowObject catalog: {cat}"
        backfill_dir(cat)


if __name__ == "__main__":
    main()

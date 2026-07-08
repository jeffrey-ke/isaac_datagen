"""One-time migration: backfill the mandatory ``OptFlowObject.grasp_point`` field
onto legacy OptFlowObject catalogs (assets/optflow_objects/{amazon, amazon-v2,
kleenex, ycb}), which predate the field.

The frame is recovered from data the catalog already holds — the inverse of
``graspableobj_to_optflow_obj.ref_pose_from_grasp``: the reference camera looks
along the grasp frame's -X at the bbox centroid, so ``+X = -ref_pose[:3, 2]``
(OpenCV +Z-forward), ``+Z = world up``, ``+Y = Z x X``, origin = bbox face
center along +X (bbox measured from the catalog's own usdz). For
``patch_grasp_frames``-patched catalogs this recovers the frame of the face the
reference was ACTUALLY rendered from — the semantically right value.

Residual-writes ``serialize(only={"grasp_point"})`` — a full re-serialize would
``shutil.SameFileError`` on the usdz copy. Runs OUTSIDE Isaac; needs standalone
pxr for the usdz bbox:

    uv run --with usd-core python debug_scripts/backfill_grasp_point.py \
        ../../assets/optflow_objects/amazon ../../assets/optflow_objects/amazon-v2 \
        ../../assets/optflow_objects/kleenex ../../assets/optflow_objects/ycb
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from isaac_datagen.objects import OptFlowObject, UsdPath


def usdz_world_bbox(usdz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of the usdz's composed content in its own world (== catalog local)
    frame — the same frame render_one measured via local_bbox_range(geo)."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usdz_path))
    prim = stage.GetPrimAtPath("/World")
    assert prim.IsValid(), f"no /World prim in {usdz_path}"
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(rng.GetMin()), np.array(rng.GetMax())


def grasp_from_ref_pose(ref_pose: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Inverse of ref_pose_from_grasp (side-face convention: +Z = up ⟂ +X)."""
    x = -ref_pose[:3, 2].astype(np.float64)
    assert abs(x[2]) < 0.5, f"reference view too vertical for a side-face frame: {x}"
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    se3 = np.eye(4)
    se3[:3, :3] = np.column_stack([x, y, z])
    se3[:3, 3] = c + x * np.abs(half @ x)      # bbox face center along +X (== face_grasp_frames)
    return se3.astype(np.float32)


def backfill_dir(cat: Path) -> None:
    n = len(sorted((cat / "meta").glob("meta_*.yaml")))     # same count idiom as collect_objects
    print(f"{cat}: {n} objects")
    for idx in range(n):
        ref_pose = np.load(cat / "ref_pose" / f"ref_pose_{idx:04d}.npy")
        usdz = cat / "usd_path" / f"usd_path_{idx:04d}.usdz"
        lo, hi = usdz_world_bbox(usdz)
        grasp = grasp_from_ref_pose(ref_pose, lo, hi)
        align = float(np.dot(grasp[:3, 0], -ref_pose[:3, 2]))
        assert align > 0.95, f"{cat} idx {idx}: derived +X misaligned with ref view: {align:.4f}"
        # Bootstrap trick (iid_to_visibility precedent): legacy dirs can't full-deserialize
        # once the field is mandatory, so build a placeholder instance and residual-write
        # ONLY the new field.
        stub = OptFlowObject(
            usd_path=UsdPath(str(usdz)), meta={}, reference_image=PILImage.new("RGB", (1, 1)),
            reference_depth=np.zeros((1, 1), np.float32), ref_intrinsics=np.eye(3, dtype=np.float32),
            ref_pose=ref_pose, grasp_point=grasp,
        )
        stub.serialize(idx, cat, only={"grasp_point"})
        print(f"  [{idx:04d}] +X={grasp[:3, 0].round(3).tolist()} align={align:.4f}")
    # smoke test: the full deserialize contract holds again
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

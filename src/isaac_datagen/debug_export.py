"""Dry-run debug export (complementary to debug_scene.py).

Two functions split at the natural mechanism/policy seam:

  decorate_debug_scene(...)   MECHANISM — bake debug cameras + grasp-frame axis
                              gizmos into the LIVE Isaac stage and return the
                              decorated scene plus the info a consumer needs. No I/O.
  export_debug_bundle(...)    POLICY    — persist that decorated scene: a standalone
                              scene.usdz + the planned poses (dryrun.npz/json).

Imported by clean_datagen.reference_segmentation() only when runtime.dry_run is
set; the real render path never touches this module, so the baked prims can never
affect data generation (no render products are attached to them, and the dry run
exits before any capture).

The exported bundle is consumed offline by blender_render.py: Blender imports
scene.usdz (geometry + the baked USD cameras through one importer transform) and
renders an orbit/turntable around the centroid plus one frame per planned pose.
"""

from __future__ import annotations

import json

import numpy as np

from isaac_datagen.isaac_utils import setup_camera, set_transform, export_subtree_usdz
from isaac_datagen.capture import get_target2world, set_prim_pose


def decorate_debug_scene(scene, world_poses):
    """MECHANISM: bake left-camera prims + grasp-frame axis gizmos into the live stage.

    Adds, under /World (so they ride along in an export of /World):
      - /World/DebugCameras/cam_NNNN: one USD camera per planned pose, with the
        dataset's intrinsics, at the LEFT camera's world pose (the RGB the dataset
        actually uses; ObsMaskWriter saves rps[0] = zed.left_rp).
      - /World/DebugGraspFrames/frame_BBB: RGB Cartesian axes on EVERY candidate
        grasp frame (scene.grasp_points), not just the sampled targets — the cameras
        already mark which targets the capture actually uses.

    Returns the decorated scene as a dict (stage + the arrays/paths a consumer needs).
    Performs no I/O and makes no decision about what to do with the scene.
    """
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.prims import create_prim

    stage = get_current_stage()
    zed = scene.zed
    # Dataset RGB is the LEFT camera. zed.left2rig owns the rig->camera offset, so
    # this can't drift from how the real cameras are placed in ZedMini.__init__.
    left_world = world_poses @ zed.left2rig                              # (F, 4, 4)

    create_prim("/World/DebugCameras", "Xform")
    cam_paths = []
    for f, T in enumerate(left_world):
        path = f"/World/DebugCameras/cam_{f:04d}"
        setup_camera(f"cam_{f:04d}", path, zed.width, zed.height, zed.intrinsics)
        set_prim_pose(path, T)                                          # shared SE3 mechanism (== capture)
        cam_paths.append(path)

    grasp_world = get_target2world(scene.grasp_points)                 # (G, 4, 4) — ALL candidates
    create_prim("/World/DebugGraspFrames", "Xform")
    frame_paths = []
    for b, T in enumerate(grasp_world):
        p = f"/World/DebugGraspFrames/frame_{b:03d}"
        _add_axis_gizmo(stage, p, T, length=0.10, radius=0.006)
        frame_paths.append(p)

    # Blender's USD importer skips untyped prims when building its object hierarchy,
    # which severs the transform chain at the untyped reference-child wrappers (the
    # bug-doc fix child) and collapses every box to the origin. Typing them Xform is
    # semantically neutral and restores placement on import. Dry-run only.
    _retype_untyped_for_blender(stage)

    return {
        "stage": stage,
        "intrinsics": zed.intrinsics,
        "width": zed.width,
        "height": zed.height,
        "world_poses": world_poses,
        "left_world": left_world,
        "grasp_world": grasp_world,
        "cam_paths": cam_paths,
        "frame_paths": frame_paths,
    }


def export_debug_bundle(info, render_dir):
    """POLICY: persist the decorated scene for offline (Blender) inspection.

    Writes <render_dir>/debug/{scene.usdz, dryrun.npz, dryrun.json}.
    """
    out = render_dir / "debug"
    out.mkdir(parents=True, exist_ok=True)

    usdz = export_subtree_usdz(info["stage"], "/World", str(out), base_name="scene")

    np.savez(
        out / "dryrun.npz",
        K=info["intrinsics"],
        width=info["width"],
        height=info["height"],
        world_poses=info["world_poses"],
        left_world=info["left_world"],
        grasp_world=info["grasp_world"],
    )
    # Human-readable companion (counts/paths) for inspection and the Blender driver.
    (out / "dryrun.json").write_text(json.dumps({
        "scene_usdz": usdz,
        "num_poses": int(len(info["world_poses"])),
        "num_grasp_frames": int(len(info["grasp_world"])),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "cam_paths": info["cam_paths"],
        "frame_paths": info["frame_paths"],
    }, indent=2))
    return usdz


def _retype_untyped_for_blender(stage):
    """Type every untyped prim as Xform so Blender's USD importer keeps the
    transform chain (it skips untyped prims, collapsing reference-wrapped boxes to
    the origin on import). Semantically neutral; the Shader/Material/Camera/Xform
    prims we author are already typed, so only build_scene's `geo` wrappers change."""
    for p in stage.Traverse():
        if not p.GetTypeName():
            p.SetTypeName("Xform")


def _add_axis_gizmo(stage, base, pose, length, radius):
    """Three RGB cylinders named x/y/z along local X/Y/Z at world `pose`.

    UsdGeom.Cylinder is centered on its local +Z; each axis cylinder is rotated so
    its +Z aims down the target axis and shifted +length/2 so it grows from the
    origin. displayColor tags X=red/Y=green/Z=blue for native USD viewers; the
    Blender renderer recolors by child name (x/y/z) because UsdShade material
    bindings (relationships) don't survive the usdz reference-flatten.
    """
    from pxr import UsdGeom, Gf
    from isaacsim.core.utils.prims import create_prim

    create_prim(base, "Xform")
    set_prim_pose(base, pose)                                           # frame origin via the shared SE3 mechanism
    # (name, rgb, local euler (deg) aiming +Z onto the axis, local +length/2 shift)
    axes = [
        ("x", (1.0, 0.0, 0.0), (0.0, 90.0, 0.0), (length / 2, 0.0, 0.0)),
        ("y", (0.0, 1.0, 0.0), (-90.0, 0.0, 0.0), (0.0, length / 2, 0.0)),
        ("z", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, length / 2)),
    ]
    for name, rgb, rot, tr in axes:
        cyl = UsdGeom.Cylinder.Define(stage, f"{base}/{name}")
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(length)
        cyl.GetAxisAttr().Set("Z")
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
        set_transform(cyl.GetPrim(), translation=list(tr), rotation=list(rot))  # fixed LOCAL offset

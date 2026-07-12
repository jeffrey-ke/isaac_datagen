
from __future__ import annotations

import json

import numpy as np

from isaac_datagen.isaac_utils import setup_camera, set_transform, export_subtree_usdz
from isaac_datagen.capture import get_target2world, set_prim_pose


def decorate_debug_scene(scene, world_poses):
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.prims import create_prim

    stage = get_current_stage()
    zed = scene.zed
    left_world = world_poses @ zed.left2rig

    create_prim("/World/DebugCameras", "Xform")
    cam_paths = []
    for f, T in enumerate(left_world):
        path = f"/World/DebugCameras/cam_{f:04d}"
        setup_camera(f"cam_{f:04d}", path, zed.width, zed.height, zed.intrinsics)
        set_prim_pose(path, T)
        cam_paths.append(path)

    grasp_world = get_target2world(scene.grasp_points)
    create_prim("/World/DebugGraspFrames", "Xform")
    frame_paths = []
    for b, T in enumerate(grasp_world):
        p = f"/World/DebugGraspFrames/frame_{b:03d}"
        _add_axis_gizmo(stage, p, T, length=0.10, radius=0.006)
        frame_paths.append(p)

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
    for p in stage.Traverse():
        if not p.GetTypeName():
            p.SetTypeName("Xform")


def _add_axis_gizmo(stage, base, pose, length, radius):
    from pxr import UsdGeom, Gf
    from isaacsim.core.utils.prims import create_prim

    create_prim(base, "Xform")
    set_prim_pose(base, pose)
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
        set_transform(cyl.GetPrim(), translation=list(tr), rotation=list(rot))

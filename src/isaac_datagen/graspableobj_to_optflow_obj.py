"""Offline reference render: a GraspableObject dataset -> a OptFlowObject dataset.

Boot Isaac once, then for each object render an isolated perspective RGB-D view from a camera
anchored on the (Stage-0 patched) grasp frame, using the SAME ``setup_camera`` pinhole +
``distance_to_image_plane`` annotator as the observation renderer -> exact intrinsics/depth
parity with the capture stage. Decoupled-offline, like ``mesh_convert.py``.

    uv run src/isaac_datagen/optflow_render.py <config.yaml> <graspable_dataset> <out_dataset> [key=val ...]

The grasp frame defines the viewpoint: its origin is the face center and its +X column is the
outward face normal (face_grasp_frames, mesh_convert.py), so the camera sits out along +X and
looks back at the origin. The stored ``ref_pose`` is OpenCV (+Z-forward); the OpenGL pose is
transient, used only to position the Isaac camera prim (Isaac cameras look down -Z).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from isaac_datagen.runtime_config import load_config
from isaac_datagen.scene import boot_sim, warmup_render, make_dome_light
from isaac_datagen.capture import set_prim_pose
from isaac_datagen.objects import GraspableObject, OptFlowObject, UsdPath
from isaac_datagen.isaac_utils import setup_camera, setup_render_product, load_asset, local_bbox_range
from vision_core.pose_utils import look_at, cv2opengl

REF_DOME_INTENSITY = 1000.0   # even, shadow-free reference light; TUNE on first render (boot_sim exposure is fixed)
MARGIN = 1.1                  # standoff slack so the grasp face fills (not to-edge) the frame


def ref_pose_from_grasp(grasp_point, lo, hi, K, width, height, margin=MARGIN):
    """OpenCV (+Z-forward) camera2local pose framing the grasp face.

    Camera sits out along grasp +X and looks back at the grasp origin; the standoff is sized so
    the in-plane face extents fit the FOV. ``lo``/``hi`` are the mesh-local bbox corners.
    """
    origin = grasp_point[:3, 3]
    normal = grasp_point[:3, 0] / np.linalg.norm(grasp_point[:3, 0])      # outward +X face normal
    up = np.array([0.0, 0.0, 1.0])
    half_w = 0.5 * float(np.dot(hi - lo, np.abs(np.cross(up, normal))))   # in-plane horizontal half-extent
    half_h = 0.5 * float(hi[2] - lo[2])                                   # vertical (world-up) half-extent
    fx, fy = float(K[0, 0]), float(K[1, 1])
    hfov, vfov = 2 * np.arctan(width / (2 * fx)), 2 * np.arctan(height / (2 * fy))
    d = margin * max(half_w / np.tan(hfov / 2), half_h / np.tan(vfov / 2))
    return look_at(at_coord=origin, from_coord=origin + d * normal)       # OpenCV camera2local


def render_one(app, rep, obj, K, width, height, runtime):
    """Render one isolated object -> (rgb HxWx3 uint8, masked depth HxW float32, ref_pose_cv 4x4)."""
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    from isaacsim.core.utils.semantics import add_labels                  # as scene.add_object

    create_new_stage()                                                    # clean slate per object
    make_dome_light(get_current_stage(), "/World", intensity=REF_DOME_INTENSITY)

    geo = load_asset("/World/geo", str(obj.usd_path), ref_prim_path="/World")  # usdz content lives under /World
    add_labels(geo, labels=[obj.meta["name"]], instance_name="instance")  # -> nonzero iid for masking

    rng = local_bbox_range(geo)                                           # mesh-local extent (object at origin)
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    ref_pose_cv = ref_pose_from_grasp(obj.grasp_point, lo, hi, K, width, height)

    setup_camera("ref_cam", "/World/ref_cam", width, height, K)           # OpenCV pinhole, same builder as ZedMini
    set_prim_pose("/World/ref_cam", cv2opengl(ref_pose_cv))               # GL pose on the prim (Isaac cam = -Z fwd)

    rp = setup_render_product("/World/ref_cam", (width, height), "ref")
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    a_seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": False})
    for a in (a_rgb, a_dep, a_seg):
        a.attach(rp)

    warmup_render(app, runtime.warmup_frames)                             # settle RTX (lit-vs-black fix)
    rep.orchestrator.step(rt_subframes=runtime.rt_subframes)              # PT subframes accumulate (boot_sim setting)

    rgb = np.asarray(a_rgb.get_data())[:, :, :3]
    depth = np.asarray(a_dep.get_data(), dtype=np.float32)
    seg = a_seg.get_data()
    seg = seg["data"] if isinstance(seg, dict) else seg                   # isolated stage -> object is the only nonzero id
    depth = np.where(seg != 0, depth, 0.0).astype(np.float32)             # bg -> 0 (warp_kpts nonzero_mask)
    return rgb, depth, ref_pose_cv


def main() -> None:
    """
       uv run src/isaac_datagen/graspableobj_to_optflow_obj.py \
          src/isaac_datagen/configs/randomized.yaml \
          datasets/ycb_dataset \
          datasets/ycb_preoptflow \
          dataset_dir=datasets/ycb_dataset
    """
    runtime = load_config(sys.argv[1], sys.argv[4:])                      # render settings + intrinsics_path
    in_dir, out_dir = Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    K = np.load(runtime.intrinsics_path).astype(np.float32)               # ref K = obs K by default
    width, height = runtime.width, runtime.height

    app = boot_sim(runtime, out_dir)
    import omni.replicator.core as rep
    from PIL import Image as PILImage

    n = len(sorted((in_dir / "meta").glob("meta_*.yaml")))                # same count idiom as collect_objects
    for idx in range(n):
        obj = GraspableObject.deserialize(idx, in_dir)
        rgb, depth, ref_pose_cv = render_one(app, rep, obj, K, width, height, runtime)
        OptFlowObject(
            usd_path=UsdPath(str(obj.usd_path)),
            meta=obj.meta,
            reference_image=PILImage.fromarray(rgb),
            reference_depth=depth,
            ref_intrinsics=K,
            ref_pose=ref_pose_cv.astype(np.float32),
        ).serialize(idx, out_dir)
        print(f"  [{idx:04d}] {obj.meta['name']} rendered", flush=True)
    app.close()


if __name__ == "__main__":
    main()

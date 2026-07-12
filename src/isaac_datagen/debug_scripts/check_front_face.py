from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from isaac_datagen import grasp_policies
from isaac_datagen.extract_store_objects import matched_products, parse_sku
from isaac_datagen.grasp_policies import FixedFaceGrasp
from isaac_datagen.mesh_convert import FACE_NORMALS
from isaac_datagen.isaac_utils import (
    untransformed_bbox_range, setup_camera, setup_render_product, find_prims)
from isaac_datagen.graspableobj_to_optflow_obj import ref_pose_from_grasp
from isaac_datagen.capture import get_target2world, set_prim_pose
from isaac_datagen.measure_luminance import frame_luminance
from isaac_datagen.runtime_config import load_config
from isaac_datagen.scene import boot_sim, warmup_render
from vision_core.pose_utils import cv2opengl

CAM_PATH = "/World/ref_cam"
BLACK_EPS = 1.0
BLACK_EXIT = 3


def class_representatives(store, patterns, facings):
    by_class: dict[str, list[tuple[str, str]]] = {}
    for path in matched_products(store, patterns):
        name, cls = parse_sku(path.rsplit("/", 1)[1])
        by_class.setdefault(cls, []).append((name, path))
    reps = [(cls, name, path)
            for cls in sorted(by_class)
            for name, path in sorted(by_class[cls])[:facings]]
    return reps, {cls: len(members) for cls, members in by_class.items()}


def category_of(cls):
    return re.sub(r"\d+$", "", cls)


def scale_store_lights(store, specs, factor):
    from pxr import Usd, UsdLux
    assert specs, "--light-scale needs light_jitter_patterns (root/pattern) in the config"
    stage = store.GetStage()
    n = 0
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for spec in specs:
            for p in find_prims(spec.root, spec.pattern):
                attr = UsdLux.LightAPI(stage.GetPrimAtPath(p)).GetIntensityAttr()
                if attr and attr.IsValid():
                    attr.Set(float(attr.Get()) * factor); n += 1
    assert n, f"--light-scale matched no UsdLux lights via {specs}"
    print(f"[light] fixed-scaled {n} UsdLux lights x{factor}", flush=True)


def face_policies(spec, all_faces):
    if all_faces:
        return [(f, FixedFaceGrasp(f)) for f in sorted(FACE_NORMALS)]
    face = spec.grasp_frame_policy_args.get("face", spec.grasp_frame_policy)
    policy = grasp_policies.get(spec.grasp_frame_policy)(**spec.grasp_frame_policy_args)
    return [(face, policy)]


def v0_prim(store, model_path):
    stage = store.GetStage()
    v0 = stage.GetPrimAtPath(f"{model_path}/v_0")
    if v0.IsValid():
        return v0
    kids = [f"{c.GetName()} ({c.GetTypeName()})"
            for c in stage.GetPrimAtPath(model_path).GetChildren()]
    print(f"[skip] {model_path}: no v_0 node -- not a standard product; children={kids}", flush=True)
    return None


def bbox_at_v0(v0):
    rng = untransformed_bbox_range(v0)
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    assert (hi > lo).all(), f"empty bbox at {v0.GetPath()}"
    return lo, hi


def store_camera_pose(model_path, lo, hi, grasp, K, w, h):
    ref_pose_cv = ref_pose_from_grasp(grasp, lo, hi, K, w, h)
    l2w = get_target2world([f"{model_path}/v_0"])[0]
    return l2w @ ref_pose_cv


def render_view(app, rep, a_rgb, pose_gl, runtime):
    set_prim_pose(CAM_PATH, pose_gl)
    warmup_render(app, runtime.warmup_frames)
    rep.orchestrator.step(rt_subframes=runtime.rt_subframes)
    return np.asarray(a_rgb.get_data())[:, :, :3]


def whole_frame_luminance(rgb):
    h, w = rgb.shape[:2]
    obs = np.concatenate(
        [rgb.transpose(2, 0, 1), np.full((1, h, w), 255, np.uint8)], axis=0)
    fg_mean, _ = frame_luminance(obs, pixel_threshold=8.0)
    return fg_mean


def contact_sheet(records, cross_xy, out_png, cols):
    from vision_core.viz import panel_grid, save_figure
    fig, axes = panel_grid(len(records), cols, panel_w=4.0, panel_h=4.2, wspace=0.05, hspace=0.15)
    cx, cy = cross_xy
    for ax, (tile_path, title) in zip(axes, records):
        ax.imshow(PILImage.open(tile_path))
        ax.axhline(cy, color="cyan", lw=0.6, alpha=0.7)
        ax.axvline(cx, color="cyan", lw=0.6, alpha=0.7)
        ax.plot([cx], [cy], "+", color="red", ms=12, mew=1.6)
        ax.set_title(title, fontsize=7); ax.axis("off")
    save_figure(fig, out_png, dpi=130)


def main():
    ap = argparse.ArgumentParser(description="Probe the aisle-facing front-face assumption.")
    ap.add_argument("config", help="store config yaml (scene_builder: build_store_scene)")
    ap.add_argument("outdir", type=Path, help="output dir for per-category montages + tiles/")
    ap.add_argument("--facings", type=int, default=1,
                    help="reps rendered per class (first N by sorted name); >1 checks "
                         "placement variation (facings may be rotated -- end-caps, opposite aisle)")
    ap.add_argument("--all-faces", action="store_true",
                    help="render all 4 side faces per rep (row) to see WHICH face is the aisle front")
    ap.add_argument("--light-scale", type=float, default=1.0,
                    help="FIXED multiply of the store's own light intensities (e.g. 8.0 to rule "
                         "out dim-bottom-shelf darkness); 1.0 = authored. No per-frame jitter.")
    ap.epilog = ("Trailing OmegaConf dotlist key=val tokens are passed through to load_config "
                 "(e.g. scene_builder_args.product_patterns=[model_snack*,model_cereal*]).")
    args, overrides = ap.parse_known_args()
    tiles_dir = args.outdir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_config(args.config, overrides)
    assert runtime.scene_builder == "build_store_scene", \
        "check_front_face needs a store config (scene_builder: build_store_scene)"
    K = np.load(runtime.intrinsics_path).astype(np.float32)
    W, H = runtime.width, runtime.height

    app = boot_sim(runtime, args.outdir)
    import omni.replicator.core as rep

    from isaac_datagen.store_scene import StoreSceneSpec, load_store
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    store = load_store(spec)
    if args.light_scale != 1.0:
        scale_store_lights(store, runtime.light_jitter_patterns, args.light_scale)

    reps, mult = class_representatives(store, spec.product_patterns, args.facings)
    print(f"[survey] {sum(mult.values())} product prims across {len(mult)} classes "
          f"(rendering {args.facings} facing(s)/class)", flush=True)
    for cls in sorted(mult):
        print(f"    {cls:24s} x{mult[cls]}", flush=True)

    faces = face_policies(spec, args.all_faces)
    print(f"[faces] {[f for f, _ in faces]} per representative", flush=True)

    setup_camera("ref_cam", CAM_PATH, W, H, K)
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_rgb.attach(setup_render_product(CAM_PATH, (W, H), "ref"))

    by_cat: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    first = True
    for cls, name, model_path in reps:
        v0 = v0_prim(store, model_path)
        if v0 is None:
            continue
        lo, hi = bbox_at_v0(v0)
        for face_label, policy in faces:
            grasp = policy(lo, hi, cls)
            pose_gl = cv2opengl(store_camera_pose(model_path, lo, hi, grasp, K, W, H))
            rgb = render_view(app, rep, a_rgb, pose_gl, runtime)
            if first:
                first = False
                lum = whole_frame_luminance(rgb)
                print(f"[black-gate] first-frame whole-frame luma = {lum:.3f}", flush=True)
                if not (lum > BLACK_EPS):
                    print("process came up black -- relaunch", flush=True)
                    app.close(); sys.exit(BLACK_EXIT)
            tile_path = tiles_dir / f"{cls}__{face_label}.png"
            PILImage.fromarray(rgb).save(tile_path)
            by_cat[category_of(cls)].append((tile_path, f"{cls} / {name} / {face_label}"))
            print(f"    rendered {name:28s} face={face_label}", flush=True)

    cross = (float(K[0, 2]), float(K[1, 2]))
    for cat, records in sorted(by_cat.items()):
        out_png = args.outdir / f"{cat}.png"
        contact_sheet(records, cross, out_png, cols=len(faces))
        print(f"wrote {out_png}  ({len(records)} tiles)", flush=True)
    print(f"[done] {len(by_cat)} category sheets + {sum(len(r) for r in by_cat.values())} tiles "
          f"in {args.outdir}", flush=True)
    app.close()


if __name__ == "__main__":
    main()

"""One-off probe: verify the "-Y is the aisle-facing front face" assumption before
the expensive full store extraction.

For ONE representative product prim per SKU class (first by sorted name -- the prim
reference_catalog would pick), boot Isaac once, load the LIVE store USD, place a
camera along the hypothesized grasp-frame outward normal, and render the FULL store
RGB (NOT an isolated object). Camera into open aisle => front; buried in shelf =>
wrong face. Saves per-tile PNGs + ONE montage per category, and prints the class
count + per-class facing multiplicity (answering ~414 prims vs ~64 classes).

    uv run debug_scripts/check_front_face.py <store_config.yaml> <outdir>
        [--facings N] [--all-faces] [key=val ...]

Products that do NOT follow the model_*/v_0 convention (e.g. model_drink101) are
SKIPPED with a warning that dumps their actual children -- one odd prim must not
tank the run, and the same v_0 assumption lives in Stage-A extract_one.

Camera crux (verified invariant  cam2world = l2w(v_0) @ ref_pose): ref_pose_from_grasp
returns camera2LOCAL; compose with l2w read at EXACTLY v_0 (op-free modeling frame).

HARD GATE: ~60% of processes render pure-black for their whole lifetime (decided once
at renderer init, immutable). We render+measure ONE frame first; if whole-frame luma
is ~0 we print "process came up black -- relaunch", close, and sys.exit(3) so a shell
`until ...; do ...; done` wrapper relaunches.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # BEFORE any import that pulls pyplot (mesh_convert via grasp_policies)

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

CAM_PATH = "/World/ref_cam"     # single reusable camera prim (same idiom as render_one)
BLACK_EPS = 1.0                 # whole-frame BT.709 luma (0..255) below this => black process
BLACK_EXIT = 3                  # relaunch signal for the `until` shell wrapper


def class_representatives(store, patterns, facings):
    """(reps, multiplicity): group matched product prims by SKU class, take the first
    `facings` members per class by sorted NAME (reference_catalog convention).
    reps: [(cls, name, model_path), ...]; multiplicity: {cls: n_prims} (shelf facings)."""
    by_class: dict[str, list[tuple[str, str]]] = {}
    for path in matched_products(store, patterns):          # sorted, deduped abs prim paths
        name, cls = parse_sku(path.rsplit("/", 1)[1])
        by_class.setdefault(cls, []).append((name, path))
    reps = [(cls, name, path)
            for cls in sorted(by_class)
            for name, path in sorted(by_class[cls])[:facings]]
    return reps, {cls: len(members) for cls, members in by_class.items()}


def category_of(cls):
    """Vendor category = SKU class minus its trailing SKU number (cereal001 -> cereal,
    instant_beverages012 -> instant_beverages). Groups the per-category montage sheets."""
    return re.sub(r"\d+$", "", cls)


def scale_store_lights(store, specs, factor):
    """FIXED (non-jittered) intensity scale of the store's own UsdLux lights, applied
    ONCE. Rules out a face reading dark because its product sits on a dim bottom shelf
    (vs dark from shelf occlusion). Same find + LightAPI + root-layer-override seam as
    scene.register_light_pattern_jitter, but a single constant factor, not per-frame
    log-uniform. Root-layer EditContext beats the store's reference arc."""
    from pxr import Usd, UsdLux
    assert specs, "--light-scale needs light_jitter_patterns (root/pattern) in the config"
    stage = store.GetStage()
    n = 0
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for spec in specs:
            for p in find_prims(spec.root, spec.pattern):        # raises if a pattern matches nothing
                attr = UsdLux.LightAPI(stage.GetPrimAtPath(p)).GetIntensityAttr()
                if attr and attr.IsValid():
                    attr.Set(float(attr.Get()) * factor); n += 1
    assert n, f"--light-scale matched no UsdLux lights via {specs}"
    print(f"[light] fixed-scaled {n} UsdLux lights x{factor}", flush=True)


def face_policies(spec, all_faces):
    """[(face_label, policy), ...]. Default = the config's FixedFaceGrasp face (faithful
    "-Y" check); --all-faces = all 4 side faces (shows WHICH face is the aisle front)."""
    if all_faces:
        return [(f, FixedFaceGrasp(f)) for f in sorted(FACE_NORMALS)]
    face = spec.grasp_frame_policy_args.get("face", spec.grasp_frame_policy)
    policy = grasp_policies.get(spec.grasp_frame_policy)(**spec.grasp_frame_policy_args)
    return [(face, policy)]


def v0_prim(store, model_path):
    """The op-free v_0 modeling node under a product, or None if the product does not
    follow the model_*/v_0 convention (e.g. model_drink101). On a miss, dump the
    product's actual children so the odd structure is visible -- one non-standard prim
    must not crash the run, and Stage-A extract_one shares this v_0 assumption."""
    stage = store.GetStage()
    v0 = stage.GetPrimAtPath(f"{model_path}/v_0")
    if v0.IsValid():
        return v0
    kids = [f"{c.GetName()} ({c.GetTypeName()})"
            for c in stage.GetPrimAtPath(model_path).GetChildren()]
    print(f"[skip] {model_path}: no v_0 node -- not a standard product; children={kids}", flush=True)
    return None


def bbox_at_v0(v0):
    """usdz-frame bbox (lo, hi) of the op-free v_0 node -- excludes its own ops
    (matches extract_store_objects.extract_one)."""
    rng = untransformed_bbox_range(v0)
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    assert (hi > lo).all(), f"empty bbox at {v0.GetPath()}"
    return lo, hi


def store_camera_pose(model_path, lo, hi, grasp, K, w, h):
    """OpenCV camera2WORLD for one product: ref_pose_from_grasp -> camera2LOCAL (looks at
    centroid along grasp +X outward normal, standoff sized to fill FOV); compose with l2w
    read at v_0. l2w's scale multiplies centroid and offset together, so framing holds."""
    ref_pose_cv = ref_pose_from_grasp(grasp, lo, hi, K, w, h)     # camera2LOCAL (OpenCV)
    l2w = get_target2world([f"{model_path}/v_0"])[0]             # (4,4) local(v_0)->world
    return l2w @ ref_pose_cv                                      # camera2WORLD (OpenCV)


def render_view(app, rep, a_rgb, pose_gl, runtime):
    """Reposition the reusable camera to the OpenGL pose and render the full store RGB
    (mirrors graspableobj_to_optflow_obj.render_one's proven RTX sequence)."""
    set_prim_pose(CAM_PATH, pose_gl)                            # authored in the root layer
    warmup_render(app, runtime.warmup_frames)                  # settle RTX (lit-vs-black)
    rep.orchestrator.step(rt_subframes=runtime.rt_subframes)   # PT subframes accumulate
    return np.asarray(a_rgb.get_data())[:, :, :3]              # (H, W, 3) uint8


def whole_frame_luminance(rgb):
    """Whole-frame BT.709 luma via shared measure_luminance.frame_luminance (alpha forced
    opaque => foreground = every pixel; never nan). No hand-rolled weights."""
    h, w = rgb.shape[:2]
    obs = np.concatenate(
        [rgb.transpose(2, 0, 1), np.full((1, h, w), 255, np.uint8)], axis=0)  # (4,H,W) RGBA
    fg_mean, _ = frame_luminance(obs, pixel_threshold=8.0)
    return fg_mean


def contact_sheet(records, cross_xy, out_png, cols):
    """One montage from saved tile PNGs. records: [(tile_path, title), ...] laid out
    class-major (rows) x face-minor (cols). Crosshair at the look-at point (principal
    point = target under test). Built on vision_core.viz -- NOT a third write_grid."""
    from vision_core.viz import panel_grid, save_figure
    fig, axes = panel_grid(len(records), cols, panel_w=4.0, panel_h=4.2, wspace=0.05, hspace=0.15)
    cx, cy = cross_xy
    for ax, (tile_path, title) in zip(axes, records):
        ax.imshow(PILImage.open(tile_path))
        ax.axhline(cy, color="cyan", lw=0.6, alpha=0.7)
        ax.axvline(cx, color="cyan", lw=0.6, alpha=0.7)
        ax.plot([cx], [cy], "+", color="red", ms=12, mew=1.6)   # look-at = target under test
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
    # parse_known_args: leftover key=val tokens -> OmegaConf dotlist (order-independent vs --flags;
    # a plain positional 'overrides' would drop tokens that follow an optional like --all-faces).
    args, overrides = ap.parse_known_args()
    tiles_dir = args.outdir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_config(args.config, overrides)
    assert runtime.scene_builder == "build_store_scene", \
        "check_front_face needs a store config (scene_builder: build_store_scene)"
    K = np.load(runtime.intrinsics_path).astype(np.float32)
    W, H = runtime.width, runtime.height

    app = boot_sim(runtime, args.outdir)                        # boot FIRST (creates SimulationApp)
    import omni.replicator.core as rep

    from isaac_datagen.store_scene import StoreSceneSpec, load_store
    spec = StoreSceneSpec(**runtime.scene_builder_args)         # validates store_usd/patterns/policy
    store = load_store(spec)                                    # create_new_stage + /World + ref store
    if args.light_scale != 1.0:                                 # fixed brighten (no jitter) before any render
        scale_store_lights(store, runtime.light_jitter_patterns, args.light_scale)

    reps, mult = class_representatives(store, spec.product_patterns, args.facings)
    print(f"[survey] {sum(mult.values())} product prims across {len(mult)} classes "
          f"(rendering {args.facings} facing(s)/class)", flush=True)
    for cls in sorted(mult):
        print(f"    {cls:24s} x{mult[cls]}", flush=True)

    faces = face_policies(spec, args.all_faces)
    print(f"[faces] {[f for f, _ in faces]} per representative", flush=True)

    setup_camera("ref_cam", CAM_PATH, W, H, K)                  # OpenCV pinhole (same builder as ZedMini)
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_rgb.attach(setup_render_product(CAM_PATH, (W, H), "ref"))

    by_cat: dict[str, list[tuple[Path, str]]] = defaultdict(list)   # category -> [(tile_path, title)]
    first = True
    for cls, name, model_path in reps:
        v0 = v0_prim(store, model_path)
        if v0 is None:                                          # non-standard product (e.g. drink101)
            continue
        lo, hi = bbox_at_v0(v0)
        for face_label, policy in faces:
            grasp = policy(lo, hi, cls)                         # (4,4) local grasp SE3, +X = outward
            pose_gl = cv2opengl(store_camera_pose(model_path, lo, hi, grasp, K, W, H))
            rgb = render_view(app, rep, a_rgb, pose_gl, runtime)
            if first:                                           # black-process gate on FIRST frame
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

    cross = (float(K[0, 2]), float(K[1, 2]))                    # principal point = look-at projection
    for cat, records in sorted(by_cat.items()):                # one scannable sheet per category
        out_png = args.outdir / f"{cat}.png"
        contact_sheet(records, cross, out_png, cols=len(faces))
        print(f"wrote {out_png}  ({len(records)} tiles)", flush=True)
    print(f"[done] {len(by_cat)} category sheets + {sum(len(r) for r in by_cat.values())} tiles "
          f"in {args.outdir}", flush=True)
    app.close()


if __name__ == "__main__":
    main()

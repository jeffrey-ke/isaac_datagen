"""USD scene building — boxes, stacks, lighting, assets, sim bootstrap."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_datagen.isaac_utils import load_asset, set_transform, bounding_half_extents, find_prims, create_empty
from isaac_datagen.hardwares import ZedMini
from isaac_datagen.objects import OccupancyGrid, UntilExhaustedStacker, GraspableObject
from isaac_datagen.pose_planning import plan_poses


@dataclass(frozen=True)
class SceneHandle:
    zed: ZedMini
    grasp_points: list
    objects: List[GraspableObject]

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def bbox_size_of(prim_path):
    from isaacsim.core.utils.stage import get_current_stage
    prim = get_current_stage().GetPrimAtPath(prim_path)
    hx, hy, hz = bounding_half_extents(prim)
    return (2 * hx, 2 * hy, 2 * hz)

def add_object(*, at_parent: str, obj: GraspableObject) -> str:
    # Wrap the reference: Isaac Sim's transform path ignores xformOps authored
    # directly on a reference-carrying prim. The placement transform must land
    # on the plain Xform wrapper; the reference lives on a child that inherits
    # it. The source .usdz files have no defaultPrim, so reference "/World".
    from pxr import Usd, Tf
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.semantics import add_labels
    # USD prim names must be valid identifiers; YCB names like "007_tuna_fish_can"
    # lead with a digit (illegal -> SdfPath parses to <> and DefinePrim throws).
    # Sanitize the path component only; the true name still rides on the instance
    # semantic label below, which is what the iid->name->class chain reads.
    wrapper_path = f"{at_parent}/{Tf.MakeValidIdentifier(obj.meta['name'])}"
    geo_path = f"{wrapper_path}/geo"
    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.DefinePrim(wrapper_path, "Xform")
    geo = load_asset(geo_path, obj.usd_path, ref_prim_path="/World")
    add_labels(geo, labels=[obj.meta["class"]], instance_name="class")
    add_labels(geo, labels=[obj.meta["name"]], instance_name="instance")
    return wrapper_path

def organize_objects(policy: Callable, prim_paths: List[str]):
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    translations_rotations = [policy(path) for path in prim_paths]
    for (translation, rotation), prim_path in zip(translations_rotations, prim_paths):
        set_transform(stage.GetPrimAtPath(prim_path), translation=translation, rotation=rotation)


def create_stack_of_objects(parent_path, objects: List[GraspableObject], runtime):
    from isaacsim.core.utils.prims import create_prim
    stack_prim = create_prim(f"{parent_path}/stack", "Xform")
    stack_path = stack_prim.GetPath().pathString

    if runtime.placement == "occupancy_grid":
        # STATIC FULL-WALL policy: the grid is all-ones, so every slot must
        # receive a box -- otherwise is_top/is_graspable describe phantom boxes
        # that were never placed. Enforce enough objects to fill the wall.
        dims = runtime.pallet_dims
        capacity = dims[0] * dims[1] * dims[2]
        if len(objects) < capacity:
            raise ValueError(
                f"create_stack_of_objects: full-wall pallet {tuple(dims)} needs {capacity} "
                f"objects, got {len(objects)}. Supply more objects or shrink pallet_dims."
            )
        prim_paths_added = [add_object(at_parent=stack_path, obj=o) for o in objects[:capacity]]
        # Uniform-box policy: cell size taken from the first object's bbox.
        policy = OccupancyGrid(dims, bbox_size_of(prim_paths_added[0]))

    elif runtime.placement == "until_exhausted_stacker":
        # Heterogeneous policy: places ALL objects in columns of <= column_height.
        if len(objects) < 1:
            raise ValueError("until_exhausted_stacker needs >= 1 object")
        prim_paths_added = [add_object(at_parent=stack_path, obj=o) for o in objects]
        # Built AFTER add_object so it can measure each prim's bbox from the stage.
        policy = UntilExhaustedStacker(prim_paths_added, runtime.column_height)

    else:
        raise ValueError(f"unknown placement policy: {runtime.placement!r}")

    organize_objects(policy=policy, prim_paths=prim_paths_added)
    is_graspable = policy.graspability()
    return stack_path, prim_paths_added, is_graspable

def collect_shader_paths(stack_path):
    shader_paths = []
    def collect(box_path):
        find_prims(f"{box_path}/_materials/Material", "Image_Texture",
                   action=lambda p: shader_paths.append(p))
    find_prims(stack_path, "Box_*", action=collect)
    return shader_paths


def randomize_box_textures(stack_path, texture_paths, rng):
    from pxr import Sdf, UsdShade
    from isaacsim.core.utils.stage import get_current_stage

    def set_random_texture(box_path):
        asset = Sdf.AssetPath(texture_paths[rng.randint(len(texture_paths))])

        def set_file_input(shader_path):
            prim = get_current_stage().GetPrimAtPath(shader_path)
            UsdShade.Shader(prim).GetInput("file").Set(asset)

        find_prims(f"{box_path}/_materials/Material", "Image_Texture", action=set_file_input)

    find_prims(stack_path, "Box_*", action=set_random_texture)


class ReplicatorWrapper:
    def __init__(self, rep):
        self.rep = rep
        self._randomizers = []

    def register(self, fn):
        self.rep.randomizer.register(fn)
        self._randomizers.append(getattr(self.rep.randomizer, fn.__name__))

    def apply_randomizers(self):
        for fn in self._randomizers:
            fn()


def register_dome_jitter(rep, replicator, prim_path, runtime):
    """Per-frame dome-intensity jitter, randomly sampled each frame.

    Stateless rep.distribution.uniform: it redraws on every orchestrator step,
    seeded run-to-run by set_global_seed in make_replicator. (A Python-sampled
    rep.distribution.sequence would let us log the exact per-frame values, but
    sequence does not advance on an attribute write the way it does on
    modify.pose — it sticks on the first value — so uniform is the path that
    actually varies frame to frame.)
    """
    node = rep.get.prim_at_path(prim_path)
    lo, hi = runtime.dome_intensity_range
    def jitter_dome():
        with node:
            rep.modify.attribute("intensity", rep.distribution.uniform(lo, hi))
        return node.node
    replicator.register(jitter_dome)


def register_background_jitter(rep, replicator, prim_path, texture_paths):
    dome_node = rep.get.prim_at_path(prim_path)
    def randomize_background():
        with dome_node:
            rep.modify.attribute("texture:file", rep.distribution.choice(list(texture_paths)))
        return dome_node.node
    replicator.register(randomize_background)


def register_box_texture_jitter(rep, replicator, texture_paths, shader_paths):
    shader_nodes = [rep.get.prim_at_path(p) for p in shader_paths]
    def randomize_box_textures():
        for node in shader_nodes:
            with node:
                rep.modify.attribute("file", rep.distribution.choice(list(texture_paths)))
        return shader_nodes[-1].node
    replicator.register(randomize_box_textures)


def make_replicator(runtime, num_frames, render_dir):
    """Build the per-frame randomizer graph.

    Lighting is a STATIC aimed-DistantLight key + low DomeLight fill (see build_scene) —
    no per-frame light jitter by default. `num_frames` is len(world_poses)
    (= num_targets × num_frames) threaded from the call site for the lighting_log header.
    The optional dome-intensity jitter (jitter_dome, off) and background jitter register
    here when enabled, seeded run-to-run by runtime.effective_seed.
    """
    import omni.replicator.core as rep
    from omni.replicator.core.utils.rng import set_global_seed
    set_global_seed(runtime.effective_seed)

    replicator = ReplicatorWrapper(rep)
    log = {}
    # The DistantLight key + DomeLight fill are static (authored in build_scene); only the
    # optional dome-intensity jitter registers here, and only when jitter_dome is on.
    if runtime.dome_light and runtime.jitter_dome:
        register_dome_jitter(rep, replicator, "/World/DomeLight", runtime)
        log["DomeLight"] = {"distribution": "uniform",
                            "intensity_range": list(runtime.dome_intensity_range),
                            "normalize": runtime.dome_normalize}
    if runtime.background_textures:
        register_background_jitter(rep, replicator, "/World/DomeLight", runtime.background_textures)

    if runtime.log_lighting:
        import json
        (Path(render_dir) / "lighting_log.json").write_text(json.dumps(
            {"num_frames": num_frames, "seed": runtime.effective_seed, "lights": log}, indent=2))
    return replicator


def make_dome_light(stage, parent, intensity=1000.0, normalize=True):
    from pxr import UsdLux
    from isaacsim.core.utils.prims import create_prim
    path = f"{parent}/DomeLight"
    dome_prim = create_prim(path, "DomeLight")
    dome = UsdLux.DomeLight.Get(stage, path)
    dome.GetIntensityAttr().Set(intensity)
    dome.GetNormalizeAttr().Set(normalize)
    return dome_prim


def make_sphere_light(stage, parent, intensity=5000.0, radius=0.1):
    from pxr import UsdLux
    from isaacsim.core.utils.prims import create_prim
    path = f"{parent}/SphereLight"
    create_prim(path, "SphereLight")
    light = UsdLux.SphereLight.Get(stage, path)
    light.GetIntensityAttr().Set(intensity)
    light.GetRadiusAttr().Set(radius)
    return light


def look_at_euler(eye, target):
    """Euler XYZ (deg) orienting a -Z-emitting USD prim (light/camera) to look eye→target.

    Same convention as LookAtPoser: vision_core.look_at is +Z-forward (OpenCV); cv2opengl
    flips it to USD's -Z-forward. Eye magnitude is irrelevant for a distant light (direction only).
    """
    from vision_core.pose_utils import look_at, cv2opengl
    pose = cv2opengl(look_at(np.asarray(target, float), np.asarray(eye, float)))
    return tuple(R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True))


def make_distant_light(stage, parent, intensity=3000.0, angle=0.53, rotation=(0.0, 0.0, 0.0)):
    """Directional KEY light. Emits along local -Z; `rotation` (rotateXYZ euler deg) aims it —
    pass look_at_euler(eye, centroid) so it points at the wall, not the floor."""
    from pxr import UsdLux
    from isaacsim.core.utils.prims import create_prim
    path = f"{parent}/DistantLight"
    create_prim(path, "DistantLight")
    light = UsdLux.DistantLight.Get(stage, path)
    light.GetIntensityAttr().Set(intensity)
    light.GetAngleAttr().Set(angle)
    set_transform(stage.GetPrimAtPath(path), rotation=rotation)
    return light


def boot_sim(runtime, render_dir):
    from isaacsim.simulation_app import SimulationApp
    app = SimulationApp({
        "headless": True,
        "width": runtime.width,
        "height": runtime.height,
        # multi_gpu ruled out as the cause of the intermittent all-black render
        # (11/15 black with False vs 9/15 with True — unchanged).
        "multi_gpu": True,
        # "active_gpu": 0,
        # "physics_gpu": 0,
    })

    import carb.settings
    import omni.replicator.core as rep

    s = carb.settings.get_settings()
    s.set("/rtx/renderMode", "PathTracing")
    s.set("/rtx/pathtracing/totalSpp", runtime.path_tracing_spp)
    s.set("/rtx/pathtracing/maxBounces", runtime.path_tracing_max_bounces)
    s.set("/rtx/denoiser/enabled", True)
    s.set("/rtx-transient/resourcemanager/enableTextureStreaming", runtime.enable_texture_streaming)
    s.set("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", runtime.texture_streaming_budget)
    s.set("/rtx/debugMaterialType", runtime.debug_material_type)
    s.set("/omni/replicator/backends/disk/root_dir", os.path.abspath(render_dir))
    rep.settings.set_render_pathtraced()

    # Make rendering block: each app.update() returns only after Hydra has
    # finished the frame, instead of dispatching it async and continuing. This
    # is what the `isaacsim.exp.base.zero_delay.kit` experience flips on (and
    # what SimulationContext.set_block_on_render(True) sets). Without it the
    # warmup_render() loop's app.update() calls are fire-and-forget, so a fixed
    # warmup_frames count can return before RTX has actually settled — the
    # lit-vs-black coin flip. The paired updateOrder setting moves the
    # render-complete check last for true zero-frame-delay. (base.kit already
    # disables async *replicator* rendering for SDG; this is the stronger
    # per-update blocking guarantee on top of that.)
    s.set("/app/hydraEngine/waitIdle", True)
    s.set("/app/updateOrder/checkForHydraRenderComplete", 1000)

    # Intermittent all-black-render fix: by default a PT capture does NOT
    # accumulate samples across the subframes of a single captured frame — the
    # accumulation buffer resets every subframe, so each frame is effectively one
    # noisy sample whose lit-vs-black outcome is a per-process coin flip (forum:
    # "Replicator Path Tracing samples do not accumulate" /t/229697; the official
    # PT capture preset sets this too). Turning it on makes the rt_subframes
    # subframes accumulate into a converged frame. The orchestrator bumps
    # /rtx/externalFrameCounter once per captured frame, so accumulation still
    # resets cleanly between frames.
    s.set("/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", True)

    # Dark-box fix: the captured `rgb` AOV is post-tonemap LDR (ACES, op=6) and
    # auto-exposure is off, so exposure is fixed by the photographic triangle.
    # The shipped exposureTime=0.02s default underexposed the dome-lit scene into
    # the ACES toe → uint8 crush to 0. Set a fixed exposure to land the box wall
    # in the midtones; auto-exposure stays off so the per-frame dataset is
    # deterministic. See RuntimeConfig.set_exposure / exposure_time / f_number.
    if runtime.set_exposure:
        s.set("/rtx/post/tonemap/exposureTime", runtime.exposure_time)
        s.set("/rtx/post/tonemap/fNumber", runtime.f_number)
        s.set("/rtx/post/tonemap/filmIso", runtime.film_iso)

    # Confirm the RTX post/tonemap state that actually took (op==6 ACES expected).
    print("[TONEMAP] " + " ".join(
        f"{k}={s.get(k)!r}" for k in (
            "/rtx/post/tonemap/op",
            "/rtx/post/histogram/enabled",
            "/rtx/post/tonemap/exposureTime",
            "/rtx/post/tonemap/fNumber",
            "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/cm2Factor",
        )), flush=True)

    return app


def warmup_render(app, n_frames):
    """Settle the RTX renderer before capture: step the app n_frames times so MDL shaders
    compile, the dome HDRI/textures stream in, and the PT/denoiser state initializes.
    Without this the first captured frame(s) are a lit-vs-black coin flip (see boot_sim's
    accumulation note). Mirrors Isaac's camera-sensor warmup (isaacsim.sensors.camera tests:
    N x app.update() before reading pixels). n_frames == 0 is a no-op."""
    for _ in range(n_frames):
        app.update()


def add_grasp_frame(box_path):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom
    box_prim = get_current_stage().GetPrimAtPath(box_path)
    # Bottom-front edge of the bbox in the prim's OWN local frame. Use
    # ComputeUntransformedBound, NOT ComputeLocalBound/local_bbox_range: grasp
    # frames are added AFTER the wrapper is slotted, and ComputeLocalBound bakes
    # that placement into the midpoint — doubling the offset and flinging the
    # grasp frame (and the camera tied to it) off the object. Untransformed bound
    # ignores the prim's own placement, so the offset is purely geometric (and a
    # no-op for centered boxes — origin == bbox center).
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cx, cy, cz = bb.ComputeUntransformedBound(box_prim).ComputeAlignedRange().GetMidpoint()
    half = bounding_half_extents(box_prim)
    grasp = create_empty("GraspPoint", box_path)
    set_transform(grasp, translation=(cx, cy - half[1], cz - half[2]), rotation=(0.0, 0.0, -90.0))
    return grasp.GetPath().pathString


SHADOW_SHAPES = ("Cube", "Cone", "Cylinder", "Sphere")


def add_shadow_occluders(stage, parent, grasp_frames, runtime):
    """Place runtime.occluders_per_target invisible occluders per grasp target.

    Each occluder casts a path-traced shadow on its box but is hidden from the
    camera (primvars:hideForCamera=True) and carries no semantic label, so only
    its shadow reaches obs/masks. Positioned ONCE: a single occluder-Poser sample
    in the target frame, mapped to world via that target's target2world. Must run
    AFTER the stack is positioned so each grasp frame's target2world is final.
    Per-frame shadow variety comes from the moving camera + per-frame light jitter;
    cross-render variety from the per-render-seeded poser (seed + idx).
    """
    from pxr import Sdf, UsdGeom
    from isaacsim.core.utils.prims import create_prim
    from isaac_datagen import posers
    from isaac_datagen.capture import get_target2world, set_prim_pose

    poser = posers.get(runtime.occluder_pose_policy)(**runtime.occluder_pose_policy_args)
    target2worlds = get_target2world(grasp_frames)                       # (M, 4, 4)
    create_prim(f"{parent}/ShadowOccluders", "Xform")
    for ti, t2w in enumerate(target2worlds):
        for k in range(runtime.occluders_per_target):
            path = f"{parent}/ShadowOccluders/t{ti:03d}_occ{k}"
            s = runtime.occluder_scale if runtime.occluder_scale is not None else float(np.random.uniform(0.04, 0.2))
            create_prim(path, SHADOW_SHAPES[(ti + k) % len(SHADOW_SHAPES)], scale=(s, s, s))
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
                "hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)      # doNotCastShadows left unset
            set_prim_pose(path, t2w @ poser(1)[0])                       # target-frame sample → world


def build_scene(runtime, objects: List[GraspableObject]):
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.semantics import add_labels

    create_new_stage()
    stage = get_current_stage()

    # Define /World as an Xform and mark it the defaultPrim. Without a defaultPrim
    # the stage exports to an invalid/near-empty USD(Z) and references to the
    # result fail to resolve.
    world_prim = create_prim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    if runtime.scene != "empty":
        load_asset("/World/Workbench", os.path.join(RESOURCE_PATH, "workbench_world.usd"))

    # Dome is a low ambient fill so shadowed faces don't crush to black; the DistantLight
    # key (created below, after the geometry exists, so it can be aimed) is the main source.
    make_dome_light(stage, "/World",
                    intensity=500 if runtime.dome_light else 0.0,
                    normalize=runtime.dome_normalize)

    parent_path = "/World/GeneratedPallets"
    create_prim(parent_path, "Xform")
    stack_path, objects_paths, is_graspable = create_stack_of_objects(
        parent_path,
        objects,
        runtime,
    )

    graspable_paths = [p for p, v in is_graspable.items() if v]
    grasp_frames_paths = [add_grasp_frame(p) for p in graspable_paths]

    set_transform(
        get_current_stage().GetPrimAtPath(stack_path),
        translation=(0.1, 0.1, 0.045),
    )

    if runtime.distant_light:
        from isaac_datagen.capture import get_target2world
        centroid = get_target2world(grasp_frames_paths)[:, :3, 3].mean(0)   # world wall center
        eye = centroid + np.asarray(runtime.distant_light_offset, float)    # "sun" pos; direction only
        make_distant_light(stage, "/World", intensity=runtime.distant_intensity,
                           angle=runtime.distant_angle, rotation=look_at_euler(eye, centroid))

    if runtime.occluders_per_target:
        add_shadow_occluders(stage, "/World", grasp_frames_paths, runtime)

    intrinsics = np.load(runtime.intrinsics_path)
    zed = ZedMini("gripper", "/World", intrinsics, width=runtime.width, height=runtime.height)

    # from pxr import UsdGeom, Usd
    # grasp_point = grasp_points[rng.randint(len(grasp_points))]
    # ` ah here is where the grasp_point is used.
    # target2world = np.array(
    #     UsdGeom.Xformable(grasp_point).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # ).T
    # target_pose = plan_poses(runtime.target_to_baseline_ypr_desired,
    #                          runtime.xrange, runtime.yrange, runtime.zrange, 1)[0]
    # world_pose = target2world @ target_pose
    # pos = tuple(world_pose[:3, 3].tolist())
    # rot = tuple(R.from_matrix(world_pose[:3, :3]).as_euler('xyz', degrees=True).tolist())
    # set_transform(zed.prim, translation=pos, rotation=rot)

    return SceneHandle(zed=zed, objects=objects, grasp_points=grasp_frames_paths)

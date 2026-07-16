
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_datagen.isaac_utils import load_asset, set_transform, bounding_half_extents, find_prims, create_empty
from isaac_datagen.hardwares import ZedMini
from isaac_datagen.objects import GraspableObject
from isaac_datagen import placers
from isaac_datagen.pose_planning import plan_poses


@dataclass(frozen=True)
class SceneHandle:
    zed: ZedMini
    grasp_points: list
    objects: List[GraspableObject]
    object_prim_paths: List[str] = None

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def add_object(*, at_parent: str, obj: GraspableObject) -> str:
    from isaacsim.core.utils.stage import get_current_stage
    wrapper_path = add_wrapped_reference(
        at_parent=at_parent, name=obj.meta["name"], usd_path=obj.usd_path)
    label_product(get_current_stage().GetPrimAtPath(f"{wrapper_path}/geo"), obj)
    return wrapper_path


def label_product(prim, obj) -> dict:
    from isaacsim.core.utils.semantics import add_labels, remove_labels
    displaced = _override_vendor_class_labels(prim, obj.meta["class"])
    remove_labels(prim, include_descendants=True)
    add_labels(prim, labels=[obj.meta["class"]], instance_name="class")
    add_labels(prim, labels=[obj.meta["name"]], instance_name="instance")
    return displaced


def _override_vendor_class_labels(prim, cls: str) -> dict:
    # legacy vendor opinions compose through the reference arc: deletable no, overridable yes
    from pxr import Usd
    displaced = {}
    for p in Usd.PrimRange(prim):
        for attr in p.GetAttributes():
            name = attr.GetName()
            if (name.startswith("semantic:") and name.endswith(":params:semanticType")
                    and attr.Get() == "class"):
                data = p.GetAttribute(name.replace(":semanticType", ":semanticData"))
                if data and data.Get() != cls:
                    displaced[str(data.GetPath())] = data.Get()
                    data.Set(cls)
    return displaced


def add_wrapped_reference(*, at_parent: str, name: str, usd_path: str) -> str:
    from pxr import Usd, Tf
    from isaacsim.core.utils.stage import get_current_stage
    wrapper_path = f"{at_parent}/{Tf.MakeValidIdentifier(name)}"
    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.DefinePrim(wrapper_path, "Xform")
    load_asset(f"{wrapper_path}/geo", usd_path, ref_prim_path="/World")
    return wrapper_path

def organize_objects(policy: Callable, prim_paths: List[str]):
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    translations_rotations = [policy(path) for path in prim_paths]
    for (translation, rotation), prim_path in zip(translations_rotations, prim_paths):
        set_transform(stage.GetPrimAtPath(prim_path), translation=translation, rotation=rotation)


def create_stack_of_objects(parent_path, objects: List[GraspableObject], runtime, orientation=None):
    from isaacsim.core.utils.prims import create_prim
    from isaac_datagen import orientations
    stack_prim = create_prim(f"{parent_path}/stack", "Xform")
    stack_path = stack_prim.GetPath().pathString

    prim_paths_added = [add_object(at_parent=stack_path, obj=o) for o in objects]
    if orientation is not None:
        orient = orientations.get(orientation["name"])(**orientation.get("args", {}))
        orient(prim_paths_added, objects)
    policy = placers.get(runtime.placement)(prim_paths_added, **runtime.placement_args)

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
        self._per_frame = []

    def register(self, fn):
        self.rep.randomizer.register(fn)
        self._randomizers.append(getattr(self.rep.randomizer, fn.__name__))

    def apply_randomizers(self):
        for fn in self._randomizers:
            fn()

    def register_per_frame(self, fn):
        self._per_frame.append(fn)

    def per_frame(self, i):
        for fn in self._per_frame:
            fn(i)


def register_dome_jitter(replicator, prim_path, runtime, num_frames):
    from pxr import Usd, UsdLux
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    light = UsdLux.DomeLight(stage.GetPrimAtPath(prim_path))
    lo, hi = runtime.dome_intensity_range
    intensities = np.random.default_rng([runtime.effective_seed, 1]).uniform(lo, hi, num_frames).tolist()

    def jitter_dome(i):
        with Usd.EditContext(stage, stage.GetRootLayer()):
            light.GetIntensityAttr().Set(intensities[i])
    replicator.register_per_frame(jitter_dome)
    return intensities


def register_distant_jitter(replicator, prim_path, runtime, num_frames):
    from pxr import Usd, UsdLux
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    light = UsdLux.DistantLight(prim)
    base = look_at_euler(runtime.distant_light_offset, (0.0, 0.0, 0.0))
    rng = np.random.default_rng([runtime.effective_seed, 0])
    rotations = sample_offset_eulers(runtime.distant_light_offset, runtime.distant_offset_jitter, num_frames, rng)
    intensities = temperatures = None
    if runtime.distant_intensity_jitter is not None:
        lo, hi = runtime.distant_intensity_jitter
        intensities = rng.uniform(lo, hi, num_frames).tolist()
    if runtime.distant_temperature_jitter is not None:
        lo, hi = runtime.distant_temperature_jitter
        temperatures = rng.uniform(lo, hi, num_frames).tolist()
        light.GetEnableColorTemperatureAttr().Set(True)

    def jitter_distant(i):
        set_transform(prim, rotation=rotations[i])
        with Usd.EditContext(stage, stage.GetRootLayer()):
            if intensities is not None:
                light.GetIntensityAttr().Set(intensities[i])
            if temperatures is not None:
                light.GetColorTemperatureAttr().Set(temperatures[i])
    replicator.register_per_frame(jitter_distant)
    return base, rotations, intensities, temperatures


def register_background_jitter(rep, replicator, prim_path, texture_paths):
    dome_node = rep.get.prim_at_path(prim_path)
    def randomize_background():
        with dome_node:
            rep.modify.attribute("texture:file", rep.distribution.choice(list(texture_paths)))
        return dome_node.node
    replicator.register(randomize_background)


def register_light_pattern_jitter(replicator, spec, runtime, num_frames, stream):
    from pxr import Usd, UsdLux
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    lights = {}
    for p in find_prims(spec.root, spec.pattern):
        attr = UsdLux.LightAPI(stage.GetPrimAtPath(p)).GetIntensityAttr()
        if attr and attr.IsValid():
            lights[p] = (attr, float(attr.Get()))
    assert lights, f"light_jitter_patterns matched no UsdLux lights: {spec}"
    lo, hi = spec.intensity_scale_range
    factors = np.exp(np.random.default_rng(
        [runtime.effective_seed, 2, stream]).uniform(np.log(lo), np.log(hi), num_frames)).tolist()

    def jitter_lights(i):
        with Usd.EditContext(stage, stage.GetRootLayer()):
            for attr, base in lights.values():
                attr.Set(base * factors[i])
    replicator.register_per_frame(jitter_lights)
    return {"base_intensity": {p: b for p, (_, b) in lights.items()}, "scale_factors": factors}


def register_box_texture_jitter(rep, replicator, texture_paths, shader_paths):
    shader_nodes = [rep.get.prim_at_path(p) for p in shader_paths]
    def randomize_box_textures():
        for node in shader_nodes:
            with node:
                rep.modify.attribute("file", rep.distribution.choice(list(texture_paths)))
        return shader_nodes[-1].node
    replicator.register(randomize_box_textures)


def make_replicator(runtime, num_frames, render_dir):
    import omni.replicator.core as rep
    from omni.replicator.core.utils.rng import set_global_seed
    set_global_seed(runtime.effective_seed)

    replicator = ReplicatorWrapper(rep)
    log = {}
    if runtime.distant_light and runtime.jitter_distant:
        base, rotations, intensities, temperatures = register_distant_jitter(
            replicator, "/World/DistantLight", runtime, num_frames)
        log["DistantLight"] = {
            "base_rotation_xyz_deg": list(base),
            "offset_jitter_m": runtime.distant_offset_jitter,
            "rotations_xyz_deg": [list(r) for r in rotations],
            "intensity": intensities if intensities is not None else runtime.distant_intensity,
            "temperature": temperatures,
        }
    if runtime.dome_light and runtime.jitter_dome:
        log["DomeLight"] = register_dome_jitter(replicator, "/World/DomeLight", runtime, num_frames)
    if runtime.background_textures:
        register_background_jitter(rep, replicator, "/World/DomeLight", runtime.background_textures)
    for k, spec in enumerate(runtime.light_jitter_patterns):
        log[f"LightPattern{k}:{spec.pattern}"] = register_light_pattern_jitter(
            replicator, spec, runtime, num_frames, stream=k)

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
    from vision_core.pose_utils import look_at, cv2opengl
    pose = cv2opengl(look_at(np.asarray(target, float), np.asarray(eye, float)))
    return tuple(R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True))


def sample_offset_eulers(offset, jitter, n, rng):
    offset = np.asarray(offset, float)
    deltas = rng.uniform(-jitter, jitter, size=(n, 3))
    return [look_at_euler(eye=offset + d, target=(0.0, 0.0, 0.0)) for d in deltas]


def make_distant_light(stage, parent, intensity=3000.0, angle=0.53, rotation=(0.0, 0.0, 0.0)):
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
        "multi_gpu": True,
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

    s.set("/app/hydraEngine/waitIdle", True)
    s.set("/app/updateOrder/checkForHydraRenderComplete", 1000)

    s.set("/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges", True)

    if runtime.set_exposure:
        s.set("/rtx/post/tonemap/exposureTime", runtime.exposure_time)
        s.set("/rtx/post/tonemap/fNumber", runtime.f_number)
        s.set("/rtx/post/tonemap/filmIso", runtime.film_iso)

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
    for _ in range(n_frames):
        app.update()


def add_grasp_frame(box_path):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom
    box_prim = get_current_stage().GetPrimAtPath(box_path)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cx, cy, cz = bb.ComputeUntransformedBound(box_prim).ComputeAlignedRange().GetMidpoint()
    half = bounding_half_extents(box_prim)
    grasp = create_empty("GraspPoint", box_path)
    set_transform(grasp, translation=(cx, cy - half[1], cz - half[2]), rotation=(0.0, 0.0, -90.0))
    return grasp.GetPath().pathString


SHADOW_SHAPES = ("Cube", "Cone", "Cylinder", "Sphere")


def add_shadow_occluders(stage, parent, grasp_frames, runtime):
    from pxr import Sdf, UsdGeom
    from isaacsim.core.utils.prims import create_prim
    from isaac_datagen import posers
    from isaac_datagen.capture import get_target2world, set_prim_pose

    poser = posers.get(runtime.occluder_pose_policy)(**runtime.occluder_pose_policy_args)
    target2worlds = get_target2world(grasp_frames)
    create_prim(f"{parent}/ShadowOccluders", "Xform")
    for ti, t2w in enumerate(target2worlds):
        for k in range(runtime.occluders_per_target):
            path = f"{parent}/ShadowOccluders/t{ti:03d}_occ{k}"
            s = runtime.occluder_scale if runtime.occluder_scale is not None else float(np.random.uniform(0.04, 0.2))
            create_prim(path, SHADOW_SHAPES[(ti + k) % len(SHADOW_SHAPES)], scale=(s, s, s))
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
                "hideForCamera", Sdf.ValueTypeNames.Bool).Set(True)
            set_prim_pose(path, t2w @ poser(1)[0])


def _bbox_grasp_frame(path, obj):
    return add_grasp_frame(path)


def _catalog_grasp_frame(path, obj):
    from isaac_datagen.store_scene import add_catalog_grasp_frame  # lazy (import cycle)
    return add_catalog_grasp_frame(f"{path}/geo", obj)


GRASP_FRAME_SOURCES = {"bbox": _bbox_grasp_frame, "catalog": _catalog_grasp_frame}


@dataclass(frozen=True)
class PlainSceneSpec:
    mutations: list = field(default_factory=list)  # unknown keys -> TypeError (fail loud)
    grasp_frames: str = "bbox"  # "catalog" replays the Stage-A baked grasp_point
    orientation: dict = None    # {name, args?} -> orientations registry; None = keep canonical yaw

    def __post_init__(self):
        from isaac_datagen import store_mutations  # lazy: store_mutations->extract_store_objects->scene cycle
        assert self.grasp_frames in GRASP_FRAME_SOURCES, \
            f"grasp_frames must be one of {sorted(GRASP_FRAME_SOURCES)}: {self.grasp_frames!r}"
        for m in self.mutations:
            assert isinstance(m, dict) and m.get("name") and set(m) <= {"name", "args"}, \
                f"mutation spec must be {{name, args?}}: {m!r}"
            assert getattr(store_mutations.get(m["name"]), "PLAIN_SAFE", False), \
                f"mutation {m['name']!r} is store-only (reads StoreSceneSpec fields)"
        if self.orientation is not None:
            from isaac_datagen import orientations
            assert isinstance(self.orientation, dict) and self.orientation.get("name") \
                and set(self.orientation) <= {"name", "args"}, \
                f"orientation spec must be {{name, args?}}: {self.orientation!r}"
            orientations.get(self.orientation["name"])(**self.orientation.get("args", {}))


def apply_plain_mutations(stack_path, spec, objects, objects_paths, effective_seed):
    from isaacsim.core.utils.stage import get_current_stage
    from isaac_datagen import store_mutations  # lazy (import cycle)
    targets = [store_mutations.CaptureTarget(o, p) for o, p in zip(objects, objects_paths)]
    root = get_current_stage().GetPrimAtPath(stack_path)  # walk root = the stack (all placed objects)
    targets = store_mutations.apply_mutations(root, spec, targets, effective_seed)
    assert [t.prim_path for t in targets] == list(objects_paths), \
        "plain-scene mutations must not add/remove/reorder targets (in-place stage edits only)"
    return [t.obj for t in targets]  # honors in-place obj replacement


def build_scene(runtime, objects: List[GraspableObject]):
    spec = PlainSceneSpec(**runtime.scene_builder_args)

    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.semantics import add_labels

    create_new_stage()
    stage = get_current_stage()

    world_prim = create_prim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    if runtime.scene != "empty":
        load_asset("/World/Workbench", os.path.join(RESOURCE_PATH, "workbench_world.usd"))

    make_dome_light(stage, "/World",
                    intensity=runtime.dome_fill_intensity if runtime.dome_light else 0.0,
                    normalize=runtime.dome_normalize)

    parent_path = "/World/GeneratedPallets"
    create_prim(parent_path, "Xform")
    stack_path, objects_paths, is_graspable = create_stack_of_objects(
        parent_path,
        objects,
        runtime,
        orientation=spec.orientation,
    )
    objects = apply_plain_mutations(stack_path, spec, objects, objects_paths, runtime.effective_seed)

    graspable_paths = [p for p, v in is_graspable.items() if v]
    make_grasp_frame = GRASP_FRAME_SOURCES[spec.grasp_frames]
    obj_by_path = dict(zip(objects_paths, objects))
    grasp_frames_paths = [make_grasp_frame(p, obj_by_path[p]) for p in graspable_paths]

    set_transform(
        get_current_stage().GetPrimAtPath(stack_path),
        translation=(0.1, 0.1, 0.045),
    )

    if runtime.distant_light:
        from isaac_datagen.capture import get_target2world
        centroid = get_target2world(grasp_frames_paths)[:, :3, 3].mean(0)
        eye = centroid + np.asarray(runtime.distant_light_offset, float)
        make_distant_light(stage, "/World", intensity=runtime.distant_intensity,
                           angle=runtime.distant_angle, rotation=look_at_euler(eye, centroid))

    if runtime.occluders_per_target:
        add_shadow_occluders(stage, "/World", grasp_frames_paths, runtime)

    intrinsics = np.load(runtime.intrinsics_path)
    zed = ZedMini("gripper", "/World", intrinsics, width=runtime.width, height=runtime.height)


    # geo, not wrapper: l2w consumers need the mesh frame (orientation yaw lives on geo)
    return SceneHandle(zed=zed, objects=objects, grasp_points=grasp_frames_paths,
                       object_prim_paths=[f"{p}/geo" for p in objects_paths])

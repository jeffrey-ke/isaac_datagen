"""USD scene building — boxes, stacks, lighting, assets, sim bootstrap."""

import os
import sys
from dataclasses import dataclass
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
    from pxr import Usd
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.semantics import add_labels
    wrapper_path = f"{at_parent}/{obj.meta['name']}"
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


def _target_range_to_world(runtime, target2world):
    lo = np.array([runtime.xrange[0], runtime.yrange[0], runtime.zrange[0], 1.0])
    hi = np.array([runtime.xrange[1], runtime.yrange[1], runtime.zrange[1], 1.0])
    world_lo = (target2world @ lo)[:3]
    world_hi = (target2world @ hi)[:3]
    mn = np.minimum(world_lo, world_hi)
    mx = np.maximum(world_lo, world_hi)
    mn[2] = 0.3
    mx[2] = 1.5
    return (float(mn[0]), float(mn[1]), float(mn[2])), (float(mx[0]), float(mx[1]), float(mx[2]))


def register_sphere_jitter(rep, replicator, prim_path, world_lo, world_hi):
    light_node = rep.get.prim_at_path(prim_path)
    def jitter_lights():
        with light_node:
            rep.modify.attribute("colorTemperature", rep.distribution.normal(6500, 1000))
            rep.modify.attribute("intensity", rep.distribution.uniform(0, 500000))
            rep.modify.pose(position=rep.distribution.uniform(world_lo, world_hi))
        return light_node.node
    replicator.register(jitter_lights)


def register_distant_jitter(rep, replicator, prim_path):
    light_node = rep.get.prim_at_path(prim_path)
    def jitter_distant():
        with light_node:
            rep.modify.attribute("intensity", rep.distribution.uniform(5000, 30000))
            rep.modify.pose(rotation=rep.distribution.uniform((-30, -180, 0), (30, 180, 0)))
        return light_node.node
    replicator.register(jitter_distant)


def register_dome_jitter(rep, replicator, prim_path):
    dome_node = rep.get.prim_at_path(prim_path)
    def jitter_dome():
        with dome_node:
            rep.modify.attribute("intensity", rep.distribution.uniform(500, 1000))
        return dome_node.node
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


def make_replicator(runtime):
    import omni.replicator.core as rep

    replicator = ReplicatorWrapper(rep)
    # world_lo, world_hi = _target_range_to_world(runtime, target2world)

    # register_sphere_jitter(rep, replicator, "/World/SphereLight", world_lo, world_hi)
    register_distant_jitter(rep, replicator, "/World/DistantLight")
    if runtime.dome_light:
        register_dome_jitter(rep, replicator, "/World/DomeLight")
    if runtime.background_textures:
        register_background_jitter(rep, replicator, "/World/DomeLight", runtime.background_textures)

    return replicator


def make_dome_light(stage, parent, intensity=1000.0):
    from pxr import UsdLux
    from isaacsim.core.utils.prims import create_prim
    path = f"{parent}/DomeLight"
    dome_prim = create_prim(path, "DomeLight")
    dome = UsdLux.DomeLight.Get(stage, path)
    dome.GetIntensityAttr().Set(intensity)
    dome.GetNormalizeAttr().Set(True)
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


def make_distant_light(stage, parent, intensity=3000.0, angle=0.53):
    from pxr import UsdLux
    from isaacsim.core.utils.prims import create_prim
    path = f"{parent}/DistantLight"
    create_prim(path, "DistantLight")
    light = UsdLux.DistantLight.Get(stage, path)
    light.GetIntensityAttr().Set(intensity)
    light.GetAngleAttr().Set(angle)
    return light


def boot_sim(runtime, render_dir):
    from isaacsim.simulation_app import SimulationApp
    app = SimulationApp({
        "headless": True,
        "width": runtime.width,
        "height": runtime.height,
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

    return app


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

def build_scene(runtime, objects: List[GraspableObject], rng):
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

    make_dome_light(stage, "/World", intensity=1000.0 if runtime.dome_light else 0.0)
    make_sphere_light(stage, "/World")
    make_distant_light(stage, "/World")

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

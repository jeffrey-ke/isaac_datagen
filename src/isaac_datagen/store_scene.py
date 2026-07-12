from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isaac_datagen import grasp_policies, store_mutations
from isaac_datagen.hardwares import ZedMini
from isaac_datagen.isaac_utils import create_empty, load_asset
from isaac_datagen.scene import SceneHandle, make_dome_light

STORE_ROOT = "/World/Store"
STORE_DEFAULT_PRIM = "/root"


@dataclass(frozen=True)
class StoreSceneSpec:
    store_usd: str
    product_patterns: list
    grasp_frame_policy: str
    grasp_frame_policy_args: dict = field(default_factory=dict)
    mutations: list = field(default_factory=list)
    require_tracked_only: list = field(default_factory=list)

    def __post_init__(self):
        assert Path(self.store_usd).exists(), f"store_usd missing: {self.store_usd}"
        assert self.product_patterns and all(self.product_patterns), \
            f"product_patterns must be a non-empty list of non-empty globs: {self.product_patterns}"
        grasp_policies.get(self.grasp_frame_policy)
        for m in self.mutations:
            assert isinstance(m, dict) and m.get("name") and set(m) <= {"name", "args"}, \
                f"mutation spec must be {{name, args?}}: {m!r}"
            store_mutations.get(m["name"])
        assert all(self.require_tracked_only), \
            f"require_tracked_only globs must be non-empty: {self.require_tracked_only}"


def load_store(spec: StoreSceneSpec):
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    create_new_stage()
    stage = get_current_stage()
    stage.SetDefaultPrim(create_prim("/World", "Xform"))
    load_asset(STORE_ROOT, str(spec.store_usd), ref_prim_path=STORE_DEFAULT_PRIM)
    return stage.GetPrimAtPath(STORE_ROOT)


def resolve_product_prim(store, obj):
    path = store.GetPath().AppendPath(obj.meta["store_prim"])
    prim = store.GetStage().GetPrimAtPath(path)
    assert prim.IsValid(), f"catalog object {obj.meta['name']}: no prim at {path}"
    return prim


def label_product(prim, obj):
    from isaacsim.core.utils.semantics import add_labels, remove_labels
    _override_vendor_class_labels(prim, obj.meta["class"])
    remove_labels(prim, include_descendants=True)
    add_labels(prim, labels=[obj.meta["class"]], instance_name="class")
    add_labels(prim, labels=[obj.meta["name"]], instance_name="instance")


def _override_vendor_class_labels(prim, cls: str):
    from pxr import Usd
    for p in Usd.PrimRange(prim):
        for attr in p.GetAttributes():
            name = attr.GetName()
            if (name.startswith("semantic:") and name.endswith(":params:semanticType")
                    and attr.Get() == "class"):
                data = p.GetAttribute(name.replace(":semanticType", ":semanticData"))
                if data and data.Get() != cls:
                    data.Set(cls)


def add_catalog_grasp_frame(prim_path: str, obj) -> str:
    from isaac_datagen.capture import set_prim_pose
    grasp = create_empty("GraspPoint", prim_path)
    set_prim_pose(grasp.GetPath().pathString, obj.grasp_point)
    return grasp.GetPath().pathString


def build_store_scene(runtime, objects) -> SceneHandle:
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    store = load_store(spec)
    if runtime.dome_light:
        make_dome_light(store.GetStage(), "/World", intensity=runtime.dome_fill_intensity,
                        normalize=runtime.dome_normalize)
    targets = [store_mutations.CaptureTarget(o, resolve_product_prim(store, o).GetPath().pathString)
               for o in objects]
    targets = store_mutations.apply_mutations(store, spec, targets, runtime.effective_seed)

    tracked = {t.obj.meta["class"] for t in targets}
    for glob in spec.require_tracked_only:
        leaked = sorted(p.GetName() for p in store_mutations.active_products(store, [glob])
                        if store_mutations.parse_sku(p.GetName())[1] not in tracked)
        assert not leaked, (f"require_tracked_only {glob!r}: untracked products still active "
                            f"(scene leak — present but unlabeled): {leaked}")

    stage = store.GetStage()
    object_prim_paths, grasp_frames = [], []
    for t in targets:
        prim = stage.GetPrimAtPath(t.prim_path)
        assert prim.IsValid() and prim.IsActive(), f"target prim gone: {t.prim_path}"
        assert " " not in t.obj.meta["class"], \
            f"multi-token class (Isaac truncates at whitespace): {t.obj.meta['class']!r}"
        label_product(prim, t.obj)
        grasp_frames.append(add_catalog_grasp_frame(t.prim_path, t.obj))
        object_prim_paths.append(t.prim_path)
    zed = ZedMini("gripper", "/World", np.load(runtime.intrinsics_path),
                  width=runtime.width, height=runtime.height)
    return SceneHandle(zed=zed, grasp_points=grasp_frames,
                       objects=[t.obj for t in targets],
                       object_prim_paths=object_prim_paths)

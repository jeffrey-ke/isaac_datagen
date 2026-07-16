from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isaac_datagen import grasp_policies, store_mutations
from isaac_datagen.hardwares import ZedMini
from isaac_datagen.isaac_utils import create_empty, load_asset
from isaac_datagen.scene import SceneHandle, label_product, make_dome_light

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
    site_catalog: str = ""       # curated site catalog; required by build_repopulated_store_scene

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
    assert "store_prim" in obj.meta, (
        f"object {obj.meta.get('name')!r} (class {obj.meta.get('class')!r}) has no store_prim — "
        f"not a store-native object; it cannot be native-revealed (use repopulation instead)")
    path = store.GetPath().AppendPath(obj.meta["store_prim"])
    prim = store.GetStage().GetPrimAtPath(path)
    assert prim.IsValid(), f"catalog object {obj.meta['name']}: no prim at {path}"
    return prim


def add_catalog_grasp_frame(prim_path: str, obj) -> str:
    from isaac_datagen.capture import set_prim_pose
    grasp = create_empty("GraspPoint", prim_path)
    set_prim_pose(grasp.GetPath().pathString, obj.grasp_point)
    return grasp.GetPath().pathString


def _finalize_store_scene(store, spec, targets, runtime) -> SceneHandle:
    # Shared tail: leaked-product guard + per-target label/grasp-frame + camera + SceneHandle.
    # Generic over targets, so inserted (repopulation) and revealed (native) targets both work.
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


def _load_store_with_lights(spec, runtime):
    store = load_store(spec)
    if runtime.dome_light:
        make_dome_light(store.GetStage(), "/World", intensity=runtime.dome_fill_intensity,
                        normalize=runtime.dome_normalize)
    return store


def build_store_scene(runtime, objects) -> SceneHandle:
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    store = _load_store_with_lights(spec, runtime)
    targets = [store_mutations.CaptureTarget(o, resolve_product_prim(store, o).GetPath().pathString)
               for o in objects]
    targets = store_mutations.apply_mutations(store, spec, targets, runtime.effective_seed)
    return _finalize_store_scene(store, spec, targets, runtime)


def build_repopulated_store_scene(runtime, objects) -> SceneHandle:
    """Repopulate curated shelf sites with the collected object queue (any provenance).

    Objects arrive already collected + ReplicateFilter'd + ShuffleFilter'd; zip them against
    the sites in order, insert each anchored to the site's curated grasp, strip the rest."""
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    assert spec.site_catalog, "build_repopulated_store_scene needs scene_builder_args.site_catalog"
    assert not spec.mutations, \
        "build_repopulated_store_scene ignores mutations (sites drive placement); leave the list empty"
    store = _load_store_with_lights(spec, runtime)
    sites = store_mutations.load_sites(spec.site_catalog)
    assert len(objects) <= len(sites), (
        f"store repopulation: {len(objects)} objects exceed {len(sites)} curated sites — "
        f"reduce the store ReplicateFilter count, or use the composed scene (no site limit)")
    stage = store.GetStage()
    create_empty("StoreSwaps", "/World")
    targets = [store_mutations.replace_instance(stage, store, site.store_prim, obj, site.grasp)
               for site, obj in zip(sites, objects)]            # queue order == placement order
    store_mutations.deactivate_remaining_products(store, spec)  # strip unused sites + non-catalog SKUs
    print(f"[scene] repopulated {len(targets)}/{len(sites)} sites", flush=True)
    return _finalize_store_scene(store, spec, targets, runtime)

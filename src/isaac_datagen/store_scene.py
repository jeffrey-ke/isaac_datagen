"""Store-scene builder (inverse datagen): load an existing photorealistic store
USD as the sim scene and align the FILTERED OptFlowObject catalog onto its own
product prims (semantic labels + GraspPoint frames authored from each object's
serialized grasp_point + ZedMini). Products NOT in the filtered catalog stay
unlabeled background — realistic distractors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isaac_datagen import grasp_policies, store_mutations
from isaac_datagen.hardwares import ZedMini
from isaac_datagen.isaac_utils import create_empty, load_asset
from isaac_datagen.scene import SceneHandle, make_dome_light

STORE_ROOT = "/World/Store"       # store reference prim; meta["store_prim"] is relative to it
STORE_DEFAULT_PRIM = "/root"      # store001.usd has defaultPrim=None; composed root is /root


@dataclass(frozen=True)
class StoreSceneSpec:
    """Validated view of runtime.scene_builder_args — ALL fields required,
    fail-loud. ONE schema drives both Stage A (extract_store_objects) and
    Stage C (build_store_scene), so extractor and builder cannot drift."""
    store_usd: str
    product_patterns: list         # fnmatch prim-NAME globs for find_prims (Stage A)
    grasp_frame_policy: str        # grasp_policies registry key — required, no default
    grasp_frame_policy_args: dict = field(default_factory=dict)
    # store_mutations registry specs [{name, args?}], applied in order by
    # build_store_scene only — Stage A ignores them. Default [] = unmutated store.
    # Names validated here; ctor args validated by make_mutations at build time
    # (Stage A never needs the swap catalogs).
    mutations: list = field(default_factory=list)
    # Stage-C only: prim-name globs under which EVERY active product's SKU class must be a
    # labeled target after mutations run — a fail-loud scene-content gate (the label-only
    # cid_to_class check can't see a present-but-unlabeled product). Default [] = no gate.
    require_tracked_only: list = field(default_factory=list)

    def __post_init__(self):
        assert Path(self.store_usd).exists(), f"store_usd missing: {self.store_usd}"
        assert self.product_patterns and all(self.product_patterns), \
            f"product_patterns must be a non-empty list of non-empty globs: {self.product_patterns}"
        grasp_policies.get(self.grasp_frame_policy)    # KeyError at load, not mid-run
        for m in self.mutations:
            assert isinstance(m, dict) and m.get("name") and set(m) <= {"name", "args"}, \
                f"mutation spec must be {{name, args?}}: {m!r}"
            store_mutations.get(m["name"])             # KeyError at load, not mid-run
        assert all(self.require_tracked_only), \
            f"require_tracked_only globs must be non-empty: {self.require_tracked_only}"


def load_store(spec: StoreSceneSpec):
    """Fresh stage + /World defaultPrim (same invariant as build_scene) + the store
    referenced under STORE_ROOT (explicit ref_prim_path because the store layer
    has no defaultPrim). Returns the store root PRIM — the handle every downstream
    helper takes as an explicit parameter (stage reachable via .GetStage());
    STORE_ROOT is used ONLY here, so no other function depends on module path state.

    NOTE: referencing /root drops the store layer's root-level /PhysicsScene —
    capture never steps physics; the M3 check watches for FixedJoint errors.
    """
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    create_new_stage()
    stage = get_current_stage()
    stage.SetDefaultPrim(create_prim("/World", "Xform"))
    load_asset(STORE_ROOT, str(spec.store_usd), ref_prim_path=STORE_DEFAULT_PRIM)
    return stage.GetPrimAtPath(STORE_ROOT)


def resolve_product_prim(store, obj):
    """meta['store_prim'] (the join key the extractor wrote, RELATIVE to the store
    root) resolved under the given store prim — fail loud on drift."""
    path = store.GetPath().AppendPath(obj.meta["store_prim"])
    prim = store.GetStage().GetPrimAtPath(path)
    assert prim.IsValid(), f"catalog object {obj.meta['name']}: no prim at {path}"
    return prim


def label_product(prim, obj):
    """add_object's labeling precedent (scene.py), minus the wrapper/geo
    convention: labels go on P itself — the annotator reads labels on ancestors,
    so every leaf mesh of the product subtree inherits them.

    ORDER MATTERS: add_labels (replicator modify.semantics) MERGES the prim's
    composed legacy semanticData into the list it authors, and the annotator
    UNIONS class labels across the subtree — with the vendor's ``class=snack``
    still composed, v_0 ends up ``semantics:labels:class = [snack, cereal001]``
    and idToSemantics reports ``'cereal001,snack'``, missing the exact
    class_to_cid lookup (cid_mask all zeros). So: rewrite the vendor's legacy
    values to ours FIRST (a value override is the only edit that wins over the
    reference arc — RemoveProperty can't delete referenced opinions), drop any
    stale LabelsAPI state, then author ours; the merge dedups to [class]."""
    from isaacsim.core.utils.semantics import add_labels, remove_labels
    _override_vendor_class_labels(prim, obj.meta["class"])
    remove_labels(prim, include_descendants=True)
    add_labels(prim, labels=[obj.meta["class"]], instance_name="class")
    add_labels(prim, labels=[obj.meta["name"]], instance_name="instance")


def _override_vendor_class_labels(prim, cls: str):
    """Rewrite the store asset's own class-type semanticData in P's subtree to ours.

    The vendor authors legacy class semantics BOTH on P and on the product meshes
    (e.g. ``semantic:...:semanticType=class / semanticData=snack`` on ``v_0`` and
    ``v_0/E_snack_1``); add_labels merges P's value and the annotator unions the
    rest, so every legacy value in the subtree must equal ours before add_labels
    runs. A root-layer value override wins over the reference arc."""
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
    """GraspPoint child of P authored directly from the catalog's serialized
    grasp_point — the exact frame the reference was shot from, so capture aims
    at the face the reference depicts. set_prim_pose authors LOCAL xformOps:
    l2w(GraspPoint) = l2w(P) @ grasp_point."""
    from isaac_datagen.capture import set_prim_pose
    grasp = create_empty("GraspPoint", prim_path)
    set_prim_pose(grasp.GetPath().pathString, obj.grasp_point)
    return grasp.GetPath().pathString


def build_store_scene(runtime, objects) -> SceneHandle:
    """Store-mode counterpart of scene.build_scene: same signature and return.
    Needs NO grasp policy — the aim frame is the catalog's own grasp_point
    (grasp_policies is a Stage-A concern: which face to SHOOT the reference from)."""
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    store = load_store(spec)                                # the store root prim — passed explicitly
    if runtime.dome_light:                                  # optional ambient fill over store lights
        make_dome_light(store.GetStage(), "/World", intensity=runtime.dome_fill_intensity,
                        normalize=runtime.dome_normalize)
    # Bind each filtered catalog object to its own (already present) store prim;
    # mutations rewrite the stage AND this list together, so the
    # scene.objects[i] <-> object_prim_paths[i] writer contract can't drift.
    targets = [store_mutations.CaptureTarget(o, resolve_product_prim(store, o).GetPath().pathString)
               for o in objects]                            # the FILTERED subset only
    rng = np.random.default_rng([runtime.effective_seed, 3])  # streams 0/1/2 = jitters
    for mutation in store_mutations.make_mutations(spec.mutations):
        targets = mutation(store, spec, targets, rng)
    assert targets, "store mutations left no captureable targets"
    names = [t.obj.meta["name"] for t in targets]           # mutations mint names
    assert len(names) == len(set(names)), f"duplicate names after mutations: {names}"

    tracked = {t.obj.meta["class"] for t in targets}
    for glob in spec.require_tracked_only:
        leaked = sorted(p.GetName() for p in store_mutations.active_products(store, [glob])
                        if store_mutations.parse_sku(p.GetName())[1] not in tracked)
        assert not leaked, (f"require_tracked_only {glob!r}: untracked products still active "
                            f"(scene leak — present but unlabeled): {leaked}")

    stage = store.GetStage()
    object_prim_paths, grasp_frames = [], []
    for t in targets:                                       # uniform: store prims AND wrappers
        prim = stage.GetPrimAtPath(t.prim_path)
        assert prim.IsValid() and prim.IsActive(), f"target prim gone: {t.prim_path}"
        assert " " not in t.obj.meta["class"], \
            f"multi-token class (Isaac truncates at whitespace): {t.obj.meta['class']!r}"
        label_product(prim, t.obj)       # override->remove->add: needed for swapped-in
        #  store usdz with baked vendor labels (arm A); no-op-safe on clean usdz
        grasp_frames.append(add_catalog_grasp_frame(t.prim_path, t.obj))
        object_prim_paths.append(t.prim_path)               # l2w at EXACTLY the usdz-frame node
    zed = ZedMini("gripper", "/World", np.load(runtime.intrinsics_path),
                  width=runtime.width, height=runtime.height)
    return SceneHandle(zed=zed, grasp_points=grasp_frames,
                       objects=[t.obj for t in targets],    # mutated list, NOT the input
                       object_prim_paths=object_prim_paths)

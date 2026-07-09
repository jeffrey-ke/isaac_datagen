"""Store-shelf mutations registry (posers/placers/filters idiom): remove or swap
store products before capture.

A mutation is a callable ``mutation(store, spec, targets, rng) -> targets``: it
edits the live stage AND returns the updated CaptureTarget binding list in the
same call, so the writer's ``scene.objects[i] <-> object_prim_paths[i]`` contract
cannot drift. build_store_scene applies ``make_mutations(spec.mutations)`` IN ORDER,
between binding resolution (each filtered catalog object bound to its own store
prim) and the uniform label_product loop; swapped-in wrappers are never re-mutated.
Stage A (extract_store_objects) ignores mutations entirely. Mutations draw from the
reproducible seed stream ``[effective_seed, 3]`` — streams 0/1/2 are the light jitters'.

Classes must be defined IN this module for get() to find them (the
import-to-register footgun once dropped ShelfPlacer). Module-level imports are
kit-free (parse_sku from extract_store_objects is cycle-safe — it never imports
store_scene at module level); kit-touching imports (capture, scene, clean_datagen,
isaac_utils) are DEFERRED inside functions.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaac_datagen import grasp_policies
from isaac_datagen.extract_store_objects import parse_sku
from isaac_datagen.objects import OptFlowObject


@dataclass(frozen=True)
class CaptureTarget:
    """Binding between a catalog object (what it is: class/name, reference data,
    grasp_point — what the writer records) and the live prim where it physically
    is (whose LOCAL frame equals the usdz frame: store v_0 node or swap wrapper —
    where l2w is read, labels authored, GraspPoint added)."""
    obj: OptFlowObject
    prim_path: str


def get(name):                                   # posers/placers/... idiom, KeyError on miss
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


def make_mutations(specs: list[dict]) -> list:   # mirrors filters.make_filters
    return [get(s["name"])(**s.get("args", {})) for s in specs]


def active_products(store, patterns) -> list:
    """ACTIVE direct product children scoped to the config's product globs.
    GetChildren honors the default predicate -> previously deactivated products
    drop out. Deliberately NOT matched_products/find_prims: those raise per
    pattern on zero matches, but a prior mutation may legitimately empty one;
    each mutation asserts its OWN class pattern matched instead. Scoping also
    keeps parse_sku off non-product prims (model_store001 -> class 'store001')."""
    return [c for c in store.GetChildren()
            if any(fnmatch.fnmatchcase(c.GetName(), pat) for pat in patterns)]


def _matching_products(store, spec, pattern) -> list:
    prims = [p for p in active_products(store, spec.product_patterns)
             if fnmatch.fnmatchcase(parse_sku(p.GetName())[1], pattern)]   # [1] = SKU class
    assert prims, f"class pattern {pattern!r} matches no ACTIVE store product"
    return prims


def _drop_under(targets, removed_paths):         # trailing "/" avoids _2 vs _20 prefix hits
    pref = tuple(f"{p}/" for p in removed_paths)
    return [t for t in targets if not t.prim_path.startswith(pref)]


def _bbox(bbox_range):
    lo, hi = np.array(bbox_range.GetMin()), np.array(bbox_range.GetMax())
    assert (hi > lo).all(), f"empty bbox: {lo} {hi}"
    return lo, hi


def _bottom_center(lo, hi):                      # usdz-frame shelf-contact anchor (Z-up)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def _orthonormal_rotation(l2w):
    """Product-site rotation with the ancestor model_* scale divided out.
    Fail-loud on what set_prim_pose (translate+rotate, NO scale) can't reproduce."""
    sc = np.linalg.norm(l2w[:3, :3], axis=0)
    assert np.allclose(sc, sc[0], rtol=1e-3), f"non-uniform scale in product l2w: {sc}"
    rot = l2w[:3, :3] / sc
    assert np.linalg.det(rot) > 0.9, f"improper rotation:\n{rot}"
    assert rot[2, 2] > 0.99, f"product not upright:\n{rot}"
    return rot


@dataclass(frozen=True)
class ProductSite:
    """What a replacement needs to know about the original product, captured
    BEFORE deactivation (bbox and l2w are unreadable once the prim is pruned)."""
    name: str            # SKU instance, e.g. cereal001_2
    path: str            # model_* prim path (what gets deactivated)
    lo: np.ndarray       # v_0 usdz-frame bbox
    hi: np.ndarray
    l2w: np.ndarray      # v_0 local-to-world, incl. ancestor scale
    grasp: np.ndarray    # grasp frame the config's policy mints at this bbox
    #                      ("which way does the front face point")


def measure_site(stage, prim, policy) -> ProductSite:
    """Read everything off the original product before it is deactivated."""
    from isaac_datagen.capture import get_target2world
    from isaac_datagen.isaac_utils import untransformed_bbox_range
    name, _ = parse_sku(prim.GetName())
    v0_path = f"{prim.GetPath().pathString}/v_0"
    assert stage.GetPrimAtPath(v0_path).IsValid(), f"no v_0 under {prim.GetPath()}"  # Stage-A contract
    lo, hi = _bbox(untransformed_bbox_range(stage.GetPrimAtPath(v0_path)))
    return ProductSite(name=name, path=prim.GetPath().pathString, lo=lo, hi=hi,
                       l2w=get_target2world([v0_path])[0], grasp=policy(lo, hi))


def replacement_pose(site: ProductSite, lo_r, hi_r, grasp_r) -> np.ndarray:
    """World pose for the swap wrapper. Rotation: aim the replacement's grasp
    face where the original's pointed (catalogs disagree on which local axis is
    "front", and the camera poser aims at the grasp face — without this a swap
    could face into the shelf; both grasp frames are +X-face/+Z-up, so a
    store->store swap degenerates to the original rotation). Translation: put
    the replacement's bbox bottom-center on the original's shelf-contact point,
    at the replacement's own real size."""
    rot = _orthonormal_rotation(site.l2w) @ site.grasp[:3, :3] @ grasp_r[:3, :3].T
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = (site.l2w @ np.append(_bottom_center(site.lo, site.hi), 1.0))[:3] \
                  - rot @ _bottom_center(lo_r, hi_r)
    return pose


def insert_replacement(stage, src: OptFlowObject, site: ProductSite) -> CaptureTarget:
    """Reference src's usdz under a /World/StoreSwaps wrapper, seat it at the
    site, return the new binding. Name minted unique per site (ReplicateFilter
    precedent). NO labels here — the uniform label_product loop applies the
    override ordering that store-extracted usdz require (arm A)."""
    from isaac_datagen.capture import set_prim_pose
    from isaac_datagen.isaac_utils import untransformed_bbox_range
    from isaac_datagen.scene import add_wrapped_reference
    name = f"{src.meta['name']}_at_{site.name}"
    wrapper = add_wrapped_reference(at_parent="/World/StoreSwaps",
                                    name=name, usd_path=src.usd_path)
    lo_r, hi_r = _bbox(untransformed_bbox_range(stage.GetPrimAtPath(wrapper)))
    set_prim_pose(wrapper, replacement_pose(site, lo_r, hi_r, src.grasp_point))
    return CaptureTarget(dataclasses.replace(src, meta={**src.meta, "name": name}), wrapper)


class RemoveClass:
    """Deactivate EVERY active store product whose SKU class fnmatches `pattern`;
    drop its bindings. Store-wide: unlabeled distractor instances go too — an
    emptied shelf slot, not a hidden mesh."""
    def __init__(self, pattern: str):
        assert pattern, "RemoveClass needs a non-empty class glob"
        self.pattern = pattern

    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import deactivate_prim
        removed = []
        for prim in _matching_products(store, spec, self.pattern):
            removed.append(prim.GetPath().pathString)
            deactivate_prim(prim)                             # model_* root, not v_0
            print(f"[MUT] RemoveClass({self.pattern!r}): {removed[-1]}", flush=True)
        return _drop_under(targets, removed)


class ReplaceClass:
    """Swap EVERY active store product whose SKU class fnmatches `pattern` for an
    OptFlowObject drawn (seeded, with replacement) from `catalog`, seated at the
    removed product's shelf pose. Store-wide like RemoveClass."""
    def __init__(self, pattern: str, catalog: str, source_class: str = "*"):
        assert pattern, "ReplaceClass needs a non-empty class glob"
        assert Path(catalog, "meta").is_dir(), f"not an object-dataset dir: {catalog}"
        self.pattern, self.catalog, self.source_class = pattern, catalog, source_class

    def _sources(self):
        from isaac_datagen.clean_datagen import collect_preoptflow   # deferred: cycle
        objs = [o for o in collect_preoptflow([self.catalog])
                if fnmatch.fnmatchcase(o.meta["class"], self.source_class)]
        assert objs, f"source_class {self.source_class!r} matches nothing in {self.catalog}"
        return objs

    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import create_empty, deactivate_prim
        sources = self._sources()
        policy = grasp_policies.get(spec.grasp_frame_policy)(**spec.grasp_frame_policy_args)
        stage = store.GetStage()
        create_empty("StoreSwaps", "/World")                  # Xform.Define: idempotent
        removed, added = [], []
        for prim in _matching_products(store, spec, self.pattern):
            site = measure_site(stage, prim, policy)          # reads BEFORE deactivation
            deactivate_prim(prim)
            src = sources[int(rng.integers(len(sources)))]    # seeded draw, with replacement
            added.append(insert_replacement(stage, src, site))
            removed.append(site.path)
            print(f"[MUT] ReplaceClass({self.pattern!r}): {site.path} -> "
                  f"{added[-1].obj.meta['name']}", flush=True)
        return _drop_under(targets, removed) + added

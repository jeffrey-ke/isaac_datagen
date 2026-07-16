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
    obj: OptFlowObject
    prim_path: str


def get(name):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e


def make_mutations(specs: list[dict]) -> list:
    return [get(s["name"])(**s.get("args", {})) for s in specs]


def apply_mutations(root, spec, targets, effective_seed):
    rng = np.random.default_rng([effective_seed, 3])  # stream 3 = mutations (0/1/2 = light jitters)
    for mutation in make_mutations(spec.mutations):
        targets = mutation(root, spec, targets, rng)
    assert targets, "mutations left no captureable targets"
    names = [t.obj.meta["name"] for t in targets]
    assert len(names) == len(set(names)), f"duplicate names after mutations: {names}"
    return targets


def active_products(store, patterns) -> list:
    return [c for c in store.GetChildren()
            if any(fnmatch.fnmatchcase(c.GetName(), pat) for pat in patterns)]


def _matching_products(store, spec, pattern) -> list:
    prims = [p for p in active_products(store, spec.product_patterns)
             if fnmatch.fnmatchcase(parse_sku(p.GetName())[1], pattern)]
    assert prims, f"class pattern {pattern!r} matches no ACTIVE store product"
    return prims


def _drop_under(targets, removed_paths):
    pref = tuple(f"{p}/" for p in removed_paths)
    return [t for t in targets if not t.prim_path.startswith(pref)]


def _bbox(bbox_range):
    lo, hi = np.array(bbox_range.GetMin()), np.array(bbox_range.GetMax())
    assert (hi > lo).all(), f"empty bbox: {lo} {hi}"
    return lo, hi


def _bottom_center(lo, hi):
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def _orthonormal_rotation(l2w):
    sc = np.linalg.norm(l2w[:3, :3], axis=0)
    rot = l2w[:3, :3] / sc   # per-column de-scale recovers R from R @ diag(s), non-uniform included
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-4), f"shear in product l2w:\n{l2w[:3, :3]}"
    assert np.linalg.det(rot) > 0.9, f"improper rotation:\n{rot}"
    assert rot[2, 2] > 0.99, f"product not upright:\n{rot}"
    return rot


@dataclass(frozen=True)
class ProductSite:
    name: str
    path: str
    lo: np.ndarray
    hi: np.ndarray
    l2w: np.ndarray
    grasp: np.ndarray


@dataclass(frozen=True)
class Site:
    store_prim: str          # "<product>/v_0", relative to the store root; resolves to the geometry prim
    grasp: np.ndarray        # curated grasp_point SE3 of the object that sat here
    cls: str


def load_sites(site_catalog) -> list[Site]:
    """Curated shelf sites from a store-extracted OptFlowObject catalog.

    Each object contributes (store_prim, curated grasp_point, class). Fails loud if
    handed a non-store catalog (objects without store_prim)."""
    from isaac_datagen.asset_catalogs import catalog_meta
    metas = catalog_meta(Path(site_catalog))     # sorted meta_*.yaml -> dicts
    sites = []
    for i, m in enumerate(metas):
        assert "store_prim" in m, (
            f"site catalog {site_catalog}: object {m.get('name', i)!r} has no store_prim "
            f"— not a store-extracted catalog; only store001-optflow-objects-keep-style "
            f"catalogs define shelf sites")
        grasp = OptFlowObject.deserialize_field(i, Path(site_catalog), "grasp_point")
        sites.append(Site(store_prim=m["store_prim"], grasp=np.asarray(grasp), cls=m["class"]))
    return sites


def measure_site(stage, prim, grasp_fn) -> ProductSite:
    # grasp_fn(lo, hi, cls) -> 4x4 SE3. ReplaceClass passes a bbox-face policy; repopulation
    # passes a constant fn returning the curated grasp (site geometry read once, either way).
    from isaac_datagen.capture import get_target2world
    from isaac_datagen.isaac_utils import untransformed_bbox_range
    name, cls = parse_sku(prim.GetName())
    v0_path = f"{prim.GetPath().pathString}/v_0"
    assert stage.GetPrimAtPath(v0_path).IsValid(), f"no v_0 under {prim.GetPath()}"
    lo, hi = _bbox(untransformed_bbox_range(stage.GetPrimAtPath(v0_path)))
    return ProductSite(name=name, path=prim.GetPath().pathString, lo=lo, hi=hi,
                       l2w=get_target2world([v0_path])[0], grasp=grasp_fn(lo, hi, cls))


def replacement_pose(site: ProductSite, lo_r, hi_r, grasp_r) -> np.ndarray:
    rot = _orthonormal_rotation(site.l2w) @ site.grasp[:3, :3] @ grasp_r[:3, :3].T
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = (site.l2w @ np.append(_bottom_center(site.lo, site.hi), 1.0))[:3] \
                  - rot @ _bottom_center(lo_r, hi_r)
    return pose


def insert_replacement(stage, src: OptFlowObject, site: ProductSite) -> CaptureTarget:
    from isaac_datagen.capture import set_prim_pose
    from isaac_datagen.isaac_utils import untransformed_bbox_range
    from isaac_datagen.scene import add_wrapped_reference
    name = f"{src.meta['name']}_at_{site.name}"
    wrapper = add_wrapped_reference(at_parent="/World/StoreSwaps",
                                    name=name, usd_path=src.usd_path)
    lo_r, hi_r = _bbox(untransformed_bbox_range(stage.GetPrimAtPath(wrapper)))
    set_prim_pose(wrapper, replacement_pose(site, lo_r, hi_r, src.grasp_point))
    return CaptureTarget(dataclasses.replace(src, meta={**src.meta, "name": name}), wrapper)


def replace_instance(stage, store, store_prim: str, src: OptFlowObject, grasp) -> CaptureTarget:
    """Repopulation atom: put `src` at the curated site addressed by `store_prim`,
    aligning src's own grasp to the site's curated `grasp`. Deactivates the native product."""
    from isaac_datagen.isaac_utils import deactivate_prim
    v0 = stage.GetPrimAtPath(store.GetPath().AppendPath(store_prim))
    assert v0.IsValid() and v0.IsActive(), f"store site prim gone/inactive: {store_prim}"
    product = v0.GetParent()                          # store_prim is "<product>/v_0"
    site = measure_site(stage, product, lambda lo, hi, cls: np.asarray(grasp))
    deactivate_prim(product)
    return insert_replacement(stage, src, site)


def deactivate_remaining_products(store, spec) -> int:
    """Strip every store product still active after repopulation (unused sites + non-catalog SKUs)."""
    from isaac_datagen.isaac_utils import deactivate_prim
    removed = [p.GetPath().pathString for p in active_products(store, spec.product_patterns)]
    for p in active_products(store, spec.product_patterns):
        deactivate_prim(p)
    print(f"[MUT] repopulate: deactivated {len(removed)} non-repopulated product(s)", flush=True)
    return len(removed)


class RemoveClass:
    def __init__(self, pattern: str):
        assert pattern, "RemoveClass needs a non-empty class glob"
        self.pattern = pattern

    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import deactivate_prim
        removed = []
        for prim in _matching_products(store, spec, self.pattern):
            removed.append(prim.GetPath().pathString)
            deactivate_prim(prim)
            print(f"[MUT] RemoveClass({self.pattern!r}): {removed[-1]}", flush=True)
        return _drop_under(targets, removed)


class ReplaceClass:
    def __init__(self, pattern: str, catalog: str, source_class: str = "*"):
        assert pattern, "ReplaceClass needs a non-empty class glob"
        assert Path(catalog, "meta").is_dir(), f"not an object-dataset dir: {catalog}"
        self.pattern, self.catalog, self.source_class = pattern, catalog, source_class

    def _sources(self):
        from isaac_datagen.clean_datagen import collect_preoptflow
        objs = [o for o in collect_preoptflow([self.catalog])
                if fnmatch.fnmatchcase(o.meta["class"], self.source_class)]
        assert objs, f"source_class {self.source_class!r} matches nothing in {self.catalog}"
        return objs

    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import create_empty, deactivate_prim
        sources = self._sources()
        policy = grasp_policies.get(spec.grasp_frame_policy)(**spec.grasp_frame_policy_args)
        stage = store.GetStage()
        create_empty("StoreSwaps", "/World")
        removed, added = [], []
        for prim in _matching_products(store, spec, self.pattern):
            site = measure_site(stage, prim, policy)
            deactivate_prim(prim)
            src = sources[int(rng.integers(len(sources)))]
            added.append(insert_replacement(stage, src, site))
            removed.append(site.path)
            print(f"[MUT] ReplaceClass({self.pattern!r}): {site.path} -> "
                  f"{added[-1].obj.meta['name']}", flush=True)
        return _drop_under(targets, removed) + added


class RemoveUntrackedProducts:
    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import deactivate_prim
        tracked = {t.obj.meta["class"] for t in targets}
        removed = []
        for prim in active_products(store, spec.product_patterns):
            if parse_sku(prim.GetName())[1] not in tracked:
                removed.append(prim.GetPath().pathString)
                deactivate_prim(prim)
        print(f"[MUT] RemoveUntrackedProducts: deactivated {len(removed)} untracked "
              f"product(s), kept {len(tracked)} classes", flush=True)
        return _drop_under(targets, removed)


class RemovePrims:
    def __init__(self, names: list):
        assert names, "RemovePrims needs a non-empty prim-name list"
        self.names = list(names)

    def __call__(self, store, spec, targets, rng):
        from isaac_datagen.isaac_utils import deactivate_prim
        by_name = {p.GetName(): p for p in active_products(store, spec.product_patterns)}
        missing = sorted(set(self.names) - set(by_name))
        assert not missing, f"RemovePrims: no active product prim named {missing}"
        removed = [by_name[n].GetPath().pathString for n in self.names]
        for n in self.names:
            deactivate_prim(by_name[n])
        print(f"[MUT] RemovePrims: deactivated {len(removed)} product prim(s)", flush=True)
        return _drop_under(targets, removed)


class DisablePhysics:
    PLAIN_SAFE = True  # reads no StoreSceneSpec fields; usable from build_scene

    def __init__(self, pattern: str):
        assert pattern, "DisablePhysics needs a non-empty prim-name glob"
        self.pattern = pattern

    def __call__(self, root, spec, targets, rng):
        from pxr import Usd
        from isaac_datagen.isaac_utils import disable_rigid_body
        matched = [p for p in Usd.PrimRange(root)
                   if fnmatch.fnmatchcase(p.GetName(), self.pattern)]
        assert matched, f"DisablePhysics({self.pattern!r}): no prim matches under {root.GetPath()}"
        subtree = {q.GetPath().pathString: q  # dict dedup: broad patterns nest matches
                   for m in matched for q in Usd.PrimRange(m)}
        disabled = [path for path, q in sorted(subtree.items()) if disable_rigid_body(q)]
        assert disabled, f"DisablePhysics({self.pattern!r}): matched prims carry no rigid body"
        print(f"[MUT] DisablePhysics({self.pattern!r}): disabled {len(disabled)} rigid body(ies)",
              flush=True)
        return targets

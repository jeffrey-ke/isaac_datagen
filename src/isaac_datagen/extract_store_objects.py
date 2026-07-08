"""Stage A (inverse datagen): extract shelf-product prims from an existing store
USD into a GraspableObject catalog.

    uv run src/isaac_datagen/extract_store_objects.py <store_config.yaml> <out_dataset> [key=val ...]

Runs inside booted Isaac: store001.usd's geometry composes from a remote https
subLayer only the omni resolver can fetch. The SAME store config drives Stage C
(store_scene.StoreSceneSpec single-sources store_usd / patterns / grasp policy).
"""
from __future__ import annotations

import itertools
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage

from isaac_datagen import grasp_policies
from isaac_datagen.objects import GraspableObject, UsdPath
from isaac_datagen.runtime_config import load_config
from isaac_datagen.scene import boot_sim

# model_sauces001_6 -> name 'sauces001_6', class 'sauces001'; model_snack014 -> snack014/snack014
SKU_RE = re.compile(r"^model_(?P<name>(?P<cls>[a-z][a-z0-9_]*?\d{3})(?:_\d+)?)$")


def parse_sku(prim_name: str) -> tuple[str, str]:
    m = SKU_RE.match(prim_name)
    if m is None:
        raise ValueError(f"prim name does not parse as a SKU: {prim_name!r}")
    return m["name"], m["cls"]


def placeholder_reference() -> PILImage.Image:
    """Unused downstream — Stage B (render_one) re-renders reference_image fresh."""
    return PILImage.new("RGB", (64, 64), (128, 128, 128))


def matched_products(store, patterns) -> list[str]:
    """Union of find_prims(store, pat) over the config globs — sorted, deduped
    absolute prim paths. Pure function of its inputs: the store root prim comes
    from the caller (find_prims takes a prim directly — no current-stage lookup,
    no module path constants) and find_prims raises per pattern on zero matches."""
    from isaac_datagen.isaac_utils import find_prims
    return sorted(set(itertools.chain.from_iterable(
        find_prims(store, pat) for pat in patterns)))


def extract_one(store, model_path: str, policy, tmp_dir: str) -> GraspableObject:
    from isaac_datagen.isaac_utils import export_subtree_usdz, untransformed_bbox_range
    name, cls = parse_sku(model_path.rsplit("/", 1)[1])
    p = f"{model_path}/v_0"                                   # P = the transform node: neutralized on
    prim = store.GetStage().GetPrimAtPath(p)                  # export, l2w read at this exact node
    assert prim.IsValid(), f"no v_0 transform node under {model_path}"
    rng = untransformed_bbox_range(prim)                      # usdz-frame bbox: EXCLUDES P's own ops
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    assert (hi > lo).all(), f"empty bbox at {p}"
    usdz = export_subtree_usdz(store.GetStage(), p, tmp_dir, base_name=name,
                               root_prim="/World", neutralize_root_xform=True)
    return GraspableObject(
        usd_path=UsdPath(usdz),
        meta={"name": name, "class": cls,                     # store_prim RELATIVE to the store root
              "store_prim": p[len(str(store.GetPath())) + 1:]},  # (referencing /root splices its
        #  children directly under STORE_ROOT, so this is e.g. "model_sauces001_6/v_0")
        reference_image=placeholder_reference(),
        grasp_point=policy(lo, hi).astype(np.float32),
    )


def main() -> None:
    runtime = load_config(sys.argv[1], sys.argv[3:])
    assert runtime.scene_builder == "build_store_scene", \
        "extract_store_objects needs a store config (scene_builder: build_store_scene)"
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    app = boot_sim(runtime, out_dir)

    from isaac_datagen.store_scene import StoreSceneSpec, load_store
    spec = StoreSceneSpec(**runtime.scene_builder_args)
    policy = grasp_policies.get(spec.grasp_frame_policy)(**spec.grasp_frame_policy_args)
    store = load_store(spec)                                  # the store root prim — the one handle
    products = matched_products(store, spec.product_patterns)
    names = [parse_sku(p.rsplit("/", 1)[1])[0] for p in products]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate SKU names among matched prims: {dupes}"   # collect_* contract

    # export_subtree_usdz writes each usdz into the TEMP dir; GraspableObject.serialize's
    # UsdPath serializer (shutil.copy) then copies it to {out_dir}/usd_path/usd_path_{idx:04d}.usdz
    # — the durable copy. The temp dir (and its intermediate usdz) is deleted on context exit.
    with tempfile.TemporaryDirectory() as tmp:
        for idx, mp in enumerate(products):
            obj = extract_one(store, mp, policy, tmp)
            obj.serialize(idx, out_dir)
            print(f"  [{idx:04d}] {obj.meta['name']} (class {obj.meta['class']}) extracted", flush=True)
    with open(out_dir / "runtime.yaml", "w") as f:            # provenance (optflow_generation idiom):
        yaml.safe_dump(asdict(runtime), f)                    # records grasp policy + patterns used
    app.close()


if __name__ == "__main__":
    main()

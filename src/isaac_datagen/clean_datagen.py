"""Entry point: load YAML config, boot sim, build scene, capture dataset.

Usage:
    python clean_datagen.py <config.yaml> [key=value ...]

The YAML is validated against RuntimeConfig; trailing args are OmegaConf
dotlist overrides (e.g. `num_frames=8 seed=3`).
"""

from __future__ import annotations
import os
# Let torch reuse fragmented reserved blocks; the SD ensemble allocation otherwise
# fails against Isaac's resident CUDA memory on the shared GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import asdict
from pathlib import Path
import itertools
import json
import sys

import yaml

from isaac_datagen.scene import boot_sim, build_scene, make_replicator, warmup_render
from isaac_datagen.capture import get_target2world, capture_with_poses, plan_capture
from isaac_datagen.runtime_config import load_config
from isaac_datagen.objects import GraspableObject, OptFlowObject
from isaac_datagen.filters import filter_objects
from isaac_datagen import posers
from vision_core.seed_utils import seed_everything


def collect_objects(paths: list[str | Path]) -> list[GraspableObject]:
    """Deserialize every GraspableObject from each dataset dir and concatenate.

    Safe to chain across datasets: cids derive from the sorted class-name set and iids
    from per-prim semantic labels, so neither depends on list position. The one
    requirement is globally-unique meta["name"] — it is the USD prim-path component in
    scene.add_object and the name_to_class key in reference_seg_writer, both last-wins
    on a clash.
    """
    list_of_lists = []
    for p in paths:
        path = Path(p)
        n = len(sorted((path / "meta").glob("meta_*.yaml")))
        list_of_lists.append([GraspableObject.deserialize(i, path) for i in range(n)])

    objects = list(itertools.chain.from_iterable(list_of_lists))

    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate GraspableObject names across datasets: {dupes}")
    return objects


def collect_preoptflow(paths: list[str | Path]) -> list[OptFlowObject]:
    """Deserialize every OptFlowObject from each dataset dir and concatenate.

    The optical-flow analog of collect_objects: same meta/ count idiom and the same
    globally-unique meta["name"] requirement (the name is the placed prim-path component
    in scene.add_object and the catalog join key in OptFlowMetadata)."""
    objects: list[OptFlowObject] = []
    for p in paths:
        path = Path(p)
        n = len(sorted((path / "meta").glob("meta_*.yaml")))
        objects.extend(OptFlowObject.deserialize(i, path) for i in range(n))

    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate OptFlowObject names across datasets: {dupes}")
    return objects


def reference_segmentation(runtime=None):
    if runtime is None:
        runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(runtime.effective_seed)        # seed = runtime.seed + runtime.idx; before boot_sim
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.reference_seg_writer import ObsMaskWriter

    objects = filter_objects(
            collect_objects(runtime.objects_path),
            runtime.filter_specs
    )
    scene = build_scene(runtime, objects)

    _idx, _grasp_points, world_poses = plan_capture(runtime, scene)

    if runtime.dry_run:
        # Dry run: export scene.usdz + baked debug cameras (at the planned poses) +
        # an axis gizmo on every candidate grasp frame, for offline (Blender)
        # inspection, then skip the writer and RTX capture entirely. world_poses is
        # produced by the exact same code the real run uses.
        from isaac_datagen.debug_export import decorate_debug_scene, export_debug_bundle
        export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)
        app.close()
        return

    writer = ObsMaskWriter(runtime.descriptor_config_path, runtime.descriptor_device, scene.objects,
                           render_dir, full_alpha=runtime.obs_full_alpha)
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    warmup_render(app, runtime.warmup_frames)   # settle RTX before the writer captures
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)

    # Write the per-render-dir catalog (id-space maps + per-class reference images + DIFT features).
    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


def optflow_generation(runtime=None):
    """Optical-flow capture: place OptFlowObjects in clutter, capture per-frame RGB-D + camera
    pose, and write the per-render constant catalog once. Each ``OptFlowSample`` nests a full
    ``ObsMask`` (serialized flat) and ``OptFlowMetadata`` nests an ``ObsMaskMetadata``, so the
    render dir is ALSO a reference-seg render dir consumable by run_pipeline phases 2 & 3.

    Mirrors reference_segmentation; plan_capture is reused purely as the camera-pose
    generator (it aims the rig at randomly chosen grasp frames — the observation captures
    whatever is visible). See .docs_claude/plans/active/optflow-2-writer-capture.md.
    """
    if runtime is None:
        runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(runtime.effective_seed)        # before boot_sim
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.optflow_writer import OptFlowWriter

    objects = filter_objects(
        collect_preoptflow(runtime.objects_path),
        runtime.filter_specs,
    )
    scene = build_scene(runtime, objects)          # SceneHandle.object_prim_paths now populated
    l2w = get_target2world(scene.object_prim_paths)   # (M, 4, 4) aligned to scene.objects

    _idx, _grasp_points, world_poses = plan_capture(runtime, scene)

    if runtime.dry_run:
        from isaac_datagen.debug_export import decorate_debug_scene, export_debug_bundle
        export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)
        app.close()
        return

    # obsmask.obs is RGBA whose alpha carries the instance foreground (full_alpha=False, recoverable
    # straight off obs). The RGB channels are the full frame either way (composite_rgba never masks RGB;
    # straight-alpha PNG preserves color under alpha==0), and the UFM adapter reads obs[:3], so the alpha
    # never reaches the warp — no reason to blank it to 255.
    writer = OptFlowWriter(scene.objects, l2w, scene.zed.intrinsics, render_dir,
                           runtime.descriptor_config_path, runtime.descriptor_device,
                           full_alpha=False)   # obs K = zed.intrinsics
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    warmup_render(app, runtime.warmup_frames)      # settle RTX before the writer captures
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)

    # Write the per-render-dir constants (nested ObsMaskMetadata + optflow per-class catalog).
    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    # Keep the dir self-describing (phases 2/3 don't read it, but mirrors reference_segmentation).
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


def main():
    """Console entry: load the config, then dispatch on runtime.mode."""
    runtime = load_config(sys.argv[1], sys.argv[2:])
    if runtime.mode == "optflow":
        optflow_generation(runtime)
    else:
        reference_segmentation(runtime)


if __name__ == "__main__":
    main()

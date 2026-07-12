
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from dataclasses import asdict
from pathlib import Path
import argparse
import itertools
import json
import sys

import yaml

from isaac_datagen.scene import boot_sim, build_scene, make_replicator, warmup_render
from isaac_datagen import scene_builders
from isaac_datagen.capture import get_target2world, capture_with_poses, plan_capture
from isaac_datagen.runtime_config import load_config
from isaac_datagen.objects import GraspableObject, OptFlowObject
from isaac_datagen.filters import filter_objects
from isaac_datagen import posers
from isaac_datagen.tldr import TLDR
from vision_core.seed_utils import seed_everything


def collect_objects(paths: list[str | Path]) -> list[GraspableObject]:
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
    from isaac_datagen import cid_iid_trace
    cid_iid_trace.init(render_dir)
    seed_everything(runtime.effective_seed)
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.reference_seg_writer import ObsMaskWriter

    objects = filter_objects(
            collect_objects(runtime.objects_path),
            runtime.filter_specs
    )
    scene = build_scene(runtime, objects)

    _idx, _grasp_points, world_poses = plan_capture(runtime, scene)

    if runtime.dry_run:
        from isaac_datagen.debug_export import decorate_debug_scene, export_debug_bundle
        export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)
        app.close()
        return

    writer = ObsMaskWriter(runtime.descriptor_config_path, runtime.descriptor_device, scene.objects,
                           render_dir, full_alpha=runtime.obs_full_alpha)
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    warmup_render(app, runtime.warmup_frames)
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)

    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


def optflow_generation(runtime=None):
    if runtime is None:
        runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    from isaac_datagen import cid_iid_trace
    cid_iid_trace.init(render_dir)
    seed_everything(runtime.effective_seed)
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.optflow_writer import OptFlowWriter

    objects = filter_objects(
        collect_preoptflow(runtime.objects_path),
        runtime.filter_specs,
    )
    scene = scene_builders.get(runtime.scene_builder)(runtime, objects)
    l2w = get_target2world(scene.object_prim_paths)

    _idx, _grasp_points, world_poses = plan_capture(runtime, scene)

    if runtime.dry_run:
        from isaac_datagen.debug_export import decorate_debug_scene, export_debug_bundle
        export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)
        app.close()
        return

    writer = OptFlowWriter(scene.objects, l2w, scene.zed.intrinsics, render_dir,
                           runtime.descriptor_config_path, runtime.descriptor_device,
                           full_alpha=runtime.obs_full_alpha)
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    warmup_render(app, runtime.warmup_frames)
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)

    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


def main():
    parser = argparse.ArgumentParser(
        prog="isaac-datagen",
        description="Load a YAML config (+ optional key=value dotlist overrides), "
                     "boot Isaac Sim, and render a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TLDR,
    )
    parser.add_argument("config", help="path to a YAML config (see CONFIGS below)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args, overrides = parser.parse_known_args(sys.argv[1:])
    runtime = load_config(args.config, overrides)
    if runtime.mode == "optflow":
        optflow_generation(runtime)
    else:
        reference_segmentation(runtime)


if __name__ == "__main__":
    main()

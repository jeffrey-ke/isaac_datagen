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
import json
import sys

import numpy as np
import yaml

from isaac_datagen.scene import boot_sim, build_scene, make_replicator
from isaac_datagen.capture import get_target2world, capture_with_poses, plan_capture
from isaac_datagen.runtime_config import load_config
from isaac_datagen.objects import GraspableObject
from isaac_datagen import posers


def main():
    if len(sys.argv) < 2:
        print("usage: python clean_datagen.py <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    app = boot_sim(runtime, render_dir)

    rng = np.random.RandomState(runtime.seed)
    objects = collect_objects(runtime.graspable_objects_path)
    scene = build_scene(runtime, objects, rng)

    grasp_point = scene.grasp_points[rng.randint(len(scene.grasp_points))]

    target2world = get_target2world(grasp_point)
    replicator = make_replicator(runtime, target2world)

    from isaac_datagen.stereo_writer import StereoSampleWriter

    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)          # (N, 4, 4)
    world_poses = target2world @ target_frame_poses
    offsets = [pose[:3, 3].tolist() for pose in target_frame_poses]

    stereo_writer = StereoSampleWriter(output_dir=str(render_dir),
                                       offsets=offsets, target2world=target2world)
    capture_with_poses(world_poses, stereo_writer, scene.zed, replicator)

    with open(render_dir / f'config{runtime.idx:03d}.json', 'w') as f:
        json.dump(asdict(runtime), f, indent=2)

    app.close()

def collect_objects(path: str | Path) -> list[GraspableObject]:
    path = Path(path)
    n = len(sorted((path / "meta").glob("meta_*.yaml")))
    return [GraspableObject.deserialize(i, path) for i in range(n)]

def reference_segmentation():
    runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.reference_seg_writer import ObsMaskWriter

    rng = np.random.RandomState(runtime.seed)
    objects = collect_objects(runtime.graspable_objects_path)
    scene = build_scene(runtime, objects, rng)

    _idx, _grasp_points, world_poses = plan_capture(runtime, scene, rng)

    if runtime.dry_run:
        # Dry run: export scene.usdz + baked debug cameras (at the planned poses) +
        # an axis gizmo on every candidate grasp frame, for offline (Blender)
        # inspection, then skip the writer and RTX capture entirely. world_poses is
        # produced by the exact same code the real run uses.
        from isaac_datagen.debug_export import decorate_debug_scene, export_debug_bundle
        export_debug_bundle(decorate_debug_scene(scene, world_poses), render_dir)
        app.close()
        return

    writer = ObsMaskWriter(runtime.descriptor_config_path, runtime.descriptor_device, scene.objects, render_dir)
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)

    # Write the per-render-dir catalog (id-space maps + per-class reference images + DIFT features).
    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


if __name__ == "__main__":
    reference_segmentation()

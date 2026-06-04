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
from isaac_datagen.capture import get_target2world, make_index, capture_with_poses
from isaac_datagen.pose_planning import plan_poses
from isaac_datagen.runtime_config import load_config
from isaac_datagen.objects import GraspableObject


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

    make_index(
        runtime.target_to_baseline_ypr_desired,
        runtime.xrange, runtime.yrange, runtime.zrange,
        runtime.sampling, grasp_point, scene.zed, replicator, render_dir,
    )

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

    writer = ObsMaskWriter(runtime.descriptor_config_path, runtime.descriptor_device, scene.objects, render_dir)

    idx = rng.choice(len(scene.grasp_points), size=runtime.num_targets)
    grasp_points = [scene.grasp_points[i] for i in idx]

    target2worlds = get_target2world(grasp_points)
    replicator = make_replicator(runtime)
    target_frame_poses = plan_poses(
        runtime.target_to_baseline_ypr_desired, runtime.xrange, runtime.yrange, runtime.zrange, runtime.sampling
    )
    # (B, N, 4, 4) -> (B*N, 4, 4): flatten target and pose dims into one batch.
    world_poses = np.einsum('bij,njk->bnik', target2worlds, target_frame_poses).reshape(-1, 4, 4)

    capture_with_poses(world_poses, writer, scene.zed, replicator)

    # Write the per-render-dir catalog (id-space maps + per-class reference images + DIFT features).
    writer.finalize_metadata(render_dir)

    with open(render_dir / 'runtime.yaml', 'w') as f:
        yaml.safe_dump(asdict(runtime), f)
    with open(render_dir / 'descriptor.yaml', 'w') as f:
        yaml.safe_dump(yaml.safe_load(Path(runtime.descriptor_config_path).read_text()), f)

    app.close()


if __name__ == "__main__":
    reference_segmentation()

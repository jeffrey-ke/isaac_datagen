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

import yaml

from isaac_datagen.scene import boot_sim, build_scene, make_replicator
from isaac_datagen.capture import get_target2world, capture_with_poses, plan_capture
from isaac_datagen.runtime_config import load_config
from isaac_datagen.objects import GraspableObject
from isaac_datagen import posers
from vision_core.seed_utils import seed_everything


def collect_objects(path: str | Path) -> list[GraspableObject]:
    path = Path(path)
    n = len(sorted((path / "meta").glob("meta_*.yaml")))
    return [GraspableObject.deserialize(i, path) for i in range(n)]

def reference_segmentation():
    runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(runtime.effective_seed)        # seed = runtime.seed + runtime.idx; before boot_sim
    app = boot_sim(runtime, render_dir)

    from isaac_datagen.reference_seg_writer import ObsMaskWriter

    objects = collect_objects(runtime.graspable_objects_path)
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
                           render_dir)
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

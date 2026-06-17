"""Debug the reference-segmentation scene: grasp points + USDZ export.

Rebuilds exactly the scene `reference_segmentation()` builds (same objects
slice, same seed, same pallet_dims), then:

  1. Prints how many grasp points the scene actually has, the `np.random.choice`
     indices that pick the per-target grasp frames, and the world-frame
     translation of every grasp point — so you can SEE whether the targets are
     distinct boxes or the same box repeated.
  2. Exports the built `/World` to a self-contained `.usdz` you can open in any
     USD viewer to inspect the geometry (e.g. why the 11x4 pallet renders as a
     narrow tower).

Run it the same way as clean_datagen.py, from src/isaac_datagen/:

    uv run debug_scene.py configs/randomized.yaml [key=value ...]

Diagnostics are printed AND written to <render_dir>/grasp_debug.txt, because
Isaac Sim floods stdout with kit startup noise (see usdz_export_investigation).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

import numpy as np

from isaac_datagen.scene import boot_sim, build_scene
from isaac_datagen.capture import get_target2world
from isaac_datagen.isaac_utils import export_subtree_usdz
from isaac_datagen.runtime_config import load_config
from isaac_datagen.clean_datagen import collect_objects
from vision_core.seed_utils import seed_everything


def main():
    if len(sys.argv) < 2:
        print("usage: uv run debug_scene.py <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])

    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(runtime.effective_seed)        # before boot_sim, mirroring reference_segmentation
    app = boot_sim(runtime, render_dir)

    # Mirror reference_segmentation() exactly so the diagnosis is faithful: same seed
    # (global, via seed_everything), same build, same global np.random.choice for the
    # grasp picks — so the indices below match what the real run draws.
    objects = collect_objects(runtime.graspable_objects_path)
    scene = build_scene(runtime, objects)

    n = len(scene.grasp_points)
    idx = np.random.choice(n, size=runtime.num_targets)
    selected = [scene.grasp_points[i] for i in idx]

    all_t2w = get_target2world(scene.grasp_points)        # (n, 4, 4)
    sel_t2w = get_target2world(selected)                  # (num_targets, 4, 4)

    lines = []
    cap = int(np.prod(runtime.pallet_dims)) if runtime.pallet_dims else "n/a"
    lines.append(f"pallet_dims          = {runtime.pallet_dims}  (capacity = {cap})")
    lines.append(f"objects passed       = {len(objects)}  (collect_objects(...))")
    lines.append(f"num_targets (config) = {runtime.num_targets}")
    lines.append(f"len(grasp_points)    = {n}")
    lines.append(f"np.random.choice indices   = {idx.tolist()}")
    lines.append("")
    lines.append("ALL grasp points (path  ->  world translation):")
    for path, t2w in zip(scene.grasp_points, all_t2w):
        lines.append(f"  {path}  ->  {np.round(t2w[:3, 3], 4).tolist()}")
    lines.append("")
    lines.append("SELECTED per-target grasp points (one per num_target):")
    for i, (path, t2w) in enumerate(zip(selected, sel_t2w)):
        lines.append(f"  target[{i}] idx={idx[i]}  {path}  ->  {np.round(t2w[:3, 3], 4).tolist()}")
    lines.append("")
    unique_xyz = {tuple(np.round(t[:3, 3], 4).tolist()) for t in sel_t2w}
    lines.append(f"=> {len(unique_xyz)} UNIQUE target translation(s) across {runtime.num_targets} targets")
    if len(unique_xyz) == 1:
        lines.append("   !! all targets collapse to ONE box -- np.random.choice repeats the same index")

    report = "\n".join(lines)
    print(report)
    (render_dir / "grasp_debug.txt").write_text(report + "\n")

    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    usdz_path = export_subtree_usdz(stage, "/World", str(render_dir), base_name="scene")
    print(f"\nwrote {os.path.abspath(usdz_path)}")

    # Extract each distinct selected box on its own. The grasp frame lives at
    # ".../<box>/GraspPoint", so the box wrapper prim is its parent. Exporting
    # that subtree alone lets you open it standalone and see which box was picked.
    seen = []
    for gp_path in selected:
        box_path = gp_path.rsplit("/", 1)[0]   # strip "/GraspPoint"
        if box_path in seen:
            continue
        seen.append(box_path)
        box_name = box_path.rsplit("/", 1)[1]
        box_usdz = export_subtree_usdz(stage, box_path, str(render_dir), base_name=f"selected_{box_name}")
        print(f"wrote {os.path.abspath(box_usdz)}  (grasp target: {box_path})")

    print(f"wrote {os.path.abspath(render_dir / 'grasp_debug.txt')}")

    app.close()


if __name__ == "__main__":
    main()

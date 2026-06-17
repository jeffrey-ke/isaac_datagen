"""Render OptFlowSample.visualize(md) for the frames in an optflow render dir.

Deserializes the render dir's OptFlowMetadata once (idx 0) and every per-frame
OptFlowSample, then emits one labeled [ref | obs] correspondence PNG per frame
(GT reference→observation warp via RoMa's get_gt_warp, the trainer's warp).

    uv run debug_scripts/viz_optflow.py <render_dir> [--idx N] [--cls CLASS] [--out DIR] [--dpi N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from PIL import Image

from isaac_datagen.objects import OptFlowSample, OptFlowMetadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("render_dir", type=Path)
    p.add_argument("--idx", type=int, default=None, help="single frame; default = all frames")
    p.add_argument("--cls", default=None, help="restrict to one class (default = every class)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()

    matplotlib.rcParams["figure.dpi"] = args.dpi  # visualize() rasterizes at figure dpi

    md = OptFlowMetadata.deserialize(0, args.render_dir)
    print(f"classes: {list(md.class_to_name)}")
    for cls, names in md.class_to_name.items():
        print(f"  {cls}: {len(names)} instances {names}")

    n = len(sorted((args.render_dir / "observation").glob("observation_*.png")))
    idxs = [args.idx] if args.idx is not None else range(n)

    out_dir = args.out or args.render_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(list(idxs))} frame(s) → {out_dir}")

    for i in idxs:
        sample = OptFlowSample.deserialize(i, args.render_dir)
        arr = sample.visualize(md, cls_name=args.cls, title=f"{args.render_dir.name} frame {i:04d}")
        path = out_dir / f"viz_{i:04d}.png"
        Image.fromarray(arr).save(path)
        print(f"  frame {i:04d} → {path.name}  {arr.shape}")

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()

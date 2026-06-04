"""Spot-check viz for ObsMask occlusion ratios.

Thin CLI over ``vision_core.viz``: draws a RANDOM sample of frames across a
generated dataset and renders one ``occlusion_panel`` per frame — each iid's
mask overlaid in a unique color, the occlusion ratio printed at the mask
centroid, and an iid→name→ratio legend in the right gutter. Buried /
frame-edge objects should carry high ratios, clearly-visible ones near 0.
Deliberately stays in INSTANCE space (occlusion is per-instance); see
viz_inliers for the class-space view.

Usage:
    isaac-datagen-viz-occlusion <dataset_dir> [--n 12] [--seed 0] [--cols 4]
        [--alpha 0.45] [--dpi 200] [--out PATH]
``<dataset_dir>`` is the directory holding ``render000/``, ``render001/``, … (i.e.
``runtime.dataset_dir``); the sample is drawn across ALL render dirs under it.
"""

import argparse
import functools
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from vision_core.datastructs import ObsMask, ObsMaskMetadata, count_samples
from vision_core.viz import error_panel, occlusion_panel, panel_grid, save_figure


def frame_pairs(dataset_dir: Path):
    """All (render_dir, frame_idx) pairs under a dataset root, for random sampling."""
    return [(rd, i)
            for rd in sorted(p for p in dataset_dir.glob("render*") if p.is_dir())
            if (rd / "obs").is_dir()
            for i in range(count_samples(rd))]


def main():
    p = argparse.ArgumentParser(description="Spot-check ObsMask occlusion ratios over a random sample.")
    p.add_argument("dataset_dir", type=Path, help="dir holding render000/ render001/ …")
    p.add_argument("--n", type=int, default=12, help="number of random frames to show")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pairs = frame_pairs(args.dataset_dir)
    if not pairs:
        print(f"no render*/obs frames under {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)
    sample = random.Random(args.seed).sample(pairs, min(args.n, len(pairs)))

    metadata = functools.cache(lambda rd: ObsMaskMetadata.deserialize(0, rd))

    # wide wspace leaves a gutter for each panel's external (right-side) legend
    fig, axes = panel_grid(len(sample), args.cols, 6.2, 4.6, wspace=0.6, hspace=0.2)
    for ax, (rd, idx) in zip(axes, sample):
        try:
            occlusion_panel(ax, ObsMask.deserialize(idx, rd), metadata(rd), alpha=args.alpha)
        except Exception as e:  # e.g. old-schema dirs (missing cid_mask/ or pre-rename fields)
            error_panel(ax, f"{rd.name}/{idx:04d}", e)
            continue
        ax.set_title(f"{rd.name}/{idx:04d}", fontsize=8)

    out = args.out or args.dataset_dir / "occlusion_viz.png"
    fig.suptitle(f"occlusion spot-check — {len(sample)} random frames from {args.dataset_dir.name}", fontsize=10)
    save_figure(fig, out, args.dpi)
    print(f"wrote {out}  ({len(sample)} frames from {len(pairs)} total)")


if __name__ == "__main__":
    main()


import argparse
import functools
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from vision_core.datastructs import ObsMask, ObsMaskDescriptorMetadata, count_samples
from vision_core.viz import error_panel, occlusion_panel, panel_grid, save_figure


def frame_pairs(dataset_dir: Path):
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

    metadata = functools.cache(lambda rd: ObsMaskDescriptorMetadata.deserialize(0, rd))

    fig, axes = panel_grid(len(sample), args.cols, 6.2, 4.6, wspace=0.6, hspace=0.2)
    for ax, (rd, idx) in zip(axes, sample):
        try:
            occlusion_panel(ax, ObsMask.deserialize(idx, rd), metadata(rd), alpha=args.alpha)
        except Exception as e:
            error_panel(ax, f"{rd.name}/{idx:04d}", e)
            continue
        ax.set_title(f"{rd.name}/{idx:04d}", fontsize=8)

    out = args.out or args.dataset_dir / "occlusion_viz.png"
    fig.suptitle(f"occlusion spot-check — {len(sample)} random frames from {args.dataset_dir.name}", fontsize=10)
    save_figure(fig, out, args.dpi)
    print(f"wrote {out}  ({len(sample)} frames from {len(pairs)} total)")


if __name__ == "__main__":
    main()

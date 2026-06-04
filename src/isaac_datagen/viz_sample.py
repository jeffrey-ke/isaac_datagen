"""Per-sample viz for ImageInlierSample — per-class proposals and gt masks.

Convenience compositions of ``vision_core.viz`` primitives (same parts as
viz_inliers, different composition). An ImageInlierSample maps class name →
proposals and class name → labels; for each selected frame, one figure:

  panels 0..N-1   — per class: obs with THAT class's proposals scattered,
                    color-coded by its labels (green = inlier / red = outlier),
                    with the class's canonical reference image as a thumbnail
                    in the corner.
  panels N..2N-1  — per class: obs with its gt union mask (``cid_mask == cid``)
                    alpha-blended, color-matched to its proposals panel title.

Usage:
    isaac-datagen-viz-sample <render_dir> [--out DIR] [--frames 0,5,10 |
        --max-frames K --stride S] [--cols 4] [--dpi 200] [--max-points N]
        [--alpha 0.45]
Requires phase-3 output (``labels/``); run ``isaac-datagen-inliers`` first.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from vision_core.datastructs import ImageInlierSample, ObsMaskMetadata, count_samples
from vision_core.viz import (add_thumbnail, assign_colors, overlay_id_masks, panel_grid,
                             rgba_chw_to_rgb, save_figure, scatter_labeled)
from isaac_datagen.viz_inliers import select_frames


# ── Convenience panels — thin calling sequences over vision_core.viz ────────────

def class_proposals_panel(ax, obs_rgb, coords, labels, ref_rgba, cls, color, max_points=None):
    """obs + one class's proposals (green = inlier / red = outlier from its
    labels) + that class's reference image as a corner thumbnail."""
    ax.imshow(obs_rgb)
    note = scatter_labeled(ax, coords, labels, max_points)
    ax.set_title(f"{cls}  in={int(labels.sum())}/{len(labels)}{note}", fontsize=8, color=color)
    ax.axis("off")
    add_thumbnail(ax, ref_rgba)


def class_mask_panel(ax, obs_rgb, cid_mask_np, cid, cls, color, alpha=0.45):
    """obs + one class's gt union mask (``cid_mask == cid``) alpha-blended."""
    ax.imshow(overlay_id_masks(obs_rgb, cid_mask_np, {cid: color}, alpha))
    ax.set_title(f"{cls}  gt mask", fontsize=8, color=color)
    ax.axis("off")


def sample_figure(sample, md, *, cols=4, max_points=None, alpha=0.45, title=None):
    """ImageInlierSample → Figure: one proposals panel (with ref thumbnail) per
    class in ``sample.proposals``, then one gt union-mask panel per class.
    Returns None if the sample has no proposals."""
    obs_rgb = rgba_chw_to_rgb(sample.obs)
    cidm = sample.cid_mask.numpy()
    # driven by the proposals dict — a class can be present in cid_mask yet
    # have no proposals (e.g. skipped by the proposer pass)
    classes = sorted(sample.proposals)
    if not classes:
        return None
    class_to_cid = {c: i for i, c in md.cid_to_class.items()}
    cid_to_color = assign_colors([class_to_cid[c] for c in classes])
    n = len(classes)

    fig, axes = panel_grid(2 * n, cols)
    for ax, cls in zip(axes[:n], classes):
        class_proposals_panel(
            ax, obs_rgb,
            sample.proposals[cls].numpy(), sample.labels[cls].numpy().astype(bool),
            md.class_to_ref[cls], cls, cid_to_color[class_to_cid[cls]], max_points,
        )
    for ax, cls in zip(axes[n:], classes):
        cid = class_to_cid[cls]
        class_mask_panel(ax, obs_rgb, cidm, cid, cls, cid_to_color[cid], alpha)

    if title:
        fig.suptitle(title, fontsize=10)
    return fig


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Per-class proposals/gt-masks viz for ImageInlierSample.")
    p.add_argument("render_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--frames", type=str, default=None, help="comma-separated frame indices")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.45)
    args = p.parse_args()

    render_dir = args.render_dir
    if not (render_dir / "labels").exists():
        print(f"no labels/ in {render_dir} — run isaac-datagen-inliers first", file=sys.stderr)
        sys.exit(1)

    md = ObsMaskMetadata.deserialize(0, render_dir)
    n_frames = count_samples(render_dir)
    frames = select_frames(n_frames, args.frames, args.stride, args.max_frames)

    out_dir = args.out or render_dir.parent / (render_dir.name + "_viz_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"visualizing {len(frames)} frame(s) from {render_dir} → {out_dir}")

    for idx in frames:
        if idx >= n_frames:
            print(f"  frame {idx}: out of range (n_frames={n_frames}) — skipping")
            continue
        sample = ImageInlierSample.deserialize(idx, render_dir)
        fig = sample_figure(sample, md, cols=args.cols, max_points=args.max_points,
                            alpha=args.alpha, title=f"{render_dir.name}  frame {idx:04d}")
        if fig is None:
            print(f"  frame {idx:04d}: no labeled classes — skipping")
            continue
        out_path = out_dir / f"sample_{idx:04d}.png"
        save_figure(fig, out_path, args.dpi)
        print(f"  wrote {out_path}")

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()


import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from vision_core.datastructs import PreImageInlierSample, ObsMaskDescriptorMetadata, count_samples
from vision_core.viz import inlier_figure, save_figure


def select_frames(n_frames, explicit, stride, max_frames):
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip()]
    return list(range(0, n_frames, stride))[:max_frames]


def main():
    p = argparse.ArgumentParser(description="Visualize PreImageInlierSample inlier/outlier labels.")
    p.add_argument("render_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--frames", type=str, default=None, help="comma-separated frame indices")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--class", dest="classes", type=str, default=None,
                   help='comma-separated class name(s), e.g. "fish can" or "fish can,acorn"')
    args = p.parse_args()

    render_dir = args.render_dir
    if not (render_dir / "labels").exists():
        print(f"no labels/ in {render_dir} — run isaac-datagen-inliers first", file=sys.stderr)
        sys.exit(1)

    classes = ([c.strip() for c in args.classes.split(",") if c.strip()]
               if args.classes else None)

    md = ObsMaskDescriptorMetadata.deserialize(0, render_dir)
    if classes:
        known = set(md.cid_to_class.values())
        bad = [c for c in classes if c not in known]
        if bad:
            print(f"unknown class(es) in {render_dir}: {bad}\nknown: {sorted(known)}",
                  file=sys.stderr)
            sys.exit(1)
    n_frames = count_samples(render_dir)
    frames = select_frames(n_frames, args.frames, args.stride, args.max_frames)

    out_dir = args.out or render_dir.parent / (render_dir.name + "_viz_inliers")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"visualizing {len(frames)} frame(s) from {render_dir} → {out_dir}")

    for idx in frames:
        if idx >= n_frames:
            print(f"  frame {idx}: out of range (n_frames={n_frames}) — skipping")
            continue
        sample = PreImageInlierSample.deserialize(idx, render_dir)
        fig = inlier_figure(sample, md, cols=args.cols, max_points=args.max_points,
                            title=f"{render_dir.name}  frame {idx:04d}", classes=classes)
        if fig is None:
            if classes:
                print(f"  frame {idx:04d}: no proposals for class(es) {classes} — skipping")
            else:
                print(f"  frame {idx:04d}: no labeled classes — skipping")
            continue
        suffix = "" if not classes else "_" + "_".join(c.replace(" ", "_") for c in classes)
        out_path = out_dir / f"sample_{idx:04d}{suffix}.png"
        save_figure(fig, out_path, args.dpi)
        print(f"  wrote {out_path}")

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()

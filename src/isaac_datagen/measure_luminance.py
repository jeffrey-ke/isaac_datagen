"""Step-0 luminance catalog for the dark-box investigation (no Isaac dep).

Reads a rendered ObsMask render dir and reports, per frame, how dark the
*object* is. The boxes render against a transparent background, so a whole-frame
mean luma is dominated by empty pixels and would hide exactly the failure we are
chasing (boxes fading to black). We therefore mask to the foreground (alpha > 0)
and report:

  - fg_mean   : mean BT.709 luma over foreground pixels (overall object brightness)
  - dark_frac : fraction of foreground pixels below --pixel-threshold (how much of
                the box wall is near-black)

Frames whose dark_frac exceeds --frame-threshold are flagged "dark". With
--with-lighting, each dark frame is joined against <render_dir>/lighting_log.json
to print the dome intensity that produced it (the Step-3 correlation).

Usage:
    isaac-datagen-measure-luminance <render_dir> [--pixel-threshold 8]
        [--frame-threshold 0.5] [--metric dark_frac|fg_mean] [--csv OUT] [--with-lighting]

<render_dir> is a single render dir (holding obs/), e.g. .../perspective-refseg/render001.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vision_core.datastructs import ObsMask, count_samples

# BT.709 luma weights (Rec. 709), matching the renderer's RGB primaries.
_BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def frame_luminance(obs, pixel_threshold: float):
    """(fg_mean, dark_frac) for one (4,H,W) uint8 RGBA obs tensor.

    Foreground = alpha > 0. Returns (nan, nan) for an all-background frame so it
    is reported but never miscounted as "dark" (nan fails every comparison).
    """
    arr = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)
    rgb = arr[:3].astype(np.float64)                 # (3,H,W)
    fg = arr[3] > 0                                  # (H,W) foreground mask
    if not fg.any():
        return float("nan"), float("nan")
    luma = np.tensordot(_BT709, rgb, axes=([0], [0]))  # (H,W) BT.709 luma
    fg_luma = luma[fg]
    return float(fg_luma.mean()), float((fg_luma < pixel_threshold).mean())


def load_lighting(render_dir: Path):
    """Per-frame dome-intensity schedule from lighting_log.json, or None.

    Only the old Python-sampled sequence path wrote a per-frame list. The
    uniform-jitter path records just the distribution (a dict), which has no
    realized per-frame values to join against — treat it as absent.
    """
    p = render_dir / "lighting_log.json"
    if not p.exists():
        return None
    dome = json.loads(p.read_text()).get("lights", {}).get("DomeLight")
    return dome if isinstance(dome, list) else None


def main():
    ap = argparse.ArgumentParser(description="Catalog foreground luminance / dark frames in a render dir.")
    ap.add_argument("render_dir", type=Path, help="single render dir holding obs/")
    ap.add_argument("--pixel-threshold", type=float, default=8.0,
                    help="8-bit luma below which a foreground pixel is 'near-black'")
    ap.add_argument("--frame-threshold", type=float, default=0.5,
                    help="flag a frame 'dark' when dark_frac exceeds this")
    ap.add_argument("--metric", choices=("dark_frac", "fg_mean"), default="dark_frac",
                    help="value sorted/printed as the headline (default dark_frac)")
    ap.add_argument("--csv", type=Path, default=None, help="write per-frame rows to this CSV")
    ap.add_argument("--with-lighting", action="store_true",
                    help="join dark frames against lighting_log.json (dome intensity)")
    args = ap.parse_args()

    if not (args.render_dir / "obs").is_dir():
        print(f"no obs/ under {args.render_dir}", file=sys.stderr)
        sys.exit(1)

    n = count_samples(args.render_dir)
    lighting = load_lighting(args.render_dir) if args.with_lighting else None

    rows = []          # (idx, fg_mean, dark_frac)
    for i in range(n):
        obs = ObsMask.deserialize_field(i, args.render_dir, "obs")  # (4,H,W) uint8 RGBA
        fg_mean, dark_frac = frame_luminance(obs, args.pixel_threshold)
        rows.append((i, fg_mean, dark_frac))

    dark = [r for r in rows if r[2] == r[2] and r[2] > args.frame_threshold]  # r[2]==r[2] drops nan
    dark.sort(key=lambda r: r[2], reverse=True)

    print(f"render dir : {args.render_dir}")
    print(f"frames     : {n}")
    print(f"thresholds : pixel<{args.pixel_threshold}  frame dark_frac>{args.frame_threshold}")
    print(f"dark frames: {len(dark)}  ({len(dark) / n:.1%})")
    if dark:
        print(f"dark idx   : {[r[0] for r in dark]}")
        print("\n  idx  fg_mean  dark_frac" + ("   dome_intensity" if lighting else ""))
        for idx, fg_mean, dark_frac in dark:
            line = f"  {idx:04d}  {fg_mean:7.2f}  {dark_frac:8.3f}"
            if lighting is not None and idx < len(lighting):
                line += f"   {lighting[idx]['intensity']:.1f}"
            print(line)
    if args.with_lighting and lighting is None:
        print("(--with-lighting: no lighting_log.json found)", file=sys.stderr)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "fg_mean", "dark_frac", "dark"])
            for idx, fg_mean, dark_frac in rows:
                is_dark = dark_frac == dark_frac and dark_frac > args.frame_threshold
                w.writerow([idx, f"{fg_mean:.4f}", f"{dark_frac:.4f}", int(is_dark)])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

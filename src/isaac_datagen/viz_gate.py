"""Render the proposer's per-frame gate decision for every frame in a dataset (no matcher, pure disk).

The gate (``proposal_gate.gate_classes_reproj``) admits a class iff ANY of its member instances has more
than ``--min-visible-ratio`` of its reference texture visible in the observation — the class's reference
RGB-D reprojected through that instance's ``class_to_l2w`` placement, fraction not occluded/off-frame.
This tool makes that visible: per frame it outlines every present instance over the observation — GREEN
if that instance clears the ratio, RED if not — and a class is proposed iff ≥1 of its instances is green.
Each instance is labelled with its visibility ratio; the title lists the resulting gated-class set. Use
it to eyeball the cut before regenerating proposals.

Needs an OptFlow dataset (per-frame ``observation_depth``/``cam2world`` + per-instance ``class_to_l2w``).
One PNG per frame → ``<out-root>/<dataset>/<render>/f####.png`` (default out-root is a gitignored
``gate_viz/`` at the meta-repo root). Run from a sibling env with torch + vision_core (vision_core's own
env has no torchvision):

    isaac-datagen-viz-gate <dataset_root|render_dir> [...] [--min-visible-ratio 0.30] [--out-root DIR] [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

from vision_core.datastructs import OptFlowSample, OptFlowMetadata, count_samples
from vision_core.viz import rgba_chw_to_rgb

from isaac_datagen.proposal_gate import instance_visibility

DEFAULT_OUT_ROOT = Path("/home/jeffk/repo/refseg-workspace/gate_viz")


def _render_frame(sample, md, vis: dict[int, float], min_visible_ratio: float,
                  title: str, out_path: Path) -> None:
    """One frame: instance outlines colored by the gate (green=passes ratio, red=dropped).

    ``vis`` maps iid -> reference-texture visibility ratio (``proposal_gate.instance_visibility``);
    an iid absent from ``vis`` has no reference/placement to score and is drawn grey as ``n/a``."""
    om = sample.obsmask
    iid_mask = om.iid_mask.numpy()
    mm = md.obsmaskmeta
    name_to_class = dict(mm.name_to_class)
    iid_to_name = {int(k): v for k, v in mm.iid_to_name.items()}
    present = sorted({int(i) for i in np.unique(iid_mask)} & set(iid_to_name))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgba_chw_to_rgb(om.obs)); ax.axis("off")
    for iid in present:
        region = iid_mask == iid
        ratio = vis.get(iid)
        passes = ratio is not None and ratio > min_visible_ratio
        color = "lime" if passes else "red"
        ax.contour(region, levels=[0.5], colors=[color], linewidths=2)
        ys, xs = np.where(region)
        cls = name_to_class.get(iid_to_name.get(iid), "?")
        label = f"{cls}\n{ratio:.0%}" if ratio is not None else f"{cls}\nn/a"
        ax.text(xs.mean(), ys.mean(), label, color="white", fontsize=8,
                ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor=("darkgreen" if passes else "darkred"), alpha=0.85, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=9, loc="left")
    fig.savefig(out_path, dpi=95, bbox_inches="tight"); plt.close(fig)


def viz_render_dir(render_dir: Path, out_root: Path, min_visible_ratio: float, limit: int | None = None) -> int:
    """Write one gate-decision PNG per frame into <out_root>/<dataset>/<render>/. Returns count."""
    render_dir = Path(render_dir)
    dataset = render_dir.parent.name
    out_dir = out_root / dataset / render_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    md = OptFlowMetadata.deserialize(0, render_dir)
    mm = md.obsmaskmeta
    iid_to_name = {int(k): v for k, v in mm.iid_to_name.items()}
    ref_cache: dict = {}   # class -> dense reference points, computed once across frames

    n = count_samples(render_dir)
    n = n if limit is None else min(limit, n)
    for idx in tqdm(range(n), desc=f"{dataset}/{render_dir.name}", unit="frame", leave=False):
        sample = OptFlowSample.deserialize(idx, render_dir)
        vis = instance_visibility(sample, md, ref_cache=ref_cache)   # iid -> visible ratio
        gated = sorted({mm.name_to_class[iid_to_name[iid]]
                        for iid, r in vis.items() if r > min_visible_ratio})  # for the title
        title = (f"{dataset}/{render_dir.name}  f{idx:04d}   gate: >{min_visible_ratio:.0%} ref-texture "
                 f"visible (green=passes, red=dropped)\ngated classes ({len(gated)}): {', '.join(gated) or '—'}")
        _render_frame(sample, md, vis, min_visible_ratio, title, out_dir / f"f{idx:04d}.png")
    return n


def _resolve(root: Path, out_root: Path, min_visible_ratio: float, limit: int | None) -> int:
    """A render dir (has obs/) → render it; else a dataset root → walk its render dirs."""
    if (root / "obs").is_dir():
        return viz_render_dir(root, out_root, min_visible_ratio, limit)
    from vision_core.migrate import for_each_render_dir
    return for_each_render_dir(root, lambda rd: viz_render_dir(rd, out_root, min_visible_ratio, limit))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="dataset roots and/or render dirs")
    ap.add_argument("--min-visible-ratio", type=float, default=0.30,
                    help="gate threshold: min fraction of reference texture visible (default 0.30)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"output root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--limit", type=int, default=None, help="cap frames per render dir (debug)")
    args = ap.parse_args()

    total = sum(_resolve(r, args.out_root, args.min_visible_ratio, args.limit) for r in args.roots)
    print(f"done: {total} gate-decision PNGs under {args.out_root}")


if __name__ == "__main__":
    main()

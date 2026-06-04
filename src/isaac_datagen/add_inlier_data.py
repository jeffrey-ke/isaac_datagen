"""Phase-3 pass: label each phase-2 proposal inlier/outlier for verifier training.

Runs AFTER ``add_proposals``. NN-free: for each frame it tags every proposer point
True iff it lands inside its class's union mask (``cid_mask == cid`` — ANY same-class
box counts, so a correct match onto a visually identical sibling box is not mislabeled
an outlier), then writes the result *residually* as
``ImageInlierSample.serialize(idx, dir, only={"labels"})`` — so ``obs/``,
``cid_mask/``, and ``proposals/`` are never rewritten. After the loop it writes a
per-render-dir ``ImageInlierMetadata`` with the aggregate inlier counts.

Usage: isaac-datagen-inliers <render_dir>
"""

import sys
from pathlib import Path

from vision_core.datastructs import (
    ObsMaskMetadata, PreReferenceSegSample, ImageInlierSample, ImageInlierMetadata,
)
from vision_core.mask_utils import coords_in_mask


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-inliers <render_dir>", file=sys.stderr)
        sys.exit(1)
    render_dir = Path(sys.argv[1])

    md = ObsMaskMetadata.deserialize(0, render_dir)
    class_to_cid = {cls: cid for cid, cls in md.cid_to_class.items()}  # 1:1 by construction

    n_frames = len(list((render_dir / "obs").iterdir()))
    n_inliers = n_total = 0
    for idx in range(n_frames):
        pre = PreReferenceSegSample.deserialize(idx, render_dir)
        labels = {
            cls: coords_in_mask(pre.cid_mask == class_to_cid[cls], coords)
            for cls, coords in pre.proposals.items()
        }
        ImageInlierSample(
            obs=pre.obs, cid_mask=pre.cid_mask, proposals=pre.proposals, labels=labels,
        ).serialize(idx, render_dir, only={"labels"})
        n_in = sum(int(v.sum()) for v in labels.values())
        n_tot = sum(int(v.numel()) for v in labels.values())
        n_inliers += n_in
        n_total += n_tot
        print(f"[{idx + 1}/{n_frames}] {render_dir.name}: {n_in}/{n_tot} inliers, {len(labels)} class(es)")

    # Per-render-dir stats catalog (written once, like ObsMaskMetadata).
    ImageInlierMetadata(stats={"n_inliers": n_inliers, "n_total": n_total}).serialize(0, render_dir)
    print(f"{render_dir.name}: {n_inliers}/{n_total} inliers total → stats/stats_0000.json")


if __name__ == "__main__":
    main()

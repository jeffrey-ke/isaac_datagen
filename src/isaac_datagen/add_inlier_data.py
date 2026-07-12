
import argparse
from pathlib import Path

from vision_core.datastructs import (
    ObsMaskDescriptorMetadata, PreReferenceSegSample, PreImageInlierSample, ImageInlierMetadata,
)
from vision_core.mask_utils import coords_in_mask


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("render_dir", type=Path)
    p.add_argument("--eps", type=float, required=True,
                   help="border margin (px) a proposal must keep inside its class union mask")
    args = p.parse_args()
    render_dir = args.render_dir

    md = ObsMaskDescriptorMetadata.deserialize(0, render_dir)
    class_to_cid = {cls: cid for cid, cls in md.cid_to_class.items()}

    n_frames = len(list((render_dir / "obs").iterdir()))
    n_inliers = n_total = 0
    for idx in range(n_frames):
        pre = PreReferenceSegSample.deserialize(idx, render_dir)
        labels = {
            cls: coords_in_mask(pre.cid_mask == class_to_cid[cls], coords, args.eps)
            for cls, coords in pre.proposals.items()
        }
        PreImageInlierSample(
            obs=pre.obs, cid_mask=pre.cid_mask, proposals=pre.proposals, labels=labels,
        ).serialize(idx, render_dir, only={"labels"})
        n_in = sum(int(v.sum()) for v in labels.values())
        n_tot = sum(int(v.numel()) for v in labels.values())
        n_inliers += n_in
        n_total += n_tot
        print(f"[{idx + 1}/{n_frames}] {render_dir.name}: {n_in}/{n_tot} inliers, {len(labels)} class(es)")

    ImageInlierMetadata(
        stats={"n_inliers": n_inliers, "n_total": n_total, "eps": args.eps},
    ).serialize(0, render_dir)
    print(f"{render_dir.name}: {n_inliers}/{n_total} inliers total (eps={args.eps:g}px) "
          f"→ stats/stats_0000.json")


if __name__ == "__main__":
    main()

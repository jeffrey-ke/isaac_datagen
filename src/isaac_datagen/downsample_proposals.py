
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from vision_core.datastructs import PreReferenceSegSample
from vision_core.sampling import fps_indices


def fps_downsample(xy: torch.Tensor, k: int) -> torch.Tensor:
    return xy[fps_indices(xy, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("--max-points", type=int, default=256, help="K: per-class proposal cap")
    parser.add_argument("--dry-run", action="store_true", help="count and summarize, write nothing")
    args = parser.parse_args()

    n_frames = len(list((args.render_dir / "obs").iterdir()))
    pts_before = pts_after = n_shrunk = n_empties = n_written = 0
    for idx in tqdm(range(n_frames), desc=args.render_dir.name, unit="frame"):
        pre = PreReferenceSegSample.deserialize(idx, args.render_dir)
        new_proposals = {}
        for cls, xy in pre.proposals.items():
            pts_before += xy.shape[0]
            if xy.shape[0] == 0:
                n_empties += 1
                continue
            kept = fps_downsample(xy, args.max_points)
            n_shrunk += kept.shape[0] < xy.shape[0]
            pts_after += kept.shape[0]
            new_proposals[cls] = kept
        changed = len(new_proposals) != len(pre.proposals) or any(
            new_proposals[c].shape != pre.proposals[c].shape for c in new_proposals
        )
        if changed and not args.dry_run:
            pre.proposals = new_proposals
            pre.serialize(idx, args.render_dir, only={"proposals"})
            n_written += 1

    print(
        f"{args.render_dir.name}: {pts_before} → {pts_after} points (K={args.max_points}), "
        f"{n_shrunk} class entries downsampled, {n_empties} empties dropped, "
        f"{n_written}/{n_frames} frames rewritten{' [dry run]' if args.dry_run else ''}"
    )


if __name__ == "__main__":
    main()

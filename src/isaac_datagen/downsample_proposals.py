"""Phase-2.5 pass: FPS-downsample each class's proposal points to at most K.

Runs AFTER ``add_proposals`` and BEFORE ``add_inlier_data`` (phase-3 labels each
proposal point, so downsampling later would desync labels from kept points).
NN-free and fast — pure I/O plus cheap CPU math, no sharding needed. For each
frame, each class's ``(N, 2)`` proposal tensor is reduced to ``K`` spatially
spread points via furthest point sampling; classes with ``N <= K`` are kept
as-is, empty entries are dropped. Writes residually via
``serialize(idx, dir, only={"proposals"})`` — ``obs/`` and ``cid_mask/`` are
never rewritten; writes are atomic, so the pass is safely re-runnable and
idempotent (a second run finds nothing to shrink and writes nothing).

FPS via the ``fpsample`` library (pinned <1: 1.x ships no wheel here). Always
``start_idx=0`` — the default start is random, and this pass rewrites the
dataset, so it must be reproducible.

Usage: isaac-datagen-downsample-proposals <render_dir> [--max-points 256] [--dry-run]
"""

import argparse
from pathlib import Path

import fpsample
import torch
from tqdm import tqdm

from vision_core.datastructs import PreReferenceSegSample


def fps_downsample(xy: torch.Tensor, k: int) -> torch.Tensor:
    """At most k spatially spread rows of an (N, 2) coord tensor, deterministically."""
    if xy.shape[0] <= k:
        return xy  # fpsample asserts on K > N, and identity needs no call
    idx = fpsample.fps_sampling(xy.numpy(), k, start_idx=0)
    return xy[torch.from_numpy(idx.astype("int64"))]


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
            if xy.shape[0] == 0:  # legacy frames predate the drop-empties producer fix
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

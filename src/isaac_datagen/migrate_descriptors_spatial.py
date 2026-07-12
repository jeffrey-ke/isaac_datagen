import argparse
import math
from pathlib import Path

import torch

from vision_core.datastructs import ObsMaskDescriptorMetadata


def migrate_tensor(t: torch.Tensor) -> torch.Tensor | None:
    if t.ndim == 3:
        return None
    assert t.ndim == 2, f"expected (N, C) or (C, h, w), got {tuple(t.shape)}"
    n, c = t.shape
    h = math.isqrt(n)
    assert h * h == n, f"non-square feature grid: N={n}"
    spatial = t.T.reshape(c, h, h).contiguous()
    assert torch.equal(spatial.flatten(1).T, t), "round-trip mismatch — refusing to overwrite"
    return spatial


def migrate_render_dir(rd: Path) -> int:
    n_migrated = 0
    for pt_path in sorted((rd / "class_to_descriptors").glob("class_to_descriptors_*.pt")):
        idx = int(pt_path.stem.rsplit("_", 1)[1])
        md = ObsMaskDescriptorMetadata.deserialize(idx, rd)
        file_migrated = 0
        for cls, t in md.class_to_descriptors.items():
            spatial = migrate_tensor(t)
            if spatial is None:
                print(f"    {cls:>12}: {tuple(t.shape)}  (already spatial, skip)")
            else:
                print(f"    {cls:>12}: {tuple(t.shape)} -> {tuple(spatial.shape)}")
                md.class_to_descriptors[cls] = spatial
                file_migrated += 1
        if file_migrated:
            md.serialize(idx, rd, only={"class_to_descriptors"})
        n_migrated += file_migrated
    return n_migrated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_root", type=Path)
    args = p.parse_args()
    render_dirs = sorted(
        d for d in args.dataset_root.glob("render*")
        if (d / "class_to_descriptors").is_dir() and not d.name.endswith("_viz_clusters")
    )
    if not render_dirs:
        raise SystemExit(f"no render*/class_to_descriptors under {args.dataset_root}")
    total = 0
    for rd in render_dirs:
        print(f"{rd}:")
        total += migrate_render_dir(rd)
    print(f"\nDone: {len(render_dirs)} render dir(s) processed, {total} tensor(s) migrated.")


if __name__ == "__main__":
    main()

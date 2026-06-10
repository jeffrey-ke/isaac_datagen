"""Copy every GraspableObject of a given `class` into a new dataset directory.

Deserializes each sample in `src`, keeps those whose meta["class"] matches, and
re-serializes them with fresh contiguous indices into `dst` (full serialize, so
usdz/png/npy/yaml are all copied — a self-contained subset dataset).

    uv run src/isaac_datagen/filter_class.py <src_dir> <dst_dir> <class_name>
"""

import argparse
from pathlib import Path

from isaac_datagen.objects import GraspableObject
from isaac_datagen.relabel_classes import collect_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="source dataset directory")
    parser.add_argument("dst", help="destination dataset directory")
    parser.add_argument("class_name", help="meta['class'] value to keep")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    out = 0
    for idx in collect_indices(src):
        sample = GraspableObject.deserialize(idx, src)
        if sample.meta["class"] != args.class_name:
            continue
        sample.serialize(out, dst)
        print(f"[{idx:04d}] {sample.meta['name']} -> {dst.name}/{out:04d}")
        out += 1
    print(f"Wrote {out} '{args.class_name}' samples to {dst}")

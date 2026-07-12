
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

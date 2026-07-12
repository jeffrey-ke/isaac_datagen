from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from isaac_datagen.objects import OptFlowObject


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--idx", type=int, default=None, help="single object; default = all")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    n = len(sorted((args.dataset / "meta").glob("meta_*.yaml")))
    idxs = [args.idx] if args.idx is not None else range(n)

    out_dir = args.out or args.dataset / "viz_objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(list(idxs))} object(s) → {out_dir}")

    for i in idxs:
        obj = OptFlowObject.deserialize(i, args.dataset)
        arr = obj.visualize(title=obj.meta["name"])
        path = out_dir / f"obj_{i:04d}_{obj.meta['name']}.png"
        Image.fromarray(arr).save(path)
        print(f"  [{i:04d}] {obj.meta['name']} → {path.name}  {arr.shape}")

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()

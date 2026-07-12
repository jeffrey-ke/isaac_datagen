
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from PIL import Image

from vision_core.datastructs import (
    ObsMaskDescriptorMetadata, PreImageInlierSample, PreReferenceSegSample,
)
from vision_core.mask_utils import coords_in_mask


def labels_at_eps(pre, class_to_cid, eps):
    return {
        cls: coords_in_mask(pre.cid_mask == class_to_cid[cls], coords, eps)
        for cls, coords in pre.proposals.items()
    }


def print_count_table(classes, eps_list, counts):
    header = "class".ljust(16) + "".join(f"eps={e:g}".rjust(10) for e in eps_list) + "   total"
    print(header)
    print("-" * len(header))
    for c in classes:
        row = c.ljust(16) + "".join(f"{counts[e][c][0]}".rjust(10) for e in eps_list)
        print(row + f"   /{counts[eps_list[0]][c][1]}")
    totals = ["TOTAL".ljust(16)]
    totals += [f"{sum(counts[e][c][0] for c in classes)}".rjust(10) for e in eps_list]
    print("".join(totals) + f"   /{sum(counts[eps_list[0]][c][1] for c in classes)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("render_dir", type=Path)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--eps", type=float, nargs="+", default=[0, 1, 2, 3, 5, 8])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--no-composite", dest="composite", action="store_false")
    args = p.parse_args()

    matplotlib.rcParams["figure.dpi"] = args.dpi

    md = ObsMaskDescriptorMetadata.deserialize(0, args.render_dir)
    class_to_cid = {cls: cid for cid, cls in md.cid_to_class.items()}
    pre = PreReferenceSegSample.deserialize(args.idx, args.render_dir)

    out_dir = args.out or args.render_dir.parent / f"{args.render_dir.name}_eps_sweep_{args.idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"sweeping eps={args.eps} on {args.render_dir.name} frame {args.idx:04d} → {out_dir}")

    classes = sorted(pre.proposals.keys())
    counts = {}
    panels = []
    for eps in args.eps:
        labels = labels_at_eps(pre, class_to_cid, eps)
        counts[eps] = {c: (int(labels[c].sum()), int(labels[c].numel())) for c in classes}
        arr = PreImageInlierSample(
            obs=pre.obs, cid_mask=pre.cid_mask, proposals=pre.proposals, labels=labels,
        ).visualize(md, cols=args.cols, max_points=args.max_points,
                    title=f"{args.render_dir.name} frame {args.idx:04d}  eps={eps:g}px")
        Image.fromarray(arr).save(out_dir / f"eps_{eps:g}.png")
        panels.append(arr)
        print(f"  eps={eps:g}: {sum(c[0] for c in counts[eps].values())}"
              f"/{sum(c[1] for c in counts[eps].values())} inliers → eps_{eps:g}.png")

    if args.composite and panels:
        w = max(a.shape[1] for a in panels)
        comp = np.concatenate(
            [np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)), constant_values=255)
             for a in panels], axis=0)
        Image.fromarray(comp).save(out_dir / "composite.png")
        print(f"  composite ({comp.shape[0]}x{comp.shape[1]}) → composite.png")

    print_count_table(classes, args.eps, counts)
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()


import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from isaac_datagen.objects import GraspableObject


def collect_indices(directory: Path) -> list[int]:
    return sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in (directory / "meta").glob("meta_*.yaml")
    )


def write_grid(samples, indices, out_path: Path, cols: int = 8):
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, sample, idx in zip(axes.flat, samples, indices):
        ax.imshow(sample.reference_image)
        ax.set_title(f"{idx}: {sample.meta['class']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def relabel(samples, indices, directory: Path):
    for idx, sample in zip(indices, samples):
        answer = input(
            f"[{idx:04d}] {sample.meta['name']} class={sample.meta['class']!r} -> "
        ).strip()
        if not answer:
            continue
        sample.meta["class"] = answer
        sample.serialize(idx, directory, only={"meta"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="dataset directory (e.g. object_dataset_amazon)")
    parser.add_argument("--grid-only", action="store_true",
                        help="write the reference-image grid png and exit")
    args = parser.parse_args()
    directory = Path(args.path)

    indices = collect_indices(directory)
    samples = [GraspableObject.deserialize(i, directory) for i in indices]
    grid_path = directory / "reference_grid.png"
    write_grid(samples, indices, grid_path)
    print(f"Wrote {grid_path} ({len(samples)} samples)")

    if not args.grid_only:
        plt.show(block=False)
        plt.pause(0.1)
        relabel(samples, indices, directory)

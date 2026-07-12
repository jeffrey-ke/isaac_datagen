
import argparse
from pathlib import Path

import torch

from vision_core.datastructs import ObsMaskDescriptorMetadata
from vision_core.viz import fit_pca_basis


def migrate_render_dir(rd: Path, dry_run: bool) -> int:
    n = 0
    for pt in sorted((rd / "class_to_descriptors").glob("class_to_descriptors_*.pt")):
        idx = int(pt.stem.rsplit("_", 1)[1])
        c2d = ObsMaskDescriptorMetadata.deserialize_field(idx, rd, "class_to_descriptors")
        tokens = torch.cat([d.flatten(1).T for d in c2d.values()], dim=0)
        basis = fit_pca_basis(tokens, n=3)
        shapes = {k: tuple(v.shape) for k, v in basis.items()}
        print(f"  {rd.name} idx={idx}: {len(c2d)} classes, {tokens.shape[0]} tokens → {shapes}")
        if not dry_run:
            md = ObsMaskDescriptorMetadata.__new__(ObsMaskDescriptorMetadata)
            md.principal_components = basis
            md.serialize(idx, rd, only={"principal_components"})
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="fit + report, write nothing")
    args = parser.parse_args()

    render_dirs = sorted(
        d for d in args.dataset_root.glob("render*")
        if (d / "class_to_descriptors").is_dir()
    )
    if not render_dirs:
        raise SystemExit(f"no render*/class_to_descriptors under {args.dataset_root}")
    total = sum(migrate_render_dir(rd, args.dry_run) for rd in render_dirs)
    print(f"{len(render_dirs)} render dir(s), {total} catalog(s) backfilled"
          f"{' [dry run]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()

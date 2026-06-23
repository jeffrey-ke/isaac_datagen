"""Migrate a dataset's descriptor catalog to the per-backbone SubfolderDict layout.

`class_to_descriptors` and `principal_components` were single pickled blobs (one backbone per render
dir). They are now `SubfolderDict`s keyed by descriptor backbone (the registry name from the dir's
`descriptor.yaml`, e.g. `DiftDescriptor`), so several backbones coexist under
`<field>/<backbone>/`. This `relocate` pass brings existing dirs forward:

    class_to_descriptors/class_to_descriptors_0000.pt   (legacy {class: tensor} blob)
      ->  class_to_descriptors/DiftDescriptor/class_to_descriptors_0000.pt   (+ a key manifest)

Idempotent (a dir already in the manifest layout is skipped) and atomic (residual single-field
writes). Works uniformly on reference-seg AND optflow datasets — the optflow `OptFlowMetadata` nests
its `obsmaskmeta` FLAT, so `class_to_descriptors/` sits at the render-dir level in both.

    cd isaac_datagen && env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone \
        relocate /data/user/jeffk/datasets/expanded-refseg

Adding a NEW backbone to an existing dataset (e.g. CleanDIFT, no re-render — re-encode the stored
`class_to_ref` images with another descriptor config and drop a new `<backbone>/` subfolder) is a
separate step, not implemented here; see the descriptor-variant plan.
"""
import argparse
from pathlib import Path

import yaml

from vision_core.datastructs import ObsMaskDescriptorMetadata
from vision_core.migrate import for_each_render_dir, relocate_field_to_subfolder

# Both per-backbone fields of the catalog; PCA is backbone-specific (fit on that backbone's tokens).
_FIELDS = ("class_to_descriptors", "principal_components")


def _backbone_of(rd: Path) -> str:
    """The backbone key for this render dir: the `name` in its `descriptor.yaml`."""
    descriptor_yaml = rd / "descriptor.yaml"
    if not descriptor_yaml.exists():
        raise SystemExit(f"{rd}: no descriptor.yaml — cannot determine the backbone name")
    return yaml.safe_load(descriptor_yaml.read_text())["name"]


def _relocate_render_dir(rd: Path) -> int:
    backbone = _backbone_of(rd)
    return sum(
        relocate_field_to_subfolder(rd, ObsMaskDescriptorMetadata, field, backbone)
        for field in _FIELDS
        if (rd / field).is_dir()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relocate", help="legacy single-blob -> SubfolderDict layout, keyed by descriptor.yaml name")
    r.add_argument("dataset_root", type=Path)
    args = p.parse_args()

    if args.cmd == "relocate":
        for_each_render_dir(args.dataset_root, _relocate_render_dir)


if __name__ == "__main__":
    main()

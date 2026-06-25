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

The `add-backbone` subcommand adds a NEW backbone to an existing dataset with NO re-render: it re-encodes
the stored `class_to_ref` images with another descriptor config and drops a new `<backbone>/` subfolder
beside the existing one(s) in both `class_to_descriptors/` and `principal_components/`. This is the
CleanDIFT-on-disk step (e.g. our finetuned CleanDIFT) — minutes, no Isaac Sim. The descriptor forward is
the expensive part, so the descriptor is built once and reused across render dirs.

`class_to_descriptors[backbone][cls]` is a SINGLE `(C, h, w)` grid (consumers do `.flatten(1).T` /
`.shape[0]`), so the config MUST be a single-scale descriptor (e.g. `cleandift_finetuned.yaml` ->
CleanDiftFinetunedDescriptor). An FPN config returns a list of volumes and would fail the `.squeeze(0)`
below — FPN backbones are the verifier's runtime observation `provider`, not stored ref tokens.

    cd isaac_datagen && env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone \
        add-backbone /data/user/jeffk/datasets/expanded-refseg \
        ../reference_matching/src/reference_matching/configs/cleandift_finetuned.yaml --device cuda
"""
import argparse
from pathlib import Path

import torch
import yaml

from vision_core.datastructs import ObsMaskDescriptorMetadata
from vision_core.migrate import (
    add_backbone_to_subfolder,
    for_each_render_dir,
    relocate_field_to_subfolder,
)
from vision_core.viz import fit_pca_basis

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


def _has_backbone(rd: Path, field: str, backbone: str, idx: int = 0) -> bool:
    """True if `backbone` is already a key in this render dir's `<field>/` manifest (a list).
    A legacy single-blob field (a dict) counts as absent — the add path raises a relocate hint."""
    manifest = torch.load(rd / field / f"{field}_{idx:04d}.pt", weights_only=False)
    return isinstance(manifest, list) and backbone in manifest


def _add_backbone_render_dir(rd: Path, descriptor, backbone: str, device: str) -> int:
    """Re-encode the stored `class_to_ref` with `descriptor` and write its `backbone` subfolder into both
    catalog fields. Skips the (expensive) forward when both fields already carry `backbone`."""
    if all(_has_backbone(rd, f, backbone) for f in _FIELDS):
        return 0
    # Read only the reference images — not the whole catalog (which would eagerly load every backbone).
    class_to_ref = ObsMaskDescriptorMetadata.deserialize_field(0, rd, "class_to_ref")
    with torch.inference_mode():                          # (C, h, w) per class, mirrors reference_catalog
        class_to_descriptors = {
            cls: descriptor(ref.unsqueeze(0).to(device)).squeeze(0).cpu()
            for cls, ref in class_to_ref.items()
        }
    # PCA→RGB basis is per-backbone, fit over ALL classes' tokens (same flatten(1).T tokenization).
    tokens = torch.cat([d.flatten(1).T for d in class_to_descriptors.values()], dim=0)
    pca = fit_pca_basis(tokens, n=3)
    return (add_backbone_to_subfolder(rd, "class_to_descriptors", backbone, class_to_descriptors)
            + add_backbone_to_subfolder(rd, "principal_components", backbone, pca))


def _add_backbone(dataset_root: Path, descriptor_config: Path, device: str) -> None:
    backbone = yaml.safe_load(descriptor_config.read_text())["name"]   # SubfolderDict key == registry name
    from reference_matching import descriptor as descriptor_module
    descriptor = descriptor_module.from_config(str(descriptor_config)).to(device)   # built ONCE, reused
    for_each_render_dir(dataset_root,
                        lambda rd: _add_backbone_render_dir(rd, descriptor, backbone, device))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relocate", help="legacy single-blob -> SubfolderDict layout, keyed by descriptor.yaml name")
    r.add_argument("dataset_root", type=Path)
    a = sub.add_parser("add-backbone", help="re-encode stored class_to_ref with another descriptor; add a new <backbone>/ (no re-render)")
    a.add_argument("dataset_root", type=Path)
    a.add_argument("descriptor_config", type=Path, help="reference_matching descriptor config yaml (its `name` is the backbone key)")
    a.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.cmd == "relocate":
        for_each_render_dir(args.dataset_root, _relocate_render_dir)
    elif args.cmd == "add-backbone":
        _add_backbone(args.dataset_root, args.descriptor_config, args.device)


if __name__ == "__main__":
    main()

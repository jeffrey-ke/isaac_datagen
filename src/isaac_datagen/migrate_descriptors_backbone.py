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

_FIELDS = ("class_to_descriptors", "principal_components")


def _backbone_of(rd: Path) -> str:
    descriptor_yaml = rd / "descriptor.yaml"
    if not descriptor_yaml.exists():
        raise SystemExit(f"{rd}: no descriptor.yaml — cannot determine the backbone name")
    return yaml.safe_load(descriptor_yaml.read_text())["name"]


def _write_provenance(rd: Path, backbone: str, record: dict) -> None:
    keydir = rd / "class_to_descriptors" / str(backbone)
    keydir.mkdir(parents=True, exist_ok=True)
    (keydir / "descriptor.yaml").write_text(yaml.safe_dump(record, sort_keys=True))


def _relocate_render_dir(rd: Path) -> int:
    backbone = _backbone_of(rd)
    n = sum(
        relocate_field_to_subfolder(rd, ObsMaskDescriptorMetadata, field, backbone)
        for field in _FIELDS
        if (rd / field).is_dir()
    )
    _write_provenance(rd, backbone, yaml.safe_load((rd / "descriptor.yaml").read_text()))
    return n


def _has_backbone(rd: Path, field: str, backbone: str, idx: int = 0) -> bool:
    manifest = torch.load(rd / field / f"{field}_{idx:04d}.pt", weights_only=False)
    return isinstance(manifest, list) and backbone in manifest


def _pca_basis(class_to_descriptors: dict):
    sample = next(iter(class_to_descriptors.values()))
    if torch.is_tensor(sample):
        tokens = torch.cat([d.flatten(1).T for d in class_to_descriptors.values()], dim=0)
        return fit_pca_basis(tokens, n=3)
    return {k: fit_pca_basis(
                torch.cat([leaf[k].flatten(1).T for leaf in class_to_descriptors.values()], dim=0), n=3)
            for k in sample}


def _add_backbone_render_dir(rd: Path, descriptor, backbone: str, device: str,
                             record: dict, overwrite: bool = False) -> int:
    if not overwrite and all(_has_backbone(rd, f, backbone) for f in _FIELDS):
        return 0
    class_to_ref = ObsMaskDescriptorMetadata.deserialize_field(0, rd, "class_to_ref")
    with torch.inference_mode():
        class_to_descriptors = {
            cls: descriptor.to_leaf(descriptor(descriptor.prep(ref).unsqueeze(0).to(device)))
            for cls, ref in class_to_ref.items()
        }
    pca = _pca_basis(class_to_descriptors)
    n = (add_backbone_to_subfolder(rd, "class_to_descriptors", backbone, class_to_descriptors, overwrite=overwrite)
         + add_backbone_to_subfolder(rd, "principal_components", backbone, pca, overwrite=overwrite))
    _write_provenance(rd, backbone, record)
    return n


def _add_backbone(dataset_root: Path, descriptor_config: Path, device: str, overwrite: bool = False) -> None:
    record = yaml.safe_load(descriptor_config.read_text())
    backbone = record["name"]
    from reference_matching import descriptor as descriptor_module
    descriptor = descriptor_module.from_config(str(descriptor_config)).to(device)
    for_each_render_dir(dataset_root,
                        lambda rd: _add_backbone_render_dir(rd, descriptor, backbone, device, record, overwrite),
                        root_fallback=True)


def _record(dataset_root: Path, descriptor_config: Path) -> None:
    record = yaml.safe_load(descriptor_config.read_text())
    backbone = record["name"]

    def one(rd: Path) -> int:
        if not _has_backbone(rd, "class_to_descriptors", backbone):
            return 0
        _write_provenance(rd, backbone, record)
        return 1

    for_each_render_dir(dataset_root, one, root_fallback=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relocate", help="legacy single-blob -> SubfolderDict layout, keyed by descriptor.yaml name")
    r.add_argument("dataset_root", type=Path)
    a = sub.add_parser("add-backbone", help="re-encode stored class_to_ref with another descriptor; add a new <backbone>/ (no re-render)")
    a.add_argument("dataset_root", type=Path)
    a.add_argument("descriptor_config", type=Path, help="reference_matching descriptor config yaml (its `name` is the backbone key)")
    a.add_argument("--device", default="cuda")
    a.add_argument("--overwrite", action="store_true",
                   help="re-encode and REPLACE an existing backbone's value files in place (e.g. re-baking "
                        "CleanDiftFinetunedFpn with different keys 0/1/2 -> 1/2/3); default skips a dir that "
                        "already carries the backbone")
    rec = sub.add_parser("record", help="backfill the per-backbone provenance record (class_to_descriptors/"
                         "<backbone>/descriptor.yaml) into dirs already carrying that backbone; no re-encode")
    rec.add_argument("dataset_root", type=Path)
    rec.add_argument("descriptor_config", type=Path,
                     help="the config the existing bake is KNOWN to have used (names collide across configs — "
                          "e.g. fpn_cleandift_finetuned{,_123}.yaml — so the operator supplies the mapping)")
    args = p.parse_args()

    if args.cmd == "relocate":
        for_each_render_dir(args.dataset_root, _relocate_render_dir)
    elif args.cmd == "add-backbone":
        _add_backbone(args.dataset_root, args.descriptor_config, args.device, args.overwrite)
    elif args.cmd == "record":
        _record(args.dataset_root, args.descriptor_config)


if __name__ == "__main__":
    main()

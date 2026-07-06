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

Every backbone subfolder also carries a provenance record — `class_to_descriptors/<backbone>/descriptor.yaml`,
the full `{name, args}` of the config that baked it. `add-backbone` writes it whenever it actually encodes
(refreshed on `--overwrite`; never on the skip path, so stale bytes can't be mislabeled); `relocate` seeds the
original backbone's from the top-level marker; the `record` subcommand backfills dirs baked before this existed.
`segmentation.utils.descriptor_provenance` reads it at train time to freeze the bake into each run dir.

`class_to_descriptors[backbone][cls]` is EITHER a single `(C, h, w)` grid (single-scale descriptors —
`DiftDescriptor`/CleanDIFT) OR, for a keyed multi-scale FPN, a `{scale_key: (C_k, h_k, w_k)}` dict — the
shape each descriptor declares via its `to_leaf`. The multi-scale FPN leaf feeds the GLIGEN M2F segmenter's
`MultiScaleRefEncoder` (per-scale round-robin reference conditioning); `torch.save` persists the dict
natively, so no serializer change. Both forms coexist under `class_to_descriptors/<backbone>/`.

    # single-scale (CleanDIFT):
    cd isaac_datagen && env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone \
        add-backbone /data/user/jeffk/datasets/expanded-refseg \
        ../reference_matching/src/reference_matching/configs/cleandift_finetuned.yaml --device cuda
    # multi-scale (DiftFpn keys 0/1/2 -> the DiftFpn backbone the M2F encoder reads):
    cd isaac_datagen && env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone \
        add-backbone /data/user/jeffk/datasets/expanded-refseg \
        ../reference_matching/src/reference_matching/configs/fpn_dift.yaml --device cuda
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


def _write_provenance(rd: Path, backbone: str, record: dict) -> None:
    """Snapshot the {name, args} this backbone's features were baked with, BESIDE them:
    `class_to_descriptors/<backbone>/descriptor.yaml`. segmentation.utils.descriptor_provenance
    reads exactly this path; SubfolderDict readers never open it (manifest-listed `.pt` only)."""
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
    # Seed the original backbone's per-backbone record from the top-level marker — the relocate
    # keying makes them the same bake by definition. Idempotent (the marker is immutable).
    _write_provenance(rd, backbone, yaml.safe_load((rd / "descriptor.yaml").read_text()))
    return n


def _has_backbone(rd: Path, field: str, backbone: str, idx: int = 0) -> bool:
    """True if `backbone` is already a key in this render dir's `<field>/` manifest (a list).
    A legacy single-blob field (a dict) counts as absent — the add path raises a relocate hint."""
    manifest = torch.load(rd / field / f"{field}_{idx:04d}.pt", weights_only=False)
    return isinstance(manifest, list) and backbone in manifest


def _pca_basis(class_to_descriptors: dict):
    """Per-backbone PCA→RGB viz basis, fit over ALL classes' tokens. Single-scale tensor leaves → one
    basis (unchanged). Keyed multi-scale leaves (`{scale: (C_k,h,w)}`) → one basis PER scale, since
    per-scale channels differ and can't share a basis; stored as `{scale: basis}`."""
    sample = next(iter(class_to_descriptors.values()))
    if torch.is_tensor(sample):
        tokens = torch.cat([d.flatten(1).T for d in class_to_descriptors.values()], dim=0)
        return fit_pca_basis(tokens, n=3)
    return {k: fit_pca_basis(
                torch.cat([leaf[k].flatten(1).T for leaf in class_to_descriptors.values()], dim=0), n=3)
            for k in sample}


def _add_backbone_render_dir(rd: Path, descriptor, backbone: str, device: str,
                             record: dict, overwrite: bool = False) -> int:
    """Re-encode the stored `class_to_ref` with `descriptor` and write its `backbone` subfolder into both
    catalog fields. Skips the (expensive) forward when both fields already carry `backbone` — UNLESS
    `overwrite`, which re-encodes and REPLACES the existing value files in place (the manifest is left as-is
    since the key already exists). Use `overwrite` to re-bake the same backbone name with different keys
    (e.g. CleanDiftFinetunedFpn {0,1,2} -> {1,2,3}); the value file holds the whole `{scale: tensor}` leaf,
    so the replace is total (no stale scale lingers). The stored leaf shape (single `(C,h,w)` grid vs keyed
    `{scale: (C_k,h,w)}` dict) is whatever the descriptor's `to_leaf` declares — no shape-sniffing here."""
    if not overwrite and all(_has_backbone(rd, f, backbone) for f in _FIELDS):
        return 0
    # Read only the reference images — not the whole catalog (which would eagerly load every backbone).
    class_to_ref = ObsMaskDescriptorMetadata.deserialize_field(0, rd, "class_to_ref")
    # Every descriptor owns two contract methods: `prep` (public preprocessing) and `to_leaf` (forward
    # output -> stored catalog leaf). No prep/shape sniffing here — single-scale and keyed FPN are uniform.
    with torch.inference_mode():
        class_to_descriptors = {
            cls: descriptor.to_leaf(descriptor(descriptor.prep(ref).unsqueeze(0).to(device)))
            for cls, ref in class_to_ref.items()
        }
    pca = _pca_basis(class_to_descriptors)
    n = (add_backbone_to_subfolder(rd, "class_to_descriptors", backbone, class_to_descriptors, overwrite=overwrite)
         + add_backbone_to_subfolder(rd, "principal_components", backbone, pca, overwrite=overwrite))
    # Provenance is written ONLY when the bake actually ran (the skip path above writes nothing —
    # stamping a new config's record onto stale bytes is the mislabel hazard); refreshed on --overwrite.
    _write_provenance(rd, backbone, record)
    return n


def _add_backbone(dataset_root: Path, descriptor_config: Path, device: str, overwrite: bool = False) -> None:
    record = yaml.safe_load(descriptor_config.read_text())
    backbone = record["name"]                                          # SubfolderDict key == registry name
    from reference_matching import descriptor as descriptor_module
    descriptor = descriptor_module.from_config(str(descriptor_config)).to(device)   # built ONCE, reused
    # root_fallback: a flat dataset (marker at the root, no render*/ wrapper — e.g. the real-world
    # testset re-emitted as a single render dir) is baked as one dir, matching how the readers consume it.
    for_each_render_dir(dataset_root,
                        lambda rd: _add_backbone_render_dir(rd, descriptor, backbone, device, record, overwrite),
                        root_fallback=True)


def _record(dataset_root: Path, descriptor_config: Path) -> None:
    """Backfill: stamp `descriptor_config`'s {name, args} as the per-backbone provenance record into
    every render dir that ALREADY carries that backbone in its class_to_descriptors manifest (dirs
    without it are skipped). Data-only — one manifest read + one yaml write per dir, no re-encode;
    trusts the caller's backbone→config mapping exactly as add-backbone does at bake time."""
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

import re
from pathlib import Path

import yaml

_REF_RE = re.compile(r"^reference_image_(\d{4})\.\w+$")


def read_asset_list(txt: str | Path) -> list[str]:
    paths = Path(txt).read_text().split()
    assert paths, f"{txt}: empty asset list"
    return paths


def parse_asset(path: str | Path) -> tuple[Path, int]:
    p = Path(path)
    m = _REF_RE.match(p.name)
    assert p.parent.name == "reference_image" and m, \
        f"{path}: not a <catalog>/reference_image/reference_image_NNNN.* path"
    assert p.is_file(), f"{path}: does not exist"
    return p.parent.parent, int(m.group(1))


def _asset_meta(catalog: Path, idx: int) -> dict:
    meta = catalog / "meta" / f"meta_{idx:04d}.yaml"
    assert meta.is_file(), f"{meta}: reference image has no meta sibling"
    return yaml.safe_load(meta.read_text())


def asset_classes(paths: list[str]) -> list[str]:
    classes = [_asset_meta(*parse_asset(p))["class"] for p in paths]
    dupes = sorted({c for c in classes if classes.count(c) > 1})
    assert not dupes, f"duplicate classes in asset list: {dupes}"
    return classes


def assemble_catalog(paths: list[str], dest: str | Path) -> list[str]:
    from isaac_datagen.objects import OptFlowObject   # heavy import stays out of test path

    classes = asset_classes(paths)
    dest = Path(dest)
    assert not dest.exists(), f"{dest} exists — refusing to overwrite a catalog"
    dest.mkdir(parents=True)
    for out_idx, p in enumerate(paths):
        catalog, src_idx = parse_asset(p)
        OptFlowObject.deserialize(src_idx, catalog).serialize(out_idx, dest)
    return classes


def catalog_meta(path: Path) -> list[dict]:
    metas = sorted((Path(path) / "meta").glob("meta_*.yaml"))
    assert metas, f"{path}: not an OptFlowObject catalog (no meta/meta_*.yaml)"
    return [yaml.safe_load(m.read_text()) for m in metas]


def catalog_classes(path: Path) -> list[str]:
    return sorted({m["class"] for m in catalog_meta(path)})

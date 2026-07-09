"""Arm B of the store-mutations plan: scrub the vendor's baked ``semantic:*``
attributes out of a Stage-A catalog usdz.

Stage A exported ``store001-optflow-objects`` BEFORE any of our labeling ran, so
every usdz still carries the store vendor's coarse ``semantic:*`` attributes
(e.g. ``snack`` on a cereal box) baked into its flattened layer. In the LIVE
store scene those attributes compose through a *reference* arc, so they cannot be
deleted — only out-shouted with a stronger root-layer opinion (see
``label_product``'s override ordering). But inside the flattened catalog usdz the
attributes are plain, locally-authored data with NO reference arc, so there
``RemoveProperty`` really removes them. This scrubs them at the source, making a
swapped-in store object as clean as any amazon/ycb/kleenex catalog object.

The usdz is an uncompressed zip whose textures are referenced by relative path,
so unzipping keeps those paths valid; ``UsdUtils.CreateNewUsdzPackage``
re-bundles the layer + its discovered texture dependencies (the
``export_subtree_usdz`` precedent). Runs OUTSIDE Isaac — the catalog usdz are
self-contained after Stage A's texture localization, so standalone pxr suffices:

    uv run --with usd-core python debug_scripts/pull_flatten_usd.py \
        datasets/store001-optflow-objects
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SEMANTIC_PREFIX = "semantic:"


def pull_flatten_usd(usdz_path: str, out_path: str) -> int:
    """Repackage ``usdz_path`` into ``out_path`` with every ``semantic:*``
    attribute deleted from every prim; return the number removed.

    Unzip (usdz is an uncompressed zip; relative texture paths stay valid),
    open the inner root layer, ``RemoveProperty`` each ``semantic:*`` attribute
    (legal: locally authored, no reference arc — the exact edit the live store
    scene forbids), save, and repackage via ``UsdUtils.CreateNewUsdzPackage``.
    """
    from pxr import Usd, UsdUtils

    removed = 0
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(usdz_path) as zf:
            zf.extractall(td)
        inner = _root_layer(Path(td), usdz_path)
        stage = Usd.Stage.Open(str(inner))
        for prim in stage.Traverse():
            for name in [a.GetName() for a in prim.GetAttributes()]:
                if name.startswith(SEMANTIC_PREFIX):
                    assert prim.RemoveProperty(name), \
                        f"RemoveProperty failed on {prim.GetPath()}.{name} in {usdz_path}"
                    removed += 1
        stage.GetRootLayer().Save()
        assert UsdUtils.CreateNewUsdzPackage(str(inner), str(out_path)), \
            f"CreateNewUsdzPackage returned False for {out_path}"
    return removed


def _root_layer(extracted_dir: Path, usdz_path: str) -> Path:
    """The single USD root layer at the top of an unzipped catalog usdz.

    A Stage-A flattened package holds exactly one ``.usd*`` layer plus its
    (non-``.usd``) texture payloads. More than one root layer means the package
    is not the flattened shape this scrub assumes — fail loud rather than pick one
    (the plan's escalation rule)."""
    layers = sorted(extracted_dir.glob("*.usd*"))
    assert len(layers) == 1, \
        f"expected exactly one root layer in {usdz_path}, found {[p.name for p in layers]}"
    return layers[0]


def scrub_in_place(usdz: Path) -> int:
    """Back the pristine ``usdz`` up to ``<name>.orig.bak`` (skip-if-exists) and
    rewrite it with all ``semantic:*`` attributes removed; return removed count.

    Idempotent and safe to re-run: the backup is written once (never
    overwritten), and the scrub always reads FROM that pristine backup, so every
    run re-derives the same scrubbed file (rotate-graspable-meshes precedent:
    always edit from the backup)."""
    backup = Path(f"{usdz}.orig.bak")
    if not backup.exists():
        shutil.copy2(usdz, backup)
    return pull_flatten_usd(str(backup), str(usdz))


def scrub_catalog(catalog_dir: Path) -> None:
    """Scrub every ``usd_path/*.usdz`` of an OptFlowObject catalog in place,
    printing a per-file count of removed ``semantic:*`` attributes."""
    usd_dir = catalog_dir / "usd_path"
    assert usd_dir.is_dir(), f"not an OptFlowObject catalog (no usd_path/): {catalog_dir}"
    usdzs = sorted(usd_dir.glob("*.usdz"))       # *.orig.bak backups are skipped by the glob
    assert usdzs, f"no usdz files under {usd_dir}"
    print(f"{catalog_dir}: scrubbing {len(usdzs)} usdz")
    for usdz in usdzs:
        removed = scrub_in_place(usdz)
        print(f"  {usdz.name}: removed {removed} semantic:* attrs", flush=True)


def main() -> None:
    assert len(sys.argv) == 2, "usage: pull_flatten_usd.py <catalog_dir>"
    scrub_catalog(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

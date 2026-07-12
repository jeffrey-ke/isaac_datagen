from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SEMANTIC_PREFIX = "semantic:"


def pull_flatten_usd(usdz_path: str, out_path: str) -> int:
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
    layers = sorted(extracted_dir.glob("*.usd*"))
    assert len(layers) == 1, \
        f"expected exactly one root layer in {usdz_path}, found {[p.name for p in layers]}"
    return layers[0]


def scrub_in_place(usdz: Path) -> int:
    backup = Path(f"{usdz}.orig.bak")
    if not backup.exists():
        shutil.copy2(usdz, backup)
    return pull_flatten_usd(str(backup), str(usdz))


def scrub_catalog(catalog_dir: Path) -> None:
    usd_dir = catalog_dir / "usd_path"
    assert usd_dir.is_dir(), f"not an OptFlowObject catalog (no usd_path/): {catalog_dir}"
    usdzs = sorted(usd_dir.glob("*.usdz"))
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

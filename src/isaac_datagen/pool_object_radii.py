import os
import subprocess
from pathlib import Path

from isaac_datagen.asset_catalogs import catalog_meta

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # isaac_datagen/ (has pyproject.toml)

# Runs mesh_radius (Task 2, which prints ONLY a bare float) on the xargs-substituted path ($1),
# then emits "<path>\t<radius>" via a SINGLE printf call. This must be one call -- a naive
# `printf "%s\t" "$1"; uv run ...` (two separate writes) races under `xargs -P`: the fast printf
# calls from several parallel workers can land before any of the slower `uv run` subprocess calls
# finish, scrambling which path pairs with which radius (verified live: reproduced the scrambled
# interleaving with the two-write version, confirmed clean pairing with this one-write version).
# Command substitution `$(...)` blocks until mesh_radius fully finishes and captures its stdout
# (trailing newline stripped by the shell) before the single printf writes the whole line at once.
_MESH_RADIUS_LINE = (
    r'out="$(uv run --with usd-core python -m isaac_datagen.mesh_radius "$1")"; st=$?; '
    r'printf "%s\t%s\n" "$1" "$out"; exit $st'
)


def pool_usd_paths(ingest_catalog: Path) -> dict[str, Path]:
    """class -> usdz path, for every class in an assembled ingest catalog."""
    metas = catalog_meta(ingest_catalog)
    return {m["class"]: Path(ingest_catalog) / "usd_path" / f"usd_path_{i:04d}.usdz"
            for i, m in enumerate(metas)}


def _reassemble(stdout: str, class_to_path: dict) -> dict:
    radius_by_path = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        path, radius = line.rsplit("\t", 1)
        radius_by_path[path] = float(radius)
    missing = [cls for cls, p in class_to_path.items() if str(p) not in radius_by_path]
    assert not missing, f"mesh_radius produced no output for classes: {missing}"
    return {cls: radius_by_path[str(p)] for cls, p in class_to_path.items()}


def compute_pool_object_radii(ingest_catalog: Path, nproc: int | None = None) -> dict[str, float]:
    class_to_path = pool_usd_paths(ingest_catalog)
    stdin = "\n".join(str(p) for p in class_to_path.values())
    result = subprocess.run(
        ["xargs", "-P", str(nproc or os.cpu_count()), "-I{}",
         "sh", "-c", _MESH_RADIUS_LINE, "_", "{}"],
        input=stdin, capture_output=True, text=True, cwd=PROJECT_ROOT, check=True,
    )
    return _reassemble(result.stdout, class_to_path)

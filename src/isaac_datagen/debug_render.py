
import shutil
import subprocess
import sys
from pathlib import Path

from isaac_datagen.runtime_config import load_config

BLENDER = shutil.which("blender") or "/usr/local/bin/blender"
BLENDER_SCRIPT = Path(__file__).with_name("blender_render.py")


def _find(script: str) -> str:
    exe = shutil.which(script, path=str(Path(sys.executable).parent)) or shutil.which(script)
    if exe is None:
        sys.exit(f"console script not found: {script} (uv sync?)")
    return exe


def _run(cmd, banner: str) -> None:
    print(f"\n=== {banner} ===", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"{banner} failed (exit {e.returncode})")


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-debug-render <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    debug_dir = render_dir / "debug"

    _run([_find("isaac-datagen"), *sys.argv[1:], "dry_run=true"], "isaac-datagen dry_run=true")
    if not (debug_dir / "scene.usdz").exists():
        sys.exit(f"dry run did not produce {debug_dir / 'scene.usdz'}")

    if shutil.which("blender") is None and not Path(BLENDER).exists():
        sys.exit(f"blender not found (looked for 'blender' on PATH and {BLENDER})")
    _run([BLENDER, "--background", "--python", str(BLENDER_SCRIPT), "--", str(debug_dir)],
         "blender render")

    print(f"\ndebug bundle + renders in {debug_dir} (poses/*.png + orbit.gif)", flush=True)


if __name__ == "__main__":
    main()

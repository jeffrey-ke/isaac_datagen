"""Preflight: decodes every file under random_backgrounds/ via the loader's exact read_image path
(vision_core.transforms.BackgroundLibrary), flagging files that would crash a dataloader worker.
Dry-run by default; --quarantine moves bad files to <root>/_rejected/, --delete removes them.
Exits non-zero if any unresolved bad file remains. See plan
`read-how-our-current-whimsical-kurzweil.md` S5."""

import argparse
import shutil
from pathlib import Path

from torchvision.io import read_image, ImageReadMode

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "assets" / "random_backgrounds"
REJECTED_DIRNAME = "_rejected"


def iter_candidates(root: Path):
    rejected = root / REJECTED_DIRNAME
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if rejected in p.parents:
            continue
        yield p


def check(path: Path) -> str | None:
    """None if decodable, else a short error string."""
    try:
        read_image(str(path), mode=ImageReadMode.RGB)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def quarantine(path: Path, root: Path) -> Path:
    dst = root / REJECTED_DIRNAME / path.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--quarantine", action="store_true")
    action.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    scanned = ok = bad = resolved = 0
    for path in iter_candidates(args.root):
        scanned += 1
        error = check(path)
        if error is None:
            ok += 1
            continue
        bad += 1
        print(f"{path}: {error}")
        if args.quarantine:
            quarantine(path, args.root)
            resolved += 1
        elif args.delete:
            path.unlink()
            resolved += 1

    unresolved = bad - resolved
    print(f"scanned={scanned} ok={ok} bad={bad} resolved={resolved} unresolved={unresolved}")
    raise SystemExit(1 if unresolved > 0 else 0)


if __name__ == "__main__":
    main()

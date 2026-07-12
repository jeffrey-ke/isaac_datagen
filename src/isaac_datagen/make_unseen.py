import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from torchvision.transforms import v2

from vision_core.datastructs import OptFlowMetadata, OptFlowSample, SubfolderDict
from vision_core.transforms import SwapRedBlue
from isaac_datagen.runtime_config import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(*argv: str, cwd: Path | None = None) -> None:
    print(f"\n=== python -m {' '.join(argv)}{f'   (cwd={cwd})' if cwd else ''} ===", flush=True)
    subprocess.run([sys.executable, "-m", *argv], check=True, cwd=cwd)


def _flip_catalog(src: Path, dst: Path, swap) -> None:
    md = OptFlowMetadata.deserialize(0, src)
    mm = md.obsmaskmeta
    mm.class_to_ref = {cls: swap(ref) for cls, ref in mm.class_to_ref.items()}
    mm.class_to_descriptors = SubfolderDict()
    mm.principal_components = SubfolderDict()
    md.serialize(0, dst)


def _select_val_frames(manifest_path: str, split: str, src: Path, limit: int | None) -> list[int]:
    key = f"{src.parent.name}/{src.name}"
    man = json.loads(Path(manifest_path).read_text())
    frames = sorted({fr for k, fr, _cls in man[split] if k == key})
    if not frames:
        sys.exit(f"no '{split}' frames for key {key!r} in {manifest_path}")
    return frames[:limit] if limit else frames


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(prog="isaac-datagen-unseen", add_help=True)
    p.add_argument("config")
    p.add_argument("source_render_dir")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--split-manifest", dest="split_manifest")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int)
    args, overrides = p.parse_known_args(argv)
    return args, overrides


def main() -> None:
    args, overrides = _parse_args(sys.argv[1:])
    src = Path(args.source_render_dir)
    n_src = len(list((src / "obs").glob("obs_*.png")))

    if args.split_manifest:
        frames = _select_val_frames(args.split_manifest, args.split, src, args.limit)
    elif args.start is not None and args.end is not None:
        if args.start >= args.end:
            sys.exit(f"empty frame window [{args.start}, {args.end})")
        frames = list(range(args.start, args.end))
    else:
        sys.exit("provide --start/--end (window) or --split-manifest [--split val]")
    if max(frames) >= n_src:
        sys.exit(f"frame {max(frames)} exceeds source frame count {n_src} in {src / 'obs'}")

    runtime = load_config(args.config, overrides)
    dst = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    if dst.exists():
        sys.exit(f"dst {dst} exists — pick a fresh idx/dataset_dir (won't overwrite)")
    dst.mkdir(parents=True)

    swap = v2.Compose([SwapRedBlue()])
    for dst_idx, src_idx in enumerate(frames):
        s = OptFlowSample.deserialize(src_idx, src)
        s.obsmask.obs = swap(s.obsmask.obs)
        s.serialize(dst_idx, dst)
    print(f"wrote {len(frames)} flipped frames {frames} → 0..{len(frames) - 1} in {dst}", flush=True)

    _flip_catalog(src, dst, swap)
    for name in ("runtime.yaml", "descriptor.yaml", "lighting_log.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    _run("isaac_datagen.migrate_descriptors_backbone", "add-backbone",
         str(Path(runtime.dataset_dir).resolve()), str(Path(runtime.descriptor_config_path).resolve()),
         "--device", runtime.descriptor_device, cwd=REPO_ROOT)

    _run("isaac_datagen.add_proposals", args.config, *overrides)
    _run("isaac_datagen.add_inlier_data", str(dst), "--eps", str(runtime.inlier_border_eps))
    print(f"\nunseen render dir ready: {dst}", flush=True)


if __name__ == "__main__":
    main()

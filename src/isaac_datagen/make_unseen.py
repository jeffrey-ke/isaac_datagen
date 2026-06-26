"""Build a channel-swapped 'unseen 0-shot' eval render dir from a SUBSET of an existing phase-1 dir.

Selects frames [start, end) of a source render dir, swaps R/B (tv2 SwapRedBlue) on each obs frame AND
every class reference (re-encoding the catalog's clean-DIFT descriptors + PCA from the flipped refs so
both sides live in the swapped domain), then re-runs phases 2+3 (grid proposals + inlier labels). Output
is a normal labeled render dir with frames RENUMBERED contiguously from 0 — keep it OUT of the training
``data.paths`` / split manifest; it is consumed only as the verifier's fixed ``viz/unseen`` eval batch
(``segmentation.viz_callbacks.select_unseen_batches``).

This is the easiest of a family of 'unseen' transforms; later variants are sibling tv2 transforms.

Usage:
    isaac-datagen-unseen <config.yaml> <source_render_dir> <start> <end> [key=value ...]

``config`` supplies the OUTPUT location (``dataset_dir`` + ``idx`` → dst = ``dataset_dir/render{idx:03d}``),
the proposer + descriptor configs, and devices — the SAME config you train the verifier against, so the
unseen dir matches (grid proposer + the verifier's descriptor backbone, e.g. CleanDiftFinetunedDescriptor).
Every sub-step runs through a sibling tool's PUBLIC ``python -m`` CLI — no private cross-module imports.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from torchvision.transforms import v2

from vision_core.datastructs import OptFlowMetadata, OptFlowSample, SubfolderDict
from vision_core.transforms import SwapRedBlue
from isaac_datagen.runtime_config import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]   # .../isaac_datagen (one level under the workspace)


def _run(*argv: str, cwd: Path | None = None) -> None:
    """Run a sibling tool's PUBLIC module CLI in this interpreter/venv (robust, no PATH lookup)."""
    print(f"\n=== python -m {' '.join(argv)}{f'   (cwd={cwd})' if cwd else ''} ===", flush=True)
    subprocess.run([sys.executable, "-m", *argv], check=True, cwd=cwd)


def _flip_catalog(src: Path, dst: Path, swap) -> None:
    """Copy the per-render catalog verbatim (id dicts / intrinsics / optflow refs are flip-invariant)
    but FLIP ``obsmaskmeta.class_to_ref`` and EMPTY the two descriptor SubfolderDicts. Emptying leaves
    a ``[]`` key manifest (marker dir present) so the public ``add-backbone`` pass refills exactly the
    configured backbone from the flipped refs — dropping any stale extra backbones cleanly."""
    md = OptFlowMetadata.deserialize(0, src)
    mm = md.obsmaskmeta
    mm.class_to_ref = {cls: swap(ref) for cls, ref in mm.class_to_ref.items()}
    mm.class_to_descriptors = SubfolderDict()
    mm.principal_components = SubfolderDict()
    md.serialize(0, dst)


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit("usage: isaac-datagen-unseen <config.yaml> <source_render_dir> <start> <end> [key=value ...]")
    src = Path(sys.argv[2])
    start, end = int(sys.argv[3]), int(sys.argv[4])
    if start >= end:
        sys.exit(f"empty frame window [{start}, {end})")
    runtime = load_config(sys.argv[1], sys.argv[5:])
    dst = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    if dst.exists():
        sys.exit(f"dst {dst} exists — pick a fresh idx/dataset_dir (won't overwrite)")
    n_src = len(list((src / "obs").glob("obs_*.png")))
    if end > n_src:
        sys.exit(f"end={end} exceeds source frame count {n_src} in {src / 'obs'}")
    dst.mkdir(parents=True)

    swap = v2.Compose([SwapRedBlue()])
    # 1 + 2. subset + RENUMBER frames [start, end) → 0..N-1. A file copy would keep the SOURCE numbering
    # and desync a subset, so go through (de)serialize: read the full per-frame OptFlowSample (carries
    # every geometry field phase-2's reproj gate reads — depth, cam2world, masks), flip its obs, write it
    # at the new contiguous index. Phase-2/3 products are intentionally NOT copied; they re-run on the
    # flipped obs below.
    for dst_idx, src_idx in enumerate(range(start, end)):
        s = OptFlowSample.deserialize(src_idx, src)
        s.obsmask.obs = swap(s.obsmask.obs)               # flip targets the obs sub-field only
        s.serialize(dst_idx, dst)
    print(f"wrote {end - start} flipped frames [{start},{end}) → 0..{end - start - 1} in {dst}", flush=True)

    # 3. catalog: flipped refs + emptied descriptors, then refill via the PUBLIC add-backbone CLI.
    _flip_catalog(src, dst, swap)
    # Per-render single files (no numbering). descriptor.yaml MUST land before add-backbone — it is the
    # backbone key add-backbone re-encodes into.
    for name in ("runtime.yaml", "descriptor.yaml", "lighting_log.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    # add-backbone runs from the isaac_datagen REPO ROOT so the descriptor config's relative
    # `cleandift_ckpt: ../checkpoints/...` (which is anchored one level under the workspace, NOT at the
    # isaac configs' src/isaac_datagen base) resolves. Pass dataset + descriptor config ABSOLUTE so only
    # that in-yaml checkpoint path depends on the cwd.
    _run("isaac_datagen.migrate_descriptors_backbone", "add-backbone",
         str(Path(runtime.dataset_dir).resolve()), str(Path(runtime.descriptor_config_path).resolve()),
         "--device", runtime.descriptor_device, cwd=REPO_ROOT)

    # 4 + 5. phases 2/3 via their public module entrypoints, inheriting THIS cwd (the isaac configs'
    # src/isaac_datagen base). Forward the SAME dotlist so the subprocess resolves the identical
    # dataset_dir/idx/proposer settings this run used.
    _run("isaac_datagen.add_proposals", sys.argv[1], *sys.argv[5:])
    _run("isaac_datagen.add_inlier_data", str(dst), "--eps", str(runtime.inlier_border_eps))
    print(f"\nunseen render dir ready: {dst}", flush=True)


if __name__ == "__main__":
    main()

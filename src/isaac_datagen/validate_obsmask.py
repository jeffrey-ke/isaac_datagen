"""Per-frame ObsMask cid/iid consistency check (NN-free).

Each graspable instance visible on a frame (keys of ``obs.iid_to_occlusion``) must
map to a valid class id in ``cid_mask`` (cid >= 2; 0=BACKGROUND, 1=UNLABELLED).
Catches writer bugs where ``iid_mask`` segments an object but ``cid_iid_masks`` fails
to assign a class (e.g. tuna with 38k iid pixels and 0 ``fish can`` cid pixels).

Membership is per-sample only — Isaac iids are session-local; do not union iids
across frames or use ``ObsMaskDescriptorMetadata.iid_to_name`` for the check (catalog names are
for error display only).

Usage:
    isaac-datagen-validate-obsmask <render_dir>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torchvision import tv_tensors

from vision_core.datastructs import ObsMask, ObsMaskDescriptorMetadata, count_samples

MIN_CLASS_CID = 2


@dataclass(frozen=True)
class CidOrphan:
    frame: int
    iid: int
    name: str
    n_pixels: int
    cids_seen: tuple[int, ...]


def graspable_iids(obs: ObsMask) -> set[int]:
    """Graspable instance iids present on this frame (writer: ``iid_to_occlusion`` keys)."""
    return {int(k) for k in obs.iid_to_occlusion}


def load_obsmask(render_dir: Path, idx: int) -> ObsMask:
    """One frame's masks + occlusion dict; skips RGBA load (dummy obs placeholder)."""
    iid_mask = ObsMask.deserialize_field(idx, render_dir, "iid_mask")
    cid_mask = ObsMask.deserialize_field(idx, render_dir, "cid_mask")
    iid_to_occlusion = ObsMask.deserialize_field(idx, render_dir, "iid_to_occlusion")
    return ObsMask(
        obs=tv_tensors.Image(torch.zeros(4, 1, 1, dtype=torch.uint8)),
        iid_mask=iid_mask,
        cid_mask=cid_mask,
        iid_to_occlusion=iid_to_occlusion,
    )


def load_obsmasks(render_dir: Path) -> list[ObsMask]:
    render_dir = Path(render_dir)
    for sub in ("iid_mask", "cid_mask"):
        if not (render_dir / sub).is_dir():
            sys.exit(f"missing {render_dir / sub}/ — not an ObsMask render dir")
    n = count_samples(render_dir)
    return [load_obsmask(render_dir, i) for i in range(n)]


def check_obsmask(obs: ObsMask, frame: int, *, min_class_cid: int = MIN_CLASS_CID) -> list[CidOrphan]:
    """Return cid orphans on this single frame (empty list if clean)."""
    iidm = obs.iid_mask.numpy()
    cidm = obs.cid_mask.numpy()
    orphans: list[CidOrphan] = []
    for iid in sorted(graspable_iids(obs)):
        pixels = iidm == iid
        cids = cidm[pixels]
        if (cids < min_class_cid).any():
            orphans.append(CidOrphan(
                frame=frame,
                iid=iid,
                name="?",  # filled by validate_render_dir from catalog (display only)
                n_pixels=int(pixels.sum()),
                cids_seen=tuple(sorted(int(c) for c in set(cids.tolist()))),
            ))
    return orphans


def validate_render_dir(render_dir: Path, *, min_class_cid: int = MIN_CLASS_CID) -> list[CidOrphan]:
    render_dir = Path(render_dir)
    md = ObsMaskDescriptorMetadata.deserialize(0, render_dir)
    iid_to_name = {int(k): v for k, v in md.iid_to_name.items()}
    out: list[CidOrphan] = []
    for f, obs in enumerate(load_obsmasks(render_dir)):
        for o in check_obsmask(obs, f, min_class_cid=min_class_cid):
            out.append(replace(o, name=iid_to_name.get(o.iid, "?")))
    return out


def _format_orphan(o: CidOrphan) -> str:
    return (f"frame {o.frame:04d}  iid {o.iid}  {o.name}  {o.n_pixels} px  "
            f"cids_seen={o.cids_seen}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("render_dir", type=Path)
    args = p.parse_args()

    orphans = validate_render_dir(args.render_dir)
    n_frames = count_samples(args.render_dir)
    n_iids = len({(o.frame, o.iid) for o in orphans})
    if orphans:
        print(f"{args.render_dir}: {len(orphans)} orphan row(s) across {n_iids} (frame,iid) "
              f"in {n_frames} frames", file=sys.stderr)
        for o in orphans:
            print(f"  {_format_orphan(o)}", file=sys.stderr)
        sys.exit(1)
    print(f"{args.render_dir}: ok ({n_frames} frames, no cid/iid orphans)")


if __name__ == "__main__":
    main()

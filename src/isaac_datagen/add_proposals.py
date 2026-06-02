"""Phase-2 pass: add proposer point-prompts to a rendered ObsMask dataset.

Runs AFTER Isaac has produced the per-frame ``ObsMask``s and the per-render-dir
``ObsMaskMetadata`` catalog. Does NOT boot Isaac Sim — it only needs torch +
``reference_matching``. For each frame it runs the (expensive) proposer once per
distinct reference image present, then writes the result *residually* onto the
existing render dir via ``PreReferenceSegSample.serialize(idx, dir, only={"proposals"})``
— so ``obs/`` and ``id_mask/`` are never rewritten.

Usage (mirrors clean_datagen):
    isaac-datagen-proposals <config.yaml> [key=value ...]
The render dir is ``{dataset_dir}/render{idx:03d}`` and the proposer is built from
``runtime.proposer_config_path`` on ``runtime.proposer_device``.
"""

import sys
from pathlib import Path

import torch

from vision_core.datastructs import ObsMask, ObsMaskMetadata, PreReferenceSegSample
from isaac_datagen.runtime_config import load_config


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-proposals <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"

    md = ObsMaskMetadata.deserialize(0, render_dir)

    from reference_matching import proposal as proposal_module
    device = runtime.proposer_device
    proposer = proposal_module.from_config(runtime.proposer_config_path).to(device)

    n_frames = len(list((render_dir / "obs").iterdir()))
    for idx in range(n_frames):
        om = ObsMask.deserialize(idx, render_dir)
        present = {int(i) for i in om.id_mask.unique().tolist()} & set(md.id_to_name)
        names = {md.id_to_name[i] for i in present} & set(md.name_to_ref)

        obs_b = om.obs.unsqueeze(0).to(device)
        proposals = {}
        with torch.inference_mode():
            for name in names:
                ref_b = md.name_to_ref[name].unsqueeze(0).to(device)
                # proposer returns list[(xy (M,2), scores (M,))] per batch element
                xy, _scores = proposer(obs_b, ref_b)[0]
                proposals[name] = xy.cpu()

        PreReferenceSegSample(obs=om.obs, id_mask=om.id_mask, proposals=proposals) \
            .serialize(idx, render_dir, only={"proposals"})
        print(f"[{idx + 1}/{n_frames}] {render_dir.name}: {len(proposals)} reference(s)")


if __name__ == "__main__":
    main()

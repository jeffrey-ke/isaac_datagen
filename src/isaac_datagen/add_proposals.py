"""Phase-2 pass: add proposer point-prompts to a rendered ObsMask dataset.

Runs AFTER Isaac has produced the per-frame ``ObsMask``s and the per-render-dir
``ObsMaskMetadata`` catalog. Does NOT boot Isaac Sim — it only needs torch +
``reference_matching``. Operates in class space: for each frame it runs the
(expensive) proposer once per class present (against the class's canonical
reference), skipping classes whose every member is more occluded than
``runtime.proposer_max_occlusion``, then writes the result *residually* onto the
existing render dir via
``PreReferenceSegSample.serialize(idx, dir, only={"proposals"})`` — so ``obs/``
and ``cid_mask/`` are never rewritten.

Usage (mirrors clean_datagen):
    isaac-datagen-proposals <config.yaml> [key=value ...]
The render dir is ``{dataset_dir}/render{idx:03d}`` and the proposer is built from
``runtime.proposer_config_path`` on ``runtime.proposer_device``.
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

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
    print(f"loading proposer from {runtime.proposer_config_path} on {device} …", flush=True)
    proposer = proposal_module.from_config(runtime.proposer_config_path).to(device)
    on_cuda = torch.device(device).type == "cuda"

    n_frames = len(list((render_dir / "obs").iterdir()))
    total_pts = 0
    bar = tqdm(range(n_frames), desc=render_dir.name, unit="frame")
    for idx in bar:
        om = ObsMask.deserialize(idx, render_dir)
        present_iids = {int(i) for i in om.iid_mask.unique().tolist()} & set(md.iid_to_name)
        # Occlusion gate (iid space, then join iid → name → class): a class is kept
        # if ANY member is visible enough (best-visible member). NaN (unknown
        # occlusion) compares False and is dropped.
        visible_iids = {iid for iid in present_iids
                        if om.iid_to_occlusion[iid] < runtime.proposer_max_occlusion}
        classes = {md.name_to_class[md.iid_to_name[iid]] for iid in visible_iids}
        names = sorted(classes & set(md.class_to_ref))
        if not names:
            tqdm.write(f"  frame {idx:04d}: no visible labeled references — writing empty proposals")

        obs_b = om.obs.unsqueeze(0).to(device)
        proposals = {}
        frame_pts = 0
        with torch.inference_mode():
            inner = tqdm(names, desc=f"  ↳ f{idx:04d}", unit="ref", leave=False)
            for name in inner:
                inner.set_postfix_str(name)  # which class is matching right now
                ref_b = md.class_to_ref[name].unsqueeze(0).to(device)
                # proposer returns list[(xy (M,2), scores (M,))] per batch element
                xy, _scores = proposer(obs_b, ref_b)[0]
                proposals[name] = xy.cpu()
                frame_pts += int(xy.shape[0])
                if xy.shape[0] == 0:
                    tqdm.write(f"  frame {idx:04d}: '{name}' returned 0 proposal points")

        PreReferenceSegSample(obs=om.obs, cid_mask=om.cid_mask, proposals=proposals) \
            .serialize(idx, render_dir, only={"proposals"})

        total_pts += frame_pts
        postfix = {"classes": len(proposals), "pts": frame_pts, "Σpts": total_pts}
        if on_cuda:
            postfix["vram"] = f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f}G"
        bar.set_postfix(postfix)

    print(f"done: {n_frames} frames, {total_pts} proposal points → {render_dir / 'proposals'}", flush=True)


if __name__ == "__main__":
    main()

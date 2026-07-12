
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from vision_core.datastructs import OptFlowSample, OptFlowMetadata, PreReferenceSegSample
from vision_core.seed_utils import seed_everything
from isaac_datagen.proposal_gate import gate_classes_reproj
from isaac_datagen.runtime_config import load_config


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-proposals <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"

    md = OptFlowMetadata.deserialize(0, render_dir)
    mm = md.obsmaskmeta

    n_frames = len(list((render_dir / "obs").glob("obs_*.png")))
    start = runtime.start_frame
    end = n_frames if runtime.end_frame is None else min(runtime.end_frame, n_frames)
    assert start < end, f"empty frame window [{start}, {end}) of {n_frames} frames"
    prop_ext = PreReferenceSegSample._get_serializer(dict)[0]
    done = {idx for idx in range(start, end)
            if (render_dir / "proposals" / f"proposals_{idx:04d}{prop_ext}").exists()}
    if done:
        print(f"resume: {len(done)}/{end - start} frames in [{start}, {end}) already have proposals — skipping",
              flush=True)
    if len(done) == end - start:
        print(f"done: 0 new + {len(done)} skipped frames in [{start}, {end}) → {render_dir / 'proposals'}",
              flush=True)
        return

    from reference_matching import proposal as proposal_module
    device = runtime.proposer_device
    print(f"loading proposer from {runtime.proposer_config_path} on {device} …", flush=True)
    proposer = proposal_module.from_config(runtime.proposer_config_path).to(device)
    on_cuda = torch.device(device).type == "cuda"

    total_pts = 0
    ref_cache: dict = {}
    bar = tqdm(range(start, end), desc=f"{render_dir.name}[{start}:{end}]", unit="frame")
    for idx in bar:
        if idx in done:
            continue
        seed_everything(runtime.effective_seed + idx)
        s = OptFlowSample.deserialize(idx, render_dir)
        om = s.obsmask
        gated = gate_classes_reproj(s, md, runtime.proposer_min_visible_ratio,
                                    runtime.proposer_tau_d, runtime.proposer_tau_r, ref_cache)
        names = sorted(set(gated) & set(mm.class_to_ref))
        if not names:
            tqdm.write(f"  frame {idx:04d}: no visible labeled references — writing empty proposals")

        obs_b = om.obs.unsqueeze(0).to(device)
        proposals = {}
        frame_pts = 0
        with torch.inference_mode():
            inner = tqdm(names, desc=f"  ↳ f{idx:04d}", unit="ref", leave=False)
            for name in inner:
                inner.set_postfix_str(name)
                ref_b = mm.class_to_ref[name].unsqueeze(0).to(device)
                xy, _scores = proposer(obs_b, ref_b)[0]
                if xy.shape[0] == 0:
                    tqdm.write(f"  frame {idx:04d}: '{name}' returned 0 proposal points — dropping")
                    continue
                proposals[name] = xy.cpu()
                frame_pts += int(xy.shape[0])

        PreReferenceSegSample(obs=om.obs, cid_mask=om.cid_mask, proposals=proposals) \
            .serialize(idx, render_dir, only={"proposals"})

        total_pts += frame_pts
        postfix = {"classes": len(proposals), "pts": frame_pts, "Σpts": total_pts}
        if on_cuda:
            postfix["vram"] = f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f}G"
        bar.set_postfix(postfix)

    print(f"done: {end - start - len(done)} new + {len(done)} skipped frames in [{start}, {end}), "
          f"{total_pts} new proposal points → {render_dir / 'proposals'}", flush=True)


if __name__ == "__main__":
    main()

"""Run the full reference-seg datagen pipeline: render → proposals → inlier labels.

Thin orchestrator: each phase runs as its OWN subprocess (Isaac Sim only releases
the GPU when its process exits; the proposer then needs it), chained fail-fast.
Phase 1 is skipped when the render dir already has obs/ frames — phases 2 and 3
are individually resumable, but re-rendering under an existing proposals/ would
silently desync them. Delete the render dir to force a fresh render.

Multi-GPU phase 2: ``proposer_devices=[cuda:0,cuda:1]`` fans out one proposals
subprocess per device, splitting the frame window contiguously. Phase 3 labels
ALL frames, so a manually windowed run (start_frame/end_frame without full
coverage) will fail phase 3 on the missing proposals — the window fields are
for sharding, not subsetting.

Usage: isaac-datagen-pipeline <config.yaml> [key=value ...]
"""

import shutil
import subprocess
import sys
from pathlib import Path

from isaac_datagen.runtime_config import load_config


def _find(script: str) -> str:
    exe = shutil.which(script, path=str(Path(sys.executable).parent)) or shutil.which(script)
    if exe is None:
        sys.exit(f"console script not found: {script} (uv sync?)")
    return exe


def _run(script: str, *args: str) -> None:
    exe = _find(script)
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    try:
        subprocess.run([exe, *args], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"{script} failed with exit code {e.returncode} — fix and re-run "
                 f"(completed phases/frames are skipped on resume)")


def _run_proposals_sharded(devices, n_obs: int, runtime) -> None:
    start = runtime.start_frame
    end = n_obs if runtime.end_frame is None else min(runtime.end_frame, n_obs)
    bounds = [start + round(i * (end - start) / len(devices)) for i in range(len(devices) + 1)]
    exe = _find("isaac-datagen-proposals")
    print(f"\n=== isaac-datagen-proposals × {len(devices)} shards over [{start}, {end}) ===", flush=True)
    procs = []
    for k, dev in enumerate(devices):
        shard_args = [*sys.argv[1:], f"start_frame={bounds[k]}", f"end_frame={bounds[k + 1]}",
                      f"proposer_device={dev}"]
        print(f"  shard {k}: frames [{bounds[k]}, {bounds[k + 1]}) on {dev}", flush=True)
        procs.append((k, dev, subprocess.Popen([exe, *shard_args])))
    # Wait for ALL shards even if one fails — the survivors' work is resumable.
    failed = [(k, dev, rc) for k, dev, p in procs if (rc := p.wait()) != 0]
    if failed:
        sys.exit("; ".join(f"shard {k} ({dev}) exited {rc}" for k, dev, rc in failed)
                 + " — fix and re-run (completed frames are skipped on resume)")


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-pipeline <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"

    obs = render_dir / "obs"
    n_obs = len(list(obs.glob("obs_*.png"))) if obs.is_dir() else 0
    if n_obs:
        print(f"phase 1: {obs} already has {n_obs} frames — skipping render "
              f"(delete {render_dir} to re-render)", flush=True)
    else:
        _run("isaac-datagen", *sys.argv[1:])
        n_obs = len(list(obs.glob("obs_*.png")))

    devices = runtime.proposer_devices or ()
    if len(devices) > 1:
        _run_proposals_sharded(devices, n_obs, runtime)
    else:
        extra = [f"proposer_device={devices[0]}"] if devices else []
        _run("isaac-datagen-proposals", *sys.argv[1:], *extra)

    _run("isaac-datagen-inliers", str(render_dir), "--eps", str(runtime.inlier_border_eps))


if __name__ == "__main__":
    main()

"""Run the full reference-seg datagen pipeline: render → proposals → inlier labels.

Thin orchestrator: each phase runs as its OWN subprocess (Isaac Sim only releases
the GPU when its process exits; the proposer then needs it), chained fail-fast.
Phase 1 is skipped when the render dir already has obs/ frames — phases 2 and 3
are individually resumable, but re-rendering under an existing proposals/ would
silently desync them. Delete the render dir to force a fresh render.

After phase 1, every run validates cid/iid mask consistency (graspable iids on
each frame must map to a class cid in cid_mask). On failure the pipeline exits
before proposals/inliers.

After a FRESH render, on an interactive terminal the pipeline prints a
foreground-luminance / dark-frame summary (via isaac-datagen-measure-luminance)
and pauses for a y/N confirmation, so you can catch junk (e.g. the intermittent
all-black renders) before the downstream phases spend compute on them; answer N
to abort (render dir kept). Non-interactive runs (CI/cron/pipes) skip the prompt
and continue.

Multi-GPU phase 2: ``proposer_devices=[cuda:0,cuda:1]`` fans out one proposals
subprocess per device, splitting the frame window contiguously. Phase 3 labels
ALL frames, so a manually windowed run (start_frame/end_frame without full
coverage) will fail phase 3 on the missing proposals — the window fields are
for sharding, not subsetting.

Usage: isaac-datagen-pipeline <config.yaml> [key=value ...]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from isaac_datagen.runtime_config import load_config
from isaac_datagen.tldr import TLDR
from isaac_datagen.validate_obsmask import validate_render_dir, _format_orphan


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


def _confirm_or_abort(render_dir: Path, n_obs: int) -> None:
    """Pause after a fresh render so the user can eyeball the frames before the
    pipeline spends compute on the downstream (no-Isaac) phases — useful given the
    intermittent all-black-render bug. Only prompts on an interactive terminal;
    non-interactive runs (CI/cron/pipes) continue uninterrupted.
    """
    if not sys.stdin.isatty():
        return
    # Auto-print the foreground-luminance / dark-frame catalog so obvious junk
    # (the intermittent all-black renders) shows up without opening images.
    exe = shutil.which("isaac-datagen-measure-luminance", path=str(Path(sys.executable).parent)) \
        or shutil.which("isaac-datagen-measure-luminance")
    if exe:
        subprocess.run([exe, str(render_dir)], check=False)
    else:
        print("(isaac-datagen-measure-luminance not found — skipping dark-frame summary)", flush=True)
    try:
        ans = input(f"\nRendered {n_obs} frames to {render_dir / 'obs'} (dark-frame summary above).\n"
                    f"Continue to proposals + inlier labeling? [y/N] ").strip().lower()
    except EOFError:
        ans = "n"
    if ans not in ("y", "yes"):
        sys.exit("aborted after render — downstream phases skipped; render dir kept, "
                 "re-run isaac-datagen-pipeline to resume.")


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
    parser = argparse.ArgumentParser(
        prog="isaac-datagen-pipeline",
        description="Run all three phases (render -> proposals -> inlier labels) as one "
                     "resumable command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TLDR,
    )
    parser.add_argument("config", help="path to a YAML config (see CONFIGS below)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args, overrides = parser.parse_known_args(sys.argv[1:])
    runtime = load_config(args.config, overrides)
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"

    obs = render_dir / "obs"
    n_obs = len(list(obs.glob("obs_*.png"))) if obs.is_dir() else 0
    fresh = not n_obs
    if n_obs:
        print(f"phase 1: {obs} already has {n_obs} frames — skipping render "
              f"(delete {render_dir} to re-render)", flush=True)
    else:
        _run("isaac-datagen", *sys.argv[1:])
        n_obs = len(list(obs.glob("obs_*.png")))

    orphans = validate_render_dir(render_dir)
    if orphans:
        print(f"\ncid/iid validation failed: {len(orphans)} orphan row(s) in {render_dir}",
              file=sys.stderr, flush=True)
        for o in orphans[:20]:
            print(f"  {_format_orphan(o)}", file=sys.stderr, flush=True)
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more", file=sys.stderr, flush=True)
        sys.exit("fix cid_mask / re-render before proposals — "
                 "run isaac-datagen-validate-obsmask for the full list")

    if fresh:
        _confirm_or_abort(render_dir, n_obs)   # gate: inspect fresh render before downstream phases

    devices = runtime.proposer_devices or ()
    if len(devices) > 1:
        _run_proposals_sharded(devices, n_obs, runtime)
    else:
        extra = [f"proposer_device={devices[0]}"] if devices else []
        _run("isaac-datagen-proposals", *sys.argv[1:], *extra)

    _run("isaac-datagen-inliers", str(render_dir), "--eps", str(runtime.inlier_border_eps))


if __name__ == "__main__":
    main()

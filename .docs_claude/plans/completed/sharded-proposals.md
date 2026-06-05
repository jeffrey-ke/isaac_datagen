# Multi-GPU phase-2: contiguous frame windows + run_pipeline fan-out

## Context

Phase-2 is the long pole (~6 s/frame single-GPU). The machine has two GPUs; user wants
`run_pipeline` to optionally split one render dir's frames across them. Decisions
(user): **contiguous ranges** (`start_frame`/`end_frame`), not modulo sharding; the
multi-GPU option lives in **run_pipeline** (`proposer_devices`), with manual sharded
invocations falling out for free.

Facts this builds on (verified this session):
- `idx` selects the render dir, not frames — no existing frame-window parameter.
- The `done` resume set in `add_proposals.py` is a membership set, not a prefix
  counter — it composes with any disjoint/interleaved completion pattern.
- Atomic writes mean even overlapping shards can't corrupt files (only waste compute).
- `proposer_device` is already an overridable config field.

## Changes

### 1. `src/isaac_datagen/runtime_config.py` — three optional fields

```python
    # Phase-2 frame window (contiguous sharding): process frames
    # [start_frame, end_frame); end_frame=None → through the last frame.
    start_frame: int = 0
    end_frame: int | None = None

    # run_pipeline only: >1 device → one phase-2 subprocess per device, splitting
    # the frame window contiguously. None/1 device → single proposer run.
    proposer_devices: tuple[str, ...] | None = None
```

(`tuple[str, ...]` matches the existing `texture_paths` pattern; dotlist syntax
`proposer_devices=[cuda:0,cuda:1]`.) `__post_init__` gains
`assert self.start_frame >= 0 and (self.end_frame is None or self.end_frame > self.start_frame)`.

### 2. `src/isaac_datagen/add_proposals.py` — loop over the window

```python
    n_frames = len(list((render_dir / "obs").glob("obs_*.png")))
    start = runtime.start_frame
    end = n_frames if runtime.end_frame is None else min(runtime.end_frame, n_frames)
    assert start < end, f"empty frame window [{start}, {end}) of {n_frames} frames"
    prop_ext = PreReferenceSegSample._get_serializer(dict)[0]   # ".pt"
    done = {idx for idx in range(start, end)
            if (render_dir / "proposals" / f"proposals_{idx:04d}{prop_ext}").exists()}
    if done:
        print(f"resume: {len(done)}/{end - start} frames in [{start}, {end}) already have proposals — skipping", flush=True)
    if len(done) == end - start:
        print(f"done: 0 new + {len(done)} skipped frames in [{start}, {end}) → {render_dir / 'proposals'}", flush=True)
        return
    ...
    bar = tqdm(range(start, end), desc=f"{render_dir.name}[{start}:{end}]", unit="frame")
```

Final summary print likewise reports `in [{start}, {end})`. Docstring: one sentence on
the window fields. Default window `[0, n_frames)` reproduces today's behavior exactly.

### 3. `src/isaac_datagen/run_pipeline.py` — fan out phase 2 over devices

Split `_run` into `_find(script)` (the `shutil.which` lookup) + `_run`. After phase 1
(recount `n_obs` if phase 1 just ran):

```python
    devices = runtime.proposer_devices or ()
    if len(devices) > 1:
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
        failed = [(k, dev, p.wait()) for k, dev, p in procs if p.wait() != 0]
        if failed:
            sys.exit("; ".join(f"shard {k} ({dev}) exited {rc}" for k, dev, rc in failed)
                     + " — fix and re-run (completed frames are skipped on resume)")
    else:
        extra = [f"proposer_device={devices[0]}"] if devices else []
        _run("isaac-datagen-proposals", *sys.argv[1:], *extra)
```

Usage: `uv run isaac-datagen-pipeline configs/randomized.yaml idx=0 'proposer_devices=[cuda:0,cuda:1]'`.

Known cosmetic wart (accepted): both shards' tqdm bars interleave on one tty.
Known caveat (documented in run_pipeline docstring): phase 3 labels ALL frames, so a
manually windowed run (`start_frame`/`end_frame` without full coverage) will fail
phase 3 on the missing proposals — the window fields are for sharding, not subsetting.

## Verification (GPU-free now; real 2-GPU run after the current job finishes)

1. Config: `load_config` round-trips the new fields from dotlist
   (`'proposer_devices=[cuda:0,cuda:1]' start_frame=2 end_frame=4`); defaults unchanged.
2. Window resume on fully-done `cid-mask-verify/render900`:
   `isaac-datagen-proposals … idx=900 dataset_dir=cid-mask-verify start_frame=1 end_frame=3`
   → `resume: 2/2 frames in [1, 3)`, exits before proposer load.
3. Sharded pipeline dry-run on the same dir:
   `isaac-datagen-pipeline … idx=900 dataset_dir=cid-mask-verify 'proposer_devices=[cuda:0,cuda:1]'`
   → phase-1 skip, two shard lines `[0, 2)` / `[2, 4)`, both skip-all instantly (no
   proposer load → safe while cuda:0 is busy), phase 3 runs.
4. (Deferred until the running render000 job completes) real 2-GPU shard: delete two
   proposals files in different halves, run the sharded pipeline, confirm each shard
   processes exactly its frame and VRAM appears on both GPUs.

## Status (2026-06-05)

Implemented and verified through item 3 (config round-trip; window resume `[1,3)`;
sharded pipeline dry-run on render900 — shards `[0,2)`/`[2,4)` resumed independently,
phase 3 ran once after both). Item 4 (real 2-GPU run with both proposers loaded) is
DEFERRED until the running render000 phase-2 job releases cuda:0 — move this plan to
completed/ once that passes.

## Outcome (2026-06-05, live test)

Item 4 passed on the real dataset: deleted `proposals_0250.pt`/`proposals_0750.pt`
from `expanded-refseg/render000`, ran the sharded pipeline with
`proposer_devices=[cuda:0,cuda:1]` — each shard reported `1 new + 499 skipped` in its
window, both GPUs held a loaded proposer simultaneously (user-observed), phase 3 ran
once: 10,239,419/15,612,572 inliers; exit 0.

Follow-up found during verification: `mkstemp`'s 0600 security default leaked into
every atomically-written dataset file. Fixed in `vision_core.atomic_write_path`
(chmod tmp to `0o666 & ~umask` before the writer opens it); existing render000
outputs chmod'd back to 664.

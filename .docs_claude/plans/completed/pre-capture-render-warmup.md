# Plan: pre-capture render warmup (settle RTX before the writer captures)

## Context

`scene.py:boot_sim` documents an unresolved flake: PT renders are a "lit-vs-black per-process
coin flip" (the in-code note records **9–11 of 15 frames black**, unchanged by the `multi_gpu`
toggle). The accumulation-reset setting
(`/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges`, `boot_sim:290`) tamed it but
did not eliminate it.

What's missing is a **render warmup before the first capture**. Today nothing steps the renderer
between `build_scene` and the first `rep.orchestrator.step()` — so the dome HDRI, MDL shaders,
denoiser and PT state may not be initialized when frame 0 is captured. Isaac's own
`isaacsim.sensors.camera/.../tests/test_camera_sensor.py` (6.0) handles this by stepping the app
`NUM_WARMUP_FRAMES = 10` times (`await omni.kit.app.get_app().next_update_async()`) after
`set_intensities(500)` and before reading pixels.

Note: dome intensity is **already** set explicitly (`make_dome_light` → `1000`, `build_scene:387`),
renderMode is already pinned to PathTracing, so this plan adds **only** the warmup — no intensity
or rendermode changes.

Approach chosen (option A): a plain `app.update()` settle loop before capture — matches the Isaac
example, low risk, no change to `capture_session`. We cannot warm up *through the orchestrator*
after the trigger is bound, because `move_prims` uses `rep.distribution.sequence` which advances
one pose per `orchestrator.step()` and the attached writer would serialize junk frames.

## Changes

### 1. `src/isaac_datagen/runtime_config.py` — new tunable field (near `rt_subframes`, ~L165)

```python
# app.update() frames to settle RTX before capture: lets MDL shaders compile, the dome
# HDRI/textures stream in, and PT/denoiser state initialize, so frame 0 isn't a black
# coin flip (see boot_sim's accumulation note). 0 disables.
warmup_frames: int = 16
```

### 2. `src/isaac_datagen/scene.py` — new helper, directly below `boot_sim` (after L314)

```python
def warmup_render(app, n_frames):
    """Settle the RTX renderer before capture: step the app n_frames times so MDL shaders
    compile, the dome HDRI/textures stream in, and the PT/denoiser state initializes.
    Without this the first captured frame(s) are a lit-vs-black coin flip (see boot_sim's
    accumulation note). Mirrors Isaac's camera-sensor warmup (isaacsim.sensors.camera tests:
    N x app.update() before reading pixels). n_frames == 0 is a no-op."""
    for _ in range(n_frames):
        app.update()
```

### 3. `src/isaac_datagen/clean_datagen.py` — import + call before capture

Import (L24):
```python
# before
from isaac_datagen.scene import boot_sim, build_scene, make_replicator
# after
from isaac_datagen.scene import boot_sim, build_scene, make_replicator, warmup_render
```

Call site (L85–88), placed after the `dry_run` early-return so dry runs don't pay for it:
```python
    writer = ObsMaskWriter(runtime.descriptor_config_path, runtime.descriptor_device, scene.objects,
                           render_dir, full_alpha=runtime.obs_full_alpha)
    replicator = make_replicator(runtime, len(world_poses), render_dir)
    warmup_render(app, runtime.warmup_frames)   # settle RTX before the writer captures
    capture_with_poses(world_poses, writer, scene.zed, replicator, rt_subframes=runtime.rt_subframes)
```

`app` (the `SimulationApp` from `boot_sim`) is already in scope here (`clean_datagen.py:63`);
`SimulationApp.update()` is the synchronous form of the `next_update_async()` the Isaac test uses.

### 4. (optional) `src/isaac_datagen/configs/randomized.yaml`

The `warmup_frames: int = 16` default applies without a YAML entry. Add an explicit
`warmup_frames: 16` line near the lighting block only if you want it visible/tunable from the
config; otherwise override per-run via dotlist (`warmup_frames=24`).

## Verification

1. Run a render:
   `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml idx=0 num_frames=8`
2. Inspect `render000/rgb/` — frames should be lit, not black. Compare the black-frame rate
   against a prior run (the boot_sim note's baseline was ~9–11/15 black).
3. The writer's existing `[PROBE]` print (dome intensity / HDR readback in
   `reference_seg_writer.write`) and the `[TONEMAP]` print from `boot_sim` confirm lighting state.
4. Tune `warmup_frames`: start at 16; raise (e.g. `warmup_frames=24`) if black frames persist;
   lower toward 8–10 once stable to save per-render time. `warmup_frames=0` reproduces today's
   behavior for an A/B check.

## Risk / fallback

`app.update()` settles **global** RTX state (shaders, HDRI, denoiser, lights). It does not, by
itself, prove it primes the *PT-accumulation-on-the-rep-render-products* path that is specifically
flaking. If black frames survive a generous `warmup_frames`, the fallback (deferred, not in this
plan) is to add a couple of throwaway `rep.orchestrator.step()` calls inside `capture_session`
*before* `attach_writers` and before the `on_frame` trigger is bound — that exercises the exact
capture path with no writer attached and no pose consumed.

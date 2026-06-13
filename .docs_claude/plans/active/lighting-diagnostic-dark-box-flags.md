# Lighting diagnostic config flags — dark-box investigation

> **STATUS (2026-06-12): ACTIVE — root cause found, fix not yet implemented.** The diagnostic
> plumbing below all landed and did its job. It proved the dark boxes are **not** a light-strength
> problem: the RTX tonemapper/exposure has a near-binary response (hard black floor → immediate
> saturation), so the dome intensity has no usable midtone. See **Investigation results** at the
> bottom for the data, the two bugs found, and where to take it next.

## Context

The `reference_segmentation` path renders **objects that fade to black across the box wall** — not
dead frames and not black textures (the boxes *are* textured: "facial tissue" is faintly legible).
Illumination falls off spatially: boxes near a light are lit, the rest go dark with no discrete
boundary. Examples in `…/perspective-refseg/render001/obs/`: `obs_0336.png` (mild — right column lit,
left dark), `obs_0233.png` (acute — almost the whole wall black).

Likely mechanism: the **static sphere light** (localized, intensity 5000, inverse-square falloff) is
the dominant source, while the **dome** (500–1000, `Normalize=True` → divided by solid angle → weak)
provides almost no uniform fill, and the **distant** light's jittered rotation can point edge-on. So
per-frame "is the wall lit" depends on camera pose × light draw. Raising the dome floor 0→500 didn't
help (still too weak), and the Blender dry-run doesn't run the per-frame replicator randomizers, so
we can't see it offline.

**Strategy: strip the scene to the single simplest light — the dome — and make it observable.** Sphere
and distant are both ablated; if a strong, uniform, non-normalized dome lights the whole wall evenly,
the cause was weak/localized fill and the fix is dome settings. If the wall is *still* dark under a
cranked dome, the cause is upstream (materials/denoiser/exposure), not light placement.

Diagnostics, exposed as `RuntimeConfig` flags (OmegaConf overrides, e.g. `dome_intensity_range=[50000,50000]`):

- **Step 0 — measure:** standalone script cataloging per-frame **foreground** luminance (mask by
  alpha; the background is transparent, so a whole-frame mean is meaningless) and flagging dark frames.
- **Step 1 — determinism:** seed is pinned by default, so two runs share an identical dome schedule but
  different cameras; dark-frame drift ⇒ camera-driven, stable ⇒ light-draw-driven.
- **Step 2 — constant strong dome:** `dome_intensity_range=[50000,50000] dome_normalize=false` ⇒ one
  run decides "weak/normalized uniform fill" vs upstream.
- **Step 3 — sweep + log:** vary dome range/normalize; a per-frame `lighting_log.json` joins each dark
  frame to the dome intensity that produced it.

**Decisions with the user:** sphere **and** distant lights are removed (dome alone is enough to debug;
no `target2world` rig). Dome jitter moves from graph-side `rep.distribution.uniform` (opaque to Python)
to **Python-sampled `rep.distribution.sequence`** — statistically identical, now seedable + loggable,
mirroring how `move_prims` already drives camera poses. **`main()`/stereo is out of scope and left
untouched.**

## Key facts

- **Sequence length = `len(world_poses)` = `num_targets × num_frames`** (1100 for `randomized.yaml`), NOT `num_frames`. `plan_capture` (capture.py:35-52) flattens `(B,N,4,4)→(B*N,4,4)`; one orchestrator step per element. The dome sequence must match this length or it desyncs from the camera sequence — thread the count from the call site.
- `rep.distribution.sequence(list)` consumes one element per `orchestrator.step()`, inside the same `on_frame` trigger as `move_prims` → lockstep with camera poses.
- Seed: `from omni.replicator.core.utils.rng import set_global_seed; set_global_seed(n)`. Camera offsets use the **unseeded global numpy RNG** (`generate_random_offsets` → `np.random.uniform`) — left unseeded so Step 1 varies cameras while the dome is pinned.
- Only the `reference_segmentation` call site is touched. `main()`/stereo also calls `make_replicator` and is already broken pre-existing (it passes `target2world` into `make_replicator(runtime)`); we leave it as-is.

## 1. `runtime_config.py` — new fields

ADD after the texture fields (house idiom: 2-tuple ranges, immutable defaults). Seed is a **non-optional `int`, default `0`** — the dome schedule is deterministic by default. Defaults preserve current dome behavior.

```python
    # ── Lighting diagnostics (dark-box investigation) ────────────────────────
    replicator_seed: int = 0                     # pins graph RNG + numpy RNG drawing the dome sequence
    jitter_dome: bool = True
    dome_intensity_range: tuple[float, float] = (500.0, 1000.0)
    dome_normalize: bool = True                  # flip False to test "normalize starves the dome"
    log_lighting: bool = True                    # write <render_dir>/lighting_log.json
```

`__post_init__` (mirror existing asserts):
```python
        assert self.replicator_seed >= 0
        lo, hi = self.dome_intensity_range
        assert lo <= hi, f"dome_intensity_range must have lo<=hi: {(lo, hi)}"
```

## 2. `scene.py`

Add `from pathlib import Path` (numpy already imported).

### 2a. `make_replicator` — gains `num_frames`, `render_dir`; dome-only; always-seeded
```python
# BEFORE
def make_replicator(runtime):
    import omni.replicator.core as rep
    replicator = ReplicatorWrapper(rep)
    # world_lo, world_hi = _target_range_to_world(runtime, target2world)
    # register_sphere_jitter(rep, replicator, "/World/SphereLight", world_lo, world_hi)
    register_distant_jitter(rep, replicator, "/World/DistantLight")
    if runtime.dome_light:
        register_dome_jitter(rep, replicator, "/World/DomeLight")
    if runtime.background_textures:
        register_background_jitter(rep, replicator, "/World/DomeLight", runtime.background_textures)
    return replicator

# AFTER
def make_replicator(runtime, num_frames, render_dir):
    import omni.replicator.core as rep
    from omni.replicator.core.utils.rng import set_global_seed
    set_global_seed(runtime.replicator_seed)
    rng = np.random.RandomState(runtime.replicator_seed)   # separate from camera-offset + grasp RNGs

    replicator = ReplicatorWrapper(rep)
    log = {}
    # sphere + distant ablated for the dark-box debug — the dome is the only light
    if runtime.dome_light and runtime.jitter_dome:
        register_dome_jitter(rep, replicator, "/World/DomeLight", runtime, num_frames, rng, log)
    if runtime.background_textures:
        register_background_jitter(rep, replicator, "/World/DomeLight", runtime.background_textures)

    if runtime.log_lighting:
        import json
        (Path(render_dir) / "lighting_log.json").write_text(json.dumps(
            {"num_frames": num_frames, "replicator_seed": runtime.replicator_seed, "lights": log}, indent=2))
    return replicator
```

### 2b. `register_dome_jitter` — `uniform` → Python-sampled `sequence` (precompute, record, feed)
```python
# BEFORE
def register_dome_jitter(rep, replicator, prim_path):
    dome_node = rep.get.prim_at_path(prim_path)
    def jitter_dome():
        with dome_node:
            rep.modify.attribute("intensity", rep.distribution.uniform(500, 1000))
        return dome_node.node
    replicator.register(jitter_dome)

# AFTER
def register_dome_jitter(rep, replicator, prim_path, runtime, num_frames, rng, log):
    node = rep.get.prim_at_path(prim_path)
    vals = rng.uniform(*runtime.dome_intensity_range, size=num_frames).tolist()
    assert len(vals) == num_frames
    log["DomeLight"] = [{"intensity": v} for v in vals]
    def jitter_dome():
        with node:
            rep.modify.attribute("intensity", rep.distribution.sequence(vals))
        return node.node
    replicator.register(jitter_dome)
```
`register_distant_jitter` becomes uncalled (distant ablated) — leave it dormant.

### 2c. `make_dome_light` — `Normalize` becomes a param
```python
# BEFORE
def make_dome_light(stage, parent, intensity=1000.0):
    ...
    dome.GetNormalizeAttr().Set(True)

# AFTER
def make_dome_light(stage, parent, intensity=1000.0, normalize=True):
    ...
    dome.GetNormalizeAttr().Set(normalize)
```

### 2d. `build_scene` — ablate sphere + distant, wire dome normalize
```python
# BEFORE
    make_dome_light(stage, "/World", intensity=1000.0 if runtime.dome_light else 0.0)
    make_sphere_light(stage, "/World")
    make_distant_light(stage, "/World")

# AFTER
    make_dome_light(stage, "/World", intensity=1000.0 if runtime.dome_light else 0.0,
                    normalize=runtime.dome_normalize)
    # make_sphere_light(stage, "/World")    # ablated: localized inverse-square falloff = the dark-wall culprit
    # make_distant_light(stage, "/World")   # ablated: grazing-angle directional light left faces dark
```
Also delete the now-dead `register_sphere_jitter` (144-152) and `_target_range_to_world` (132-141) — their only consumer was the sphere.

> `lighting_log.json` is the *planned* schedule; list index == frame index == `obs/obs_{idx:04d}.png`. On a crash `obs/` may be shorter — the Step-0 join guards with `i < len(frames)`.

## 3. `clean_datagen.py` — fix the one call site (`reference_segmentation` only)
```python
# reference_segmentation (line 97) — world_poses already in scope from line 84
# BEFORE
    replicator = make_replicator(runtime)
# AFTER
    replicator = make_replicator(runtime, len(world_poses), render_dir)
```
`main()`/stereo is **not** touched (out of scope; already broken pre-existing).

## 4. Step 0 — `measure_luminance.py` (new) + console script

New `src/isaac_datagen/measure_luminance.py`; register `isaac-datagen-measure-luminance =
"isaac_datagen.measure_luminance:main"` in `pyproject.toml [project.scripts]`. No Isaac dep — pure
`vision_core` + torch/numpy, mirroring `viz_occlusion.py`.

```
args: render_dir, --pixel-threshold 8, --frame-threshold 0.5, --csv OUT, --with-lighting

per frame i in range(count_samples(render_dir)):
    obs = ObsMask.deserialize_field(i, render_dir, "obs")   # (4,H,W) uint8 RGBA
    rgb, alpha = obs[:3], obs[3]
    fg = alpha > 0                                           # foreground = the box wall; bg is transparent
    luma = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]     # BT.709, over fg pixels only
    fg_mean = luma[fg].mean()                                # mean object brightness
    dark_frac = (luma[fg] < pixel_threshold).mean()          # fraction of the wall that is near-black
report: per-frame (fg_mean, dark_frac); flag "dark" frames where dark_frac > frame_threshold; print count/fraction/indices
--with-lighting: join each dark index to lighting_log.json["lights"]["DomeLight"] → print the realized dome intensity
```

Foreground masking is the key fix: a whole-frame mean luma is dominated by the transparent background
and would miss exactly the dark-box case we're chasing.

## 5. Experiment invocations (`randomized.yaml`, `isaac-datagen` entry)

- **Step 1 (determinism):** `idx=821` then `idx=822` (seed pinned at default 0 ⇒ identical dome schedule, cameras differ). `diff` the two Step-0 dark-index lists. Stable ⇒ light-driven; drift ⇒ camera-driven.
- **Step 2 (constant strong dome):** `dome_intensity_range=[50000,50000] dome_normalize=false idx=823`. Strong, constant, view-independent fill. Dark gone ⇒ confirmed weak/normalized dome (raise it / keep normalize off). Dark persists ⇒ upstream (materials/denoiser/exposure).
- **Step 3 (sweep):** narrow/shift `dome_intensity_range`, toggle `dome_normalize`; `isaac-datagen-measure-luminance <render_dir> --with-lighting` reads each dark frame's realized dome intensity.

## 6. Verification

- Tool sanity (no Isaac): `isaac-datagen-measure-luminance …/render001 --csv /tmp/lum.csv` — should flag `obs_0233` (acute) above `obs_0336` (mild) on `dark_frac`.
- Logging: `isaac-datagen <cfg> idx=820`; assert `render820/lighting_log.json` exists and `len(lights["DomeLight"]) == num_targets*num_frames` (1100).
- Default regression: `isaac-datagen <cfg> idx=824` (no overrides) still renders, lit by the dome alone (sphere + distant ablated) at the prior 500–1000 range.

## 7. Risks

1. **Ablating distant + sphere changes the look** of every render (dome-only) — intended for the debug; the production lighting recipe is revisited once the cause is known.
2. **uniform→sequence changes the RNG stream** (distribution identical) — can't bit-reproduce pre-refactor renders. Acceptable; it's the point.
3. **Dome sequence length must be `len(world_poses)`** (1100), not `num_frames` (25) — wrong length silently desyncs the frame↔light join. Threaded explicitly + `assert len(vals) == num_frames`.

## Files

- `runtime_config.py` — new fields + validation
- `scene.py` — dome-only sequence jitter, sphere + distant ablated, `make_replicator` signature, dome `normalize` param, lighting log
- `clean_datagen.py` — fix the `reference_segmentation` `make_replicator` call site
- `measure_luminance.py` — new Step-0 script (foreground-masked luminance)
- `pyproject.toml` — register the console script

---

## Investigation results (2026-06-12)

All renders ran on **GPU 1** via `CUDA_VISIBLE_DEVICES=1 … descriptor_device=cuda:0 proposer_device=cuda:0`,
because GPU 0 is occupied by the user's `gtrain` (gligen) job. Without this the DIFT/SD pipeline builds on
its config-default device (`cuda` → GPU 0) and OOMs *before* any frame renders. Config: `num_targets=5
num_frames=5` → 25 frames/render. Renders live in `src/isaac_datagen/datasets/debug/render{idx}/`.

### Headline finding — the dome knob is near-binary; there is no goldilocks intensity

Constant-dome sweep (`dome_intensity_range=[X,X] dome_normalize=false`), foreground-mean luma over 25 frames:

| idx | dome intensity | fg_mean (0–255) | look |
|---|---|---|---|
| 820 | [500,1000] **normalized** | 0.0 | pure black (baseline default) |
| 828 | 3000 | 0.0 | pure black |
| 829 | 5500 | 0.0 | pure black |
| 831 | 6500 | 234 | near-white |
| 830 | 7500 | 237 | near-white |
| 827 | 10000 | 241 | near-white (user: "too bright") |
| 823 | 50000 | 252 | blown white (user: "oversaturated") |

The whole transition from crushed-black to near-saturated happens between **5500 and 6500** — it jumps
`0 → 234` with **no midtone anywhere**, and every dark frame is *exactly* `0.0`, not "a little dim".
That is a hard exposure/tonemap **black floor + immediate clip**, not gradual falloff.

**This floor IS the original bug.** A uniform dome gives every box equal irradiance, so they all flip
together (all-black / all-white). The original *localized* sphere gave boxes unequal irradiance → those
clearing the floor render lit, those just under it crush to pure black → the "boxes fade to black across
the wall" symptom. So the root cause is the **renderer's exposure/tonemapping**, not light strength or
placement. `boot_sim` sets **no exposure/tonemap at all** (RTX defaults) — that is the lever to fix.

### Second bug found (separate, set aside) — `rep.modify.attribute` sequence does not advance

The per-frame dome sweep (Step 3) never worked: `lighting_log.json` records a varied schedule
(995→24476) but every frame rendered at the **first** element (~995 → black). Verified the camera *does*
vary (18/25 distinct alpha masks), so `rep.distribution.sequence` advances for `rep.modify.pose`
(`move_prims`) but **not** for `rep.modify.attribute("intensity", …)`. "Fix B" (routing the dome through
a direct in-trigger `add_modifier` path instead of `rep.randomizer.register`, mirroring `move_prims`) was
implemented, tested (render826 — identical flat 0.0), and **reverted** — so register-vs-direct was never
the cause. The real difference is `modify.pose` vs `modify.attribute`; unsolved.
**Consequence:** per-frame dome *jitter* is non-functional, but **constant** dome renders (lo==hi) are
reliable (all elements identical → stuck-at-first is harmless) — that is why the bisection above used
constants. If production wants per-frame dome variation, this bug must be solved or the dome driven a
different way.

### Code state right now
- **Landed & working:** all of §1–§4 (config flags, dome-only ablation of sphere+distant,
  `make_replicator(runtime, num_frames, render_dir)` signature + call site, seeded loggable
  `register_dome_jitter`, `make_dome_light(normalize=…)`, `measure_luminance.py` + console script).
  `measure_luminance` validated against production `render001` and every debug render.
- **Reverted:** the "Fix B" `_modifiers`/`add_modifier` experiment in `scene.py` — `ReplicatorWrapper`
  and `register_dome_jitter` are back to the `register`-based form documented in §2b.
- **Uncommitted:** all of the above is working-tree only.

### Next directions (unresolved — pick up here)
1. **Exposure/tonemap (most promising):** `boot_sim` sets no exposure. Investigate the RTX post-process —
   auto-exposure / eye-adaptation, tonemap operator, explicit exposure — to turn the binary cliff into a
   gradient so scene radiance maps to a real midtone. This is the likely actual fix.
2. **Material albedo:** check whether the box material is near-black/low-albedo (needs huge light, clips
   fast), which would compound the binary look under a tonemapper.
3. **Restore + rebalance sphere/distant:** brings back the original fragility (boxes straddling the floor);
   only viable once the floor itself is fixed.
4. **`modify.attribute` sequence bug:** only needed if per-frame dome variation is wanted in production.

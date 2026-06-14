# Render-darkness investigation (reference_segmentation path)

> **STATUS (2026-06-13):** Two distinct bugs in the `reference_segmentation` RTX render path.
> **Bug 1 (dark boxes) — SOLVED** (underexposure; fix = `exposure_time=1.0`).
> **Bug 2 (intermittent all-black renders) — characterized, not yet fixed.** ~40% of *processes*
> render correctly; ~60% render the **entire wall pure black for the whole process** (the path tracer
> computes ~0 scene radiance from all lights — HDR RGB≈0, though dome intensity is correct at 1000).
> The outcome is a **per-process coin flip locked in at renderer init, before the first frame, and
> immutable for the process lifetime** — proven by a 40-frame run that stayed 100% black across all 40
> frames (and a lit run 100% lit). It is **independent of every lever tried**: exposure, materials,
> `rt_subframes`, PT accumulation, light type (dome vs distant), and multi_gpu. The two realistic paths
> are **detect-and-retry** (relaunch until lit) or a deeper dig into the init race. All code changes
> below are **uncommitted** working-tree state.

> **RECONCILIATION (2026-06-13):** Production lighting is **no longer dome-only**. `scene.build_scene`
> now ships an aimed **DistantLight key + low DomeLight fill** (see `RuntimeConfig.distant_*` /
> `dome_fill_intensity`); the sphere stays ablated, the distant light is restored as the key and aimed
> at the grasp-target centroid via `look_at`+`cv2opengl` (so it can't point edge-on). Bug 1's fix
> (`exposure_time=1.0`) is live in `boot_sim`. **Bug 2 is unaffected by this change** — it is
> light-type-independent per the analysis below, so the distant key neither fixes nor worsens it;
> mitigation remains detect-and-retry. The "dome-only" descriptions below are the *debug* scene, not
> the live recipe.

Extends `lighting-diagnostic-dark-box-flags.md` (which covers only Bug 1's diagnostic plumbing).

---

## How we measure

- `isaac-datagen-measure-luminance <render_dir>` → per-frame **foreground** (alpha>0) BT.709 luminance
  (`fg_mean`) + `dark_frac`. A frame with `fg_mean ≈ 0` is a black wall. `--csv` dumps all frames.
- Renders run on GPU 1 to dodge the `gtrain` job on GPU 0:
  `CUDA_VISIBLE_DEVICES=1 … descriptor_device=cuda:0 proposer_device=cuda:0`.
- Debug renders land in `src/isaac_datagen/datasets/debug/render<NNN>/obs/obs_XXXX.png` (RGBA).
- Scene is dome-only for the debug (sphere+distant ablated); seed fixed, **camera poses unseeded**, so
  repeated runs of one config share the scene but differ in camera viewpoint.

---

## Bug 1 — dark boxes (SOLVED)

**Symptom:** boxes "fade to black across the wall" — foreground luminance crushes to *exactly* 0.0,
then with slightly more light snaps to ~234+ with **no midtone** (a near-binary cliff).

**Root cause (EXP0 probe, render832):** the captured `rgb` AOV is the **post-tonemap LDR**
(`LdrColorSD`) buffer. The tonemap operator is already **ACES** (`/rtx/post/tonemap/op=6`, *not* Clamp),
auto-exposure is **off** (`/rtx/post/histogram/enabled=False`), and exposure is fixed by the
photographic triangle — but the shipped carb default **`/rtx/post/tonemap/exposureTime=0.02`** (a
daylight shutter, ~EV100 10) **underexposed** the dome-lit scene into the ACES toe → uint8 crush to 0.
Not a Clamp operator, not light strength, not materials — a stale exposure default.

**Exposure sweep** (dome 1000, normalize off), renders 833–836:

| exposure_time | fg_mean | note |
|---|---|---|
| 0.1 | 0.0 | in the ACES toe |
| 0.5 | 0.0 | in the toe |
| **1.0** | **~178** | **clears the toe — the fix** |
| 2.0 | ~181 | ACES-shoulder plateau (1.0≈2.0) |

**Fix:** `RuntimeConfig.set_exposure=True` (default) + `exposure_time=1.0`, applied in `boot_sim` via
`/rtx/post/tonemap/{exposureTime,fNumber,filmIso}`. Auto-exposure left off for per-frame determinism.

---

## Bug 2 — intermittent all-black renders (IN PROGRESS)

### Discovery
A dome-intensity sweep at `exposure_time=1.0` came back **anti-monotonic** (renders 840–844):

| dome | result |
|---|---|
| 150 | lit (~85) |
| 300 | **BLACK** |
| 500 | lit (~150) |
| 1000 | **BLACK** (yet render835 = same 1000 was lit!) |
| 200–1000 jitter | **BLACK** |

More light giving *less* brightness is physically impossible → the *applied result isn't tracking the
dome value*. So it's not lighting magnitude.

### It's flaky, per-process
Repeating the **identical** config (dome 1000, norm off, exp 1.0) flips outcome run-to-run
(renders 845–848): `845 LIT, 846 BLACK, 847 BLACK, 848 (static dome) LIT`. Properties:
- **Per-process, all-or-nothing** — all frames of a render share the outcome (decided once per process).
- **Geometry/alpha/labels correct; only RGB is black.**
- **Logs byte-identical** between a lit and a black run (render861 vs 868) apart from timestamp/idx —
  no load warning, no error. `--/rtx/materialDb/syncLoads=True` is already on. So it's a *silent*
  render-output difference, not a detectable load failure.
- Black rate ≈ 60%.

### The decisive probe (renders 892–897)
Added an `HdrColor` annotator + per-frame log of `ldr_max / hdr_max / dome_I`:

| render | ldr_max | hdr_max | dome_I |
|---|---|---|---|
| 892, 895 (lit) | 255 | **2.5–3.1** | 1000 |
| 893,894,896,897 (black) | 0 | **1.00** (just the alpha channel) | 1000 |

**Conclusion:** on black frames the dome intensity is correct (`1000`) but the **HDR scene RGB radiance
is ≈ 0** — the dome light is configured but **contributes zero illumination**. The geometry, materials,
exposure, accumulation, and LDR capture are all fine; the scene simply receives no light. This is a
**DomeLight / IBL initialization race in path tracing**, intermittent per process.

> The commit "dome light can not disappear now" (e4990cd) is **unrelated** — it only raised the jitter
> floor `uniform(0,1000)→(500,1000)` so the *intensity* is never drawn as 0. Our black frames have
> `dome_I=1000`. Different failure mode.

### Fixes that did NOT work (ruled out empirically)
| attempt | hypothesis | change | result |
|---|---|---|---|
| Phase F | material load | warmup via `orchestrator.step` | **crashed** — warmup steps advance the global counter that `rep.distribution.sequence` (camera) is indexed by → offset poses → "no labeled instances" assert |
| Phase G | material load | `app.update` warmup=32 + `rt_subframes=20` | 10/15 black (unchanged); warmup is a **no-op in headless** (didn't change render time) |
| Phase H | PT accumulation | `resetPtAccumOnlyWhenExternalFrameCounterChanges=True` (forum 229697) | 9/15 black (unchanged) |
| distant | not dome/IBL-specific | un-ablate analytic distant light | 7/15 black; distant ALSO contributes ~0 on black frames |
| multi_gpu | multi-GPU async race (OMREQ-1202) | `SimulationApp(multi_gpu=False)` | 11/15 black (unchanged) |

### The 40-frame test — the outcome is locked at init, so warmup is dead (renders 930, 933)
"If warmup-by-rendering helped, why don't the later frames of a render come out lit?" — exactly. Ran a
single process of **40 frames** (`num_frames=8`, no warmup):
- A **black** process (render930) stayed **100% black across all 40 frames** — `ldr_max=0, hdr_max=1.0`
  every frame, no recovery, no transition.
- A **lit** process (render933, found on the 3rd relaunch — ~40% hit rate) was **100% lit across all 40**
  (fg_mean ~128–172).

So a process is **wholly** black or lit; it never mixes or recovers. This **kills the warmup hypothesis**
(you cannot render your way out of a black process) and explains why every per-frame lever was
irrelevant: the black/lit state is decided **once at renderer initialization, before the first frame,
and is immutable for the process**. It's a startup coin flip, not a settle/convergence problem.

### Distant-light test (renders 898–912) — did NOT fix it, and reframed the bug
Un-ablated the distant light (analytic, not IBL) as base illumination, dome still on, 15× repeat.
**Result: 7/15 still black.** Crucially, the probe on the black frames shows `hdr_max=1.0` (alpha only)
→ the **distant light *also* contributes ~0 radiance** on those frames (a few are barely-lit, fg_mean
~2–3 with `hdr_max≈1.0–1.2`). So an IBL light (dome) **and** an analytic light (distant) fail to
illuminate *together*, per process.

**Revised diagnosis: this is NOT dome/IBL-specific.** All lighting intermittently produces ~0 radiance
in the path tracer, per-process, regardless of light type — a fundamental **PT render/lighting
initialization race** (light list / NEE / multi-GPU composite not ready on the first captured frame),
not anything about a specific light. (Side note: dome+distant lit frames came out ~130–165, slightly
*dimmer* than dome-only ~178 — unexplained, not pursued.)

---

## Web / forum corroboration

- **[Forum 283428]** PT + texture randomization → black objects, "severe in complex scenes, all frames
  black." Matches our scenario; NVIDIA gave no clean root-cause fix.
- **[Forum 229697]** "PT samples do not accumulate" without
  `/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges` (+ `/rtx/externalFrameCounter`).
  Fixed in Replicator 1.5.3 (we're on 1.12.27). We tested the setting → did **not** fix our black.
- **[Isaac 5.1 Replicator troubleshooting]** black images → `--reset-user`; randomized materials not
  loaded → `rt_subframes ≥ 2`; frame skipping → `--/exts/isaacsim.core.throttling/enable_async=false`.
- **[GitHub IsaacSim #426]** "black renders when adding more objects"; canonical pattern is a warmup
  `step()` + `step(wait_for_render=True, rt_subframes=20)`. NVIDIA never reproduced; closed.

---

## Key facts / gotchas discovered

- The captured **`rgb` annotator = `LdrColorSD`** (post-tonemap uint8); **`HdrColor`** is the raw linear
  buffer (used here only as a debug probe). `HdrColor` is RGBA — its `.max()` is dominated by alpha=1.0
  on an unlit frame, so "hdr_max=1.0" means RGB radiance ≈ 0.
- **`rep.distribution.sequence` advances per `orchestrator.step()`** (a global counter). The camera
  poses use it (`move_prims`), so any extra orchestrator step (e.g. a warmup) desyncs the schedule.
  `rep.modify.attribute("intensity", sequence)` does **not** advance the way `modify.pose` does — the
  dome jitter therefore uses **stateless `rep.distribution.uniform`** (reverted from a sequence attempt).
- **`rep.settings.set_render_pathtraced()` with no arg resets `/rtx/pathtracing/totalSpp` to 64**,
  clobbering the configured 256. `SimulationApp` also pins `spp=totalSpp=clampSpp=64`. (Not the black
  cause, but a latent quality bug.)
- The **PT denoiser key** our `boot_sim` sets (`/rtx/denoiser/enabled`) is the **RT-mode** key — a no-op
  in PathTracing; the active one is `/rtx/pathtracing/optixDenoiser/enabled`.
- Renders are ~20 s each (boot+DIFT dominate; `rt_subframes` had no measurable effect on time).

---

## Code changes (all uncommitted)

- **`runtime_config.py`** — exposure knobs (`set_exposure=True`, `exposure_time=1.0`, `f_number=5`,
  `film_iso=100`); `rt_subframes=20` knob; + asserts. (Warmup knob `render_warmup_frames` removed.)
- **`scene.py`** `boot_sim` — apply exposure; `resetPtAccumOnlyWhenExternalFrameCounterChanges=True`
  (Phase H, no effect — keep or drop); `[TONEMAP]` confirmation print; `multi_gpu` back to `True`
  (ruled-out test reverted). `register_dome_jitter` reverted sequence→`uniform`. `build_scene` distant
  light **re-ablated** (un-ablation didn't help).
- **`capture.py`** — **warmup machinery removed** (the 40-frame test killed it): `capture_session` and
  `capture_with_poses` are back to plain capture; `rt_subframes` still threaded.
- **`clean_datagen.py`** — pass `rt_subframes` into `capture_with_poses`.
- **`reference_seg_writer.py`** — **DEBUG**: `HdrColor` annotator + `[PROBE]` print (remove before commit).
- **`measure_luminance.py`** — `load_lighting` guards the new dict-shaped log entry.

---

## Next directions

The 40-frame test reframed everything: the black/lit state is a **per-process init coin flip**, locked
before frame 0 and immutable — not a settle/warmup/convergence problem. Every per-frame and per-setting
lever is therefore a dead end (and all have been tried). Two realistic paths:

1. **Detect-and-retry (pragmatic, robust — recommended stopgap).** After each render, measure foreground
   luminance; if black, **relaunch the whole process** (fresh `idx`) until lit. ~40% hit rate → ~2.5
   attempts/good render. Wraps cleanly around the per-render datagen invocation; no RTX internals
   needed. Cost: ~2.5× render processes (boot dominates, so ~2.5× wall-clock).
2. **Dig into the process-init race (real fix).** The decision happens at renderer/Hydra init before the
   first frame and is sticky. Candidate angles not yet explored: the RTX scene/light-BVH build at first
   render (is the light list ever empty for a process?); `--reset-user` / a clean Kit config;
   driver/OptiX context init nondeterminism; whether a *synchronous, fully-settled first render*
   (`/app/asyncRendering=False` + a blocking settle that does NOT use the sequence-coupled
   orchestrator.step) changes the rate. Lower confidence — five setting-level hypotheses already failed.
3. **Cleanup before commit:** ✅ warmup machinery removed; ✅ distant re-ablated; ✅ multi_gpu back to
   True. Still TODO: remove the `HdrColor`/`[PROBE]` debug from `reference_seg_writer.py`; decide whether
   to keep `resetPtAccum` and the `rt_subframes` knob; fix the `set_render_pathtraced` totalSpp clobber.

---

## Render-dir index (`datasets/debug/`)

| dirs | what they tested | result |
|---|---|---|
| 820, 825, 826 | pre-exposure dome-only diagnostics | black |
| 827–831, 823 | constant-dome bisection → mapped the tonemap cliff | 3000/5500→0; 6500→234; 50000→252 |
| 832 | EXP0 tonemap probe | found op=6 ACES, exposureTime=0.02 |
| 833–836 | exposure_time sweep | 0.1/0.5→0; **1.0→178**; 2.0→181 |
| 840–844 | dome sweep @ exp 1.0 → exposed flakiness | 150 lit, 300 black, 500 lit, 1000 black |
| 845–848 | flakiness confirmation (identical config) | 845 lit, 846/847 black, 848 (static) lit |
| 850–855 | Phase F (orchestrator.step warmup) | **crashed** (sequence offset) |
| 860–875 | Phase G (app.update warmup + rt_subframes=20) | 10/15 black |
| 876–891 | Phase H (resetPtAccum) | 9/15 black |
| 892–897 | HDR probe | dome_I=1000 but HDR RGB≈0 on black frames |
| 898–912 | distant-light un-ablation | 7/15 black; distant ALSO contributes ~0 → not light-type-specific |
| 913–927 | `multi_gpu=False` | 11/15 black → multi-GPU ruled out |
| **930** | **40-frame single process (black draw)** | **all 40 black, no recovery** |
| 931, 932 | 40-frame relaunches | black |
| **933** | **40-frame single process (lit draw)** | **all 40 lit (~128–172)** → per-process all-or-nothing confirmed |

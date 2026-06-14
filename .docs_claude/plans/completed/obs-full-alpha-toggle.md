# Plan: opaque-alpha toggle for obs (uncropped inspection renders)

> **STATUS (2026-06-14): IMPLEMENTED + verified.** `obs_full_alpha: bool = False` added and threaded
> through the writer. Flag-on render confirmed: `obs/obs_0000.png` alpha is all 255, full frame directly
> viewable, masks unchanged. Default behavior untouched. **This toggle did its job — it made visible a
> pre-existing OCCLUDER problem (see "What's going wrong" below).**

## Context
The `obs` RGBA had its **alpha set from the instance/foreground mask** (`reference_seg_writer.composite_rgba`
→ `alpha_from_instance_seg`): opaque only on graspable-instance pixels, transparent elsewhere. Correct for
the dataset contract, but it crops the view to odd shapes when inspecting renders — you can't see the full
frame (shadows / background / composition). This adds a first-class toggle so inspection renders come out
uncropped with no post-step. Default False → unchanged; foreground stays recoverable from `iid_mask`.

## Changes (as landed)
- **`runtime_config.py`** — `obs_full_alpha: bool = False` (after `rt_subframes`).
- **`reference_seg_writer.py`** — `composite_rgba(..., full_alpha=False)` emits `np.full(seg.shape, 255)`
  when set; `ObsMaskWriter.__init__(..., full_alpha=False)` stored as `self._full_alpha`; passed in `write()`.
  (Stereo path's separate `composite_rgba` left untouched.)
- **`clean_datagen.py`** — `ObsMaskWriter(..., full_alpha=runtime.obs_full_alpha)`.
- No YAML change — dotlist override `obs_full_alpha=true`, default off.

## Verify (done)
`… uv run clean_datagen.py configs/randomized.yaml num_targets=2 num_frames=4 obs_full_alpha=true …` →
`obs_0000.png` alpha min==max==255; full frame viewable directly; `iid_mask`/`cid_mask` unchanged.

## Risks
- With the flag ON, alpha-as-foreground consumers (e.g. `isaac-datagen-measure-luminance`, masks by alpha>0)
  treat the whole frame as foreground. Expected for inspection; use `iid_mask` for true foreground. Default
  OFF, so production/training data is unaffected.

---

## What's going wrong (occluder diagnosis surfaced by this toggle)

With the alpha crop removed, the full-frame renders show the **invisible shadow occluders are misbehaving**.
This is a problem with the *occluder* feature (see `per-target-shadow-occluders.md` /
`distant-light-key-light.md`), not with this alpha toggle. The lighting itself is fine — boxes render bright,
the gray dome is visible in good frames.

**Symptoms (measured across the `obs_full_alpha=true` render, 8 frames, scene=empty):**
- The occluders (the only UNLABELLED geometry, iid==1) cover **55–97% of every frame**, rendering as dark
  blobs that occlude the box wall. Boxes are only 7–27% of each frame.
- Per-frame A-vs-B comparison (A = an earlier favorable run, B = this run):

  | | near-black | gray dome visible | box faces |
  |---|---|---|---|
  | Render A (4 frames, all good) | 7–37% | 28–41% every frame | 19–29% |
  | Render B (7 of 8 frames bad)  | 73–89% | ~0% | 7–23% |
  | Render B frame 7 (escaped)    | 16% | 48% | 19% |

**Three root causes:**
1. **`primvars:hideForCamera` is not taking effect under PathTracing.** The occluders should be invisible to
   the camera (cast shadows only), but they render as solid black where they sit. (Not 100% provable in a
   `scene: empty` void — behind an occluder is also black — so the definitive test is a render WITH a dome
   background texture: see-through ⇒ flag works; black blob ⇒ it doesn't.)
2. **Occluders are placed too close to / on the camera→target sightline.** Occluder ranges
   `x[0.10,0.40] y[-0.15,0.15] z[-0.15,0.15]` overlap the camera halo `x[0.35,0.85]`, and the camera always
   looks at the target origin (0,0,0) where the occluders cluster (y,z near 0). A close occluder subtends a
   huge angle → it fills the frame regardless of exact centering.
3. **Placement is UNSEEDED, so it's high-variance and irreproducible.** `posers` →
   `pose_planning.plan_poses` → `vision_core.pose_utils.generate_random_offsets` = `np.random.uniform(...)`
   (global RNG, never seeded) for BOTH camera and occluder positions. Only the occluder *scales* are seeded
   (`build_scene`'s `RandomState(seed)`). So every run draws different camera + occluder positions: Render A
   got a favorable draw (occluders peripheral → looked great), Render B an unfavorable one (occluders
   centered/close → dominate). `obs_full_alpha` touches only the alpha channel, so it is NOT the cause of the
   A-vs-B difference — the RNG draw is.

**Recommended fixes (next task, not done here):**
- **Seed the placement** (thread the seeded `rng` into the posers / `generate_random_offsets`) so results are
  reproducible and a good config stays good.
- **Constrain occluder placement off the sightline / out of the frustum** — push them toward the face (small
  x), and/or offset in y,z so they sit between the key light and the box rather than between the camera and
  the target (cf. original ideation approach A2: out-of-frame occluders).
- **Resolve `hideForCamera` under PT** (confirm with the dome-background test); if genuinely unsupported,
  switch to out-of-frame occluders so visibility never matters.

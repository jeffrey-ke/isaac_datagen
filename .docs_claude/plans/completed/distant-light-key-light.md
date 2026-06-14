# Plan: DistantLight as the main (key) light + DomeLight as fill

> **STATUS (2026-06-13): IMPLEMENTED, render-test pending GPU.** Code landed in `runtime_config.py`,
> `scene.py`, `configs/randomized.yaml`; investigation docs reconciled. Statically verified: config
> schema accepts the new fields + YAML resolves; `look_at_euler` aim is exact (`dot(emission, centroid−eye)=1.0`,
> emission `-Z=[-0.229, 0.688, -0.688]`). Still TODO on a GPU: the Blender dry-run orientation check and the
> small lit render + `ObsMask.visualize` intensity/orientation tuning (Verify steps 3–4).

## Context
The invisible shadow occluders (`scene.add_shadow_occluders`, already implemented) need a **directional**
light to cast crisp shape shadows. The working tree was **dome-only** — a uniform dome gives only soft
ambient occlusion, no shape silhouette — with sphere + distant lights commented out (ablated) from the
dark-box debug. This change restores **`DistantLight` as the key (main) source** with the **dome demoted to
a low ambient fill**, so occluders cast recognizable shadows on the box wall.

Why distant (not sphere/dome) — grounded in the existing code + physics, not assertion:
- **Sphere was ablated for exactly this** — `scene.py`: `# ablated: localized inverse-square falloff =
  the dark-wall culprit`. A point/sphere light's irradiance falls as 1/d² (near boxes blow out, the far
  wall starves → uneven, toe-crushed wall). A **distant** light is parallel rays from infinity → irradiance
  independent of distance → uniform across the whole wall, plus crisp directional shadows.
- **Not a black-render regression** — un-ablating the distant light was already tried and the all-black
  frames persisted (7/15); the black bug is light-type-independent, so adding the distant key doesn't
  reintroduce it.
- **The dark-box fixes it relies on are already live in `boot_sim`** — PT-accumulation reset
  (`resetPtAccumOnlyWhenExternalFrameCounterChanges=True`) and fixed exposure (`set_exposure=True`/
  `exposure_time=1.0`, defaulted in `runtime_config.py`).

Decisions (confirmed with user): **dome kept as low fill**, **static light for the first render**. Honest
scope of "static": within one render the light + occluder are fixed, so the cast shadow is **world-fixed** —
the orbiting camera gives *viewpoint* variety on that shadow (its projection / which faces it crosses
shifts), **not new shadow shapes**. Distinct shadows come **across renders** from the unseeded occluder
poser. True per-frame distinct-shadow variety would need per-frame light jitter (deferred) — fine to add
once the static recipe is validated.

**Aim the light, don't hand-author euler angles.** A `DistantLight` emits along its **local -Z** — exactly
like the USD cameras that `LookAtPoser` already orients (`posers.py`). So aim it the same way the camera is
aimed: `cv2opengl(look_at(centroid, eye))` (`vision_core/pose_utils.py`), then convert to euler via
`R.from_matrix(...).as_euler('xyz', degrees=True)`. The look target is the **world centroid of the grasp
targets**, `get_target2world(grasp_frames)[:, :3, 3].mean(0)` (`capture.py`) — computed from real geometry,
auto-adapting if the wall moves, instead of the brittle hand-guess `(65,0,20)`. The only knob is
`distant_light_offset` = the "sun" position relative to that centroid; for a distant light **only its
direction matters, not its magnitude**. The wall's camera-facing faces are **-Y** (camera at world -Y
looking +Y), so a **front-top** offset `(1,-3,3)` makes the rays travel +Y/−Z into those faces and drop
occluder shadows onto them.

## Changes (as landed)

### `runtime_config.py` — distant-key + dome-fill config
```python
    # ── Lighting: DistantLight key + DomeLight fill ──────────────────────────
    # DistantLight is the main source: parallel rays → uniform wall irradiance
    # (no inverse-square dark-wall falloff) and crisp directional shadows for the
    # occluders. It is AIMED at the grasp-target centroid via look_at+cv2opengl
    # (see build_scene), not hand-rotated. distant_light_offset is the "sun"
    # position relative to that centroid — DIRECTION ONLY matters (distant light),
    # default front-top into the -Y camera-facing faces. Dome is a low ambient
    # fill so shadowed faces don't crush. TUNE distant_intensity (and/or
    # exposure_time) on the first render to land fg_mean ~120-180; keep
    # dome_fill_intensity ~10-20% of the key.
    distant_light: bool = True
    distant_intensity: float = 3000.0
    distant_angle: float = 0.53                        # angular diameter (deg); raise → softer penumbra
    distant_light_offset: tuple[float, float, float] = (1.0, -3.0, 3.0)  # sun pos rel. to centroid; dir only
    dome_fill_intensity: float = 200.0
```
The adjacent dome-only diagnostics comment was rewritten to mark it SUPERSEDED (live recipe is distant-key +
dome-fill); `jitter_dome`/`dome_intensity_range` left in place (unused while static).

### `scene.py` — aim helper; `make_distant_light` rotation; `build_scene` key+fill
`look_at_euler` (reuses the camera's look-at convention; module-level `np`/`R`, lazy `look_at`/`cv2opengl`):
```python
def look_at_euler(eye, target):
    """Euler XYZ (deg) orienting a -Z-emitting USD prim (light/camera) to look eye→target.

    Same convention as LookAtPoser: vision_core.look_at is +Z-forward (OpenCV); cv2opengl
    flips it to USD's -Z-forward. Eye magnitude is irrelevant for a distant light (direction only).
    """
    from vision_core.pose_utils import look_at, cv2opengl
    pose = cv2opengl(look_at(np.asarray(target, float), np.asarray(eye, float)))
    return tuple(R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True))
```
`make_distant_light` gained `rotation=(0.0, 0.0, 0.0)` and authors it via `set_transform`.
`build_scene`: dome demoted to fill in place (`intensity=runtime.dome_fill_intensity if runtime.dome_light
else 0.0`), ablation comments removed; the distant **key** is created **after** the stack `set_transform` +
`grasp_frames_paths`, beside the occluder block, so it can be aimed:
```python
    if runtime.distant_light:
        from isaac_datagen.capture import get_target2world
        centroid = get_target2world(grasp_frames_paths)[:, :3, 3].mean(0)   # world wall center
        eye = centroid + np.asarray(runtime.distant_light_offset, float)    # "sun" pos; direction only
        make_distant_light(stage, "/World", intensity=runtime.distant_intensity,
                           angle=runtime.distant_angle, rotation=look_at_euler(eye, centroid))
    if runtime.occluders_per_target:
        add_shadow_occluders(stage, "/World", grasp_frames_paths, runtime, rng)
```
`make_replicator` docstring + body comment updated (static aimed-DistantLight key + dome fill; no per-frame
jitter by default).

### `configs/randomized.yaml`
```yaml
distant_light: True
distant_intensity: 3000.0
distant_angle: 0.53
distant_light_offset: [1.0, -3.0, 3.0]
dome_fill_intensity: 200.0
```

### Docs reconciled
`.docs_claude/plans/active/render-darkness-investigation.md` and `lighting-diagnostic-dark-box-flags.md` got
dated RECONCILIATION notes: production lighting is now distant-key + dome-fill (the "dome-only" text is the
*debug* scene); Bug 1's exposure fix + the PT-accum reset live in `boot_sim`; Bug 2 (intermittent all-black)
is light-type-independent so this change neither fixes nor worsens it — mitigation stays detect-and-retry.

## Verify
1. **Config — DONE:** schema accepts the 5 fields; YAML resolves them.
2. **Aim — DONE:** `look_at_euler((1,-3,3),(0,0,0))` → euler `(46.5, 0, 18.4)`, emission `-Z=[-0.229, 0.688,
   -0.688]`, `dot(emission, centroid−eye)=1.0` (aims exactly at the centroid; +Y into faces, downward, slight -X).
3. **Blender (placement/orientation) — PENDING GPU:** `clean_datagen.py configs/randomized.yaml idx=0
   dry_run=true` → `/World/DistantLight` tilted front-top toward the -Y faces; each occluder between the
   light direction and its box.
4. **Render (the real test) — PENDING GPU:** `uv run clean_datagen.py configs/randomized.yaml num_targets=2
   num_frames=4`, then `ObsMask.visualize(md)`. Expect crisp shadows on the -Y faces, no visible occluder,
   shadow absent from masks. Check `fg_mean` → tune `distant_intensity` (and `distant_light_offset`, occluder
   `*_pose_policy_args`) so shadows land on the faces, in the midtones.

## Risks
- **Intensity/exposure balance:** exposure was tuned for dome=1000; distant units differ. `distant_intensity`
  (or `exposure_time`) will need a tuning pass on the first lit render.
- **look_at degeneracy:** `look_at` builds x via `cross(z, [0,0,1])`, so a near-vertical `distant_light_offset`
  would degenerate. Keep the offset non-vertical (the default's large -Y component is safe); same limitation
  `LookAtPoser` already lives with.
- **Black-render bug:** per the investigation it's a per-process PT-init coin flip independent of light type.
  If a render comes back all-black, re-run — it is not caused by this lighting change.

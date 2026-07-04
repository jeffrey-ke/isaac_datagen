# Lighting jitter: why the Replicator graph route is a no-op, and the mechanism that works

**Status: verified on GPU, 2026-07-03** (fixed-camera A/B renders + per-frame USD probe;
Isaac Sim 5.1.0, omni.replicator.core 1.12.27+107.3.3). This note replaces earlier
*inductions* about per-frame lighting randomization with instrumented facts, and corrects
two of them.

## The finding

Per-frame modifications to the build_scene lights via the Replicator graph —

```python
node = rep.get.prim_at_path("/World/DistantLight")
def jitter():
    with node:
        rep.modify.pose(rotation=rep.distribution.sequence(...))
        rep.modify.attribute("intensity", rep.distribution.uniform(lo, hi))
    return node.node
replicator.register(jitter)            # rep.randomizer.register + handle call
# handle invoked inside `with rep.trigger.on_frame():` by apply_randomizers()
```

— **never execute**. Not "write to the wrong layer", not "wrong attribute name":
a per-frame probe (dump composed attrs + every layer's authored spec after each
`orchestrator.step()`) showed **no layer ever received any opinion** from these nodes, on
any channel (rotation, `inputs:intensity`, `inputs:colorTemperature`), across a whole
capture. Zero `Cannot find prim` warnings (a `rep.get.prim_at_path` miss is loud —
`OgnGetPrimAtPath.py:28-61` carb-warns and disables the branch — so it wasn't a path bug).

Meanwhile the **camera** — driven by the *same* `rep.modify.pose` + `rep.distribution.sequence`
shapes, but created **directly inside the `on_frame` trigger body** (`move_prims`,
`capture.py`) rather than through `rep.randomizer.register` — works, and its writes land in
the **root layer** at step time. So:

- direct in-trigger modify nodes: execute per frame, write to the root layer ✓
- the same nodes built inside a `rep.randomizer.register`'d fn (invoked inside the same
  trigger): never write anything, silently ✗

The registered-randomizer indirection is the broken link. Which internal detail breaks it
(exec-chain splicing of the registered subgraph vs the returned node, etc.) was not chased
further — the fix below doesn't depend on it.

### Corrections to prior inductions
- **"`rep.distribution.sequence` doesn't advance on attribute writes — it sticks on the
  first value"** (dark-box plan, render-darkness plan): wrong reading. The attribute write
  never happened at all; the "stuck" value was just the static build_scene value showing
  through. `uniform` vs `sequence` was never the variable.
- **"stateless `uniform` redraws each frame"** for the dome: never verified and — via this
  registration path — never executed either. Per-frame dome jitter had never actually run
  in any render.
- ⚠️ `register_background_jitter` / `register_box_texture_jitter` use the same
  `replicator.register` route and are therefore **suspect no-ops** — unverified, but if a
  render is ever supposed to vary backgrounds/box textures per frame, probe first.

## The mechanism that works: direct USD writes from the capture step loop

`capture_session` (`capture.py`) drives capture with a plain Python loop —
`for i in range(n_frames): rep.orchestrator.step(...)` — running *after* `rep.new_layer()`
has exited. It now accepts a `per_frame(i)` callback invoked right before each step; USD
writes made there reach frame i's render. `ReplicatorWrapper.register_per_frame` +
`ReplicatorWrapper.per_frame` fan this out; `capture_with_poses` threads it.

`register_distant_jitter` / `register_dome_jitter` (`scene.py`) precompute the **entire
schedule** in Python with seeded rngs (`default_rng([effective_seed, k])`, k decorrelating
key vs dome), then register closures doing plain-Python schema-API writes under
`Usd.EditContext(stage, stage.GetRootLayer())` (the `set_transform` convention):

```python
set_transform(prim, rotation=rotations[i])              # existing root-layer rotateXYZ op
light.GetIntensityAttr().Set(intensities[i])            # inputs:intensity
light.GetColorTemperatureAttr().Set(temperatures[i])    # inputs:colorTemperature
```

This is also the SDK's own idiom for *existing* lights: no NVIDIA example modifies a
pre-existing UsdLux light via `rep.get`+`rep.modify` — infinigen randomizes them with plain
`GetAttribute("inputs:*").Set(...)` per iteration
(`standalone_examples/replicator/infinigen/infinigen_sdg_utils.py:614-643`), and graph-based
light randomization always **creates** lights (`rep.create.light`). Bonus over graph
distributions: `lighting_log.json` now records the exact **applied** per-frame values for
every channel (schedule == applied), not distribution descriptors.

### Verified results (fixed camera: `num_targets=1`, degenerate x/y/z ranges, 8 frames)

| Metric | Graph route (broken) | Step-loop writes (fix) |
|---|---|---|
| composed light attrs across frames | constant | vary per schedule, every channel |
| fg luminance | 139.7–140.0 | 133.3–150.5; corr(intensity, lum) = **0.94** |
| R/B ratio | 1.04 constant | 1.08–1.27; corr(temperature, R/B) = **−0.94** |
| mean abs pixel diff vs frame 0 | 0.3–0.6 (PT noise) | 10–26 |

## Magnitude matters: why a working jitter can still look negligible

The first shipped ranges (`offset_jitter=0.75`, intensity `[1200, 2800]`) were applied
correctly (corr 0.94) yet looked near-uniform. Three attenuators stack:

1. **DistantLight geometry** — ±0.75 m sun wobble ≈ ±10–13° direction change; incidence on
   the box faces spans 40–51°, so the cosine factor spans only 0.62–0.76 (±10%). Parallel
   rays: no falloff, minimal shadow motion.
2. **Additive floor** — regression on the fixed-cam run (R²=0.89): fg LDR ≈ 105 constant
   (dome fill + PT bounce + bright albedo) + 14–34 from the key + 2–7 from the dome. The
   jittered channel is a small rider on a big floor.
3. **ACES shoulder** (auto-exposure off, op=6, fixed exposure — `scene.py` boot_sim): at
   fg ≈ 130/255 the local gain is low. Key 2.4× linear → 1.15× LDR. The dome channel looks
   fine because the background wall is dome-lit *alone* and sits lower on the curve
   (2.9× linear → 1.73× LDR).

NVIDIA's demos look dramatic for magnitude reasons, not renderer settings (they set no
tonemap/exposure): none jitters a DistantLight — they reposition 3 SphereLights by meters,
randomize color over (near-)full RGB, use 5–30× intensity ratios (infinigen 500–2500;
pose_generation 1e5–3e6; object_based_sdg N(35000, 5000)), and swap whole HDR domes.

**Confirmed fix (fixed-cam A/B, same code):** `offset_jitter=2.0` (±30° cone; front-face
cos 0.19–0.98, never behind/below) + intensity `[500, 4000]` (8×) → fg LDR 90–172 (1.9×
vs 1.15×), R/B 1.03–1.69, visibly obvious montage. Shipped in both refseg-v2 configs.

## Dome-renders-black flakiness (pre-existing, NOT caused by the jitter)

Some runs render the dome pitch black for the whole capture: background ~0 and ambient
fill lost, boxes still key-lit and still jittering. Controls with `jitter_dome=False`
(no per-frame dome writes at all) reproduce it (1 gray / 1 black across identical
commands), so the per-frame `.Set()` writes are NOT the trigger — this is the old
"Bug 2 ~60% all-black wall" init-time flakiness (that era the dome was the only light,
so a black dome meant a black frame). Observed rates (small samples): new code with dome
writes 1/5 gray; without 1/2 gray; old broken-graph code 3/3 gray. Disposition: re-run
the render; do not misattribute to the jitter.

## Probe recipe (reusable)

Temporarily call, after each `orchestrator.step()`, a dumper that records for each watched
prim: composed attr values (`prim.GetAttribute(n).Get()`) and, for each
`stage.GetLayerStack()` layer, `layer.GetAttributeAtPath("<prim>.<attr>")?.default`. One
fixed-camera render then answers "did the write happen, and into which layer?" — separating
*not-executed* from *shadowed* from *renderer-ignored*, which image diffing alone cannot.

# Store front-face check: is local −Y the aisle-facing front? (isaac_datagen)

**Status 2026-07-08: COMPLETE.** Built `debug_scripts/check_front_face.py`, rendered all
store SKUs from all 4 side faces at 1× and 8× lighting, and answered the M6-gate question
decisively: **the config's global `grasp_frame_policy_args: {face: "-Y"}` is correct for only
29/63 = 46% of SKUs.** The front face is a **per-SKU** property (not global, not even
per-category), and the darkness of "wrong" faces is **shelf occlusion, not dim lighting**
(confirmed by the 8× control). This unblocks M6 planning but redirects it: the fixed
`FixedFaceGrasp("-Y")` must be replaced by a per-SKU front-face selection.

## Why this was done

`store-usd-inverse-datagen.md` (M6, "Remaining milestone", blocker (a)) flagged the −Y
front-face assumption as user-gated, "cereal verified right (GUI check)". Before extracting
all products at −Y (which shoots each class's canonical reference image from that local face),
we needed to confirm −Y is the aisle-facing front for every category — a wrong face yields a
reference shot from the side/back. The prior "verification" was a manual Omniverse GUI check
on **cereal only** — which turns out to be the one category where −Y happens to be right.

User's insight (the method): render the LIVE store from each hypothesized grasp-frame outward
normal and look at whether the camera sees open aisle (front) or is buried in shelf geometry
(wrong face). Needs **no usdz export** — products are already in `store001.usd` in place; just
place a camera along the face normal and render the full-scene RGB.

## What landed

`+` **`src/isaac_datagen/debug_scripts/check_front_face.py`** NEW — the probe. One representative
prim per SKU class (first by sorted name = the prim `reference_catalog` would pick), boot once,
`load_store`, render the full store RGB from each of the 4 side faces, save per-tile PNGs +
one montage per category. Helpers (entry point is orchestration only):
- `class_representatives(store, patterns, facings)` — group `matched_products` by `parse_sku`
  class; per-class facing multiplicity (the shelf-duplicate count).
- `category_of(cls)` — `re.sub(r"\d+$","",cls)` (cereal001 → cereal) for the per-category sheets.
- `face_policies(spec, all_faces)` — default = the config's `FixedFaceGrasp` face; `--all-faces`
  = all 4 `FixedFaceGrasp(f)`.
- `v0_prim(store, model_path)` — the op-free `v_0` node, or **None + dump children** if the
  product isn't `model_*/v_0` (skips non-standard prims instead of crashing).
- `bbox_at_v0(v0)` → (lo,hi) via `untransformed_bbox_range`.
- `store_camera_pose(model_path, lo, hi, grasp, K, w, h)` — the verified invariant
  `cam2world = get_target2world([v_0])[0] @ ref_pose_from_grasp(grasp, lo, hi, K, w, h)`;
  authored as `cv2opengl(...)` on `/World/ref_cam`.
- `render_view(...)` — `set_prim_pose` → `warmup_render` → `orchestrator.step` → rgb annotator
  (mirrors `graspableobj_to_optflow_obj.render_one`; RGB is always the full frame).
- `whole_frame_luminance(rgb)` — reuses `measure_luminance.frame_luminance` (alpha forced
  opaque) for the black-process gate.
- `scale_store_lights(store, specs, factor)` — **fixed** (non-jittered) ×N intensity scale of
  the store's own UsdLux lights, same find + `UsdLux.LightAPI` + root-layer-`EditContext` seam
  as `scene.register_light_pattern_jitter`, applied once. Driven by `--light-scale`.
- `contact_sheet(records, cross_xy, out_png, cols)` — montage from saved tiles on
  `vision_core.viz.panel_grid`/`save_figure`, crosshair at the principal point (= the target
  under test, since the camera looks down the optical axis at the object centroid).

CLI: `uv run debug_scripts/check_front_face.py <store_config.yaml> <outdir> [--facings N]
[--all-faces] [--light-scale F] [key=val ...]` (trailing dotlist via `parse_known_args`).

## Results (the answer)

Rendered 63 classes × 4 faces = 252 tiles at 1×, then again at fixed 8× lighting.
Front face inferred objectively per SKU = **argmax whole-frame BT.709 luminance** over the 4
tiles (the aisle-facing face sits in the lit aisle; into-shelf faces are near-black).

**Front-face distribution (identical at 1× and 8×):** +X:8, +Y:16, −X:10, **−Y:29**.
**`−Y` hit-rate = 29/63 = 46%.** No single face works.

| Category | n | Dominant front | Spread |
|---|---|---|---|
| cereal | 2 | −Y (2/2) | the ONLY previously-verified category — and −Y IS right here |
| instant_beverages | 11 | −Y (10/11) | +Y:1 |
| detergent | 11 | +Y (7/11) | +Y:7, −Y:3, +X:1 — genuinely mixed |
| flour | 7 | +Y (5/7) | +Y:5, −X:1, −Y:1 |
| snack | 30 | mixed | −Y:13, −X:8, +X:7, +Y:2 |
| sauces | 2 | jars | +Y:1, −X:1 (wrap-labels, all faces lit) |

So the front face is **per-SKU**. −Y "worked" historically only because cereal (the sole
GUI-checked category) is a −Y category.

## 8× lighting control (rules out dim-bottom-shelf as the cause of darkness)

User's concern: are some faces dark because the product is on a dim bottom shelf (a real front
we'd miss), rather than occluded? Re-ran at **fixed 8×** (`--light-scale 8.0`, all 19 store
UsdLux lights scaled once, no jitter). Diff vs 1×:
- **58/63 (92%) front-face inferences UNCHANGED**; the distribution and 46% hit-rate are
  **identical**.
- **Dark faces stayed dark:** of 94 faces near-black (<3 luma) at 1×, **72 stayed dark** (<8)
  at 8× = occluded by shelf geometry; **22 rose only to ~14–25 luma** while the true front
  faces sit at **90–150** at 8× — so none were hidden labels.
- **The 5 flips are all near-tie cases**, not emerging labels: detergent003 (−Y↔+Y),
  detergent009 (all 4 close), sauces001 (jar, all faces >190 at 8×), snack005 (+X/−Y tied),
  snack007 (+X/−Y tied). All were already in the "ambiguous" bucket.

**Conclusion: the darkness is occlusion, not dim lighting.** The 46%/per-SKU finding is robust
to lighting.

### Caveat on the luminance heuristic
Brightness argmax is a strong SEED, not ground truth. It mis-picks when a bright neighbor/endcap
outshines the label face — e.g. **detergent010**: luminance chose a bright green +Y blur over
the actual +X Tide label (and its top-2 margin was wide, so a margin filter won't catch it).
**20/63 SKUs are ambiguous** (top-2 within 15 luma — wrap-label jars, competing faces). So the
auto-selected table needs human confirmation on those; the montages/tiles are the truth.

## Discoveries

- **415 product prims across 64 SKU classes** (not the ~15 we guessed). Avg ~6.5 shelf facings
  per class; e.g. cereal001 ×6, cereal002 ×5, flour005 ×15 (most). 414-vs-classes: the
  descriptor forward is per-class (64), but a per-prim extraction would be 415 usdz exports vs
  64 per-class — the redundancy argument is even stronger than at ~15.
- **`model_drink101` is a normal product whose version node is `v_69323`, not `v_0`** (its geo
  is under `model_drink101/v_69323`, an Xform). The probe skipped it (per user decision), but
  it IS extractable — the `v_0` lookup just needs generalizing to "find the `v_*` child". This
  matters because **Stage-A `extract_store_objects.extract_one` hardcodes `{model_path}/v_0`**
  and would skip/crash on drink101 the same way.
- **Per-process all-black path-tracer coin flip** (`render-darkness-investigation.md`, Bug 2,
  unsolved): the probe gates on first-frame whole-frame luminance and `sys.exit(3)`; a shell
  `while … rc==3 …` wrapper (capped) relaunches. Both real runs drew LIT processes on the first
  try (luma 37 at 1×, 94 at 8×).
- Renders are fast (~1.4 s/tile) — 252 tiles is ~6 min, not the ~20–30 min feared.
- Benign `PhysX foundLostAggregatePairsCapacity` warning at attach (documented in the store
  plan) — ignore.

## Problems faced → solutions

1. **argparse dropped the dotlist override after `--all-faces`.** A `nargs="*"` positional does
   not capture tokens following an optional. → `parse_known_args()`; leftover key=val tokens →
   OmegaConf dotlist (order-independent). Validated against the exact argv.
2. **drink101 crashed the run** (`AssertionError: no v_0`), taking a kit segfault with it after
   52 good tiles. → `v0_prim` returns None + dumps children; the run skips non-standard prims.
3. **A 63-class single montage is a 34 000 px-tall image.** → per-tile PNGs saved to `tiles/`
   (so any re-montage is free, no re-render) + one scannable sheet per category.

## Artifacts (all under `src/isaac_datagen/`, gitignored)

- `debug_out/` — 1× run: `{cereal,detergent,flour,instant_beverages,sauces,snack}.png` +
  `tiles/` (252) + `front_face_luma.csv` (per-SKU inferred front + 4 face lumas + top-2 margin).
- `debug_out_8x/` — the fixed-8× run, same layout.
- Montage column order in every sheet: **+X, +Y, −X, −Y** (so −Y is the rightmost column);
  rows = SKU classes in that category.

## Verification commands (cwd = src/isaac_datagen, GPU via CUDA_VISIBLE_DEVICES=1)

```bash
# Full all-faces check (63 classes; drink101 auto-skipped), with black-process relaunch guard:
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1
n=0; while :; do n=$((n+1)); \
  uv run debug_scripts/check_front_face.py configs/store001-optflow.yaml debug_out --all-faces \
    'scene_builder_args.product_patterns=[model_snack*,model_instant_beverages*,model_flour*,model_detergent*,model_sauces*,model_cereal*,model_drink101*]'; \
  rc=$?; { [ $rc -eq 3 ] && [ $n -lt 6 ]; } || break; done
# Fixed-8x lighting control (rule out dim bottom shelf): add  --light-scale 8.0  and outdir debug_out_8x
# Minimal faithful -Y-only check (~15 renders): drop --all-faces.  Placement variation: --facings 3
```

## Remaining / implications for Phase 2 (decisions open)

1. **Replace the fixed `FixedFaceGrasp("-Y")` with per-SKU front-face selection** — either a NEW
   grasp policy that auto-selects the aisle-facing face (brightness/occlusion argmax over a
   4-face probe, human-confirmed on the ~20 ambiguous SKUs — `front_face_luma.csv` is the seed,
   and `mesh_convert.prompt_winners`→`winners.yaml`→`finalize` is the existing confirm workflow),
   or a hand-curated per-SKU face table. Open: which.
2. **This also feeds the store-mutations plan** (`read-the-latest-plan-logical-knuth.md`): its
   `ReplaceClass.replacement_pose` aims a swapped-in object's grasp face where the original
   pointed, via `site.grasp = policy(lo,hi)` — so it inherits whatever front-face mechanism we
   pick. A wrong −Y would seat replacements facing into the shelf.
3. **drink101 (and any `v_*`-not-`v_0` product):** generalize the `v_0` lookup in
   `extract_one` (Stage A) to find the actual `v_*` child if drink101 is to be extracted.

## Relationship to other plans

- Resolves blocker (a) of `store-usd-inverse-datagen.md` M6 (the −Y front-face check) — but with
  a negative result that changes M6's design (per-SKU face, not fixed −Y).
- Reuses `store-usd-inverse-datagen`'s seams verbatim: `load_store`, `matched_products`/`parse_sku`,
  `grasp_policies.FixedFaceGrasp`/`face_grasp_frames`, the `cam2world = l2w(v_0) @ ref_pose`
  invariant, `register_light_pattern_jitter`'s light-find/scale seam (as a fixed one-shot).

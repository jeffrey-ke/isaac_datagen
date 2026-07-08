# Inverse datagen: extract optflow objects from an existing store USD

**Status 2026-07-08: COMPLETE (M0–M5 + M7), verified end-to-end** — 30-frame smoke passes
validate_obsmask + RoMa warp montages; store-light jitter proven at fixed camera pose
through the real capture path. **Follow-on: M6 scale-out only**, gated on two user
decisions (per-category front-face check in the GUI; whether `model_drink101` joins
product_patterns). This doc is the durable record of how an externally-authored store
USD was massaged into the optflow framework: every problem hit, what didn't work out
of the box, and the step-by-step fixes.

## Goal & architecture

Today's optflow datagen COMPOSES a scene from a catalog. The inverse: start from
`usds/store001.usd` (photorealistic store, products arranged on shelves), extract the
shelf-product prims into optflow objects, and capture optflow data INSIDE the store
scene, with the store's own lights jittered. Approved plan (with diffs) at
`~/.claude/plans/explore-how-we-can-validated-dijkstra.md`.

```
Stage A extract_store_objects.py        store001.usd ─(booted Isaac)→ GraspableObject catalog
Stage B graspableobj_to_optflow_obj.py  (+1-line grasp_point passthrough) → OptFlowObject catalog
Stage C build_store_scene (store_scene.py) via NEW scene_builders registry → SceneHandle
        → rest of optflow_generation (plan_capture/OptFlowWriter/capture_with_poses/
          finalize_metadata) UNCHANGED
```

One config drives all three stages: `configs/store001-optflow.yaml` —
`scene_builder: build_store_scene` + `scene_builder_args` (validated fail-loud by
`StoreSceneSpec`: store_usd / product_patterns / grasp_frame_policy+args).

**Frame-consistency invariant**: export product node P = `model_*/v_0` with P's own
xformOps neutralized (usdz-frame == v_0 local frame == modeling frame); at capture
`l2w = get_target2world([P_path])` at EXACTLY that node. `ref_pose` (camera2local,
OpenCV) then composes as `T_ref→obs = inv(cam2world) @ l2w @ ref_pose`.

## Key design decisions (user-driven evolution)

1. **meta['class'] = SKU stem** (`model_sauces001_6` → class `sauces001`, name
   `sauces001_6`); one reference per SKU class. `SKU_RE` in extract_store_objects.py.
2. **grasp-frame selection = swappable policy registry** `grasp_policies.py`
   (posers/placers `get(name)` idiom): policy(lo, hi) → (4,4) grasp SE3, +X = outward
   side-face normal (±Z fails loud — look_at singularity). First policy `FixedFaceGrasp(face)`.
   Stage-A-only concern (which face to SHOOT the reference from).
3. **grasp_point is CARRIED as a mandatory field on OptFlowObject** (user decision, 3rd
   iteration): NOT a `meta['grasp_policy']` record + capture-time config assert (guards a
   self-inflicted coupling — rejected), NOT runtime re-derivation from ref_pose (correct
   but pays inverse math every capture — rejected as intermediate draft). Derivation
   survives only as the one-time backfill tool for legacy catalogs. Capture authors
   GraspPoint child of P directly from `obj.grasp_point` (`add_catalog_grasp_frame`).
4. **Helpers take explicit handles** (user feedback): `load_store(spec)` returns the
   store root PRIM (STORE_ROOT="/World/Store" used only there);
   `matched_products(store, patterns)`, `extract_one(store, ...)`,
   `resolve_product_prim(store, obj)` are pure functions of their inputs; find_prims
   takes the prim directly (avoids its get_current_stage string-root branch).
5. **Store mode selection**: `RuntimeConfig.scene_builder: str = "build_scene"` registry
   key → `scene_builders.get(name)(runtime, objects)` at clean_datagen.py:154 (the one
   changed orchestrator line). Legacy configs untouched.
6. Provenance: extractor dumps `runtime.yaml` into the catalog dir (optflow_generation idiom).

### Post-approval decisions (user-driven, after the M5 smoke)

7. **Light-jitter factors are drawn LOG-uniformly** (uniform in photographic stops).
   User feedback: with uniform [0.5, 1.5] the fixed-pose frames looked only slightly
   different; they wanted frames spread across the whole dim→bright envelope without
   most frames looking like the extreme. A raw uniform draw over a wide ratio range
   (e.g. [0.25, 8]) puts half the frames above ~4× (mostly washed-out looks); light is
   perceptually multiplicative, so `exp(U(ln lo, ln hi))` makes a doubling equally
   likely anywhere in range. One-line change in `register_light_pattern_jitter`.
8. **intensity_scale_range widened [0.5, 1.5] → [0.25, 8.0]**, chosen from a rendered
   9-step calibration ladder (0.05×–8×, fixed pose, capture-grade rt_subframes; see
   discoveries: lighting floor + soft-clip ceiling). Range endpoints are config data —
   tighten per-run without code changes.
9. Broken pre-fix captures deleted rather than kept (render000/001 with cid_mask=0);
   fixed-pose diagnostic captures (render004: uniform [0.5,1.5] proof; render005:
   log-uniform [0.25,8] demo) are NOT training data — delete before any run that
   globs `render*`.

## What landed (all in isaac_datagen)

- `src/isaac_datagen/grasp_policies.py` NEW — registry + FixedFaceGrasp.
- `src/isaac_datagen/store_scene.py` NEW — StoreSceneSpec, load_store→store prim,
  resolve_product_prim, label_product + `_override_vendor_class_labels` (see problem 6),
  add_catalog_grasp_frame, build_store_scene.
- `src/isaac_datagen/scene_builders.py` NEW — {build_scene, build_store_scene} registry.
- `src/isaac_datagen/extract_store_objects.py` NEW — Stage-A CLI (parse_sku,
  matched_products, extract_one, main; placeholder 64×64 grey reference_image — Stage B
  re-renders it).
- `src/isaac_datagen/configs/store001-optflow.yaml` NEW — full 3-stage config
  (paths relative to src/isaac_datagen cwd; dataset_dir datasets/store001-optflow;
  cereal.* RegexFilter smoke subset; log-uniform light jitter [0.25, 8.0]).
- `src/isaac_datagen/isaac_utils.py` ~ — export_subtree_usdz gains
  `root_prim="/World"` + `neutralize_root_xform=True` kwargs (defaults preserve legacy
  callers); NEW `_localize_remote_assets` step 4b (see problem 3); NEW
  `untransformed_bbox_range`.
- `src/isaac_datagen/runtime_config.py` ~ — `LightJitterSpec` dataclass;
  `scene_builder`/`scene_builder_args`/`light_jitter_patterns` fields + __post_init__
  asserts.
- `src/isaac_datagen/clean_datagen.py` ~ — scene_builders import + registry dispatch line.
- `src/isaac_datagen/objects.py` ~ — `OptFlowObject.grasp_point: np.ndarray` mandatory
  field + docstring rewrite (serializers already handle ndarray).
- `src/isaac_datagen/graspableobj_to_optflow_obj.py` ~ — 1-line grasp_point passthrough.
- `src/isaac_datagen/debug_scripts/verify_centroid_ref.py` ~ — grasp_point kwarg.
- `src/isaac_datagen/debug_scripts/backfill_grasp_point.py` NEW — one-time legacy
  migration (RUN, all 4 catalogs backfilled).
- `src/isaac_datagen/scene.py` ~ — M4's `register_light_pattern_jitter`
  (UsdLux.LightAPI intensity scale factors vs authored base, LOG-uniform draw, rng
  stream [seed, 2, k], register_per_frame, schedule returned for lighting_log.json) +
  the make_replicator hookup loop.

## Discoveries about store001.usd (from live-kit spikes)

- Remote https subLayer (synthesis-multiverse.extwin.com) resolves fine inside Isaac
  (omni resolver); plain usd-core cannot fetch it. EULA: must run kit with
  `OMNI_KIT_ACCEPT_EULA=YES` (first headless run prompts interactively otherwise).
- Referencing `/root` (STORE_DEFAULT_PRIM; layer has defaultPrim=None) SPLICES its
  children directly under /World/Store → product paths are
  `/World/Store/model_cereal001/v_0` (NO `root/` segment). 446 children; also a
  `model_drink101` category beyond the six known product stems.
- **xformOps live on `model_*` (translate/scale/rotateZYX); `v_0` is op-free** — the
  modeling frame. neutralize_root_xform on v_0 is a no-op safety net; l2w at v_0 picks
  up the full placement. cereal001 = 10×9×15cm canister; cereal002 = 12×4×20cm box;
  meters, Z-up, resting z=0.
- Store lights: exactly 19 UsdLux prims at `/World/Store/model_store001/v_0/light/`
  (RectLight ×7 @5000, DiskLight ×12 @60000). The `E_light*` Xforms under E_ceiling are
  fixture GEOMETRY, not lights — and that geometry is EMISSIVE: with all 19 UsdLux
  lights scaled to 0.05×, the scene still renders at mean brightness ~52 (vs ~101 at
  1×) with labels legible. Jitter therefore has a brightness FLOOR (~0.1×, below which
  frames are nearly identical) and, with the fixed-exposure tonemapper
  (histogram/auto-exposure disabled), NO hard ceiling — 8× soft-bleaches highlights but
  clips zero pixels at 255.
- No /PhysicsScene composes in; ~70 *Joint* prims ride along in fixture subtrees
  (inert — capture never steps physics; PhysX logs one benign
  foundLostAggregatePairsCapacity warning at attach). Products carry
  PhysicsRigidBodyAPI + PhysicsMaterial in their subtrees (inert, ride into usdz).
- **Vendor semantics**: store assets author LEGACY-style class semantics
  (`semantic:Semantics_xxxx:params:{semanticType=class, semanticData=snack}`) on BOTH
  `v_0` AND the leaf mesh (`v_0/E_snack_1`) — the cause of problem 6 below.
- Every capture's FIRST frame renders ~9 mean-brightness units darker than its light
  schedule predicts (renderer warm-up state; survives warmup_render). Not a jitter
  bug; affects frame 0 of any capture equally.
- `iid_mask` = full-frame RAW segmentation ids is BY DESIGN (graspable selection lives
  in iid_to_name/cid_mask/alpha), NOT a bug.

## Problems faced → step-by-step solutions

Everything that did NOT work out of the box, in the order hit. (1)–(3) are about
getting data OUT of the store USD; (4)–(6) about aligning it with the framework's
contracts; (7)–(10) about proving the capture behaves; (11)–(14) operational footguns.

1. **Interactive EULA killed the first headless kit boot.** The first
   `SimulationApp` launch on this machine blocked on an interactive EULA prompt and
   died under nohup. → Prefix every kit run with `OMNI_KIT_ACCEPT_EULA=YES`.

2. **The store USD is unreadable outside a booted kit.** `store001.usd` is a thin
   34KB over-layer whose geometry+lights compose from a REMOTE https subLayer; plain
   `uv run --with usd-core` pxr cannot fetch it (no omni resolver). → All Stage-A
   extraction and any stage inspection run inside booted Isaac (`boot_sim` first);
   only the EXPORTED usdz (self-contained after fix 3) is usd-core-readable.

3. **usdz packaging failed on remote texture URLs.** `export_subtree_usdz`'s
   `UsdUtils.CreateNewUsdzPackage` failed with `Failed to map 'https://...png'`:
   the flatten KEEPS UsdShade bindings (contrary to the repo's older "textures don't
   survive flatten" belief — that's about re-import, not bindings) but leaves asset
   paths as remote URLs the zip writer can't ingest. Steps: reproduced on one SKU →
   inspected the flattened layer's asset paths → added step 4b
   `_localize_remote_assets(temp_usd)`: walk `UsdUtils.ModifyAssetPaths`, download
   each http(s) dependency via `omni.client.copy` (kit's resolver session) to a dep
   dir with collision-safe names (`{idx:03d}_{basename-sans-query}`), rewrite the
   layer to relative paths, save, THEN package. Verified: `unzip -l` shows texture
   payloads; pxr reads the bound `UsdUVTexture` out of the usdz; Stage-B reference
   render shows the printed label (not grey).

4. **Wrong path assumption: referencing `/root` splices its children.** Spike
   inspection code looked for `/World/Store/root/model_*` and found nothing —
   USD reference semantics splice the target prim's CHILDREN under the referencing
   prim, so products live at `/World/Store/model_*` directly. → Fixed inspection;
   the extractor was already correct because it computes `meta['store_prim']`
   RELATIVE to the store root prim (join key survives any mount point).

5. **Which node is "the object"? xformOps sat one level above the export node.**
   Products are `model_<sku>[_k]/v_0`; the M0 probe showed translate/scale/rotateZYX
   authored on `model_*` while `v_0` is op-free. Exporting `model_*` would have baked
   placement into the usdz (breaking `T_ref→obs = inv(cam2world) @ l2w @ ref_pose` by
   double-counting). → P = `v_0` (modeling frame, origin-centered);
   `neutralize_root_xform=True` (author EMPTY xformOpOrder on the export root — a
   local opinion beating the reference arc) as a safety net; capture reads l2w at
   exactly `v_0`, which picks up the ancestors' placement/scale via
   ComputeLocalToWorldTransform. Verified in-sim: `get_target2world(GraspPoint) ==
   l2w(P) @ grasp_point` to 2e-16 for all 11 objects.

6. **cid_mask all zeros — vendor semantics fight ours (the big one).** First M3
   captures had perfect iid/occlusion/visibility but cid_mask == 0 everywhere.
   Steps: (a) noticed idToSemantics classes like `'cereal001,snack'` — a comma-joined
   union that misses the exact `class_to_cid` lookup in `cid_iid_masks`; (b) first
   fix attempt `_override_vendor_class_labels` (rewrite legacy `semanticData` values
   in P's subtree to our class; value overrides win over the reference arc) —
   INSUFFICIENT alone; (c) built an in-kit probe (dump every `semantic*` attr under
   one product + one live `instance_segmentation_fast` frame) → smoking gun: v_0's
   NEW-style attr was `semantics:labels:class = [snack, cereal001]` — the vendor
   value had been merged into OUR label list; (d) read the installed
   `isaacsim.core.utils.semantics` + replicator functional source: `add_labels` →
   `F.modify.semantics(mode="replace")` MERGES the prim's composed legacy
   semanticData into the list it authors, and the annotator additionally UNIONS
   class labels across the subtree (legacy + LabelsAPI); also `remove_all_semantics`
   is deprecated AND uses `RemoveProperty`, which CANNOT delete opinions composed
   through a reference arc — so the "remove vendor semantics first" design would
   have silently no-opped; (e) fix = ORDERING in `label_product`: run the value
   override FIRST (every legacy value in the subtree == ours), then
   `remove_labels(include_descendants=True)` (stale LabelsAPI, idempotent re-runs),
   then our two `add_labels` — the merge now dedups to `[class]`. Verified: probe
   shows clean `idToSemantics class='cereal001'`; render002 cid histogram
   {2: cereal001, 3: cereal002} on 560–735k product px/frame; overlay montage lands
   exactly on catalog SKUs, non-catalog products stay unlabeled.

7. **Legacy catalogs would break on the new mandatory field.** `grasp_point` as a
   mandatory OptFlowObject field means `deserialize` fails on the 4 pre-existing
   catalogs. → big-bang backfill (the iid_to_visibility precedent):
   `backfill_grasp_point.py` recovers the frame from data the catalog already holds
   (`+X = -ref_pose[:3,2]` re-flattened horizontal, `+Z` up, origin = bbox face
   center along +X, bbox read from the usdz via pxr) and residual-writes
   `serialize(only={"grasp_point"})` (full re-serialize hits shutil.SameFileError on
   the usdz copy). amazon(44)/amazon-v2(10)/kleenex(7)/ycb(7): alignment 1.0
   everywhere incl. ycb's patch_grasp_frames-rotated members; deserialize smoke
   tests pass.

8. **Light-jitter had a historical silent-no-op failure mode to rule out.**
   `.docs_claude/lighting-jitter-mechanism.md`: per-frame light modifies via the
   Replicator graph (`rep.randomizer.register` + `rep.modify`) NEVER execute. The
   new `register_light_pattern_jitter` uses the proven `register_per_frame` direct-
   USD-write route (root-layer `Usd.EditContext` so overrides beat the store's
   reference arc), but proof was still owed. Steps: (a) naive check — correlation of
   factor vs frame mean brightness on a real 4-frame capture — was POSE-CONFOUNDED
   (+0.17, useless); (b) in-kit fixed-camera scale sweep 0.0/0.5/1.0/1.5 → mean
   brightness 26.6/47.3/62.9/77.5, monotone, lights-off darkens to 42% — writes
   reach the renderer; (c) end-to-end proof through the REAL capture path: collapse
   `LookAtPoser` ranges to zero width (`xrange=[0.6,0.6]` etc. — uniform(lo,lo) is
   deterministic) → 8-frame capture at cam2world spread EXACTLY 0; brightness
   rank-matches lighting_log scale_factors on frames 1–7 (corr +0.96). Frame 0 is
   the warm-up outlier (discovery above).

9. **Uniform factor draws don't match perception (post-approval user feedback).**
   With uniform [0.5, 1.5] the fixed-pose frames looked barely different; naively
   widening to [0.25, 8] uniform would make half the frames brighter than 4×.
   Steps: rendered a deterministic 9-step calibration ladder (0.05×–8×, fixed pose,
   rt_subframes=20) to find the visual envelope → floor at ~0.1× (emissive fixtures),
   soft-bleach at 8×, no hard clipping → switched the draw to log-uniform
   (`exp(U(ln lo, ln hi))`) and set [0.25, 8.0]; verified with a 10-frame fixed-pose
   capture (render005): draws 0.34–7.64 spread across the envelope, extreme looks
   are rare, dimmest frames keep labels legible.

10. **Verification tooling gaps.** `OptFlowSample.visualize` imports `romatch`,
    which is not in the default venv — it's the repo's own optional extra: run
    `uv run --extra viz python debug_scripts/viz_optflow.py <render_dir> --idx N`.
    `viz_sample.py` is phase-3-only (PreImageInlierSample) — not applicable to
    optflow dirs; `validate_obsmask.py` IS applicable (ObsMask serializes flat) and
    passes 30/30 frames.

11. **Kit eats un-flushed stdout on fastShutdown.** Probe prints vanished. → always
    `print(..., flush=True)` in-kit, or redirect the whole run to a file.

12. **The interactive shell's cwd resets between calls.** Relative-path launches
    (`clean_datagen.py`, `datasets/...`) intermittently hit "No such file". → always
    `cd /home/jeffk/repo/refseg-workspace/isaac_datagen/src/isaac_datagen &&` or use
    absolute paths; all documented commands assume that cwd.

13. **grep-based log filtering backfired.** Filtering a live kit log with
    `grep -E "ratio|..." | head` matched "gene**ratio**n"/"configu**ratio**n" in
    deprecation spam AND `head` closed the pipe early (SIGPIPE risk to the probe).
    → redirect run output to a log file first, grep the file afterwards.

14. **Misread a montage label, nearly filed a fake bug.** A downscaled catalog
    montage appeared to show `cereal002_2 (class cereal001)`; the meta yaml on disk
    says `class: cereal002`. → check the serialized artifact, not the thumbnail.

## Verification commands (cwd = src/isaac_datagen, GPU via CUDA_VISIBLE_DEVICES=1)

```
# Stage A (11 cereal):
OMNI_KIT_ACCEPT_EULA=YES uv run python extract_store_objects.py configs/store001-optflow.yaml \
    datasets/store001-objects-cereal 'scene_builder_args.product_patterns=[model_cereal*]'
# Stage B:
OMNI_KIT_ACCEPT_EULA=YES uv run python graspableobj_to_optflow_obj.py \
    configs/store001-optflow.yaml datasets/store001-objects-cereal datasets/store001-optflow-objects
# Stage C capture (idx bumps render dir):
OMNI_KIT_ACCEPT_EULA=YES uv run python clean_datagen.py configs/store001-optflow.yaml \
    idx=0 num_targets=3 num_frames=10
# Fixed-pose lighting diagnostic (collapse the pose distribution to a point):
OMNI_KIT_ACCEPT_EULA=YES uv run python clean_datagen.py configs/store001-optflow.yaml idx=5 \
    num_targets=1 num_frames=10 'pose_generation_policy_args.xrange=[0.6,0.6]' \
    'pose_generation_policy_args.yrange=[0.0,0.0]' 'pose_generation_policy_args.zrange=[0.1,0.1]'
# Post-capture checks:
uv run python validate_obsmask.py datasets/store001-optflow/render000
uv run --extra viz python debug_scripts/viz_optflow.py datasets/store001-optflow/render000 --idx 0
# Legacy backfill (already run; idempotent):
uv run --with usd-core python debug_scripts/backfill_grasp_point.py ../../assets/optflow_objects/{amazon,amazon-v2,kleenex,ycb}
```

Existing artifacts (all under datasets/, gitignored): store001-objects-smoke (1 obj),
store001-objects-cereal (11), store001-optflow-objects-smoke (1 ref),
store001-optflow-objects (11 refs), store001-optflow/render000 (M5 smoke: 3 targets ×
10 frames), render002 (M3 tiny), render003 (M4 4-frame), render004+render005
(fixed-pose lighting diagnostics — NOT training data, delete before globbing render*).

## Remaining milestone

- **M6 (user-gated scale-out)**: full 414-product extraction (one-time, slow) +
  Stage B; per-capture SKU subsets via filter_specs only. Keep captures filtered
  (414-class descriptor forward is large). Blocked on two user decisions:
  (a) the -Y front-face assumption per category (GUI check — cereal verified right);
  (b) whether `model_drink101` joins product_patterns.

## Session memories saved (auto-memory, for context)

- derive-from-data-not-rerun-assert (updated: carry field on owning sample preferred).
- helpers-take-explicit-handles (matched_products feedback).

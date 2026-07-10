# Per-class front faces → one-frame-per-object in-store verification dataset (isaac_datagen)

**Status 2026-07-09: COMPLETE.** Ran the store-USD M6 bulk extraction (283 objects / 42 curated
classes, per-class aisle-facing faces baked at Stage A) and rendered a one-frame-per-facing in-store
verification dataset. A read-only pose-vs-catalog check (below) proved capture order == catalog
order and surfaced 14 store-authoring quirks — 12 label-inward facings (fronts point into the
gondola, unphotographable in-store) + 2 coincident-duplicate prim pairs — dropped via a new
`RemovePrims` mutation. Final deliverable: `datasets/store001-optflow-verify2/render000`
(**269 frames**, gate 269/269 pose + 269/269 target-visible) with per-category montages; the
283-object Stage-A/B catalog is untouched and reusable for training. Detail: the three post-plan
sections (**EXECUTION STATUS**, **RESOLVED**, **FIX PLAN / FIX EXECUTED**) below.

## Context

`store-front-face-check.md` (completed 2026-07-08) proved the store config's global
`FixedFaceGrasp("-Y")` is the aisle-facing front for only **29/63 = 46 %** of SKU classes —
the front face is **per-SKU**. The user then eyeballed the per-category montages and produced a
**hand-curated keep-map**: 42 SKU classes to keep, each tagged with its correct front-face axis
(+Y / −Y / −X); every class *not* named is to be dropped. The map lists only what to keep, and
the ~18 dropped snack SKUs are **not enumerated**, so the drop must be *keep-list-driven*, not a
per-class `RemoveClass`.

The store catalog does **not exist yet** — `datasets/store001-objects` is empty and
`store001-optflow-objects` holds only the 11 cereal smoke objects. The M6 bulk extraction was
explicitly gated on exactly the front-face question the user just answered. So this plan runs
that extraction, scoped to the 42 kept classes with their curated faces, and renders an
**in-store, one-frame-per-object, every-facing** verification dataset with the 21 unmentioned
classes deactivated — so the user can eyeball that (a) every grasp frame aims at the labelled
front and (b) the extracted optflow objects look good, before committing to a full training run.

User decisions (this session): **in-store shelf capture** (not isolated reference renders) as
the verification dataset; **every facing** (per-instance, ~250 prims), not one-per-class.

## The three stages, in plain terms — and why the order can't change

The store `.usd` is one big scene with every product glued onto its shelf. Getting from that to a
checkable dataset takes three passes, and each pass needs the previous one's output:

- **Stage A — extract (cut each product out).** Walk the store; for each kept product, save *its
  geometry alone* as its own `.usdz` file and write down *which face is the front* (from your
  keep-map). Output: a pile of bare objects — shape + front-face, **no picture yet**. The front
  face is chosen **here and only here**; the two later stages just re-use it.

- **Stage B — render the reference (photograph each product alone).** Take each bare object from
  Stage A, drop it into an empty scene by itself, and shoot one clean photo *from its front face*
  (plus depth + camera pose). That photo is the single "this is what the product looks like"
  reference the whole matching pipeline compares against. Output: the reference catalog.

- **Stage C — in-store capture (photograph each product back on the shelf).** Load the whole store
  again, switch off the 21 unwanted classes, and take **one photo per kept product from its
  front-face viewpoint, on the shelf, in context** — the verification dataset. Every frame pairs
  the on-shelf photo with that product's Stage-B reference photo, so you can see both at once.

**Why the order is forced (each output is the next input):**
- A → B: B can only photograph an object *after* A has cut it out into its own `.usdz`, and it aims
  the camera using the front-face A recorded.
- B → C: C binds each catalog object back onto its shelf prim and the capture writer records that
  object's *reference photo* (made in B) next to the on-shelf photo — so B must exist first. C also
  reads the front-face grasp frame, which was baked in A and carried through B untouched.

**Why A and B are split** (not one step): A only *exports geometry* — fast, no rendering, runs on
the store scene. B *path-traces images* — slow, runs on isolated objects. Splitting them means a
reference photo can be re-shot without re-exporting geometry, and a wrong front-face can be re-baked
in A without redoing everything. This is the repo's existing three-stage store pipeline; we are not
changing its shape, only (1) teaching Stage A a per-class front face and (2) making Stage C shoot
exactly one frame per object with the unwanted classes removed.

**"Didn't our extract-from-USD work already do extraction and rendering in one step?"** No —
nothing about the process or our requirements changed; the store pipeline (`store-usd-inverse-datagen`)
always split them. Stage A (`extract_store_objects.py`) writes a grey **placeholder** image and never
renders; Stage B (`graspableobj_to_optflow_obj.py`) does the rendering, on a fresh isolated stage. You
may be thinking of the earlier **mesh→object** tool (`mesh_blender.py`), which *did* produce the usdz
and its reference tiles in the *same* Blender pass — because a mesh is already a standalone file, so
there is nothing to cut out. A **store** product is different: it is glued inside the full store scene,
so it must first be *cut out* into its own `.usdz` (Stage A) before it can be photographed alone
(Stage B) — you can't shoot a clean reference of it while it is still buried on the shelf next to its
neighbours. So the split is inherent to extracting from a pre-authored store, and predates this plan.

## How it flows (single source of truth = the 42-entry `{class: face}` table)

```
Config A  store001-optflow-keep.yaml   →  Stage A extract  +  Stage B render
   product_patterns = 42 keep-globs         (PerClassFaceGrasp bakes each class's
   grasp_frame_policy = PerClassFaceGrasp     curated face into grasp_point per prim)
        ↓  datasets/store001-optflow-objects-keep   (42 classes × all facings ≈ 250 OptFlowObjects)
Config B  store001-optflow-verify.yaml  →  Stage C build_store_scene → capture
   product_patterns = BROAD (all cats)       RemoveUntrackedProducts deactivates every
   mutations = [RemoveUntrackedProducts]      store product whose class ∉ the 42-class catalog
   num_targets = null, num_frames = 1         plan_capture → one frame per tracked object,
   pose ranges collapsed to a point           camera one standoff out along the grasp +X normal
        ↓  datasets/store001-optflow-verify/render000   (≈250 in-store frames, one per facing)
```

The face lives in `grasp_point`, baked at **Stage A**; `build_store_scene` only *replays* it
(`add_catalog_grasp_frame`), so Config B needs no face table — it reuses the baked catalog.

## Prior art honored (from the mandated scan)

- **`grasp_policies.py` was built as a registry anticipating >1 policy** (`store-usd-inverse-datagen.md`);
  `FixedFaceGrasp` is policy #1. Adding `PerClassFaceGrasp` is the intended extension, not a flag
  (`no-flag-driven-variants`, `explicit-variant-selector-no-guess`).
- **`store-scene-mutations.md` is COMPLETE and landed** — `RemoveClass`/`deactivate_prim`/
  `active_products`/`_drop_under` all exist. `RemoveUntrackedProducts` is a thin new registry
  entry reusing them; that plan flagged the keep-list-complement need itself.
- **Face convention:** `face_grasp_frames` → +X = outward side-face normal, +Z = up; **±Z is a
  hard `look_at` singularity** — the curated map uses only ±X/±Y, so it is safe.
- **Viz already exists** (`viz-primitives-to-vision-core`, `optflow-centroid-ref-and-visualize`):
  `viz_optflow.py` (render-dir frames) and `viz_optflow_objects.py` (per-object catalog QA) —
  **no new viz code**.

---

## Change 1 — `grasp_policies.py`: `PerClassFaceGrasp` + widen `__call__` to `(lo, hi, cls)`

The policy `__call__` is today blind to class identity, but the class name is already in scope at
every call site. Widen the contract by one positional; `FixedFaceGrasp` ignores it. Verified: the
three sites below are the ONLY grasp-policy `__call__` calls (`patch_grasp_frames.py` /
`backfill_grasp_point.py` open-code `face_grasp_frames`, they don't call a policy).

```python
~ header docstring:
-   policy(**args)(lo, hi) -> (4, 4) SE3 grasp frame in OBJECT-LOCAL (usdz) frame,
+   policy(**args)(lo, hi, cls) -> (4, 4) SE3 grasp frame in OBJECT-LOCAL (usdz) frame,

~ class FixedFaceGrasp:
-     def __call__(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
-         return face_grasp_frames(lo, hi)[self.face]
+     def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
+         return face_grasp_frames(lo, hi)[self.face]      # cls ignored: one fixed face for all

+ class PerClassFaceGrasp:
+     """Per-SKU-class front-face policy: the grasp frame of each class's hand-curated
+     aisle-facing side face (from the store front-face check). faces = {class: face},
+     face in FACE_NORMALS. Fail-loud on a class not in the table — never guess a face;
+     the caller must only pass classes it covers (product_patterns must match the keys)."""
+     def __init__(self, faces: dict):
+         assert faces, "PerClassFaceGrasp needs a non-empty {class: face} table"
+         bad = {c: f for c, f in faces.items() if f not in FACE_NORMALS}
+         assert not bad, f"faces must be side faces {sorted(FACE_NORMALS)} (±Z singular): {bad}"
+         self.faces = dict(faces)
+     def __call__(self, lo: np.ndarray, hi: np.ndarray, cls: str) -> np.ndarray:
+         assert cls in self.faces, \
+             f"PerClassFaceGrasp: no face for class {cls!r} (keys={sorted(self.faces)})"
+         return face_grasp_frames(lo, hi)[self.faces[cls]]
```

Thread `cls` at the three call sites — `cls` is verified in scope at each:

```python
~ extract_store_objects.py:71  (cls parsed at :56)
-         grasp_point=policy(lo, hi).astype(np.float32),
+         grasp_point=policy(lo, hi, cls).astype(np.float32),

~ store_mutations.py:116,121  (measure_site — class was discarded as `_`)
-     name, _ = parse_sku(prim.GetName())
+     name, cls = parse_sku(prim.GetName())
      ...
-                        l2w=get_target2world([v0_path])[0], grasp=policy(lo, hi))
+                        l2w=get_target2world([v0_path])[0], grasp=policy(lo, hi, cls))

~ debug_scripts/check_front_face.py:232  (cls from `for cls, name, model_path in reps`)
-             grasp = policy(lo, hi)                              # (4,4) local grasp SE3, +X = outward
+             grasp = policy(lo, hi, cls)                         # (4,4) local grasp SE3, +X = outward
```

*Cosmetic note:* `check_front_face.py:103` derives the montage tile label as
`grasp_frame_policy_args.get("face", grasp_frame_policy)`; under `PerClassFaceGrasp` (no `face`
arg) that label falls back to the literal `"PerClassFaceGrasp"`. Rendered views are unaffected
(the `policy(lo,hi,cls)` call still selects the right face); the probe's normal mode is
`--all-faces` anyway. Leave as-is.

## Change 2 — `store_mutations.py`: `RemoveUntrackedProducts` (keep-list-complement drop)

Registers automatically via `store_mutations.get` (reflection); reuses `active_products` /
`parse_sku` / `deactivate_prim` / `_drop_under`. Products are direct children of the store root,
so `active_products` sees them; untracked prims carry no binding, so `targets` pass through intact
and the `build_store_scene:153` `IsActive()` assert on the kept 42 still holds.

```python
+ class RemoveUntrackedProducts:
+     """Deactivate EVERY active store product (scoped to spec.product_patterns) whose SKU
+     class is NOT among the tracked targets — the keep-list COMPLEMENT. Drops the SKUs a
+     curated keep-list leaves out in one spec, without enumerating them (the front-face
+     keep-map lists only what to keep). Store-wide, like RemoveClass; untracked prims have
+     no CaptureTarget binding, so targets pass through unchanged."""
+     def __call__(self, store, spec, targets, rng):
+         from isaac_datagen.isaac_utils import deactivate_prim
+         tracked = {t.obj.meta["class"] for t in targets}
+         removed = []
+         for prim in active_products(store, spec.product_patterns):
+             if parse_sku(prim.GetName())[1] not in tracked:    # [1] = SKU class
+                 removed.append(prim.GetPath().pathString)
+                 deactivate_prim(prim)                          # model_* root, not v_0
+         print(f"[MUT] RemoveUntrackedProducts: deactivated {len(removed)} untracked "
+               f"product(s), kept {len(tracked)} classes", flush=True)
+         return _drop_under(targets, removed)
```

## Change 3 — capture: `num_targets: null` ⇒ every grasp frame once

`num_targets` is consumed only by `plan_capture` (real path) and `debug_scene.py` (dry-run,
`build_scene` only); no `__post_init__` positivity assert exists. **Do NOT add a default** — the
field sits *before* required (no-default) fields, so `= None` would raise
`TypeError: non-default argument follows default argument` at class-definition time. Widen the
**annotation only**; it stays required-in-yaml (OmegaConf mandatory-missing), and every existing
config already supplies it.

```python
~ runtime_config.py:37
-     num_targets: int
+     num_targets: int | None       # int => sample N grasp targets (train); null => every scene
+     #                               grasp frame ONCE (still REQUIRED in yaml; NO default here)

~ capture.py:45  (plan_capture — THE shared pose computation both real + dry-run use)
-     idx = np.random.choice(len(scene.grasp_points), size=runtime.num_targets)
+     idx = (np.arange(len(scene.grasp_points)) if runtime.num_targets is None
+            else np.random.choice(len(scene.grasp_points), size=runtime.num_targets))

~ debug_scripts/debug_scene.py:58  (build_scene dry-run; never None today, but keep green)
-     idx = np.random.choice(n, size=runtime.num_targets)
+     idx = np.arange(n) if runtime.num_targets is None else np.random.choice(n, size=runtime.num_targets)
```

## Change 4 — no code: Config B collapses the poser to a single reference viewpoint

`LookAtPoser` with `xrange:[0.6,0.6] yrange:[0,0] zrange:[0,0]` + `num_frames:1` yields one
deterministic pose per target: the camera sits 0.6 m out along the grasp **+X** (outward) normal
and looks at the grasp-frame origin (bbox face-center). Verified: `generate_random_offsets` →
`np.random.uniform(low,low,...)` returns the constant `[0.6,0,0]` (no error); `num_frames:1`
satisfies the `num_frames ^ grid_dims` XOR.

---

## Config A — `configs/store001-optflow-keep.yaml` (Stage A + Stage B)

Full copy of `store001-optflow.yaml` (no `base:` include exists); only product/grasp/catalog
fields differ. The `{class: face}` table is the single source of truth and its keys match
`product_patterns` (per-class globs where a category is partly kept; category globs where fully
kept). `SKU_RE` guarantees 3-digit classes, so `model_snack012*` matches only the `snack012`
family; the Stage-A dupe-name assert (`:89`) catches any accidental double-match.

```yaml
~ dataset_dir: datasets/store001-optflow-keep          # (Stage A/B out dirs are CLI args)
~ scene_builder_args:
    store_usd: ../../../usds/store001.usd
    product_patterns:                                   # the 42 KEEP globs (fully-kept cats collapse)
      - "model_snack012*"  - "model_snack017*"  - "model_snack020*"  - "model_snack023*"
      - "model_snack025*"  - "model_snack027*"  - "model_snack031*"  - "model_snack032*"
      - "model_snack033*"  - "model_snack034*"  - "model_snack035*"  - "model_snack036*"
      - "model_sauces001*"
      - "model_instant_beverages001*" - "model_instant_beverages002*" - "model_instant_beverages003*"
      - "model_instant_beverages004*" - "model_instant_beverages005*" - "model_instant_beverages007*"
      - "model_instant_beverages009*" - "model_instant_beverages010*" - "model_instant_beverages011*"
      - "model_flour*"  - "model_detergent*"  - "model_cereal*"      # all classes kept in these 3
    grasp_frame_policy: PerClassFaceGrasp               # NEW registry entry (Change 1)
    grasp_frame_policy_args:
      faces:                                            # 42 entries — the hand-curated keep-map
        snack012: "+Y"    snack017: "-Y"    snack020: "-Y"    snack023: "-Y"    snack025: "-Y"
        snack027: "-Y"    snack031: "-Y"    snack032: "+Y"    snack033: "-Y"    snack034: "-Y"
        snack035: "-Y"    snack036: "-Y"
        sauces001: "+Y"
        instant_beverages001: "-Y"   instant_beverages002: "-Y"   instant_beverages003: "-Y"
        instant_beverages004: "+Y"   instant_beverages005: "+Y"   instant_beverages007: "-Y"
        instant_beverages009: "-Y"   instant_beverages010: "-Y"   instant_beverages011: "-Y"
        flour001: "-X"    flour002: "+Y"    flour003: "+Y"    flour004: "+Y"    flour005: "+Y"
        flour006: "+Y"    flour007: "+Y"
        detergent001: "+Y"   detergent002: "+Y"   detergent003: "-Y"   detergent004: "+Y"
        detergent005: "+Y"   detergent006: "+Y"   detergent007: "+Y"   detergent008: "-Y"
        detergent009: "-Y"   detergent010: "-Y"   detergent011: "-Y"
        cereal001: "-Y"   cereal002: "-Y"
~ objects_path: [datasets/store001-optflow-objects-keep]
~ filter_specs: []                                      # keep ALL 42 (no cereal smoke subset)
# YAML note: `faces` must be authored one `key: "value"` per line — the multi-per-line layout
# above is compaction for the plan. Also: every keep-glob MUST match ≥1 store prim or Stage A's
# find_prims raises (fail-loud, intended) — the 42-glob list must be exact.
```

## Config B — `configs/store001-optflow-verify.yaml` (Stage C, the deliverable)

```yaml
~ dataset_dir: datasets/store001-optflow-verify
~ num_targets: null                                     # every tracked object ONCE (Change 3)
~ num_frames: 1                                          # one pose per object
~ scene_builder_args:
    store_usd: ../../../usds/store001.usd
    product_patterns: ["model_snack*", "model_instant_beverages*", "model_flour*",
                       "model_detergent*", "model_sauces*", "model_cereal*", "model_drink101*"]
    #  ^ BROAD: RemoveUntrackedProducts must SEE the dropped SKUs (incl. drink101) to deactivate
    #    them; active_products does NOT raise on an empty pattern, so broad is safe here.
    grasp_frame_policy: FixedFaceGrasp                  # UNUSED at Stage C (grasp_point is replayed
    grasp_frame_policy_args: {face: "-Y"}               #   from the baked catalog); trivial validator
    mutations:
      - {name: RemoveUntrackedProducts}                 # drop the 21 unmentioned classes (Change 2)
~ objects_path: [datasets/store001-optflow-objects-keep]
~ filter_specs: []                                      # track all 42 catalog classes
~ pose_generation_policy: LookAtPoser                   # collapsed to a single reference viewpoint
  xrange: [0.6, 0.6]
  yrange: [0.0, 0.0]
  zrange: [0.0, 0.0]
```

*(Both configs are full copies, matching the existing `store001-optflow-{remove,replace}.yaml`.
Nothing asserts Stage A and Stage C share `product_patterns`; the only cross-stage join is
`meta["store_prim"]` via `resolve_product_prim`, independent of Stage C's patterns.)*

---

## Verification (cwd = `isaac_datagen/src/isaac_datagen`, GPU 1, EULA accepted)

```bash
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1

# 0. Unit smoke (no kit): the new policy + mutation import and behave
uv run python -c "from isaac_datagen.grasp_policies import PerClassFaceGrasp; import numpy as np; \
  p=PerClassFaceGrasp({'cereal001':'-Y'}); print(p(np.zeros(3), np.ones(3), 'cereal001').shape)"   # (4, 4)

# 1. Stage A — extract the 42 kept classes (all facings) with per-class front faces
uv run extract_store_objects.py configs/store001-optflow-keep.yaml datasets/store001-objects-keep
#    expect one "[NNNN] <name> (class <cls>) extracted" per facing (~250); no KeyError / no find_prims raise

# 2. Stage B — render one isolated reference per extracted object (the catalog)
uv run graspableobj_to_optflow_obj.py configs/store001-optflow-keep.yaml \
    datasets/store001-objects-keep datasets/store001-optflow-objects-keep
#    GUARD: check the first few reference_image PNGs are NOT black (per-process all-black RTX
#    coin-flip, store-front-face-check.md); if black, re-run Stage B.

# 3. Eyeball the EXTRACTED OBJECTS (goal b) — per-object ref RGB | depth | 3D grasp pose
uv run --extra viz debug_scripts/viz_optflow_objects.py datasets/store001-optflow-objects-keep

# 4. Stage C — in-store, ONE frame per object, drop the 21 unmentioned classes
mkdir -p datasets/store001-optflow-verify
isaac-datagen configs/store001-optflow-verify.yaml idx=0
#    expect ONE "[MUT] RemoveUntrackedProducts: deactivated N ..." line; render000 ≈ 250 frames

# 5. Eyeball the IN-STORE GRASP FRAMES (goal a) — per-frame [cid|iid] + [ref|obs] correspondence
uv run --extra viz debug_scripts/viz_optflow.py datasets/store001-optflow-verify/render000
#    then per-category contact sheets from the emitted PNGs (no new code):
#      montage datasets/store001-optflow-verify/render000/viz/viz_*.png -tile 6x -geometry +2+2 sheet.png
#    (group by category = class with trailing digits stripped; class_to_name is printed by the tool)

# 6. Serialized-artifact check: render000 cid_to_class contains the 42 kept classes and NONE of the
#    dropped ones; a spot obs frame shows the kept object facing the camera (label visible, not shelf).
```

## Risks / notes

- **Per-process all-black RTX coin-flip** (`render-darkness-investigation.md`, unsolved): Stage B and
  Stage C each render in one process; a black boot blanks every frame. Mitigation: eyeball the first
  output, re-run. Not gating it here (out of scope).
- **Fixed 0.6 m standoff** does not auto-frame like Stage B's `ref_pose_from_grasp` (bbox-sized
  standoff); big SKUs may crop, tiny ones sit small. Fine for a front-vs-shelf eyeball; bump the
  collapsed `xrange` if framing is poor.
- **A rotated facing** of a kept class will render "wrong" under its class-wide face — that is the
  signal this per-instance pass is designed to surface; handle offenders after review.
- **Viz deps / marking:** `viz_optflow.py` imports `romatch` (the `--extra viz` optional dep) for the
  GT ref→obs warp. It does not draw a dot at the exact grasp pixel (the class-correspondence overlay
  marks the object well enough); a precise grasp-pixel marker = new code (project `grasp_point` through
  `ref_pose`/`ref_intrinsics`, pass `points=`), out of scope.
- **`model_drink101`** is dropped by Config B; its `v_69323` topology is irrelevant to deactivation.
  It is not extractable by Stage A's `/v_0` hardcode and is not in the keep-map anyway.

## Execution & deviation

Implementation delegated to **Sonnet subagents**, each armed with this plan + the relevant change's
diff; the orchestrator reviews, runs the verification ladder, and reports. Changes 1–3 are tiny and
independent (each leaves the tree green); the two configs and the render runs follow. **If anything
unexpected appears** — `PerClassFaceGrasp` KeyError on a matched prim (patterns/table drift), Stage A
finding a kept class with no `/v_0`, `find_prims` raising on a keep-glob that matches nothing,
`RemoveUntrackedProducts` deactivating a kept facing, an empty/near-black catalog, or `num_targets:
null` tripping a code path not surveyed — **STOP and ask before deviating.**

## Key changes

`~` `grasp_policies.py` (`+ PerClassFaceGrasp`; widen `__call__(lo,hi)` → `(lo,hi,cls)`) ·
`~` `extract_store_objects.py:71`, `store_mutations.py:116,121`, `check_front_face.py:232`
(thread `cls`) · `+` `store_mutations.py::RemoveUntrackedProducts` ·
`~` `runtime_config.py:37` (`num_targets: int | None`, no default) · `~` `capture.py:45` +
`debug_scripts/debug_scene.py:58` (`None` ⇒ every grasp frame once) ·
`+` `configs/store001-optflow-keep.yaml` (Stage A/B: 42 keep-globs + 42-entry face table) ·
`+` `configs/store001-optflow-verify.yaml` (Stage C: broad patterns + `RemoveUntrackedProducts`,
`num_targets:null`, `num_frames:1`, collapsed poser) · Stage-A/B/C code paths, writers,
`build_store_scene`: UNCHANGED · viz: reuse `viz_optflow_objects.py` + `viz_optflow.py`, no new code.

---

# EXECUTION STATUS (2026-07-09)

**Code + configs: DONE and verified.** All 6 tracked diffs + `check_front_face.py:232` match the
plan; no-kit smoke passed (`PerClassFaceGrasp` returns (4,4); both configs load; `RemoveUntrackedProducts`
resolves). Header comments in both new configs corrected to their real roles.

**Full pipeline: RAN, exit 0 on all three stages (GPU 1, this is a 2×RTX-4090 box, not snuff).**
- Stage A: **283 objects / exactly 42 classes** extracted, per-class faces baked, no errors (~4 min).
- Stage B: **283 references, all lit** (min luma 108, none black). Spot-checked faces correct:
  cereal001(−Y), detergent001(+Y), flour001(−X) all show branded FRONTS. (~? min)
- Stage C: **283 in-store frames**; `[MUT] RemoveUntrackedProducts: deactivated 132 / kept 42`
  (132+283 = 415 = known store prim count); `cid_to_class` = **exactly the 42 keep-map classes,
  zero dropped leaked** (~9 min).
- Datasets (all under `isaac_datagen/src/isaac_datagen/`, gitignored):
  `datasets/store001-optflow-verify/render000` (in-store), `datasets/store001-optflow-objects-keep`
  (Stage-B catalog), `datasets/store001-objects-keep` (Stage-A geometry).

**KNOWN framing caveat (measured):** fixed 0.6 m standoff renders the target tight — median ~45%
of frame, up to 86% for big boxes (snack012/snack032); min ~10% (sauces001). Faces still verifiable.
Config-only fix: bump `xrange` (e.g. `[1.2,1.2]`) and re-run **Stage C only** (Stage A/B reused).

## RESOLVED (2026-07-09) — obs_NNNN vs reference_NNNN mismatch: 12 backward-placed store facings

**Symptom (user):** `obs_0214.png` = Williams-Sonoma "Pumpkin Quick Biscuit Mix" (an ORGANIC-FLOUR
product); `reference_image_0214.png` = Doves Farm "Crispy chocolate & rice" bar. Different products.
User adds: `reference_image_0086` IS that Williams-Sonoma biscuit mix, and it fills a whole shelf
in-store (a dense biscuit-mix + scone-mix wall) — i.e. the biscuit-mix product already has its own
catalog entry (~idx 86), yet it dominates the frame nominally indexed 214.

**Facts gathered (verified this session):**
- `meta_0214` (catalog) = class `snack012`, name `snack012_9`, store_prim `model_snack012_9/v_0`.
  Catalog idx 210–214 = `snack012_5…_9`; 215+ = `snack017*`.
- `reference_image_0210` (snack012_5) AND `_0212` (snack012_7) are BOTH the identical Doves Farm
  bar → every snack012 facing's extracted reference = Doves Farm bar (internally consistent class).
- `obs_0214` alpha/iid foreground = iids [1766,1768] → **`flour003_7`, `flour003_8`** (NOT snack012_9);
  alpha>0 = 84.9% of frame. flour003 = the Williams-Sonoma pumpkin biscuit mix (a FLOUR product;
  matches the "ORGANIC FLOUR" seal). `reference_image_0086` = flour003's biscuit-mix.

**Writer design (optflow_writer.py — READ this session):** OptFlowWriter has **NO per-frame →
catalog-object tie**. `write()` just increments `self._frame_id` and serializes whatever the camera
sees. References are stored **PER CLASS** (`class_to_reference`, representative = first object per
class), not per frame. So `obs_NNNN` (NNNN-th CAPTURE FRAME) and `reference_image_NNNN` (NNNN-th
CATALOG OBJECT, Stage-B sorted-prim-path order) are **DIFFERENT INDEX SPACES** — they coincide only
if capture order == catalog order. The obs alpha = instance foreground from
`reference_seg_writer.obsmask_from_data`, = prominent/visible tracked instance(s), NOT provably the
camera's aim target.

**Order analysis:** intended `num_targets:null` → `plan_capture` `idx=np.arange` → frame i aims at
`scene.grasp_points[i]` = `targets[i]` = `objects[i]` = `collect_preoptflow[i]` = catalog[i]. Code
path argues order IS preserved (collect_preoptflow deserializes by idx; filter_specs=[]; RemoveUntracked
`_drop_under` preserves order; einsum keeps target order). Early frames DID align (obs_0004 = cereal,
catalog 0–10 = cereal). Yet frame 214's foreground = flour003, not snack012_9.

### Verdict (all experiments run 2026-07-09, read-only over the serialized dataset)

**Hypothesis B (order divergence) is DEAD; hypothesis A was right in mechanism but wrong in
detail.** Capture order == catalog order EXACTLY, so `obs_i` ↔ `reference_i` is a valid pairing.
The 12 mismatched frames aim at facings the store author physically placed **label-side-in**
(rotated 180° vs their shelf-mates) — the camera, placed 0.6 m out along the *baked class face*,
lands INSIDE the gondola behind another product's front row, 0.12–0.21 m from a flour-box wall.
That wall is the "unusually zoomed-in biscuit": intended standoff 0.6 m, actual first-hit ~0.14 m.

**Experiment 1 — order test (decisive).** For every frame i, computed the expected camera position
`(class_to_l2w[cls][member] @ grasp_point_i) @ [0.6,0,0]` from *catalog-order* aiming and compared
with the serialized `cam2world_i` translation:
- error = **0.0315 m for ALL 283 frames** (max == median) — a constant, not a scramble. 0.0315 m
  = **63 mm ZED baseline / 2** (`hardwares.py:19`): `cam2world` is the LEFT eye, offset half the
  baseline from the rig pose. Order exactly preserved; earlier "make_sheets tiles are mislabeled"
  suspicion was WRONG — the tiles are labeled correctly.

**Experiment 2 — per-frame target visibility.** Center pixel of `iid_mask_i` vs catalog[i].name +
own-pixel count + center depth:
- **268/283 frames: center pixel IS the intended target**; median center depth 0.600 m (exactly
  the configured standoff).
- **12 frames: target has ZERO own pixels** and center depth 0.12–0.43 m: frames 206, 208–214
  (`snack012_{10,3,4,5,6,7,8,9}`) and 254–257 (`snack032_{4,5,6,7}`). These are exactly the two
  **+Y** snack classes; their representatives (snack012 base/_1/_2, snack032 base/_1/_2/_3) frame
  FINE.
- 3 benign flags: i=14 (`detergent001_3`, center hits sibling but target visible, 1685 px),
  i=237/238 (`snack025_1/_2`, center ray slips between boxes to the shelf back at 0.874 m; target
  hugely visible, 81k/94k px).

**Experiment 3 — world geometry of the 12.** From `class_to_l2w` + `class_to_name`:
- Good snack012 facings sit at x≈6.35–6.73, y=3.74, local +Y → world **−Y** (open aisle).
- The 12 sit at x≈4.92–4.97 on gondola A (spans x≈4.5–5.0) with local +Y → world **−X**, i.e.
  label face pointing INTO the gondola at the backs of the flour003/flour005 rows (x≈4.50–4.54,
  same y/z shelf slots, ~0.4 m in front). Camera 0.6 m along −X → x≈4.32–4.36, inside the gondola
  behind the flour front row, looking +X through flour boxes 0.14–0.21 m away. Depths match the
  measured center depths exactly.
- Same-bay products that face the aisle properly (snack023_4..8, snack034_2.., snack035_2.., local
  face → world **+X**) frame fine — confirming the 12 are *individually* rotated, not a wrong
  class face. The keep-map +Y is correct for both classes (labels ARE on local +Y).

**Experiment 4 — where the 12 ARE visible.** Scanned all 283 `iid_mask`es for the 12 iids:
`snack012_3/_4/_5`, `snack032_4` appear in **NO frame** (fully entombed). The rest appear only as
**backs** in frames aimed at their properly-facing bay-mates (231–235, 263, 266–267, 274–275; up
to 240k px) — correctly iid-labeled box backs, label never visible from any aisle.

**Experiment 5 — visual.** `obs_0204` (snack012 base): Doves Farm front, clean 0.6 m framing —
extraction + face verified correct. `obs_0232` (snack023_5): front-facing crackers with the Doves
Farm box tops of the backward snack012 copies peeking on the shelf above — matches the pixel scan.

**Conclusion:** dataset internally correct; pipeline behaved as designed; the verification dataset
did its job by surfacing 12 store-authoring quirks (label-inward facings) whose fronts are
physically unphotographable in-store. `reference_image_0214` (Doves Farm) is the correct reference
for frame 214's intended target; the frame just can't see it.

**Scratchpad artifacts:**
- `<scratch>/make_sheets.py` — per-category contact sheets, tiles labeled by CATALOG-idx meta
  (labels CONFIRMED correct — order preserved). Output → `render000/verify_sheets/…`.
- `<scratch>/obs0214_target_only.png` — obs_0214 masked to alpha>0 (= flour003 pixels).
- `<scratch>/stage{A,B,C}.log` — stage logs.
- Dict subdirs (`cid_to_class`, `class_to_name`, `iid_to_name`, …) serialize as **torch `.pt`
  pickles** → load with `torch.load(..., weights_only=False)`, not yaml/json.

---

# FIX PLAN — drop the 12 label-inward facings, re-render Stage C

## Context

The 12 backward facings can never show their label in-store: their fronts point into the gondola
at another product's shelf-backs, and their aisle-visible side is the blank box back. For the
verification dataset they produce misleading "zoomed-in neighbor" tiles; for future training runs
they would spend target frames photographing flour walls. The catalog (Stages A+B) is UNTOUCHED —
their extracted usdz/reference are placement-independent and correct (per-class references come
from the base facings anyway). The fix is scene-side: deactivate exactly those 12 prims at Stage C
and prune their capture targets — the per-instance complement of `RemoveClass`, using the same
`deactivate_prim` + `_drop_under` seams. Deactivating (vs merely un-targeting via a filter) also
removes their unlabelable backs from other frames' backgrounds, so no visible-but-untracked
same-class instances leak into any downstream training use (the union-mask "any same-class
instance is an inlier" design would otherwise inherit false-negative regions).

**Scope: two changes only** — one new mutation class + one config edit. The standoff stays 0.6 m
(framing is fine). No new committed tooling: the post-render check reuses the exact experiment
already run this session (kept in scratch), not a new tracked debug script.

## Change F1 — `store_mutations.py`: `RemovePrims` (per-instance drop, new registry class)

Sibling of `RemoveClass` (fnmatch on class, store-wide) and `RemoveUntrackedProducts` (keep-list
complement): exact prim names, fail-loud on any name that doesn't resolve to an active product.

```python
+ class RemovePrims:
+     """Deactivate specific store products by EXACT prim name (e.g. 'model_snack012_3') and
+     prune their CaptureTargets. Per-instance complement of RemoveClass — for store-authoring
+     quirks (label-inward / permanently occluded facings) where the class stays but individual
+     facings must go. Fail-loud: every name must match an active product under the config globs."""
+     def __init__(self, names: list):
+         assert names, "RemovePrims needs a non-empty prim-name list"
+         self.names = list(names)
+     def __call__(self, store, spec, targets, rng):
+         from isaac_datagen.isaac_utils import deactivate_prim
+         by_name = {p.GetName(): p for p in active_products(store, spec.product_patterns)}
+         missing = sorted(set(self.names) - set(by_name))
+         assert not missing, f"RemovePrims: no active product prim named {missing}"
+         removed = [by_name[n].GetPath().pathString for n in self.names]
+         for n in self.names:
+             deactivate_prim(by_name[n])                      # model_* root, like RemoveClass
+         print(f"[MUT] RemovePrims: deactivated {len(removed)} product prim(s)", flush=True)
+         return _drop_under(targets, removed)
```

Flow is the `RemoveClass` precedent: `build_store_scene` binds CaptureTargets from the full
catalog FIRST, mutations then deactivate + `_drop_under` prunes, the downstream `IsActive()`
assert sees only survivors. Order vs `RemoveUntrackedProducts` is immaterial (that one keys on
target *classes*, which don't change), keep it after for readability.

## Change F2 — `configs/store001-optflow-verify.yaml`: name the 12 (standoff UNCHANGED)

Only the mutation list grows. **`xrange` stays `[0.6, 0.6]`** — the user confirmed the current
framing looks fine; the sole change is that the 12 unphotographable facings stop being targeted.

```yaml
~ mutations:
    - {name: RemoveUntrackedProducts}
+   - name: RemovePrims                        # the 12 label-inward facings (RESOLVED section)
+     args:
+       names: [model_snack012_3, model_snack012_4, model_snack012_5, model_snack012_6,
+               model_snack012_7, model_snack012_8, model_snack012_9, model_snack012_10,
+               model_snack032_4, model_snack032_5, model_snack032_6, model_snack032_7]
# xrange/yrange/zrange: UNCHANGED ([0.6,0.6]/[0,0]/[0,0]) — framing is fine per user.
```

## Execution ladder (cwd = src/isaac_datagen, GPU 1, EULA accepted)

```bash
# 0. no-kit smoke: RemovePrims resolves via store_mutations.get; missing-name assert fires
uv run python -c "from isaac_datagen.store_mutations import RemovePrims, get; \
  print(get('RemovePrims') is RemovePrims)"                        # True
# 1. Stage C ONLY re-render (~9 min; Stages A/B catalog reused as-is). Re-renders in place
#    (idx=0 -> render000 overwritten); point dataset_dir at a fresh dir to keep the old one:
isaac-datagen configs/store001-optflow-verify.yaml idx=0
#    expect: "[MUT] RemoveUntrackedProducts: deactivated 132 ..." + "[MUT] RemovePrims: deactivated 12 ..."
#    and 271 frames (283 − 12).
# 2. Confirm the fix worked — re-run THIS SESSION'S check (the scratch script that found the bug;
#    per-frame expected-cam-pose vs cam2world + target own-pixel count). Expect 271/271 pose-PASS
#    and 0 frames with an invisible target (the 12 zero-pixel offenders are gone; the 3 benign
#    detergent001_3 / snack025 center-slips are NOT zero-pixel, so they still pass).
# 3. Re-make the contact sheets (scratch make_sheets.py; labels now trusted) and eyeball.
```

Note: with the 12 targets pruned, frames compact to 0–270 while the catalog stays 0–282, so the
check maps frame → surviving-catalog index (catalog order minus the 12 dropped names) — a filtered
enumerate, since `_drop_under` preserves order.

## Deviation clause

If the post-render check flags ANY frame beyond the expected outcome (a nonzero pose error = order
regression, or a new zero-pixel target = a buried facing the 12-name list missed) — STOP and report
before changing anything.

## Key changes (fix)

`+` `store_mutations.py::RemovePrims` (exact-prim-name deactivate + `_drop_under` prune; sibling of
`RemoveClass` / `RemoveUntrackedProducts`) · `~` `configs/store001-optflow-verify.yaml` (append
`RemovePrims` with the 12 names to `mutations`; standoff/poser UNCHANGED) · re-render Stage C only.
No new tracked tooling; catalog (Stages A/B) untouched.

## FIX EXECUTED (2026-07-09) — succeeded; 2 coincident-duplicate flags explained-benign

`RemovePrims` added + config edited; no-kit smoke passed. Re-rendered Stage C into
`datasets/store001-optflow-verify2/render000` (original 283-frame `…-verify/render000` preserved
for diff). Logs: `[MUT] RemoveUntrackedProducts: deactivated 132 …` + `[MUT] RemovePrims:
deactivated 12 …` → **271 frames (283 − 12)**.

**Gate** (`<scratch>/check_fix.py`, this session's pose+visibility check, remapped to surviving
catalog order): **pose 271/271 within 5 cm** (order preserved, ZED-left 31.5 mm offset) and the 12
label-inward offenders are GONE. It flagged **2 residual frames** — i=229 `snack025_1` (0 own px),
i=230 `snack025_2` (absent from `iid_to_name`) — which the deviation clause said to stop on.

**Investigated → benign.** Cause: `snack025_1` and `snack025_2` occupy the **exact same world
position** (0.00 mm apart) — coincident duplicate prims. The instance segmenter labels their shared
surface as ONE instance, so one twin claims all pixels and the other gets zero; which twin wins is
**session-local** (it flipped vs the original render, where both twins happened to catch pixels).
The `snack025` **class is fully visible and correctly −Y front-faced** (obs_0229 = Williams-Sonoma
"Sugar Cookie Mix" tins, label readable). A store-wide scan found **exactly 2 coincident pairs in
all 42 classes**: `detergent001 == detergent001_3` and `snack025_1 == snack025_2` — i.e. precisely
the "3 benign flags" (i=14, i=237/238) from the original RESOLVED investigation. Orthogonal to the
12-facing fix (RemovePrims never touched snack025/detergent001); a pure store-authoring redundancy
(a prim duplicated at identical coords).

**Coincident-twin cleanup DONE (user approved).** Added `model_detergent001_3` + `model_snack025_2`
to `RemovePrims` (14 names total) and re-rendered Stage C → **269 frames (283 − 14)**. Gate:
**pose 269/269 within 5 cm AND target visible 269/269** (0 zero-pixel, 0 black frames, min luma 15).
Final deliverable = `datasets/store001-optflow-verify2/render000` (269 frames); the original
283-frame `…-verify/render000` is preserved. Per-category montages rebuilt in
`…-verify2/render000/verify_sheets/{instore,reference}_<cat>.png`.

**Status: COMPLETE.** Stage A/B catalog (283 objects, 42 classes) untouched and reusable for
training. For a training config, carry the same 14-name `RemovePrims` list (12 label-inward + 2
coincident twins), or deactivate those prims in the store USD at source.

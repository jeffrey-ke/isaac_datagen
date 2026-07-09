# Store scene mutations: remove / replace shelf products (optflow capture)

**Status 2026-07-08: COMPLETE, verified end-to-end** — remove, replace-from-ycb,
replace-from-store-catalog (arm A) all pass validate_obsmask at fixed pose; determinism
proven; swap seating/orientation confirmed visually. **One open finding** (arm B,
`pull_flatten_usd`-scrubbed catalog loses class labels at capture) is documented under
Future work — the tool itself is correct and verified; the capture-side mechanism is the
open question. Approved plan (with diffs + review dialog) at
`~/.claude/plans/read-the-latest-plan-logical-knuth.md`. Follow-on to
`store-usd-inverse-datagen.md`.

## Goal & architecture

`build_store_scene` aligned the catalog onto the store's own product prims, never touching
geometry. This plan adds the mutation axis: (a) **remove** store products by SKU class,
(b) **replace** store products with OptFlowObjects from ANY catalog (amazon / kleenex /
ycb / store001-optflow-objects), seated at the removed product's shelf pose. Variants are
named types in a new registry (6th instance of the `get(name)` idiom), so new mutation
strategies are an entry + config, not an edit to the builder.

```
optflow_generation (UNCHANGED)
  objects = filter_objects(collect_preoptflow(...), filter_specs)
  build_store_scene:
    spec = StoreSceneSpec(**scene_builder_args)        # + mutations: [] field
    targets = [CaptureTarget(obj, store prim path)]    # binding seam
    for m in store_mutations.make_mutations(spec.mutations):
        targets = m(store, spec, targets, rng)         # edits stage + list together
    uniform loop: label_product + add_catalog_grasp_frame   # store prims AND swap wrappers
```

**CaptureTarget** (frozen dataclass) is a binding, not a placement: in store mode the
products already exist in the USD — `obj` says what it is (the catalog entry the writer
records), `prim_path` says where it is (the prim whose LOCAL frame equals the usdz frame:
store `v_0` node or swap wrapper — where l2w is read, labels authored, GraspPoint added).
`filter_specs` only chooses which existing products we TRACK; mutations are what change
the scene. Because a mutation edits the stage and returns the updated binding list in the
same call, the `scene.objects[i] <-> object_prim_paths[i]` writer contract cannot drift.

## Key design decisions

1. **Removal = `SetActive(False)` on the `model_*` root** (`deactivate_prim`, new in
   isaac_utils). Deactivation is the only edit that prunes a reference-composed subtree
   (deletion APIs can't touch referenced opinions); `hideForCamera` rejected (unreliable
   under PathTracing, still casts shadows). First prim-removal code in the repo.
2. **Store-wide semantics**: RemoveClass/ReplaceClass act on EVERY active product whose
   `parse_sku` class matches the glob — including unlabeled distractor duplicates never
   extracted into the catalog — not just catalog placements. The walk is scoped to
   `spec.product_patterns` (`active_products`), which both respects prior mutations
   (GetChildren drops deactivated prims) and keeps `parse_sku` off non-product prims
   (`model_store001` would parse as class `store001`). Deliberately NOT
   `matched_products`/`find_prims` — those raise per-pattern on zero matches, but a prior
   mutation may legitimately empty a pattern; each mutation asserts its OWN class glob
   matched instead.
3. **Replacement labeling goes through `label_product`, NOT `add_object`'s plain
   `add_labels`** — the problem-6 trap recurs for swap-ins: Stage-A export ran before any
   labeling fix, so store-extracted usdz carry the vendor's legacy `semanticData=snack`
   BAKED INSIDE. `scene.add_object` was split: new `add_wrapped_reference` does the
   wrapper+geo mechanics without labels; the builder labels everything (store prims and
   wrappers) in one uniform `label_product` loop. Verified: arm A (unscrubbed store usdz
   swap-ins) is orphan-free; clean catalogs are unaffected (override is a no-op there).
4. **Replacement pose = pure math in `replacement_pose`**: rotation aims the
   replacement's grasp face (+X) where the original's spec-policy grasp face pointed
   (catalogs don't share a "front" axis convention and LookAtPoser hangs the camera halo
   off grasp +X; both frames are +X-face/+Z-up so store→store swaps degenerate to the
   original rotation); translation puts the replacement's usdz-frame bbox bottom-center
   on the original's shelf-contact point, at catalog-native metric size (the store's own
   `model_*` scale is deliberately NOT applied — `_orthonormal_rotation` divides it out
   and fail-louds on non-uniform scale / non-upright placements, which `set_prim_pose`
   (translate+rotate only) couldn't reproduce anyway). Everything the pose needs is read
   into a `ProductSite` BEFORE deactivation (bbox and l2w are unreadable once pruned).
5. **Config seam**: `StoreSceneSpec.mutations: list = []` of `{name, args}` specs
   (mirrors `filter_specs`), names validated at spec construction, ctor args validated by
   `make_mutations` at build time — so Stage A (which constructs the same spec) never
   needs the swap catalogs. Seed stream `[effective_seed, 3]` (0/1/2 are the jitters').
6. **Arm B tool** `pull_flatten_usd`: inside a flattened catalog usdz the vendor attrs
   are plain local data (no reference arc), so `RemoveProperty` legally deletes them.
   Unzip → scrub the single inner root layer → repackage via CreateNewUsdzPackage;
   in-place catalog CLI with pristine `.usdz.orig.bak` backups, always re-deriving FROM
   the backup (rotate-graspable-meshes precedent).

## What landed (all in isaac_datagen)

- `src/isaac_datagen/store_mutations.py` NEW — CaptureTarget, ProductSite, get /
  make_mutations, active_products, measure_site, replacement_pose, insert_replacement,
  RemoveClass, ReplaceClass. Module-level imports kit-free; capture/scene/clean_datagen
  imports deferred (import-cycle avoidance).
- `src/isaac_datagen/store_scene.py` ~ — StoreSceneSpec `mutations` field + validation;
  build_store_scene refactored to targets → mutations → uniform label loop; returns
  `objects=[t.obj for t in targets]` (the mutated list, NOT the input).
- `src/isaac_datagen/scene.py` ~ — `add_object` split; NEW `add_wrapped_reference`.
- `src/isaac_datagen/isaac_utils.py` ~ — NEW `deactivate_prim`.
- `src/isaac_datagen/debug_scripts/pull_flatten_usd.py` NEW — arm-B usdz semantics scrub.
- `configs/store001-optflow-{remove,replace}.yaml` NEW — thin copies of
  store001-optflow.yaml (dataset_dir + mutations block only).

## Verification results (fixed-pose smokes, cwd src/isaac_datagen, CUDA_VISIBLE_DEVICES=1)

| Run | Config | Result |
|---|---|---|
| Regression | store001-optflow.yaml (mutations=[]) | identical behavior, only known-benign log spam |
| Remove | -remove.yaml (`RemoveClass cereal001`) | 6/6 `model_cereal001*` deactivated store-wide; `cid_to_class={2: cereal002}`; validator clean |
| Replace (ycb) | -replace.yaml (`ReplaceClass cereal00[12]` ← ycb) | 11/11 sites swapped; validator clean; 290k–980k px per ycb class; montage shows cans upright, grasp face to aisle, seated on shelf, warps land on the right instances |
| Determinism | same idx re-run | identical `[MUT]` lines |
| **Arm A (the trap test)** | scratch cfg: `cereal002` ← UNSCRUBBED store catalog (`source_class: cereal001`) | validator clean (0 orphans), 3.9M labeled px — `label_product` neutralizes baked vendor semantics on swap-ins |
| **Arm B** | same cfg after `pull_flatten_usd` scrub | **FAILS** — see Future work |

```
# the ladder (renders in datasets/store001-optflow-{remove,replace}/, gitignored):
OMNI_KIT_ACCEPT_EULA=YES uv run python clean_datagen.py configs/store001-optflow-remove.yaml \
    idx=0 num_targets=1 num_frames=4 'pose_generation_policy_args.xrange=[0.6,0.6]' \
    'pose_generation_policy_args.yrange=[0.0,0.0]' 'pose_generation_policy_args.zrange=[0.1,0.1]'
uv run python validate_obsmask.py datasets/store001-optflow-remove/render000
uv run --extra viz python debug_scripts/viz_optflow.py datasets/store001-optflow-replace/render000 --idx 0
uv run --with usd-core python debug_scripts/pull_flatten_usd.py datasets/store001-optflow-objects
```

## Discoveries & learnings

- **`label_product` on a fresh swap wrapper fully handles baked vendor semantics** (arm
  A): the override step rewrites the legacy values composed FROM INSIDE the referenced
  usdz, exactly as it does for in-store prims. This makes label_product the single
  labeling path for both store prims and wrappers — no special-casing.
- **`SetActive(False)` works as expected over the store's reference arc**: subtree prunes
  cleanly, deactivated products vanish from `GetChildren` (so later mutations skip them),
  no PhysX side effects (scene is physics-inert), no renderer artifacts in the emptied
  slots.
- The ycb optflow catalog's tuna class is already renamed **`fish_can`** (the
  tuna-fish-can orphan fix landed in the catalog data) — all current catalogs carry
  single-token classes, so the new whitespace assert is a guard, not a live constraint.
- The iid annotator names swap wrappers via the instance label even where the class label
  fails (arm B) — `iid_to_name` and `cid_mask` can disagree per-instance; the validator's
  orphan table is the right lens for exactly this split.
- `cid_to_class` is stored as a torch-serialized pickle per render dir — inspect via
  `ObsMask.deserialize` / `torch.load`, not by reading the file as yaml.

## Future work / debugging: the arm-B finding (scrubbed catalog loses class at capture)

**Symptom**: after scrubbing `store001-optflow-objects` with `pull_flatten_usd`, a
capture that swaps those objects in produces **cid orphans for every swapped instance on
every visible frame** (32 orphan rows over 8 frames in render003): the swap pixels sit in
`iid_mask` with correct `iid_to_name` (instance label RESOLVES) but `cid_mask == 0`
(class label does NOT). The unscrubbed run of the identical config (render002) is
completely clean, as is the ycb swap run.

**What is ruled out**:
- The scrub itself: a usda text-diff of `.orig.bak` vs scrubbed shows EXACTLY the 4
  legacy `semantic:*` lines removed (2 on `/World`, 2 on the leaf mesh) and nothing else;
  textures byte-identical and resolving inside the package; 0 `semantic:*` attrs remain
  across all 11 files.
- The labeling code path: identical for arm A / arm B / ycb (label_product on the
  wrapper).

**The contradiction to resolve**: ycb usdz (which never had ANY baked semantics) resolve
class fine from wrapper-only labels, while scrubbed store usdz — now also without baked
semantics — do not. In arm A the class evidently reaches the annotator via the
overridden LEGACY attrs on the geo/leaf (unioned across the subtree); with them deleted,
the wrapper's LabelsAPI class label is not reaching the mesh iids for these assets
specifically. Structural suspects to check: the store subtrees carry
PhysicsRigidBodyAPI/PhysicsMaterial and deeper nesting vs the Blender-authored ycb
meshes; instanceable flags; how `instance_segmentation_fast` picks the prim whose
ancestor chain is unioned.

**Next diagnostic** (the M3/tuna precedent): an in-kit probe — build the scene with the
scrubbed-swap config, dump every `semantic*`/`semantics:*` attr under one swap wrapper
subtree, attach `instance_segmentation_fast`, step one frame, print `idToSemantics` for
the swap iids. That shows directly whether the class is missing, truncated, or wrong.

**Current state / repro**: `datasets/store001-optflow-objects` is SCRUBBED, with pristine
`<name>.usdz.orig.bak` backups beside every file — i.e. the arm-B repro is live. In-store
captures are unaffected (they never read usdz semantics; labels are authored on store
prims), but swap-ins FROM THIS CATALOG currently orphan their class. To restore the
originals: `for f in datasets/store001-optflow-objects/usd_path/*.usdz; do cp "$f.orig.bak" "$f"; done`
(backups are kept; the scrub re-derives from them, so both directions stay idempotent).
Evidence renders: `datasets/store001-optflow-replace/render002` (arm A, clean) and
`render003` (arm B, 32 orphans) — captured with the scratch config variant
(`pattern: cereal002, catalog: datasets/store001-optflow-objects, source_class: cereal001`).

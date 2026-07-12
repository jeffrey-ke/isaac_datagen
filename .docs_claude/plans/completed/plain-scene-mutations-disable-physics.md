# DisablePhysics mutation + plain-builder mutations hook

**Status 2026-07-12: COMPLETED — V0–V6 all pass.** 50/50 frames (was 36/50), static
centroid (was −3.14 m free-fall), store smoke identical, deterministic (byte-identical
cam2world across reruns).

## Context

Rendering the snack031 1inst fine-tune pool in an **empty world** (`build_scene`,
`scene: empty`, no store USD) drops frames: 36/50 written, and *nothing closer than
1.16 m survives*. Root cause, measured this session: the store-extracted catalog usdz
carries `PhysicsRigidBodyAPI(rigidBodyEnabled=True)` + `CollisionAPI` from the live
store, and with no supporting collider the box **free-falls during capture** — its
triangulated world centroid drifts z ≈ +0.13 → **−3.14 m** across the 36 frames. Camera
poses are planned once against the build-time pose (`clean_datagen.py:118` reads
`l2w = get_target2world(...)` before capture), so close cameras lose the falling box
entirely (writer guard `reference_seg_writer.py:82` "no labeled instances") — **and the
36 frames that did write carry stale `l2w`, i.e. silently wrong labels/visibility
metadata.** This is a data-corruption bug, not just a yield bug.

Why it never showed before: the legacy plain-scene datasets (mixed / shelf /
expanded-refseg-v2) use Blender-pipeline catalogs (amazon-v2, kleenex,
object_dataset_amazon) that carry **zero** physics APIs — verified by schema scan.
Only store-extracted catalogs are live rigid bodies. The plan history even records the
wrong assumption we are now correcting: `store-usd-inverse-datagen.md` — *"Products
carry PhysicsRigidBodyAPI + PhysicsMaterial in their subtrees (**inert**, ride into
usdz)"*, echoed by `store-scene-mutations.md` (*"scene is physics-inert"*). True in the
store only because products rest on shelf colliders; falsified on the plain path.

**User directive:** physics-disabling must be toggleable and pattern-matched, *analogous
to the mutations*.

## Why this is not "as simple as adding a new mutation"

The mutations system, as documented in `store-scene-mutations.md` (2026-07-08) and
extended in `store-perclass-faces-verify-dataset.md` (2026-07-09):

- Contract: `mutation(store, spec, targets, rng) -> targets` — one call edits the live
  stage **and** returns the updated `CaptureTarget` list, so the writer's
  `objects[i] ↔ object_prim_paths[i]` alignment invariant can't drift.
- The registry (6th `get(name)` instance) exists so a new strategy is *"an entry +
  config, **not an edit to the builder**"* — but that convention is conditional on the
  builder already applying mutations.
- **Only `build_store_scene` had that hook**: `store_scene.py:84` builds
  `StoreSceneSpec(**runtime.scene_builder_args)`, `:91-96` (pre-collapse) ran rng stream
  `[effective_seed, 3]` → apply loop → non-empty + unique-name asserts. `build_scene`
  (scene.py:364) had **zero references to `scene_builder_args`** and no spec, no
  CaptureTargets, no apply loop (verified; also no plain-scene config passed
  `scene_builder_args` before this plan).

So the diff decomposed as: **~20 lines of mutation** (the easy part the mutation system
already promises) **+ ~40 lines of one-time seam** that `build_store_scene` got in the
store-usd-inverse-datagen plan and `build_scene` never had. `StoreSceneSpec` could not
be reused for the seam — it is constructed in three places (`store_scene.py:84`,
`extract_store_objects.py:68`, `debug_scripts/check_front_face.py:151`) and its required
store fields (`store_usd`, `product_patterns`, …) are the fail-loud validation those
sites depend on. After this plan, the next plain-scene mutation is again "an entry +
config."

## Design vetting (alternatives rejected)

- **Asset-side scrub** (extend `pull_flatten_usd` / Stage A to strip physics from
  catalog usdz): rejected. (1) Shared-asset blast radius — the same usdz feeds store
  `ReplaceClass` swap-ins; a scrub changes sim semantics for every consumer, forever.
  (2) Prior art warns: `pull_flatten_usd`'s only field use produced the still-open arm-B
  class-label regression, whose suspect list *includes* `PhysicsRigidBodyAPI` —
  deleting physics attrs now would confound an open investigation. (3) Becomes a
  mandatory post-extraction step that will eventually be skipped — the exact latent-bug
  class ("inert", assumed, unenforced) being fixed.
- **RuntimeConfig knob** (`disable_physics_patterns` applied inline in build_scene):
  rejected — contradicts the directive and the no-flag-variants / registry policy;
  `scene_builder_args` is the established home for builder-scoped policy.
- **Unconditional disable in `add_object`**: rejected — implicit stage edit hidden in a
  loader violates the explicit-selector policy; `DisablePhysics(pattern: "*")` gives the
  same effect explicitly.

## Key decisions

1. **Two-level fail-loud** in `DisablePhysics`: assert the pattern matches ≥1 prim
   (typo guard) and ≥1 rigid body disabled overall (stale-config guard) — but tolerate
   individual matched prims without physics, so chained-catalog configs (`objects_path`
   lists mixing Blender + store catalogs) can use `pattern: "*"`.
2. **`PLAIN_SAFE` whitelist**: all four pre-existing mutations read `spec.product_patterns`
   (store-shaped) and would die mid-build with a misleading `AttributeError` ~30s after
   Isaac boots. `PlainSceneSpec` rejects non-portable mutation names at config-parse
   time; portability is declared as a class attribute on the mutation itself.
3. **No-removal assert instead of speculative reconciliation**: no portable mutation adds
   or removes targets today, and true removal support needs a designed `is_graspable`
   reconciliation (added wrappers have no placer entry at all). Assert the prim-path list
   is unchanged; keep `objects = [t.obj for t in targets]` (preserves the contract's
   in-place obj-replacement ability, `dataclasses.replace` precedent).
4. **`rigidBodyEnabled=False` suffices**: a strings-scan of the usdz shows only
   UsdPhysics tokens (no PhysxSchema); with the attr False, omni.physx never creates the
   dynamic actor. Remaining `CollisionAPI` = static collider, motionless and invisible.
   Root-layer attribute override over a reference arc is the proven pattern
   (`_override_vendor_class_labels`, `deactivate_prim`). Contingency ladder if
   verification still shows drift: author `kinematicEnabled=True`; last resort
   `RemoveAPI`.

## What landed (all in isaac_datagen)

- `src/isaac_datagen/isaac_utils.py` ~ — NEW `disable_rigid_body(prim) -> bool`, next to
  `deactivate_prim`: root-layer `RigidBodyAPI.CreateRigidBodyEnabledAttr(False)`,
  per-prim no-op (returns False) if the prim carries no `RigidBodyAPI`.
- `src/isaac_datagen/store_mutations.py` ~ — NEW `DisablePhysics` mutation
  (`PLAIN_SAFE = True`; fnmatch on prim name, subtree walk to find the rigid-body prim,
  two-level fail-loud per decision 1); NEW `apply_mutations(root, spec, targets,
  effective_seed)` — extraction of the rng-stream + apply-loop + non-empty/unique-name
  asserts that used to live inline in `build_store_scene`.
- `src/isaac_datagen/store_scene.py` ~ — `build_store_scene`'s l.91-96 collapsed to one
  `store_mutations.apply_mutations(...)` call (behavior-preserving).
- `src/isaac_datagen/scene.py` ~ — NEW `PlainSceneSpec` (the seam `build_scene` never
  had: `mutations: list = []`, `__post_init__` validates `{name, args?}` shape and
  rejects any mutation without `PLAIN_SAFE = True`); NEW `apply_plain_mutations(
  stack_path, spec, objects, objects_paths, effective_seed)` — builds `CaptureTarget`s
  over the placed stack, walks from `stack_path` (all placed objects, not just one
  class), calls the shared `apply_mutations`, and asserts the prim-path list is
  unchanged (decision 3) before returning `[t.obj for t in targets]`. `build_scene`
  parses `spec = PlainSceneSpec(**runtime.scene_builder_args)` as its first statement
  (fail fast, mirrors `store_scene.py:84`) and calls `apply_plain_mutations` right after
  `create_stack_of_objects` returns (prims exist) and before grasp frames / pose reads.

Hook runs unconditionally — empty `mutations` is a natural no-op. Existing plain configs
pass no `scene_builder_args` → `{}` → no behavior change. The refseg-mode path
(`clean_datagen.py:74` calls `build_scene` directly) gets the same parse;
`CaptureTarget.obj` only needs `meta["name"]`, which `GraspableObject` has.

## Config recipe

```yaml
scene_builder: build_scene
scene_builder_args:
  mutations:
    - {name: DisablePhysics, args: {pattern: snack031}}   # fnmatch on prim names under the stack; "*" = all
```

## Verification — status: ALL PASS (2026-07-12)

Kit-free spec validation (V0) ran clean during implementation. V1–V6 are GPU render
checks the orchestrator runs next; this table records the plan (see the approved plan
file for full detail) and should be updated in place with PASS/FAIL + evidence once run.

Baseline (broken) artifacts already exist:
`src/isaac_datagen/datasets/tmp-snack031-1inst-emptyworld1/render000` (36/50 frames,
centroid falling to −3.14 m) + scratchpad scripts `track_box_position.py`,
`make_contact_sheet.py`, config `emptyworld-snack031-1inst.yaml`. All runs from cwd
`src/isaac_datagen`.

| # | Check | Pass criterion | Result |
|---|---|---|---|
| V0 | Kit-free spec validation | `PlainSceneSpec` accepts DisablePhysics; rejects `RemoveClass` ("store-only") and unknown names — no sim boot | **PASS** (ran during implementation, no GPU) |
| V1 | 50/50 frames | scratchpad config + mutations block → fresh `datasets/tmp-snack031-1inst-emptyworld2`; log shows `[MUT] DisablePhysics`, **zero** "no labeled instances", `obs/` count = 50 | **PASS** — `disabled 1 rigid body(ies)`, 0 drops, 50/50 (baseline 36/50) |
| V2 | Static centroid | `track_box_position.py` — centroid constant (≲2 cm noise) and matches planned l2w; baseline drifted 3.3 m | **PASS** — stationary; z scatter 0.11–0.28 m is view-dependent visible-surface sampling of the ~0.5 m box, no drift (baseline fell monotonically to −3.14 m) |
| V3 | Close frames back | contact sheet shows populated close-range views (baseline min written dist 1.16 m) | **PASS** — min written dist 0.84 m, box centered in all 50 tiles |
| V4 | Store path unaffected | `store001-optflow-remove.yaml` 4-frame fixed-pose smoke → identical `[MUT]` lines to store-scene-mutations.md §Verification | **PASS** — 6/6 `model_cereal001*` deactivated, 4/4 frames, validator core 0 orphans (output redirected to `datasets/tmp-store-remove-smoke`; `validate_obsmask.py` CLI itself crashes on stripped module docstring — pre-existing, ran `validate_render_dir` directly) |
| V5 | Plain default no-op | scratch config without mutations, `dry_run=True` — builds as before | **PASS** — 0 `[MUT]` lines, scene builds + scene.usdz exports; same pre-existing usdz-packaging crash after export as pre-change baseline |
| V6 | Determinism | rerun V1 → identical `[MUT]` lines + frame count | **PASS** — identical `[MUT]`, 50/50, all 50 cam2world byte-identical |

Contingency: if V2 still drifts, escalate `disable_rigid_body` to also author
`kinematicEnabled=True`; rerun V1–V3. Cleanup: `datasets/tmp-*emptyworld*` dirs after
sign-off.

Move to `plans/completed/` once V1–V6 are run and recorded.

# Empty-world regeneration of the 4 solo `-1inst` fine-tune pools

**Status 2026-07-12: COMPLETED — renders, CleanDiftFinetunedFpn bake, vis030 squash, and PSC
sync all done and verified** (checksum dry-run 0 diffs on all 4 pools; remote runtime.yaml
reads `build_scene` / `grasp_frames: catalog` / `DisablePhysics` / z ±0.7; one in-flight rsync
corruption on snack034 obs_0090.png auto-retried and verified clean).

## Context

The store-scene `-1inst` pools (200 LookAtPoser frames of one physical held-out-class
instance each; `store-snacks-finetune-renders.md`, 2026-07-11) were not diverse enough in
camera distance: the store's validated halo box (`x [0.3,0.9], y ±0.3, z [-0.2,0.3]`) caps
distance at ~1 m, and shelving occludes everything past ~1.75 m even with a wider box. The
user chose to regenerate the pools in an **empty world** (no store USD) with an expanded
box — enabled by the `DisablePhysics` mutation
(`plain-scene-mutations-disable-physics.md`): store-extracted catalog usdz free-fall
without shelf support, which had corrupted the first empty-world attempts (36/50 frames,
stale l2w on the survivors).

The **same dataset dirs** are regenerated in place (`datasets/<class>-1inst`,
`filtered/vis030/<class>-1inst`, and their PSC mirrors) so fine-tune `data.paths` and the
K-shot ordinal-first-K protocol are untouched. The old store renders are parked at
`datasets/parked-store-1inst/` (reproducible from the unchanged
`store001-optflow-snacks-kshot-*-1inst.yaml` configs, which are kept).

## New configs (one per held-out class)

`configs/emptyworld-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}-1inst.yaml`
— deltas vs the store siblings:

- `scene_builder: build_scene` (empty world); `scene_builder_args`:
  `grasp_frames: catalog` + `mutations: [DisablePhysics(pattern: <class>)]` — replaces the
  store's RemoveUntrackedProducts/RemovePrims/require_tracked_only (nothing else exists to
  remove or leak).
- `filter_specs: RegexFilter store_prim '^model_<class>/'` — selects exactly the base
  catalog instance; a class filter would place all 6–8 sibling entries.
- `placement_args: {max_column_height: 1}` (the plain builder actually runs the stacker).
- Poses: `x [0.3,2.0], y ±2.0, z [-0.7,0.7]` (was `x [0.3,0.9], y ±0.3, z [-0.2,0.3]`);
  z trimmed per user to avoid top-down sliver views.
- Lighting: expanded-refseg-v2 recipe incl. per-frame jitter (dome 200 jittered
  [100,350]; distant 2000 with offset/intensity/temperature jitter). Store-light pattern
  jitter removed (no store).
- Unchanged: `seed: 1001`, 200 frames, `num_targets: null`, dataset_dir, exposure block.

## `grasp_frames: catalog` (new PlainSceneSpec selector)

The pools rely on the Stage-A curated grasp frame. `build_scene` historically recomputed
a bbox frame (`add_grasp_frame`), ignoring the catalog's baked `grasp_point`. New explicit
selector `PlainSceneSpec.grasp_frames: bbox | catalog` (default `bbox` keeps every legacy
plain-scene config byte-identical) dispatches via `scene.GRASP_FRAME_SOURCES` to
`store_scene.add_catalog_grasp_frame` — the wrapper prim's local frame equals the usdz
frame (reference mounted at `<wrapper>/geo` with no extra transform), so the catalog SE3
lands identically to store Stage C. Note the store instance transform's ~0.5–0.6× scale
is absent here: object and grasp offset render at raw catalog scale, so relative geometry
is preserved but global size differs from the store renders.

## Results (2026-07-12, tesu, 2 GPU chains)

- 4 × 200/200 frames, `[MUT] DisablePhysics(<class>): disabled 1 rigid body(ies)` each,
  zero `no labeled instances` drops, zero tracebacks. Distance spread (snack031):
  0.5–1 m: 17 · 1–1.5 m: 46 · 1.5–2 m: 61 · 2–3.5 m: 76; per-frame lighting jitter
  visible in the contact sheet.
- **Bake**: `CleanDiftFinetunedFpn` added to all 4 raw renders via
  `python -m isaac_datagen.migrate_descriptors_backbone add-backbone` (run from the
  `isaac_datagen/` repo root — the descriptor config's `../checkpoints/...` ckpt path is
  CWD-relative to a submodule root). `DiftDescriptor` natively baked at render.
- **Squash**: `m2f-squash-vis --out datasets/filtered/vis030 --min-visibility 0.30` from
  `segmentation/` (stale filtered copies deleted first). 0 instances dropped (single
  always-targeted instance — the writer already refuses target-invisible frames); the
  vis030 copies exist for pipeline-path uniformity, with `squash_meta.yaml` provenance.
- **PSC sync**: `./sync-filtered-to-psc.sh --delete` (the script's solo-pool lines).
  Verification: checksummed dry-run (`rsync -n -c`) must itemize zero diffs on the 4
  pools; remote `runtime.yaml` must read `scene_builder: build_scene` (old copies said
  `build_store_scene`). The run also uploads the 4 filtered set1+X rehearsal datasets,
  which proc-6 had never synced.

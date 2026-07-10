# K-shot fine-tune pool: one surround-view render per held-out snack class

**Status 2026-07-09: Config work DONE, renders PENDING.** Revised after review (see below) to 8
configs (4 classes × {all-instances, one-instance} variants), `num_frames` 40→25, and a
`RemovePrims` fix for the one-instance variant. All 8 configs written, no-kit sanity passed
(anchored regex selects exactly 6/6/8/8 instances per class; all 8 configs parse; `RemovePrims`
name counts 5/5/7/7 as expected) and their `dataset_dir`s pre-created. **Renders not launched**:
both GPUs are occupied by the `store-sku-clean-retrain` trainings (launched tonight, ~5h ETA) —
user will launch `isaac-datagen` for each config once a GPU frees up.

## Context

`blues-cereal-generalization-roadmap.md` I.4 names the K-shot data source as **"surround-shot
capture"**: one capture session per novel object, walking around it — the grasp-frame-pose shot
stays the fixed canonical reference, and the rest of that session's shots become the K-shot
fine-tune pool.

`store-snacks-training-datasets.md` (completed) built `set1` (train slice, snack017/020/023/027)
and `set2` (snack031/033/034/035) from the `store001-optflow-objects-keep` catalog. **Set2 is the
frozen held-out validation set** (`store-novel` dev/sacred per roadmap I.2 — the K=0 zero-shot
benchmark and the surface used to select/report K-shot checkpoints) — it must never appear in a
fine-tune's `data.paths`. What this plan builds is the **separate** K-shot training data I.4 calls
for: one capture session per novel object. The already-baked Stage-A/B canonical grasp-frame
reference per class stays a fixed, separately-baked asset and is untouched by this plan.

Review of the first draft (4 configs, all-instances only, `num_frames: 40`, single-object via
`num_targets`) added these corrections:
- Reuse the store scene as-is (shelf + lighting) — still `scene_builder: build_store_scene` with
  the `LookAtPoser` halo camera, same as `set1`/`set2`. Not an isolated/turntable render.
- 40 views/instance was overkill for a K≤25 ladder; reduced to **25** — matches the ladder's
  largest K exactly. The downstream fine-tune takes the ordinal **first K** of these 25 views per
  arm (not a hash-based subset), so K=1 ⊂ K=5 ⊂ ... ⊂ K=25 falls out of capture order for free.
- Added a **second variant per class restricted to exactly one physical instance** — the literal
  "walk around **one** object" reading of I.4, alongside the all-instances variant.
- **Mechanism correction for the one-instance variant:** `num_targets: 1` only controls how many
  instances get sampled as *capture targets* (photographed); it does NOT remove the other physical
  instances of that class from the shelf. `RemoveUntrackedProducts`'s keep-set is derived per-class
  (`store_mutations.py:216`: `tracked = {t.obj.meta["class"] for t in targets}`), so the other
  same-class instances would stay active, just unphotographed. Fix: keep `num_targets: null`
  unchanged and explicitly deactivate the other same-class instances by exact prim name via
  `RemovePrims` — the same mutation `store001-optflow-verify.yaml` already chains after
  `RemoveUntrackedProducts`. After `RemovePrims` runs, only one instance remains active, so
  `num_targets: null` ("every instance of the tracked class, once") naturally yields exactly one
  target.

## Config — 8 files, one template, two variants × 4 classes

Full copy of `configs/store001-optflow-snacks-set2.yaml` (everything not shown below is copied
verbatim — the parity contract):

```yaml
~ dataset_dir: datasets/store001-optflow-snacks-kshot-<class>[-1inst]   # NEW, disjoint from set1/set2
~ num_frames: 25                          # was 40 — matches the K-shot ladder's max K, no more
~ filter_specs:
    - name: RegexFilter
      args: {key: class, value: '^(<class>)$'}   # ONE class only — this IS the isolation change
# scene_builder / scene_builder_args / pose_generation_policy / grasp_frame_policy / seed / idx:
#   UNCHANGED — same store USD, product_patterns/require_tracked_only 7-glob list, LookAtPoser
#   halo box (xrange [0.3,0.9] / yrange [-0.3,0.3] / zrange [-0.2,0.3]) as set1/set2.
```

Variant B (`-1inst` suffix) adds one more mutation after `RemoveUntrackedProducts`, naming every
OTHER instance of that class by its exact live-store prim name (confirmed against
`datasets/store001-optflow-objects-keep/meta/meta_*.yaml`'s `store_prim` field, stripped of the
`/v_0` mesh-child suffix):

```yaml
~ mutations:
    - {name: RemoveUntrackedProducts}
    - {name: RemovePrims, args: {names: [<every instance of <class> except the one kept>]}}
```

| Class | keep (1 instance) | remove (`RemovePrims` names) |
|---|---|---|
| snack031 | `model_snack031` | `model_snack031_{1,2,3,4,5}` |
| snack033 | `model_snack033` | `model_snack033_{1,2,3,4,5}` |
| snack034 | `model_snack034` | `model_snack034_{1,2,3,4,5,6,7}` |
| snack035 | `model_snack035` | `model_snack035_{1,2,3,4,5,6,7}` |

8 files: `configs/store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}.yaml`
(variant A) and the same 4 names + `-1inst` (variant B).

## Frame math (instance counts confirmed against the keep catalog; `num_frames: 25`)

| Class | instances (A) | A frames | instances (B) | B frames |
|---|---|---|---|---|
| snack031 | 6 | 150 | 1 | 25 |
| snack033 | 6 | 150 | 1 | 25 |
| snack034 | 8 | 200 | 1 | 25 |
| snack035 | 8 | 200 | 1 | 25 |
| **Σ** | 28 | **700** | 4 | **100** |

## Verification

```bash
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=<n>

# 0. no-kit sanity: DONE — anchored regex matches 6/6/8/8 instances; all 8 configs parse;
#    RemovePrims name counts are 5/5/7/7.
# 1. render each (dataset_dirs already created)
isaac-datagen configs/store001-optflow-snacks-kshot-<class>[-1inst].yaml idx=0

# 2. contamination gate: variant A should deactivate MORE products than set2's ~387-390 (the
#    other 3 held-out classes too); variant B should deactivate even more (the removed same-class
#    instances on top). cid_to_class must contain EXACTLY {<class>} in both variants.
# 3. eyeball montage (debug_scripts/viz_optflow.py): confirm the shelf/store scene is visibly
#    present. Variant A: only that class appears (any number of its own instances). Variant B:
#    literally ONE physical unit of that class visible anywhere in frame, at every view — the
#    check that actually catches a RemovePrims name mismatch or leftover instance (label/count
#    gates alone did not, per the set1/set2 lesson).
# 4. zero black frames (min mean-luma sample check, same method as set1/set2).
# 5. RemovePrims fail-loud check: confirm the `[MUT] RemovePrims: deactivated N product prim(s)`
#    log line reports the expected count (5 for snack031/033, 7 for snack034/035).
```

## Not in scope here

Feeding these renders into the actual K-shot fine-tunes (`m2fclwf`/`m2flwf` +
`configs/mask2former_lwf_kshot.yaml`/`mask2former_gligen_lwf.yaml`, taking the ordinal first K of
the 25 views per arm, vocab v3 = v2 + these 4 classes for the closed arm) and registering
`store-novel (dev)`/`(sacred)` benchmarks from `set2` are separate follow-on steps once these
renders are verified clean.

## Execution & deviation

Config authoring done directly (not delegated — mechanical, fully grounded in the reviewed plan).
Rendering deferred to the user (GPU contention with tonight's `store-sku-clean-retrain` trainings).
**Stop and ask** before trusting a render if: variant-A instance counts don't match 6/6/8/8,
variant-B leaves more than one instance of the class visible in any frame, `RegexFilter` leaves an
empty catalog, `cid_to_class` contains more than one class, a render comes back all-black, or
`RemovePrims`/`RemoveUntrackedProducts` report different counts than expected above.

## Key changes

`+` `configs/store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}.yaml` (variant A:
single-class `RegexFilter`, `RemoveUntrackedProducts`, `num_frames:25`, `LookAtPoser`) · `+`
`configs/store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}-1inst.yaml` (variant
B: same + `RemovePrims` naming every other same-class instance) · reuses
`store001-optflow-objects-keep` (Stages A/B, untouched) · **no `.py` changes**. `dataset_dir`s
pre-created; renders pending GPU availability.

# Two snacks in-store training datasets (isaac_datagen, Stage C only)

**Status 2026-07-09: COMPLETE.** Both Stage-C configs written and sanity-passed (anchored regex →
25/28 facings; both parse via `load_config`), then rendered in parallel on the 2×RTX-4090 box
(set1→GPU1, set2→GPU0), both exit 0. Final gate PASSED: **set1 1000 frames** / `cid_to_class ==
{snack017,snack020,snack023,snack027}`; **set2 1120 frames** / `cid_to_class ==
{snack031,snack033,snack034,snack035}`; no leaked classes; zero black frames (min mean-luma 29.4 /
22.8 over 40 samples); each render finalized with inline `class_to_descriptors` + `descriptor.yaml`
+ `principal_components` + `class_to_reference*`. The `LookAtPoser` spread was reviewed + approved
(5–46° off the label normal, 0.39–0.90 m). No `.py` changes — config-only. Detail in **EXECUTION
STATUS** below.

**Status update (2026-07-09): CONTAMINATION FOUND + FIXED — datasets regenerated.** Root cause is
this plan's own `(full store as context) | no mutations` decision (table row above, "training wants
context"): with no `mutations` block, every non-catalog store product — including the OTHER set's
held-out snack classes — stayed physically present but unlabeled in every frame, invalidating the
held-out-class study design. All gates were label-set checks (`cid_to_class`), blind to
present-but-unlabeled pixels; the leak surfaced in an obsmask RGB visualization, not a gate. Fix
(plan `valiant-sparking-abelson`): both configs now carry `mutations: [{name:
RemoveUntrackedProducts}]`, keep-set derived from the `filter_specs` 4-class regex; `product_patterns`
widened 6→7 globs (missing `model_drink101*`, never cataloged, let drink101 survive removal); NEW
fail-loud gate `StoreSceneSpec.require_tracked_only` + assertion in `build_store_scene`
(`isaac_datagen/src/isaac_datagen/store_scene.py`) asserts every active product under the tracked
globs has a labeled SKU class at scene-build time, before rendering. Clean regen (2026-07-09): set1
render000 = 1000 frames (390 products deactivated, 4 classes kept), set2 render000 = 1120 frames (387
deactivated); verified cid_to_class exact, no black frames, montage-checked. Contaminated renders
preserved as `contaminated-render000` (not deleted).

## Context

The completed plan `store-perclass-faces-verify-dataset.md` (2026-07-09) built the
**283-object / 42-class keep catalog** `datasets/store001-optflow-objects-keep` (Stages A+B:
per-class front faces baked, isolated references rendered) and proved the Stage-C in-store
capture path end to end. The user now wants two **training** datasets carved from that same
catalog:

1. Pick **four** snacks classes → all are targets → **40 frames each** via `LookAtPoser`.
2. Pick **four different** snacks classes → **40 frames each** via `LookAtPoser`.
   Both use the **default number of instances per class**. The two sets are **class-disjoint**
   (no shared snack class), so set 2's classes are held out from set 1.

This is **Stage C only, config-only** — the catalog already exists (no re-extraction, no
reference re-render), and every knob the request names is an existing config field proven by the
verify render. **No `.py` changes.** Deliverable: two new Stage-C configs + their render dirs.

## Request → existing config knobs (nothing new is invented)

| Request phrase | Knob | Mechanism (existing) |
|---|---|---|
| "four snacks classes, all targets" | `filter_specs: [RegexFilter{key:class, value:^(4 classes)$}]` | `filters.py:105` `RegexFilter` restricts the tracked catalog subset; precedent `store001-optflow.yaml:54` uses `RegexFilter{class: cereal.*}`. `build_store_scene:140` binds **only** that subset as targets. |
| "default number of instances per class" | `num_targets: null` | `capture.py:45` → `np.arange(len(grasp_points))`: **every** facing of each kept class is a target **once** (no `ReplicateFilter`, natural store count). Already merged + proven by the verify render (`runtime_config.py:37`). |
| "40 frames each" | `num_frames: 40` | `capture.py:50-52` `poser(40)`→(40,4,4); `world_poses = einsum('bij,njk->bnik', targets, poses).reshape(-1,4,4)` = **40 poses per target** (per facing). |
| "using the lookatposer" | `pose_generation_policy: LookAtPoser` | `posers.py:47`: each of the 40 poses is a random halo-box offset looking at the grasp origin — varied orientation per frame. |
| (full store as context) | **no `mutations`** | `store_scene.py:4-5` + `:140`: non-catalog products "stay unlabeled background — realistic distractors." Full store present automatically. The verify config's `RemoveUntrackedProducts` isolation was the deviation; training wants context, matching `store001-optflow.yaml` (no mutations). |

**Frame math** (`B` targets × `N=40`): set 1 = 25×40 = **1000 frames**; set 2 = 28×40 = **1120 frames**.

## Class selection (all clean, all −Y aisle-facing; avoids the 3 flagged classes)

Instance counts read from the keep catalog `meta/` (this session):

| Set 1 | facings | Set 2 | facings |
|---|---|---|---|
| snack017 | 8 | snack031 | 6 |
| snack020 | 4 | snack033 | 6 |
| snack023 | 9 | snack034 | 8 |
| snack027 | 4 | snack035 | 8 |
| **Σ** | **25** | **Σ** | **28** |

Deliberately **excludes** the three classes the verify investigation flagged: `snack012` and
`snack032` (label-inward `+Y` facings — a `LookAtPoser` camera on their `+X` grasp normal shoots
*into* the gondola) and `snack025` (coincident twin `snack025_2` → 0-pixel instance). All eight
chosen classes are `−Y` aisle-facing and framed fine in the verify render. Swapping any class =
edit one regex string (no code, no re-extraction).

## Config 1 — `configs/store001-optflow-snacks-set1.yaml` (Stage C)

Full copy of `configs/store001-optflow.yaml` (the store training config; repo pattern is one full
config per scenario — cf. `store001-optflow-{keep,verify,remove,replace}.yaml`). Only these fields
differ from that base:

```yaml
~ dataset_dir: datasets/store001-optflow-snacks-set1   # MUST pre-exist (mkdir in the ladder)
~ num_targets: null            # every facing of each kept class ONCE = "default instances per class"
~ num_frames: 40               # 40 LookAtPoser views per facing → 25×40 = 1000 frames
~ objects_path:
    - datasets/store001-optflow-objects-keep           # the full 283-obj catalog (Stages A/B DONE)
~ filter_specs:                                        # restrict TARGETS to the 4 snacks classes
    - name: RegexFilter                                # inclusion filter on meta["class"] (re.search)
      args: {key: class, value: '^(snack017|snack020|snack023|snack027)$'}   # anchored: exact classes
# ── UNCHANGED from store001-optflow.yaml ────────────────────────────────────────────────────────
#   pose_generation_policy: LookAtPoser  +  xrange:[0.3,0.9] yrange:[-0.3,0.3] zrange:[-0.2,0.3]
#       (the validated store training halo box — varied 0.3-0.9 m aisle views)
#   scene_builder: build_store_scene ; scene_builder_args.product_patterns  (6-cat list; UNUSED
#       without mutations, kept non-empty only for StoreSceneSpec validation, store_scene.py:40)
#   grasp_frame_policy: FixedFaceGrasp {face:"-Y"}  (UNUSED at Stage C — grasp_point is replayed
#       from the baked catalog via add_catalog_grasp_frame, store_scene.py:117; trivial validator)
#   NO mutations block  → whole store active as realistic background distractors
#   mode: optflow ; light_jitter_patterns (store lights 0.25-8×) ; exposure/rt_subframes/warmup
```

## Config 2 — `configs/store001-optflow-snacks-set2.yaml`

Byte-identical to Config 1 **except** the two lines that define which four classes and where they
land — the class-disjoint held-out set:

```yaml
~ dataset_dir: datasets/store001-optflow-snacks-set2
~ filter_specs:
    - name: RegexFilter
      args: {key: class, value: '^(snack031|snack033|snack034|snack035)$'}
```

## No code changes

`num_targets: null` (annotation + `np.arange` branch), `LookAtPoser`, `RegexFilter`, and
`build_store_scene`'s filter-only / full-context binding all already exist and were exercised by
the verify render. This plan adds **two yaml files** and nothing else.

## Verification (cwd = `isaac_datagen/src/isaac_datagen`, GPU 1, EULA accepted)

```bash
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1

# 0. no-kit sanity: the anchored regex selects EXACTLY the intended facings (expect 25 / 28)
uv run python - <<'PY'
import glob, yaml, re
S1 = re.compile(r'^(snack017|snack020|snack023|snack027)$')
S2 = re.compile(r'^(snack031|snack033|snack034|snack035)$')
cls = [yaml.safe_load(open(f))["class"]
       for f in glob.glob("datasets/store001-optflow-objects-keep/meta/meta_*.yaml")]
print("set1 facings:", sum(bool(S1.search(c)) for c in cls))   # 25
print("set2 facings:", sum(bool(S2.search(c)) for c in cls))   # 28
PY

# 1. Stage C — set 1 (full store context, 4 target classes, 40 views/facing)
mkdir -p datasets/store001-optflow-snacks-set1
isaac-datagen configs/store001-optflow-snacks-set1.yaml idx=0
#    expect render000 with 1000 frames; cid_to_class == {snack017,snack020,snack023,snack027}

# 2. Stage C — set 2
mkdir -p datasets/store001-optflow-snacks-set2
isaac-datagen configs/store001-optflow-snacks-set2.yaml idx=0
#    expect 1120 frames; cid_to_class == {snack031,snack033,snack034,snack035}

# 3. eyeball (goal a: targets framed, aisle-facing; per-frame [cid|iid] + [ref|obs])
uv run --extra viz debug_scripts/viz_optflow.py datasets/store001-optflow-snacks-set1/render000
uv run --extra viz debug_scripts/viz_optflow.py datasets/store001-optflow-snacks-set2/render000

# 4. serialized check: cid_to_class holds ONLY the set's 4 classes and NO others;
#    a spot obs frame shows the kept snack facing the camera (label visible, not a shelf back).
```

## Risks / notes

- **Per-process all-black RTX coin flip** (`render-darkness-investigation.md`, unsolved): eyeball
  the first frames of each render; re-run the offending Stage C if black. Not gated here.
- **Fixed store training halo box** (0.3-0.9 m) does not size-adapt; big/small SKUs frame loosely.
  Acceptable for training variety; bump `xrange` if a class frames poorly.
- **Full store context is intentional**: non-target snack facings (incl. the label-inward
  snack012/snack032 and coincident snack025) remain as *unlabeled background distractors* — they
  are never targets and never referenced, so they enrich the scene without polluting labels.
- **Reproducibility**: both configs keep `seed: 1, idx: 0`; the two datasets differ by target
  class set, so their frames are distinct. Give set 2 `idx=1` (or `seed=2`) if independent
  `LookAtPoser` draws are wanted — optional.

## Execution & deviation

Implementation is two config files, delegated to a **Sonnet subagent** armed with this plan; the
orchestrator runs the verification ladder and reports. **If anything unexpected appears** — the
regex selecting the wrong facing count, `RegexFilter` leaving an empty catalog, a rendered
`cid_to_class` containing an unintended class, an all-black render, or `num_targets:null`+
`num_frames:40` tripping a code path not surveyed — **STOP and ask before deviating.**

## Key changes

`+` `configs/store001-optflow-snacks-set1.yaml` (Stage C: `RegexFilter` snack017/020/023/027,
`num_targets:null`, `num_frames:40`, `LookAtPoser`, full store context) ·
`+` `configs/store001-optflow-snacks-set2.yaml` (same, snack031/033/034/035) ·
reuse `datasets/store001-optflow-objects-keep` (Stages A/B) · **no `.py` changes**.

---

# EXECUTION STATUS (2026-07-09)

**Config work: DONE.** `configs/store001-optflow-snacks-set{1,2}.yaml` created as full copies of
`store001-optflow.yaml` with only the planned deltas (`num_targets:null`, `num_frames:40`, keep
catalog, `RegexFilter` to the 4 classes each, **no** mutations). No-kit sanity: the anchored regex
selects exactly **25** (set1) / **28** (set2) facings; both configs load through `RuntimeConfig`
(`num_targets=None`, `num_frames=40`, `LookAtPoser`).

**Renders: launched in parallel, healthy.** set1→GPU1, set2→GPU0 on the 2×RTX-4090 box.
- Both booted identically to the known-good verify Stage-C run, with **no `[MUT]` line** (no
  mutations → full store as background), confirming the filter-only / full-context binding path.
- Early obs frames non-black (set1 luma 50–108, set2 33–82; max ~250).
- **LookAtPoser geometry verified** from `cam2world`: frames `b·40 .. b·40+39` converge on ONE
  target (RMS ray-miss **3.4 mm**; a 5-different-target control scatters to 1.39 m) — every facing
  gets exactly 40 views. Spread **5–46° off the label normal (median 21°), 0.39–0.90 m standoff**;
  user reviewed the montage and approved (not extreme).
- Progress at plan-copy time: set1 ~505/1000, set2 ~416/1120 obs; not yet finalized
  (`cid_to_class` etc. written at the end of the run).

**Final gate (both renders exit 0, verified):**
- set1 `store001-optflow-snacks-set1/render000`: **1000 frames** (25×40); `cid_to_class ==
  {2:snack017, 3:snack020, 4:snack023, 5:snack027}` (no leak); black-gate min mean-luma 29.4.
- set2 `store001-optflow-snacks-set2/render000`: **1120 frames** (28×40); `cid_to_class ==
  {2:snack031, 3:snack033, 4:snack034, 5:snack035}` (no leak); black-gate min mean-luma 22.8.
- Both finalized with inline descriptors (`class_to_descriptors`, `descriptor.yaml`,
  `principal_components`, `class_to_reference`/`_depth`/`_pose`/`_intrinsics`) — the optflow render
  bakes per-class references + descriptors, so no separate descriptor pass is needed for these dirs.

**Deliverables:** `datasets/store001-optflow-snacks-set1/render000` (snack017/020/023/027),
`datasets/store001-optflow-snacks-set2/render000` (snack031/033/034/035); the Stage-A/B catalog
`datasets/store001-optflow-objects-keep` (283 objects / 42 classes) is untouched and reusable.

# Fine-tune renders for the held-out snack classes: 4 solo-1inst pools + 4 set1+X rehearsal sets

**Status 2026-07-11: DONE — all 8 renders complete and verified.** Variant-A kshot yamls deleted,
the 4 `-1inst` yamls reseeded to 1001, the 4 new `set1-plus-*` yamls written; no-kit sanity passed.

**[2026-07-12: solo `-1inst` DATA superseded** — the 4 pools were regenerated in an empty world
with expanded camera ranges (`emptyworld-optflow-snacks-kshot-*-1inst.yaml`, DisablePhysics
mutation, curated catalog grasp frames); store renders parked at `datasets/parked-store-1inst/`.
The store kshot configs and the set1+X family are unchanged. See
`plans/completed/emptyworld-1inst-regeneration.md`.**]**
All 8 renders ran on tesu (2 sequential chains, one per 4090, 12:35–13:52, ~77 min total) and
passed every verification gate: exit 0, `[MUT]` counts exactly as tabled (409/409/407/407 solo ·
384/384/382/382 set1+X), `cid_to_class` exact, frame counts exact (200×4 · 1240/1240/1320/1320),
no black frames (min mean-luma 17.2), montage eyeball clean (every visible product tracked+masked),
and seed-independence vs `set2` confirmed (min camera-position distance 0.25–1.82 cm across all 8 —
no exact pose collisions). Montages + per-render logs: `isaac_datagen/launch-logs/`.

**Post-render processing (same day):**
- **`CleanDiftFinetunedFpn` baked into all 8 raw renders** via `python -m
  isaac_datagen.migrate_descriptors_backbone add-backbone <dataset>
  reference_matching/.../configs/fpn_cleandift_finetuned_123.yaml` — the fpn key the gligen
  training copy (`filtered/vis030/store001-optflow-snacks-set1`) carries; provenance yaml
  byte-identical, per-scale shapes match (1:1280×48×48, 2:640×96×96, 3:320×96×96). The gligen
  checkpoint's own consumed descriptor (`DiftDescriptor`) was already natively baked at render.
- **The 4 set1+X datasets squash-filtered** exactly like set1/set2: `m2f-squash-vis --out
  datasets/filtered/vis030 --min-visibility 0.30` → `segmentation/datasets/filtered/vis030/
  store001-optflow-snacks-set1-plus-<class>/render000` (dropped 50,003/50,462/55,819/54,526
  low-vis instance-events; frame counts preserved; descriptors copied through — filtered copies
  carry both keys). Solo `-1inst` pools deliberately NOT filtered (single always-targeted
  instance; user-scoped to the set1 variants).

## Context

This plan replaces `store-snacks-photoshoot-validation-renders.md` (2026-07-10, deleted), whose
framing the user rejected as convoluted: it cast 4 new per-class renders as **validation** sets and
bolted on a long K-shot-config compatibility review. The clean restatement (user, 2026-07-11):
**8 datasets, all fine-tune inputs** for the held-out-class absorption study
(`segmentation/.docs_claude/plans/active/blues-cereal-generalization-roadmap.md` I.4). The
validation framing is dropped entirely — `set2` (snack031/033/034/035; frozen, eval-only, never in
any fine-tune's `data.paths`) remains *the* validation benchmark; nothing new is rendered for
validation.

The 8 datasets (decisions confirmed with the user via AskUserQuestion, 2026-07-11):

1. **Solo family (4)** — LookAtPoser shots of a **single physical instance** of one held-out class
   each. These are exactly the existing `store001-optflow-snacks-kshot-<class>-1inst.yaml` configs
   from `store-snacks-kshot-surround-renders.md`. Decision: **keep only the -1inst variant**; the 4
   all-facings variant-A yamls were deleted. The K∈{1,5,25} ladder draws from these 200-frame pools.
2. **set1+X family (4)** — NEW: structurally `set1` (the train slice snack017/020/023/027) plus
   **one** held-out class tracked as a 5th class, at set1's full density (40 views/facing,
   user-confirmed). This materializes the **rehearsal** arm that the LwF plan
   (`segmentation/.../lwf-finetune-m2f-gligen-closed.md`) named as a deferred rival: fine-tune on
   old-data + novel mixed, vs LwF's novel-only + distillation. Note `train_frame_limit` has no
   class dimension, so a mixed render cannot express "K frames of the novel class" — set1+X is a
   full-data rehearsal input, not a K-pool; the K ladder runs off the solo family only.

One correctness fix rides along: every snack config used `seed: 1` — the **same effective seed as
`set2` itself** (`effective_seed = seed + idx`, `runtime_config.py:163`; seeds `np.random`
globally; `LookAtPoser` draws poses from it, `vision_core/pose_utils.py`). A fine-tune render of a
set2 class at effective seed 1 risks pose-stream (→ pixel) overlap with the frozen benchmark.

**Seed series:** 1 = original train/benchmark renders (`set1`/`set2`, untouched) · **1001 = solo
fine-tune pools** · **2001 = set1+X rehearsal family**. Wide moats because `effective_seed` is
additive in `idx`. One shared seed within a family is safe: solo configs target disjoint classes;
the set1+X configs are consumed by separate fine-tune runs, so shared set1-portion streams are
redundancy, not leakage.

## Config changes (all in `src/isaac_datagen/configs/`)

### 1. DELETED: the 4 variant-A kshot yamls (user: "only keep the 1inst")

`store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}.yaml` — unrendered, `git rm`'d
(recoverable from history). No `datasets/snackNNN` dirs existed to clean up.

### 2. RESEEDED: the 4 kept -1inst yamls (names/dirs unchanged)

`store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}-1inst.yaml`:

```yaml
~ seed: 1001        # was 1 — 1001-series = solo fine-tune pools; breaks the shared-seed
                    #   pose-stream channel with set2 (both were effective seed 1)
```

Everything else stays: `num_frames: 200`, `dataset_dir: datasets/<class>-1inst`, corrected
`objects_path`, single-class `RegexFilter`, `mutations: [RemoveUntrackedProducts,
RemovePrims{5/5/7/7 names}]` (keeps the unsuffixed `model_snackNNN` prim), 7-glob
`require_tracked_only`. Headers updated (stale variant-A sibling reference removed).

### 3. ADDED: 4 set1+X yamls — diff vs `store001-optflow-snacks-set1.yaml`

`store001-optflow-snacks-set1-plus-{snack031,snack033,snack034,snack035}.yaml` = copy of
`set1.yaml` plus:

```yaml
~ header comment    # set1's 4 train classes + held-out <C> tracked together — rehearsal
                    #   fine-tune render. NOT set2 (eval-only).
~ seed: 2001                  # was 1 — fresh streams vs set1 (1), set2 (1), and solo pools (1001)
  num_frames: 40              # UNCHANGED — set1 parity (user-confirmed full density)
~ dataset_dir: datasets/store001-optflow-snacks-set1-plus-<class>   # NEW, pre-create on render box
~ objects_path:
    - ../../assets/optflow_objects/store001-optflow-objects-keep    # corrected catalog location —
                    #   set1's own `datasets/...` value is stale (catalog moved 2026-07-10) and
                    #   stays stale by standing instruction; do not copy it
~ filter_specs:
    - name: RegexFilter
      args: {key: class, value: '^(snack017|snack020|snack023|snack027|<class>)$'}   # 5 classes;
                    #   RemoveUntrackedProducts derives its keep-set from this (single source of truth)
```

Unchanged from set1's current (contamination-fixed) form, verbatim: `mode: optflow`,
`scene: empty`, `num_targets: null` (every facing of the 5 tracked classes), `idx: 0`,
`mutations: [{name: RemoveUntrackedProducts}]`, the 7-glob `require_tracked_only` +
`product_patterns`, `FixedFaceGrasp -Y` (Stage-C-vestigial), lighting/jitter, `LookAtPoser`
xrange `[0.3,0.9]` / yrange `[-0.3,0.3]` / zrange `[-0.2,0.3]`, placement, proposer/descriptor
config paths, exposure / `rt_subframes: 20` / `warmup_frames: 32`.

## Frame math & expected [MUT] counts

Store inventory under the 7 globs = 415 products. Catalog facings: set1 = 8/4/9/4 = 25;
set2 classes = 6/6/8/8.

| Dataset | tracked facings | num_frames | frames | RemoveUntrackedProducts deactivates |
|---|---|---|---|---|
| snack031-1inst · snack033-1inst | 6 → 1 after RemovePrims | 200 | 200 each | 409 (then RemovePrims −5) |
| snack034-1inst · snack035-1inst | 8 → 1 after RemovePrims | 200 | 200 each | 407 (then RemovePrims −7) |
| set1-plus-snack031 · -snack033 | 25+6 = 31 | 40 | 1240 each | 384 |
| set1-plus-snack034 · -snack035 | 25+8 = 33 | 40 | 1320 each | 382 |
| **Σ (8 renders)** | | | **5920** | |

(Reference: set1 = 1000 frames / 390 deactivated; set2 = 1120 / 387 — the new totals ≈ 5.3× a
set2-sized render, splittable 2-way across tesu's GPUs.)

## Verification (cheapest first)

```bash
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=<n>
# 0. no-kit sanity (DONE 2026-07-11): load_config parses all 8; effective_seed == 1001 / 2001;
#    RegexFilter selects 6/6/8/8 (solo, pre-RemovePrims) and 31/31/33/33 (set1+X) instances.
# 1. pre-create the 4 set1-plus-* dataset dirs (and the -1inst dirs if absent) on the render box
#    (tesu, where set1/set2 rendered); check nvidia-smi --query-compute-apps for squatters first.
uv run isaac-datagen configs/<config>.yaml idx=0
# 2. contamination gate: exit 0 (require_tracked_only trips crash scene build) AND
#    [MUT] RemoveUntrackedProducts: deactivated 409/409/407/407 (solo) / 384/384/382/382 (set1+X).
# 3. label gate: cid_to_class == {<class>} with 1 instance (solo) /
#    {snack017,snack020,snack023,snack027,<class>} (set1+X); frame counts per the table.
# 4. black-frame check: min mean-luma over sampled frames (per-process first-frame coin-flip).
# 5. EYEBALL a montage (obs + cid_mask overlay): solo → exactly ONE instance of ONE class anywhere
#    in frame; set1+X → only the 5 tracked classes' products visible. This is the check that
#    actually catches contamination (the set1/set2 lesson) — label gates alone did not.
# 6. seed-independence spot-check: diff a few obs frames of <class> between snackNNN-1inst /
#    set1-plus-snackNNN and set2's frames of the same class — poses must visibly differ
#    (direct evidence for the 1001/2001 disjointness, not just theory).
```

## Not in scope (carried forward so it isn't lost)

- **Segmentation-side follow-on, REQUIRED before any fine-tune launch:** rebase
  `mask2former_gligen_lwf.yaml` onto `runs/m2f-jul9-2026-917pm/config.yaml`. The checked-in LwF
  yaml inherits HEAD defaults (`ref_encoder_args.channels: [1280, 640, 320]`, `hidden_dim: null`,
  `descriptor: CleanDiftFinetunedFpn`, `ref_scale_set: "1-2-3"`, `resize_shortest_edge: 800`)
  while `m2f-std-store`'s actual arch is `channels: [1280]`, `hidden_dim: 256`,
  `descriptor: DiftDescriptor`, `ref_scale_set: ''`, `resize_shortest_edge: 640` — strict
  `load_state_dict` from `init_from` will crash on the mismatch. Also set `init_from` and
  `data.paths` there.
- The fine-tune launch yamls themselves (rehearsal arm's `data.paths`, `train_frame_limit`,
  `init_from`, the `val_ratio` decision).
- seg-benchmark registration; the store-novel (sacred) carve.
- Back-fixing set1/set2's stale `objects_path` (explicit standing instruction: don't).

## Execution & deviation

Config work was mechanical (direct execution, no delegation). Render launches gated on a live
`nvidia-smi` check on tesu. **Stop and ask** if: `load_config` fails; RegexFilter counts ≠ 6/6/8/8
/ 31/31/33/33; `[MUT]` counts deviate materially from the table; scene build exits nonzero;
`cid_to_class` or frame counts mismatch; any black frame; the montage shows a non-tracked product;
or the seed-independence spot-check finds frames matching set2 (would falsify the seed analysis —
stop and re-derive, don't rationalize).

## Key changes

`-` `configs/store001-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}.yaml` (variant A
deleted) · `~` `configs/store001-optflow-snacks-kshot-{...}-1inst.yaml` ×4 (`seed: 1 → 1001`,
header) · `+` `configs/store001-optflow-snacks-set1-plus-{snack031,snack033,snack034,snack035}.yaml`
(5-class `RegexFilter`, `seed: 2001`, `num_frames: 40`, corrected `objects_path`,
`RemoveUntrackedProducts` + 7-glob `require_tracked_only`) · `-`
`.docs_claude/plans/active/store-snacks-photoshoot-validation-renders.md` (superseded by this plan)
· `~` `store-snacks-kshot-surround-renders.md` (supersession status note) · **no `.py` changes**.

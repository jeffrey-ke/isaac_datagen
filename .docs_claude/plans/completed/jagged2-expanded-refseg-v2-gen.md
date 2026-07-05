# Generate `jagged2-expanded-refseg-v2` — new-seed sibling of jagged-expanded-refseg-v2 (AS-BUILT)

## Context

`jagged-expanded-refseg-v2` (landed by `jagged-columns-jitter-stacker.md`) is a fully-generated
6-render-dir optflow dataset with jagged `UntilExhaustedStacker` column heights, larger per-object
(x,y) placement jitter (`epsilon: 0.01`), a filtered 10x-replicated 8-instance curated `amazon-v2`
catalog, and per-frame light jitter (`jitter_distant`/`jitter_dome`, mechanism documented in
`.docs_claude/lighting-jitter-mechanism.md`). This generates a sibling dataset, `jagged2-expanded-refseg-v2`,
with the identical recipe but a **new seed** so it is genuinely additional/distinct synthetic data
rather than a duplicate.

## Config

`configs/jagged2-expanded-refseg-v2.yaml` — copy of `jagged-expanded-refseg-v2.yaml` with exactly two
diffs:
- `seed: 1` → `seed: 100` (effective_seed 100-105, no overlap with the original's 1-6 — placement,
  pose, and light-jitter RNG streams all key off `effective_seed`, so this decorrelates the two
  datasets).
- `dataset_dir` → `/data/user/jeffk/datasets/jagged2-expanded-refseg-v2`.

Everything else (`mode: optflow`, jagged `placement_args`, `filter_specs`
[`ReplicateFilter(count=10)`+`ShuffleFilter`], curated `amazon-v2` `objects_path`, lighting/exposure
block incl. `jitter_distant`/`jitter_dome`, proposer/descriptor config paths, `inlier_border_eps: 0.0`)
carries over verbatim.

## Generation (DONE)

Smoke-tested first (`idx=0 num_frames=1 num_targets=1` — 1 obs, clean render→proposals→inliers, then
deleted), then the full run, sequential (GPU-bound, one Isaac Sim process at a time):

```bash
cd isaac_datagen/src/isaac_datagen
for IDX in 0 1 2 3 4 5; do
  uv run isaac-datagen-pipeline configs/jagged2-expanded-refseg-v2.yaml idx=$IDX
done
```

**Result (all 6 dirs, exit 0, ~3 min/dir):**

| dir | obs | proposal pts | inliers | rate |
|---|---|---|---|---|
| render000 | 100 | 221760 | 15046 | 6.8% |
| render001 | 100 | 193536 | 12408 | 6.4% |
| render002 | 100 | 238464 | 18374 | 7.7% |
| render003 | 100 | 188352 | 13832 | 7.3% |
| render004 | 100 | 254592 | 20288 | 8.0% |
| render005 | 100 | 221184 | 16224 | 7.3% |

600 obs total; overall inlier rate ~7.3% — consistent with the jagged/amazon convention. No
render/validate_render_dir orphans or errors in any dir.

## Verification

- `lighting_log.json` per dir confirms `seed` 100..105 (vs. the original's 1..6) — direct proof the
  light-jitter (and placement/pose) draws are genuinely new, not a duplicate of
  `jagged-expanded-refseg-v2`.
- `obs/` has 100 frames per dir (600 total); `isaac-datagen-pipeline`'s built-in cid/iid
  `validate_render_dir` gate passed for every dir (no orphans, pipeline would have aborted otherwise).
- `isaac-datagen-inliers` stats written per dir (`stats/stats_0000.json`).

## Critical files

- `isaac_datagen/src/isaac_datagen/configs/jagged2-expanded-refseg-v2.yaml` — NEW render config.
- Reuse, unchanged: `clean_datagen.optflow_generation`, `filters.{ReplicateFilter,ShuffleFilter}`,
  `placers.UntilExhaustedStacker`/`jagged_columns`, `scene.{register_distant_jitter,register_dome_jitter}`,
  `isaac-datagen-pipeline` (render → validate → proposals → inliers, one command per idx).

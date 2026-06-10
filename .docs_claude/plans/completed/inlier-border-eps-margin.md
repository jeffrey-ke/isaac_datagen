# Epsilon border-margin inlier filtering (`MASK_BORDER_EPS` + `inlier_border_eps`)

**Status:** completed (landed in working trees of `isaac_datagen` + `vision_core`; **not yet
committed** as of 2026-06-07).

## Goal

Phase-3 inlier labeling (`add_inlier_data`) previously counted a proposal as an inlier iff it
landed *anywhere* inside its class union mask (`cid_mask == cid`). Grazing points right on the
mask border — common for correspondence proposals near object silhouettes — got labeled inlier
even though they straddle background. Add a **border margin**: a proposal is an inlier only if it
sits ≥ `eps` px inside the mask, and make that `eps` **explicit, mandatory pipeline
configuration** (fail loud; never label with a silent default).

This was executed as two changes; both are summarized here.

---

## Change A — give `coords_in_mask` a border margin + a sweep tool (done inline, no prior plan)

### `vision_core/src/vision_core/mask_utils.py`
- New module constant `MASK_BORDER_EPS: float = 2.0` ("provisional default — tune via
  isaac_datagen's `sweep_label_eps` spot check").
- `coords_in_mask(mask, coords, eps=MASK_BORDER_EPS, *, mask_size=cv2.DIST_MASK_PRECISE)`:
  - Computes `cv2.distanceTransform((mask != 0), DIST_L2, mask_size)` and returns
    `dist[ys, xs] > eps` — i.e. inside the mask **and** ≥ `eps` px from the nearest in-image
    background pixel.
  - **Image frame is NOT a border**: `distanceTransform` only sees in-image zeros, so a
    frame-truncated object stays interior.
  - `eps=0.0` reproduces plain inside-ness exactly (`dist > 0 ⟺ mask[ys, xs]`) — backward
    compatible for non-pipeline callers (`segmentation/preprocess.py` via the re-export).
  - Return container follows `coords` (torch in → torch out); xs/ys clamped to `[0, w/h-1]`.

### `isaac_datagen/src/isaac_datagen/sweep_label_eps.py` (new, untracked, 102 lines)
- Read-only spot check for tuning `MASK_BORDER_EPS`. Deserializes one
  `PreReferenceSegSample` (+ `ObsMaskMetadata`) from a normalized render dir, recomputes the
  per-class inlier labels at several eps via the **exact phase-3 expression**
  `coords_in_mask(cid_mask == cid, coords, eps)`, builds a `PreImageInlierSample` per eps,
  renders each via `.visualize(md)`, and emits per-eps PNGs + a stacked composite + a per-class
  inlier-count table across eps. Never writes the dataset's `labels/`.
- Console entry: `isaac-datagen-sweep-label-eps <render_dir> [--idx 0] [--eps 0 1 2 3 5 8] …`.

**Chosen value: `eps = 2.0`** everywhere (−2.3% inliers on the sweep frame).

---

## Change B — thread `inlier_border_eps` through config + pipeline
(plan: `~/.claude/plans/deep-floating-wozniak.md`)

### `runtime_config.py`
- Added `inlier_border_eps: float` to the **mandatory** (no-default) block of `RuntimeConfig`
  (line 42). `OmegaConf.structured` marks it MISSING, so `load_config`'s `to_object` raises
  `MissingMandatoryValue` if YAML+dotlist don't supply it — mandatory enforcement is free.
- `__post_init__` (line 93): `assert self.inlier_border_eps >= 0`.

### `configs/randomized.yaml`
- `inlier_border_eps: 2.0` (line 13).

### `add_inlier_data.py`
- `sys.argv` parsing → `argparse` with **required** `--eps` (fail loud).
- Threaded into the labeling expression:
  `coords_in_mask(pre.cid_mask == class_to_cid[cls], coords, args.eps)`.
- Switched the residual sample type `ImageInlierSample` → `PreImageInlierSample`
  (`.serialize(idx, dir, only={"labels"})` — `obs/`, `cid_mask/`, `proposals/` never rewritten).
- Provenance: stats catalog now records eps —
  `ImageInlierMetadata(stats={"n_inliers", "n_total", "eps": args.eps})`.
- Docstring/usage updated to `isaac-datagen-inliers <render_dir> --eps E`; notes **no
  skip-if-exists** (re-run atomically overwrites every `labels/` + `stats/`).

### `run_pipeline.py`
- Phase-3 call (line 85):
  `_run("isaac-datagen-inliers", str(render_dir), "--eps", str(runtime.inlier_border_eps))`.
- Dotlist overrides compose for free:
  `isaac-datagen-pipeline randomized.yaml inlier_border_eps=3` flows through `load_config`.

### `vision_core/datastructs.py`
- Comment-only: `ImageInlierMetadata.stats` shape
  `{"n_inliers": int, "n_total": int}` → `… , "eps": float}`. No serializer change (dict→JSON).

### Back-fill + relabel of `expanded-refseg/render000`
- `expanded-refseg/render000/runtime.yaml`: inserted `inlier_border_eps: 2.0` (line 12, at its
  sorted-key position) so the snapshot loads under the new mandatory schema. **render001 NOT
  touched** (no phase-2/3 data there yet — user decision).
- Relabeled render000 with `--eps 2.0`. Result recorded in
  `expanded-refseg/render000/stats/stats_0000.json`:
  `{"n_inliers": 10061185, "n_total": 15612572, "eps": 2.0}`.

---

## Decisions / gotchas

- **Fail loud over silent default**: `--eps` is *required* and `inlier_border_eps` is *mandatory*
  config, even though `clean_datagen` itself doesn't consume it — only the one-command pipeline
  (`run_pipeline`) does. Rationale: labels must never be generated with an unintended margin.
- **`MASK_BORDER_EPS` stays 2.0** in the library — now redundant with explicit config for the
  pipeline, but remains the default for non-pipeline callers (`segmentation/preprocess.py`).
- **Image frame ≠ mask border** — relied on for frame-truncated objects (see `coords_in_mask`
  docstring).
- **No skip logic in `add_inlier_data`** — a plain re-run unconditionally overwrites all
  `labels_*.pt` + `stats_0000.json` (skip-if-exists lives only in phase-2 `add_proposals.py`).

## Verification (performed)

- render000 relabel landed: `stats_0000.json` carries `"eps": 2.0` and updated totals
  (10061185 / 15612572 inliers across 1000 frames).
- `inlier_border_eps` present and mandatory in `runtime_config.py` (+ assert), set to `2.0` in
  `randomized.yaml` and back-filled into `render000/runtime.yaml`.
- `run_pipeline.py` phase-3 passes `--eps {runtime.inlier_border_eps}`.

## Follow-ups / not in scope

- **Commit**: all of the above is still loose in the working trees of `isaac_datagen` and
  `vision_core` (`sweep_label_eps.py` + `expanded-refseg/` are untracked).
- render001 backfill/labels (no phase-2/3 data there yet).

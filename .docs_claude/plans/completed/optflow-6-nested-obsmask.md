# OptFlowSample nests an ObsMask → optflow renders feed run_pipeline phases 2 & 3

**Status: completed & verified 2026-06-17.** Supersedes `optflow-4-mixed-capture.md` (the
"merge two writers in one capture pass" approach was back-burnered in favor of this).

## Goal

Make an optflow render dir directly consumable by the seg pipeline (`run_pipeline.py` phases 2
`add_proposals` + 3 `add_inlier_data`), so the same renders that train the dense-warp/UFM stage also
train seg stages 2/3. Phases 2 & 3 do a *full* `ObsMaskMetadata.deserialize` (all six fields,
including `class_to_descriptors`/`principal_components` — no from-scratch descriptor backfill exists),
so the optflow writer must emit a **complete** `ObsMask` + `ObsMaskMetadata`.

## Design (user's idea)

`OptFlowSample` **nests an `ObsMask`** (replacing `observation`+`cid_mask`+`iid_mask`); `OptFlowMetadata`
nests an `ObsMaskMetadata` (replacing `cid_to_class`+`iid_to_name`). Both serialize **FLAT** so one
physical `ObsMask`/`ObsMaskMetadata` on disk serves both the UFM adapter (`OptFlowSample.deserialize`)
and the seg pipeline (`ObsMask.deserialize(idx, render_dir)`). The masks are load-bearing for UFM (1-to-1
flow needs `iid_mask` to mask out same-class siblings — see [[optflow-ufm-predicts-grasp]]); nesting
shares them with zero duplication and no `only=` juggling.

## Changes (all landed)

1. **`vision_core/datastructs.py`** — `SerializableSample.serialize/deserialize`: a field whose value
   is itself a `SerializableSample` recurses FLAT (same dir + idx), so its subdirs land at the parent
   level. Additive/safe (no existing datastruct nests one; verified by grep). Documented in the
   `serialize` docstring.
2. **`reference_seg_writer.py`** — extracted module-level helpers `reference_catalog`,
   `obsmask_from_data`, `obsmask_metadata`; `ObsMaskWriter` reduced to a thin orchestrator over them
   (pure extraction; reference_segmentation behavior identical). Dropped a dead black-render debug probe.
3. **`objects.py`** — `OptFlowSample = {obsmask: ObsMask, observation_depth, cam2world}`;
   `OptFlowMetadata` drops `cid_to_class`/`iid_to_name`, adds `obsmaskmeta: ObsMaskMetadata`;
   `OptFlowSample.visualize` reads `self.obsmask.*` + `md.obsmaskmeta.*`; dropped unused
   `ReferenceSegSample` import.
4. **`optflow_writer.py`** — `OptFlowWriter` adds the `occlusion` annotator + descriptor args; ctor calls
   `reference_catalog` (the one DIFT forward); `write()` builds the `ObsMask` via `obsmask_from_data` and
   nests it; `finalize_metadata` nests `obsmask_metadata(...)`.
5. **`clean_datagen.py`** — `optflow_generation` passes descriptor config + `full_alpha`, asserts
   `obs_full_alpha` (UFM needs the full unmasked frame), dumps `descriptor.yaml`.
6. **`debug_scripts/viz_optflow.py`** — frame-count glob `observation/` → `obs/`.

## Verification (run, all passed)

- **Change 1 round-trip** (no Isaac): nested `ObsMask` serializes flat (subdirs hoisted, no `child/`
  dir); `ObsMask.deserialize(0, dir)` reads it; parent reconstructs the nested child.
- **Smoke render** `mode=optflow ... idx=950 num_targets=2` → `datasets/debug/render950`: flat layout, all
  ObsMask + ObsMaskMetadata + optflow subdirs at top level, `descriptor.yaml` dumped, DIFT ran.
- **Spot-load** all four datastructs: `OptFlowSample.obsmask` is an `ObsMask`,
  `OptFlowMetadata.obsmaskmeta` is an `ObsMaskMetadata`, `cid_to_class` resolves through the nesting.
- **Decisive:** `isaac-datagen-pipeline` phases 2 & 3 on render950 (phase 1 skipped) → `add_proposals`
  wrote 19,667 proposal points (`proposals/`), `add_inlier_data` labeled inliers (`labels/`, `stats/`).
  Exit 0.

## Notes / accepted

- `ObsMask` references come from the OptFlowObject grasp-anchored render (less canonical than a
  GraspableObject ref) — accepted; flag to dataset owner before scale generation.
- Datastruct change breaks pre-existing optflow render dirs (old flat `observation`/`cid_mask`/`iid_mask`)
  — regenerate; optflow-5 `datasets/debug` dirs are throwaway.

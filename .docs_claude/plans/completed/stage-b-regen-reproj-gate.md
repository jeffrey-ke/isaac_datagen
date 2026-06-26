# Stage B — proposal/label regen under the reproj-coverage gate (COMPLETE)

> **Status (2026-06-25): DONE — reproj gating finished end-to-end across all 3 datasets.** Stage A gate +
> ycb_can ref-pose fix DONE. **Stage B regen DONE on all 12 render dirs** — `mixed-persp` (6), `shelf-optflow`
> (3), and `expanded-refseg` (3): `proposals/`+`labels/`+`stats/` regenerated under the reproj gate + grid
> proposer; `verified_proposals/`+`verification/` dropped everywhere. `expanded-refseg`'s two legacy
> reference-seg dirs (render000/001) were first re-rendered in `mode: optflow` (+ cleandift backbone re-added).
> **Split re-frozen over all 12 dirs** → `datasets/splits/refseg_split.json` (31,698 keys, ~10% val/root).
> Design: `~/.claude/plans/jaunty-skipping-widget.md`. Write-ups:
> `plans/completed/{reproj-coverage-gate-and-ycb-ref-pose-fix.md, expanded-refseg-optflow-regen.md}`.
>
> **Downstream (separate effort, NOT part of reproj gating):** the verifier retrain on the regenerated data
> + `verified_proposals/` re-bake are tracked by `segmentation/.docs_claude/plans/active/hpo-psc-bringup.md`
> (GPU timing job + real sweep array still to run). They consume this regen; they don't gate its completion.

## What's done
- **Stage A gate (code):** reprojection-coverage gate replaces the visible-pixel floor. Files:
  `vision_core/pose_utils.py` (`reproject_local_to_obs`; dropped dead `K` from `reprojection_occlusion`),
  `benchmark/convert.py` (refactored onto the helper — byte-identical), `isaac_datagen/proposal_gate.py`
  (`instance_visibility`, `gate_classes_reproj`), `add_proposals.py` (OptFlow* + reproj gate),
  `runtime_config.py` (`proposer_min_visible_ratio=0.30`, `tau_d/tau_r`), `viz_gate.py` (ratio coloring).
- **Stage A viz:** 1,650 gate-decision PNGs reviewed & approved → `gate_viz/stageA/`.
- **ycb_can ref-pose fix (on disk):** rotated its reference camera 180° about the can's TRUE fitted axis.
  Patched the source asset `$assets/optflow_objects/ycb/ref_pose/ref_pose_0000.npy` (backup
  `.orig.bak`) AND the baked `class_to_ref_pose['ycb_can']` in all 6 `mixed-persp/render00*` dirs.
  Recovered 0.02 → 0.77–1.0. (Scratch script `fix_ycb_ref_pose.py` — not yet formalized.)
- **Stage B regen:** per render dir — delete `proposals/`+`verified_proposals/`+`verification/`, re-run
  grid proposals under the reproj gate, relabel inliers. `mixed-persp` render000–005 DONE (validated
  render005: f0000→9 gated classes, f0019→8, **ycb_can restored** with 14–16 in-mask grid anchors).
- **`expanded-refseg` regen (amazon-only, 3 dirs) DONE** — plan `~/.claude/plans/golden-discovering-donut.md`,
  rendered locally on 2× RTX 4090:
  - **Stage 0:** new `configs/expanded-refseg.yaml` (mirrors render002's proven optflow recipe;
    `UntilExhaustedStacker`/`LookAtPoser`, grid proposer + reproj gate baked, `proposer_device: cpu`).
    Re-rendered the legacy reference-seg `render000` (1000 obs = 100×10) and `render001` (1100 obs = 25×44)
    in `mode: optflow` — full geometry present, `runtime.yaml` finalized. `render002` (300 obs) kept as-is
    (already optflow). Smoke-validated 44-target placement first. The dead pallet `OccupancyGrid` placer is
    gone from code, so the old `pallet_dims=[11,1,4]` layout is replaced by the stacked-column recipe.
  - **Stage 0.5:** re-added `CleanDiftFinetunedDescriptor` (the verifier's ref-token backbone) via
    `migrate_descriptors_backbone add-backbone … cleandift_finetuned.yaml` (render000/001 gained it,
    render002 idempotently skipped). The render only bakes `DiftDescriptor`.
  - **Stage B:** all 3 dirs regenerated — inliers @ eps=0.0 (amazon convention): render000 8.0%,
    render001 7.4%, render002 7.5%; `verified_proposals/`+`verification/` dropped. Gate viz + inlier viz
    spot-checked (`gate_viz_expanded/`).

## In progress
- (none) — Stage B regen complete. Inlier rates: mixed-persp ~3–4% (eps=0), shelf-optflow ~14–18%
  (eps=2) — expected for a deterministic anchor grid (inliers = anchors landing on the class union mask).

## Discoveries
- **ycb_can reference was 180°-about-z wrong** (picked the +Y grasp face; observations see −Y). The gate
  viz surfaced it (ratio ~0% everywhere; scatter showed reference points on the unseen back). Root cause:
  `ref_pose_from_grasp` uses `grasp_point[:3,0]`; ycb_can's grasp face points away. Placement has no yaw
  (`placers.py:70` identity), so one pose flip fixes all instances. **Rotate about the can's fitted axis,
  NOT the local origin** — the origin is offset ~(−0.017,−0.009) from the centerline and `Rz180 @ pose`
  mis-registered the surface by ~2× that (caught visually).
- **The gate is fast** (~7 frame/s for proposals; the Stage A viz was slow only due to matplotlib).
- **Per-dataset inlier eps:** `mixed-persp` `inlier_border_eps=0.0`, `shelf-optflow=2.0` — honor each
  render dir's own value.
- **Bug: `isaac-datagen-viz-inliers` is broken** — passes `classes=` to `vision_core.viz.inlier_figure`,
  which no longer accepts that kwarg → `TypeError`. Worked around by calling `inlier_figure` directly
  (`gate_viz/render005_inliers/sample_00{00,19,50}.png`). Needs a one-line fix.

## Reproduce / resume the regen (per render dir)
```
RM=…/reference_matching/src/reference_matching/configs   ID=…/isaac_datagen/src/isaac_datagen
RD=/data/user/jeffk/datasets/<ds>/render00N ; DS=$(dirname "$RD")
IDX=<N> ; EPS=<that dir's inlier_border_eps>          # mixed-persp 0.0, shelf-optflow 2.0
rm -rf "$RD"/{proposals,verified_proposals,verification}
isaac-datagen-proposals "$RD/runtime.yaml" dataset_dir="$DS" idx=$IDX \
  intrinsics_path=$ID/zed_K.npy proposer_config_path=$RM/grid_proposal.yaml \
  descriptor_config_path=$RM/descriptor.yaml proposer_device=cpu proposer_min_visible_ratio=0.3
isaac-datagen-inliers "$RD" --eps $EPS
```
Base = each render dir's on-disk `runtime.yaml` (faithful per-dir settings; its `idx` already resolves
the dir). 6 overrides, all path/device/threshold (no behavior change): grid proposer, abs `dataset_dir`/
`intrinsics_path`/`descriptor_config_path` (snapshot relatives don't resolve from cwd), cpu, 0.30 ratio.
The new gate fields auto-default since the snapshot predates them.

## Done in this effort
1. ✅ **Stage A** — reproj-coverage gate code + ycb_can ref-pose fix.
2. ✅ **Stage B regen** — grid proposals + reproj gate + inliers on all 12 render dirs; stale
   `verified_proposals/`+`verification/` dropped.
3. ✅ **expanded-refseg** — optflow re-render of render000/001 + cleandift backbone re-add + Stage B.
4. ✅ **Fresh split freeze** — `segmentation.verifier.freeze_split` over all 12 dirs →
   `datasets/splits/refseg_split.json` (31,698 keys). (`dspush splits` left to the user's discretion.)

## Downstream (separate efforts — out of scope for reproj gating)
- **Verifier retrain** on the regenerated grid proposals + **`verified_proposals/` re-bake**
  (`segmentation.verifier.process`) for gligen — tracked by
  `segmentation/.docs_claude/plans/active/hpo-psc-bringup.md` (GPU timing job + real sweep array pending).
- **Cleanups:** formalize `fix_ycb_ref_pose.py` as a tracked `isaac_datagen` migrate script; fix the
  `isaac-datagen-viz-inliers` `classes=` bug; sweep other classes for systematically-low ratios.

## Caveats
- ycb_can fix is **pose-only**: the asset's reference *image/depth* still show the original face (fine for
  the gate, which uses only depth geometry). If an image-consumer (TOTG grasp transfer) ever needs
  ycb_can, re-render the OptFlowObject from the negated grasp normal. Reversible: asset `.orig.bak` +
  invertible rotation.
- Stage B deletions are recoverable: the matching-proposer config still exists to regenerate matched
  proposals; `verified_proposals/`/`verification/` are intentionally dropped (need the retrained verifier).

## Related documents (read these for context)
- **`~/.claude/plans/jaunty-skipping-widget.md`** — the full design plan for this feature (the
  reproj-coverage gate, ref→obs direction, dense pointmap, the `expanded-refseg` re-render decision, the
  split impact). The authoritative source; this active doc is its execution tracker.
- **`isaac_datagen/.docs_claude/plans/completed/reproj-coverage-gate-and-ycb-ref-pose-fix.md`** —
  completed write-up of the gate Stage A + the ycb_can fix, with the deeper design rationale and the
  preserved passages of every throwaway diagnostic/fix script.
- **`segmentation/.docs_claude/plans/active/proposer-visible-px-gate-grid-proposals.md`** — the PRIOR
  gate plan. SUPERSEDED on the gate (px floor → ratio), but its **GridProposal switch** (576-pt anchor
  grid via `reference_matching/configs/grid_proposal.yaml`) is what Stage B's proposals use.
- **`segmentation/.docs_claude/plans/completed/shared-train-val-split-manifest.md`** — the persisted
  `KeySplit` train/val split (`vision_core/split.py`, `segmentation.verifier.freeze_split`,
  `datasets/splits/refseg_split.json`). Read before the "fresh split freeze" next step: keys are
  `(root/render_dir, frame, class)`; regen changes the key universe, so re-freeze over the new data.
- **`benchmark/.docs_claude/plans/completed/{totg-benchmark.md, totg-benchmark-1-many.md,
  totg-ref-point-border-eps.md}`** — TOTG, the origin of the reprojection chain
  (`optflow_metadata_to_totgsamples` + `reprojection_occlusion`) we extracted into
  `vision_core.pose_utils.reproject_local_to_obs`. Also why the pose-only ycb_can fix could break TOTG
  grasp transfer for ycb_can (it reprojects the reference image).
- **`isaac_datagen/.docs_claude/psc-isaac-datagen-footguns.md`** — read before the (HELD)
  `expanded-refseg` re-render: the Isaac/Singularity/PSC operational hazard map (the finalize/timeout
  landmine, optflow config, asset sync).
- **`isaac_datagen/.docs_claude/multiscale-point-descriptor.md`** — the stage-2 verifier design note
  (relevant to the verifier-retrain + `verified_proposals` re-bake step).
- **`isaac_datagen/.docs_claude/plans/completed/artifact-registry.md`** — the `art`/`dspush`/`dspull`
  dataset-artifact registry; how the re-frozen `datasets/splits/refseg_split.json` travels (`dspush
  splits`).
- **CLAUDE.md** in `isaac_datagen/`, `vision_core/`, `segmentation/` — the per-repo module indices and
  the cross-repo dataset contract (`ObsMask`→`PreReferenceSegSample`→`PreImageInlierSample`;
  `OptFlowSample`/`OptFlowMetadata`).

# Extract datagen2_isaacsim → ~/repo/isaac_datagen (+ reference_matching)

## Intent

Peel `visual_servoing/datagen2_isaacsim/` out into a standalone uv package
`~/repo/isaac_datagen`, the third coordinated extraction after `vision_core`
and `segmentation` (see `segmentation/.docs_claude/plans/active/extract-to-standalone-repo.md`).
Pure uv lockfile management, no conda.

Centered on the one entry point actually in use: `clean_datagen.py`, whose
primary mode is `reference_segmentation()`.

## What was done

### isaac_datagen — live closure only
Extracted the 11-module transitive closure of `clean_datagen.py`:
`clean_datagen, scene, capture, pose_planning, runtime_config, objects,
isaac_utils, hardwares, stereo_writer, reference_seg_writer, __init__` plus the
`resources/` asset dir (hard relative dep — `scene.py` loads
`resources/workbench_world.usd` via `RESOURCE_PATH = dirname(__file__)/resources`).
Verified byte-identical (multiset + per-file + post-rename .bak re-check).
Originals + 17 legacy/scratch scripts hidden as `.bak` in
`visual_servoing/datagen2_isaacsim/` (reversible; **not deleted**).

`object_dataset_amazon/` (the GraspableObjects) stays **external** — loaded by
config path via `GraspableObject.deserialize(path)`, not a code dep.

Import rewrites: flat → `isaac_datagen.*` (intra-package),
`datastructs`/`pose_utils` → `vision_core.*`.

### reference_matching — second extraction (the unlock)
The blocker: `reference_segmentation()` runs proposer/descriptor models inside
the Replicator render loop via `ReferenceSegWriter` (lazy
`from segmentation import proposal, descriptor`). But `segmentation` hard-floors
`numpy>=2.4`, `pillow>=12.2`, `torch>=2.11`, while isaacsim 5.1.0 hard-pins
`numpy==1.26.0`, `pillow==11.3.0`, `torch==2.7.0`. Unsatisfiable together.

Resolution: extract just `proposal.py + descriptor.py + gim_helper.py + dift/`
(a self-contained cluster — only consumer inside segmentation was `utils.py`)
into `~/repo/reference_matching` with **deliberately unpinned** torch/numpy/
pillow, so it floats to whatever the consuming env dictates. A feasibility lock
proved `isaacsim==5.1.0 + diffusers 0.37 + transformers 5.9 + lightglue + opencv`
resolves cleanly (numpy 1.26 / pillow 11.3 / torch 2.7).

- `reference_matching`: rewrote `segmentation.{gim_helper,dift}` →
  `reference_matching.*`. Locks (78 pkgs).
- `segmentation`: `utils.py` now imports proposal/descriptor from
  `reference_matching`; dropped `lightglue`+`diffusers` (cluster-only), kept
  `transformers` (grounding_sam_baseline still uses it); added
  `reference-matching` editable + kept the lightglue git source (transitive —
  uv.sources don't propagate from path deps). Locks (113 pkgs).
- `isaac_datagen`: `reference_seg_writer` imports from `reference_matching`;
  depends on `reference-matching` (no longer segmentation); numpy/torch/pillow
  unpinned so isaacsim drives them. Locks (170 pkgs). `reference_segmentation`
  is the default console script.

### Dep graph
```
vision_core  (omegaconf, tqdm)         — shared contract, lightweight
  ↑                  ↑
reference_matching   |   (torch*, diffusers, transformers, opencv, lightglue)
  ↑           ↑      |
segmentation  isaac_datagen (isaacsim==5.1.0, *floats torch→2.7)
```
`* unpinned so reference_matching co-resolves in segmentation's torch-2.11 env
AND isaac_datagen's isaacsim torch-2.7 env.`

## Remaining / not done (needs the user / a machine with isaacsim)

1. **`uv sync` + runtime verification.** All three only *lock*; none synced.
   `isaac_datagen` sync pulls isaacsim[all,extscache] (multi-GB). Then actually
   run `reference_segmentation()` — only locks were verifiable here, not execution.
2. **GIM clone sys.path.** If a proposer config selects a GIM backend
   (`load_gim_dkm/loftr/roma`), `gim_helper` lazily imports `networks.*` / `tools`
   from a GIM checkout that must be on sys.path. ALIKED+LightGlue backends don't.
3. **datagen2_isaacsim leftover.** `visual_servoing/datagen2_isaacsim/` still
   holds data dirs (object_dataset_amazon, generated outputs), docs, configs, and
   all the `.bak` files. Cleanup + the open `~/repo/datagen` duplicate question.
4. **visual-servo-rollout** still imports `from datagen2_isaacsim...` — out of
   scope per user; rewire to `isaac_datagen` when ready.
5. **`.bak` files** across visual_servoing + segmentation — keep until runtime-verified.

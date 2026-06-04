# Add per-instance occlusion to ObsMask

## Context

`clean_datagen.py` (reference-seg path) renders frames where some graspable objects are
barely visible — buried in the `OccupancyGrid` stack or clipped — yet still receive an
instance id in the mask. Downstream, `add_proposals.py:46` feeds the reference image of
**every** present id to the off-the-shelf correspondence proposer, which performs poorly
when only a sliver of the target is visible.

We want a **mechanism, not a policy**: expose how occluded each object is so any consumer
can filter as it sees fit, without `clean_datagen` baking in a "drop small masks" rule.
Concretely: add `id_to_occlusion: dict[int, float]` to the `ObsMask` struct, populated by
the custom writer from Isaac's `occlusion` annotator. Callers (e.g. the proposer pass) may
then skip ids above an occlusion threshold; nothing is dropped at generation time and the
mask stays lossless.

Scope (confirmed with user): **`ObsMask` only.** `PreReferenceSegSample` /
`ImageInlierSample` are left unchanged — they declare their own fields and share the render
dir, so the `id_to_occlusion/` subdir ObsMask writes simply coexists; a consumer wanting
occlusion calls `ObsMask.deserialize(idx, dir)`.

We also add a **visualization spot-check** (`viz_occlusion.py`): deserialize a random sample
across a generated dataset, overlay the `id_mask` on the `obs` with unique per-instance
colors, and show each instance id's occlusion ratio — the user's manual validity check.

## Key finding — the join is by prim path, not by id (verified in a kit probe)

- Objects are a wrapper `Xform` → `geo` prim that **references** the external `.usdz`; the
  `"instance"`/`"class"` semantics are applied to **`geo`** (`scene.py:32-48`, `add_object`).
  Every sampled `.usdz` is **single-mesh** (one renderable `Mesh` leaf under `/World`), so
  there is exactly **one occlusion row per object** — no sub-mesh averaging needed.
- A headless kit probe (three labeled cubes, one partially occluded) established definitively:
  - `instance_segmentation_fast` mask ids reserve 0=BACKGROUND, 1=UNLABELLED; objects got
    ids `{2,3,4}` with `idToLabels[id] = '/World/<name>/geo'`.
  - The `occlusion` annotator (`annotators.py:1214`, dtype
    `(instanceId, semanticId, occlusionRatio)`) keys by **leaf-prim id** `{1,2,3}` —
    `instanceId=3 → 0.91` for the hidden cube, `→ 0.00` for the visible ones.
  - **`OCC.instanceId ≠ SEG mask id`** (offset/different spaces). A naive
    `id_to_occlusion[occ.instanceId] = ratio` would be WRONG.
  - Both sides expose the **prim path**: SEG via `idToLabels[mask_id]`, and
    `omni.syntheticdata.scripts.helpers.get_instance_mappings()` via its `name` field, whose
    `instanceIds[]` are exactly the leaf ids OCC keys on. So the correct, robust join is:
    `OCC.instanceId → (instance_mappings: leaf∈instanceIds) → name/path → (idToLabels) → mask id`.

⇒ The writer maps occlusion onto the **same ids as `id_mask`** by joining on the geo prim
path. Single-mesh means one leaf per object, but we still average leaf ratios defensively so
the code is correct if an asset is ever multi-mesh.

Caveat documented in the writer: occlusion captures object–object + self occlusion only.
It does **not** include out-of-frame truncation (its denominator is the in-frustum render),
so a caller filtering on `id_to_occlusion` won't catch objects merely cut off by the frame
edge. Accepted limitation of this change.

## Changes (as shipped)

### 1. `vision_core` — add the field (`src/vision_core/datastructs.py`, `ObsMask`)

```python
@dataclass
class ObsMask(SerializableSample):
    obs: tv_tensors.Image
    id_mask: tv_tensors.Mask
    id_to_occlusion: dict          # {mask instance id: int -> occlusion ratio in [0,1] (or NaN): float}

    _serializers = {
        **SerializableSample._serializers,
        **ReferenceSegSample._serializers,   # tv_tensors.Mask -> .npy
        **_DICT_PT_SERIALIZER,               # dict -> .pt (preserves int keys)
    }
```

`serialize`/`deserialize` iterate `fields(self)` generically, so the new field auto-writes to
`id_to_occlusion/id_to_occlusion_{idx:04d}.pt`. `_DICT_PT_SERIALIZER` (torch.save/.pt) is used
**not** the default `dict → .json` codec, because JSON stringifies int keys — mirrors
`ObsMaskMetadata.id_to_name`.

### 2. Writer — populate it (`src/isaac_datagen/reference_seg_writer.py`, `ObsMaskWriter`)

- `"occlusion"` annotator added to `self.annotators` (no `init_params`).
- New module helper `_occlusion_by_mask_id(occ, id_to_labels, instance_mappings, present_ids)`
  does the path join: `occ_by_leaf` → `path_to_occ` (via `get_instance_mappings()`) → key by
  the `present` graspable mask ids (`np.unique(seg_hw) & frame_id_to_name`). Missing/NaN → NaN
  so the caller can tell "unknown" from "unoccluded". `write()` calls it and passes the result
  into `ObsMask(obs=, id_mask=, id_to_occlusion=)`.
- `idToLabels` is a sibling key of the seg payload (confirmed via Isaac's `basicwriter.py:539,543`).

### 3. New tool — `src/isaac_datagen/viz_occlusion.py` (+ console script) — see "Visualization" below.

## Visualization script — `viz_occlusion.py`

Standalone spot-check for the occlusion field. **No Isaac / heavy-torch deps** (matplotlib +
numpy + `vision_core`), so it runs outside the sim env. Reuses `viz_inliers.py` helpers
(`composite_over_white`, `_mask_border`, `rgba_chw_to_rgb`) rather than re-deriving them.

What it does, per the final implementation:
- Discovers all `render*/` dirs under `<dataset_dir>` and builds every `(render_dir, frame_idx)`
  pair, then draws a **random** sample (`random.Random(seed).sample`).
- For each sampled frame: `ObsMask.deserialize` + `ObsMaskMetadata.deserialize` (id→name, cached
  per render dir), overlays each present instance's `id_mask` region on the `obs` in a unique
  `tab20` color with a darker border.
- **Prints each instance's occlusion ratio on the box itself** (white-on-black at the mask
  centroid) so buried↔high-ratio is verifiable at a glance.
- **Legend is placed OUTSIDE the image** (right gutter, `bbox_to_anchor=(1.01,1.0)` + widened
  `wspace`) so it never occludes the observation; entries are `id{N} {name}: {occ}`.
- Old/pre-occlusion render dirs (missing `id_to_occlusion/`) are caught per-frame and drawn as
  an error tile instead of aborting the whole figure.

Run it:
```
uv run isaac-datagen-viz-occlusion <dataset_dir> [--n 12] [--cols 3] [--seed 0] \
    [--alpha 0.45] [--dpi 200] [--out PATH]
# or, without re-syncing the console entry:
python -m isaac_datagen.viz_occlusion <dataset_dir> --n 9 --cols 3 --out /tmp/occ.png
```
`<dataset_dir>` is `runtime.dataset_dir` (the dir holding `render000/ render001/ …`). Console
entry registered in `pyproject.toml [project.scripts]` next to `isaac-datagen-viz-inliers`.

## Files touched

- `…/vision_core/src/vision_core/datastructs.py` — `ObsMask` field + `_serializers`.
- `src/isaac_datagen/reference_seg_writer.py` — `occlusion` annotator + `_occlusion_by_mask_id` + `write()`.
- `src/isaac_datagen/viz_occlusion.py` — **new** spot-check viz.
- `pyproject.toml` — new `[project.scripts]` entry (`isaac-datagen-viz-occlusion`).

## Verification (done)

- **Serialization round-trip** (no Isaac): int keys + exact values + NaN all preserved through
  `ObsMask.serialize`/`deserialize`.
- **Writer path join** — first proven via a kit harness (8 real `.usdz` boxes labeled like
  `scene.py`, shared class `"amazon_box"` → confirms the path join is needed, not a class-label
  join), then via the **real `isaac-datagen` reference-seg run**: ~30 frames, each with sensible
  `id_to_occlusion` (front-of-wall ≈ 0.0, buried ≈ 0.8–1.0).
- **Spot-check viz** rendered from the real dataset; per-box centroid numbers agree with the
  raw `.pt` maps and the buried↔high-ratio correspondence reads clearly.

## Backward compatibility

`id_to_occlusion` is a required (non-default) field, so `ObsMask.deserialize` now requires the
`id_to_occlusion/` subdir — **datasets generated before this change can't be read as `ObsMask`**;
regenerate (this is a datagen pipeline).

## Environment note (run-blocking, resolved outside this change)

The reference-seg entry constructs `ObsMaskWriter`, whose `__init__` imports the DIFT descriptor
(`from reference_matching import descriptor`). After `reference_matching` moved detectron2 /
mask2former to **top-level** imports, that transitively loads compiled ops (MSDeformAttn,
`detectron2._C`) which **must match Isaac Sim's bundled torch** (2.7.0+cu128) — not the 2.11
that `reference_matching`'s own venv uses. Symptom chain when mismatched: `undefined symbol:
…incref_pyobjectEv` (MSDA built vs newer torch), then `libcudart.so.13: cannot open shared
object file` (detectron2 wheel built vs CUDA 13). Resolved in `pyproject.toml`:
`constraint-dependencies = ["detectron2==0.6+fd27788pt2.7.0cu128"]` (pins the wheel's
`pt<torch><cuda>` tag to this env's torch) + `no-build-isolation-package =
["multiscaledeformableattention"]` (MSDA builds against the env torch). See the comments in
`pyproject.toml [tool.uv]` for the full rationale — the detectron2 pin must move in lockstep
with torch. The occlusion feature itself is independent of this and was verified separately via
the kit harness that bypasses the descriptor import.

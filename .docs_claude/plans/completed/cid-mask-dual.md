# Dual-mask ObsMask: add class-id mask alongside instance-id mask

## Context

The 44 `GraspableObject`s were relabeled from one shared `class="amazon_box"` into 19
color classes (see `plans/completed/relabel-graspable-classes.md`) because many reference
images are near-duplicates. The datagen pipeline still keys everything by *instance* name
(`amazon_0xx`): the proposer runs once per instance (redundant forwards for same-class
boxes), and phase-3 labels a proposal an outlier when it lands on a *different but
visually identical* box — label noise that the relabel was meant to fix.

Fix: `ObsMask` carries **both** masks. The instance-id mask keeps `iid_to_occlusion`
honest (occlusion is intrinsically per-instance); the new class-id mask drives proposals,
inlier labeling, and training, where `cid_mask == cid` yields the union mask of all
same-class boxes.

Isaac's `semantic_segmentation` annotator cannot produce the class mask for our scene:
it assigns ids by the *full* semantic-set string, and our prims carry both `class` and
`instance` labels (`scene.py:46-47`), making every box's string unique (verified in
installed source `omni.replicator.core-1.12.27/.../OgnSemanticSegmentation.py:150-159,279-280`).
Instead the writer remaps the instance mask through a LUT built from `idToSemantics`
(which already carries the class per instance id) — the same computation Isaac's node
runs, but keyed on `class` alone, with a numbering we control.

## Naming convention (applies to all new/renamed symbols)

Two integer id spaces coexist; bare "id" is banned. Qualifiers:

- **`iid`** — instance id: session-local int Isaac assigns per prim in
  `instance_segmentation_fast` (the values in `iid_mask`).
- **`cid`** — class id: our deterministic int per class name (the values in `cid_mask`).
- `name` — instance name string ("amazon_003"); `class` — class name string ("red").

Existing fields/symbols in the reference-seg flow are renamed to match (`id_mask` →
`iid_mask`, `id_to_occlusion` → `iid_to_occlusion`, `id_to_name` → `iid_to_name`, …).
Subdir names follow field names; regeneration is already forced (see Breaking change).

## Settled decisions (from design discussion)

1. Class ids start at **2**, mirroring Isaac's `0=BACKGROUND, 1=UNLABELLED` convention;
   `cid_mask` dtype **uint8** (19 classes ≪ 255; ¼ the size of the int32 instance mask).
2. Honest **field renames** in metadata (`class_to_ref` etc.) — the new `cid_mask/`
   subdir already forces regeneration of old render dirs, so no compat reason to keep names.
3. `PreReferenceSegSample` / `ImageInlierSample` **swap** `id_mask` → `cid_mask`
   (phases 2/3 operate purely in class space; carrying both = dead npy read per frame).
4. Metadata includes `name_to_class` (instance name → class) — completes the relational
   schema linking the two id spaces (e.g. future per-class occlusion filtering).
5. Canonical reference per class = first member by sorted `meta["name"]` (members are
   near-duplicates by construction).
6. `proposals` / `labels` dicts become keyed by **class name** ("red").
7. Drop the bijection guard in `add_inlier_data.py` (cid ↔ class is 1:1 by construction).
8. Class ids are deterministic across render dirs only because they derive from the sorted
   class set of the scene's objects — comment this in the writer.
9. Id-space naming qualifiers (`iid`/`cid`) as above, applied pipeline-wide.

## Interaction with the occlusion mechanism (`plans/completed/obsmask-occlusion-and-viz.md`)

That plan shipped per-instance occlusion as a *mechanism* whose intended consumer is
occlusion-threshold filtering in the proposer pass (policy not yet wired in). This change
keeps that mechanism lossless: the per-instance bridge (`_occlusion_by_iid`, path join,
`occlusion` annotator) is untouched, and `iid_to_occlusion` stays keyed by `iid_mask`
values. When the filtering policy lands in the now-class-space `add_proposals`, the
cross-space join is `iid_to_occlusion → iid_to_name → name_to_class → per-class aggregate`
(e.g. min = "best-visible member") — `name_to_class` is in the metadata precisely for this.

## Changes

### 1. `~/repo/vision_core/src/vision_core/datastructs.py`

- **`ObsMask`** becomes:
  ```python
  obs: tv_tensors.Image
  iid_mask: tv_tensors.Mask        # instance-id mask (renamed from id_mask)
  cid_mask: tv_tensors.Mask        # NEW: class-id mask, uint8
  iid_to_occlusion: dict           # renamed from id_to_occlusion; {iid → ratio}
  ```
  Coerce `cid_mask` in `__post_init__` like the others. Docstring: `iid_mask` pairs with
  `iid_to_occlusion`; `cid_mask` pairs with `ObsMaskMetadata.cid_to_class`; union mask
  derived downstream as `cid_mask == cid`.
- **`PreReferenceSegSample`** and **`ImageInlierSample`**: field `id_mask` → `cid_mask`
  (docstrings: shares `obs`/`cid_mask` subdirs with ObsMask; `proposals`/`labels` keyed
  by class name).
- **`ObsMaskMetadata`** becomes:
  ```python
  iid_to_name: dict            # {iid → instance name} (session-local, as today; renamed)
  cid_to_class: dict           # {cid → class name} — interprets cid_mask
  name_to_class: dict          # {instance name → class name}
  class_to_ref: dict           # {class name → tv_tensors.Image (RGBA) canonical ref}
  class_to_descriptors: dict   # {class name → torch.Tensor DIFT features of canonical ref}
  ```
  (all dicts → existing `_DICT_PT_SERIALIZER`; no serializer changes needed anywhere —
  the Mask `.npy` serializer round-trips uint8 fine.)

### 2. `src/isaac_datagen/reference_seg_writer.py` (heart of the change)

`__init__` — static class catalog from `object_specs`:
```python
classes = sorted({obj.meta["class"] for obj in object_specs})
self.class_to_cid = {cls: cid for cid, cls in enumerate(classes, start=2)}  # 0=bg, 1 unused (Isaac convention)
self.cid_to_class = {cid: cls for cls, cid in self.class_to_cid.items()}
self.name_to_class = {obj.meta["name"]: obj.meta["class"] for obj in object_specs}
self.class_to_ref = {}
for obj in sorted(object_specs, key=lambda o: o.meta["name"]):
    self.class_to_ref.setdefault(obj.meta["class"], _pil_to_tv_rgba(obj.reference_image))
```
- `names_to_descriptors` → `class_to_descriptors`: DIFT forward only on the ~19 canonical
  refs instead of 44.
- `self.id_to_name` → `self.iid_to_name`; per-frame accumulation kept exactly as today.
- Helper rename: `_occlusion_by_mask_id` → `_occlusion_by_iid` (`present_ids` →
  `present_iids`); logic untouched.

`write()` — one LUT remap added; everything else (occlusion bridge, alpha compositing,
ObsMask serialization) unchanged in mechanism. `frame_id_to_name` → `frame_iid_to_name`.
```python
iid_to_cid = {int(k): self.class_to_cid[v["class"]]
              for k, v in labels.items()
              if "class" in v and v["class"] in self.class_to_cid}
lut = np.zeros(max(int(seg_hw.max()), max(iid_to_cid, default=0)) + 1, dtype=np.uint8)
for iid, cid in iid_to_cid.items():
    lut[iid] = cid
cid_mask = tv_tensors.Mask(torch.from_numpy(lut[seg_hw]))
ObsMask(obs=obs, iid_mask=iid_mask, cid_mask=cid_mask, iid_to_occlusion=...).serialize(...)
```

`finalize_metadata()` — serialize the five-field `ObsMaskMetadata`.

### 3. `src/isaac_datagen/add_proposals.py` (~4 lines)

```python
present_cids = {int(i) for i in om.cid_mask.unique().tolist()} & set(md.cid_to_class)
names        = sorted({md.cid_to_class[c] for c in present_cids} & set(md.class_to_ref))
ref_b        = md.class_to_ref[name]...
PreReferenceSegSample(obs=om.obs, cid_mask=om.cid_mask, proposals=...)
```
The existing set-dedup now genuinely collapses N same-class boxes → one proposer call.

### 4. `src/isaac_datagen/add_inlier_data.py`

```python
class_to_cid = {cls: cid for cid, cls in md.cid_to_class.items()}   # 1:1 by construction
labels = {name: coords_in_mask(pre.cid_mask == class_to_cid[name], coords)
          for name, coords in pre.proposals.items()}
```
Drop the bijection guard. `pre.cid_mask == cid` is the union mask → a proposal landing
on *any* same-class box is an inlier (the motivating fix).

### 5. `~/repo/segmentation-train/src/segmentation/dataset.py` (`RenderDirReferenceSegDataset`)

- Index build: `cid_mask` uniques ∩ `set(self.md.cid_to_class)` → `(frame, cid)` pairs.
- `__getitem__`: `cls = self.md.cid_to_class[cid]`; `ref_rgb=self.md.class_to_ref[cls]`,
  `seg_mask=tv_tensors.Mask(pre.cid_mask == cid)` (class-union),
  `proposal_coordinates=pre.proposals[cls]`,
  `reference_features=self.md.class_to_descriptors[cls]`.

### 6. `src/isaac_datagen/viz_inliers.py`

Switch to class space: `s.cid_mask`, `md.cid_to_class`, `md.class_to_ref` — panels and
legend become per-class (one color per class; union regions render together).

### 7. `src/isaac_datagen/viz_occlusion.py` — mechanical renames only

Stays in instance space (`iid_mask` + `iid_to_name` + `iid_to_occlusion`) — this is the
dual-mask payoff; only the field renames touch it. Old dirs (missing `cid_mask/` or using
old field names) fall into its existing per-frame error-tile handling — same graceful
degradation it already provides for pre-occlusion dirs missing `id_to_occlusion/`.

### Untouched

`scene.py` (both labels already attached), `capture.py`, `pose_planning.py`,
`clean_datagen.py` entry points, stereo pipeline, `objects.py`/`GraspableObject`
(relabeled classes flow in via `meta["class"]`).

### Breaking change

Old render dirs lack `cid_mask/` and use the old field/subdir names → cannot deserialize
with new structs. Regenerate (datasets are cheap to re-render; the relabel made
regeneration necessary for correct semantics anyway). Precedent: the occlusion change
took the same regenerate stance.

## Plan-file bookkeeping

Copy this plan to `.docs_claude/plans/active/cid-mask-dual.md` at implementation
start (project convention); move to `plans/completed/` with an Outcome section when done.

## Verification

1. **Render a small dir**: `uv run clean_datagen.py <config.yaml> idx=900 num_frames=<small>`
   into a scratch `dataset_dir`. Check on the output dir (small python snippet):
   - `cid_mask/` exists; per-frame uniques ⊆ `{0} ∪ cid_to_class.keys()`; dtype uint8; no value 1.
   - Metadata: `class_to_ref`/`class_to_descriptors` keyed by the scene's classes;
     `cid_to_class` cids start at 2; `name_to_class` covers all placed objects.
   - Cross-mask consistency: for a frame with two same-class boxes,
     `(cid_mask == cid)` ⊇ each member's `(iid_mask == iid)` and equals their union.
   - `iid_to_occlusion` keys still match `iid_mask` values (instance side untouched).
2. **Phase 2**: `isaac-datagen-proposals <config.yaml> idx=900` — proposals keyed by class
   names; frames with k same-class boxes log one proposer call per class, not per box.
3. **Phase 3**: `isaac-datagen-inliers <render_dir>` — runs without the guard; spot-check
   that a proposal inside *either* same-class box is labeled inlier.
4. **Viz**: `viz_inliers.py` (per-class panels) and `viz_occlusion.py` (per-instance,
   renames only) both render against the new dir.
5. **Training smoke test**: construct `RenderDirReferenceSegDataset` on the new dir,
   pull a few samples, check `seg_mask` is the class union and ref/features match the class.

## Outcome (2026-06-04)

All seven files landed as planned (vision_core datastructs; reference_seg_writer;
add_proposals; add_inlier_data; segmentation-train dataset; viz_inliers → class space;
viz_occlusion → renames only). Verified end-to-end on a 4-frame render at
`src/isaac_datagen/cid-mask-verify/render900` (config `randomized.yaml idx=900
num_targets=1 num_frames=4`):

- **Structs**: NN-free round-trip OK (uint8 cid_mask, int32 iid_mask, NaN occlusion,
  residual proposals/labels writes; subdirs `cid_mask/ iid_mask/ iid_to_occlusion/ …`).
- **Render**: 19 classes, cids 2–20 sorted; per frame 12–13 instances → 10 classes
  (union genuinely exercised); `cid_mask == cid` equals the union of member
  `iid_mask == iid` masks exactly; `iid_to_occlusion` keys == present iids.
- **Phase 2**: one proposer call per class (10/frame, was 12–13), 192,220 pts.
- **Phase 3**: 78,378/192,220 inliers; labels == union-membership for every
  (frame, class); **14,542 inlier points land on non-canonical same-class siblings** —
  the points the old instance-keyed labeling would have mislabeled as outliers.
- **Viz**: per-class inlier panels + per-instance occlusion grid both render.
- **Training**: `RenderDirReferenceSegDataset` → 40 (frame, class) samples; union
  seg_mask; ref/features keyed by class.

### Environment issues found during verification (pre-existing, not from this change)

1. `isaac-datagen-proposals` ImportError: the pinned detectron2 wheel ships a top-level
   `tools/` package (its training scripts) in site-packages, shadowing gim's flat-layout
   `tools` module (`.pth` paths resolve AFTER site-packages). Workaround used:
   `PYTHONPATH=/home/jeffk/repo/gim uv run isaac-datagen-proposals …`. Durable fix TBD
   (e.g. import gim's tools by explicit path in reference_matching/proposal.py).
2. The login shell exports a PYTHONPATH containing isaacsim's `pip_prebundle`, which
   shadows segmentation-train's venv torch (2.7.0 vs expected) and breaks
   xformers/diffusers imports there. Workaround: `env -u PYTHONPATH uv run …`.

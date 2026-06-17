# Add cid/iid segmentation masks to the OptFlow dataset

**Status: completed 2026-06-17.** Verified end-to-end with an optflow render into
`datasets/debug/render900` (3 frames): per-frame `cid_mask/`+`iid_mask/` and metadata
`cid_to_class/`+`iid_to_name/` serialize correctly; `cid_to_class={2:amazon_orange, 3:cheezit,
4:mustard}` (cids start at 2, sorted-class convention); nonzero cids/iids are subsets of their
catalogs and resolve to classes present in `class_to_reference`; masks share the observation's
`(1080,1920)`. `OptFlowSample.visualize` renders the `[cid_mask | iid_mask]` legend row above the
1-to-many warp rows (see `datasets/debug/render900_viz0.png`).

## Context

`optflow_writer.py:61-62` is a half-finished edit: `OptFlowWriter.write()` constructs an
`OptFlowSample(observation=…, cid_mask, iid_mask, …)` with two bare names that are never assigned
(`NameError` at runtime). The matching unstaged diff in `objects.py` already declared the two new
fields on `OptFlowSample` (`cid_mask`, `iid_mask`, objects.py:200-201) and two new untyped fields on
`OptFlowMetadata` (`cid_to_class`, `iid_to_name`, objects.py:314-315) but stopped there.

**Why:** the OptFlow dataset is currently a *1-to-many* contract — one canonical reference per class
warps into every same-class instance in the observation (`OptFlowMetadata`, `class_to_l2w`). The
downstream UFM-adapter step (out of scope here) needs to split that into *1-to-1* `(reference,
single-instance)` correspondence pairs. To do that it must know **which observation pixels belong to
which instance** — exactly the per-pixel instance-id (`iid_mask`) and class-id (`cid_mask`) masks that
the reference-seg pipeline already emits as `ObsMask`. This task wires up those masks **in the exact
same pattern as `ObsMask`/`ObsMaskWriter`** so the OptFlow samples carry an obs id-map alongside the
RGB-D + pose.

The capture path already labels objects with both `class` and `instance` semantics
(`scene.py:49-50`, shared by `optflow_generation` via `build_scene`), so the
`instance_segmentation_fast` annotator yields the same `idToSemantics` payload `ObsMaskWriter`
consumes — no scene changes needed.

## Reference pattern (verified working): `ObsMaskWriter` (reference_seg_writer.py:84-217)

- `__init__`: deterministic cid map from the SORTED class set, starting at 2
  (`class_to_cid`/`cid_to_class`, lines 113-116); `self.iid_to_name = {}` accumulator (line 107);
  adds the `instance_segmentation_fast` annotator with `init_params={"colorize": False}` (lines 96-99).
- `write()` (lines 156-194): reads `seg_hw = rp["instance_segmentation_fast"]["data"]` and
  `labels = …["idToSemantics"]`; builds `frame_iid_to_name` (filter on `"instance"` key) and
  `frame_iid_to_cid` (filter on `"class"`); LUT-remaps `seg_hw → cid_mask`; `iid_mask =
  Mask(seg_hw.astype(int32))`; accumulates `self.iid_to_name`.
- `finalize_metadata()` (lines 200-216): writes `cid_to_class` + `iid_to_name` into the metadata.

## Changes

### 1. `objects.py` — `OptFlowSample._serializers` (objects.py:205-206)  ⚠ required, not in original diff

The new `cid_mask`/`iid_mask` are `tv_tensors.Mask`. `_get_serializer` (datastructs.py:91-97) matches
by exact type/origin — `tv_tensors.Mask` does NOT fall back to the `torch.Tensor` entry — so the
prior `_serializers = SerializableSample._serializers` raised `KeyError: No serializer for
tv_tensors.Mask`. Mirror `ObsMask`, reusing the already-imported `ReferenceSegSample` (objects.py:14):

```python
_serializers = {
    **SerializableSample._serializers,
    **ReferenceSegSample._serializers,   # adds tv_tensors.Mask → .npy
}
```

### 2. `objects.py` — `OptFlowMetadata` field types (objects.py:314-315)

`_serializers` already mixes in `_DICT_PT_SERIALIZER` (objects.py:318) → both serialize as `.pt`
(int keys preserved):

```python
cid_to_class: dict                # {cid: int → class name: str}; pairs with OptFlowSample.cid_mask
iid_to_name: dict                 # {iid: int → instance name: str}; session-local
```

### 3. `isaac_utils.py` — new shared helper `cid_iid_masks` (extract & share)

Pulled the cid/iid derivation that was inline in `ObsMaskWriter.write()` into one pure function so
both writers call it. New imports in `isaac_utils.py`: `import torch`, `from torchvision import
tv_tensors`.

```python
def cid_iid_masks(seg_hw, labels, class_to_cid):
    frame_iid_to_name = {int(k): v["instance"] for k, v in labels.items() if "instance" in v}
    frame_iid_to_cid = {
        int(k): class_to_cid[v["class"]]
        for k, v in labels.items()
        if "class" in v and v["class"] in class_to_cid
    }
    lut = np.zeros(max(int(seg_hw.max()), max(frame_iid_to_cid, default=0)) + 1, dtype=np.uint8)
    for iid, cid in frame_iid_to_cid.items():
        lut[iid] = cid
    cid_mask = tv_tensors.Mask(torch.from_numpy(lut[seg_hw]))
    iid_mask = tv_tensors.Mask(torch.from_numpy(seg_hw.astype(np.int32)))
    return iid_mask, cid_mask, frame_iid_to_name
```

The empty-frame guard and `self.iid_to_name` accumulation stay in each writer (writer state, not
pure). The helper carries NO occlusion logic — that is ObsMask-only.

### 4. `reference_seg_writer.py` — `ObsMaskWriter.write()` calls the helper

Module-level `from isaac_datagen.isaac_utils import cid_iid_masks`; the inline block became:

```python
iid_mask, cid_mask, frame_iid_to_name = cid_iid_masks(seg_hw, labels, self.class_to_cid)
if not frame_iid_to_name:
    raise ValueError("write() called with no labeled instances — expected ≥1")
self.iid_to_name.update(frame_iid_to_name)
```

Occlusion (still uses `frame_iid_to_name`), `composite_rgba`, and the `ObsMask(...)` serialize are
unchanged — pure extraction, behavior identical.

### 5. `optflow_writer.py` — `OptFlowWriter`

- Module docstring updated (it had claimed "No instance-segmentation annotator").
- `__init__`: added the `instance_segmentation_fast` annotator (`colorize=False`); built
  `class_to_cid`/`cid_to_class` from `sorted({o.meta["class"] for o in objects})` (start=2); added
  `self.iid_to_name = {}`.
- `write()`: derive masks via `cid_iid_masks`, loud empty-frame guard, accumulate `iid_to_name`,
  assign `cid_mask=`/`iid_mask=` by keyword. NO `occlusion` annotator (`OptFlowSample` has no
  `iid_to_occlusion`).
- `finalize_metadata()`: pass `cid_to_class=self.cid_to_class`, `iid_to_name=self.iid_to_name`.

### 6. `objects.py` — `OptFlowSample.visualize` mask legend (follow-up)

Added a top `[cid_mask | iid_mask]` row to the figure: a local `draw_id_mask(ax, mask, id_to_label,
name)` colors each nonzero id (tab20) on a black background and attaches a legend (`{id}:
{cid_to_class/iid_to_name[id]}`). Panel grid grew to `2 + 2*len(rows)`; warp rows shifted to start
at `axes[2]`.

## Verification (run)

```
uv run clean_datagen.py configs/randomized.yaml mode=optflow \
  dataset_dir=<abs>/datasets/debug \
  intrinsics_path=<abs>/src/isaac_datagen/zed_K.npy idx=900 num_frames=3
```

Note: `randomized.yaml` no longer provides `mode` or `dataset_dir` (both mandatory, no default) — supply
via dotlist; `intrinsics_path` is relative in the yaml so pass an absolute override unless CWD is
`src/isaac_datagen`. All mask/consistency assertions passed (see Status block above).

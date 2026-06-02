## Keywords / Tags
- datagen
- isaac-sim
- replicator
- reference-segmentation
- dift
- descriptor
- writer
- serialization
- plan-active
- refactor
- gotchas

> **SUPERSEDED (2026-06-01)** by the normalized-dataset redesign
> (`~/.claude/plans/src-isaac-datagen-reference-seg-writer-abundant-church.md`).
> The per-(frame,object) `ProtoReferenceSegWriter` loop was deleted entirely: the writer
> is now `ObsMaskWriter` (one NN-free `ObsMask` per frame + a per-render-dir
> `ObsMaskMetadata` catalog), the proposer moved to a deferred phase-2 pass
> (`add_proposals.py`), and the shared datastructs live in `vision_core.datastructs`.
> Kept here for the verified external contracts (DIFT/serializer notes) only.

# Plan: finish the `ProtoReferenceSegWriter` pseudocode so it runs

## Context

`clean_datagen.py`'s `reference_segmentation()` was switched from `ReferenceSegWriter` (descriptor **+** proposer, run per-frame inside Isaac's render step) to a new `ProtoReferenceSegWriter` that **precomputes constant reference (DIFT) features once at init, drops the descriptor, and then does NN-free per-frame writes** — producing `ProtoReferenceSegSample`s. This is the cheap "phase-1 capture" writer from earlier discussion: the only neural-net cost is a one-time descriptor pass over the reference images at construction; the per-frame `write()` just composites RGBA + slices masks + serializes.

The pseudocode is ~90% there but won't run: a dead `__post_init__`, an `__init__` signature still carrying the removed proposer params, GPU/CPU mismatch against the serializer, missing imports, and backtick prose notes that are invalid syntax. This plan fixes those run-blockers and applies the agreed workbench-alpha fix. **No design changes beyond what's listed.**

External contracts verified this session:
- `reference_matching` `DiftDescriptor.__call__` (`dift/descriptor.py:40-52`): accepts uint8 RGBA `(B,4,H,W)`, internally strips alpha → resize 512 → float32 [0,1] → normalize [-1,1]; returns `(B, N, C)` float32 **on the input device**. `.squeeze()` on batch-1 → `(N, C)`. ⇒ no caller-side float/alpha handling needed.
- `vision_core` `SerializableSample.serialize(idx, dir)`: per-field subdir, file `{field}_{idx:04d}{ext}`, serializer chosen by **exact type identity** on the annotation object. `torch.Tensor`→`.npy` via `.float().numpy()`, `tv_tensors.Image`→`.png`, `tv_tensors.Mask`→`.npy` — **all require CPU tensors**. Exact-identity lookup ⇒ annotations must be real type objects (no `from __future__ import annotations`).

## Files & changes

### 1. `src/isaac_datagen/objects.py`
`ProtoReferenceSegSample` (lines 22-32) annotates `tv_tensors.Image`, `tv_tensors.Mask`, `torch.Tensor`, but the module imports neither.
- Add `import torch` and `from torchvision import tv_tensors` to the top imports.
- Do **not** add `from __future__ import annotations` — the serializer's exact-identity lookup needs the annotations to stay real type objects (matches how existing `GraspableObject` works).
- The `_serializers` merge is correct as written: `Image`/`Tensor` come from `SerializableSample`, `Mask` from `ReferenceSegSample`.

### 2. `src/isaac_datagen/reference_seg_writer.py`
- **Imports:** add `from isaac_datagen.objects import ProtoReferenceSegSample` (currently undefined at line 111, → `NameError`). Remove the now-unused `from vision_core.datastructs import ReferenceSegSample`.
- **`__init__` signature** → match the call site `(descriptor_config_path, descriptor_device, object_specs, render_dir)`. Drop `proposal_config_path` / `proposer_device`. Store `self.descriptor_device = descriptor_device`.
- **Fold the dead `__post_init__` into `__init__`.** `Writer` is not a dataclass, so `__post_init__` never fires. After building `self.descriptor`, precompute reference features and free the model:
  - iterate `self.names_to_ref.items()` (current `for name, tv_rgba in self.names_to_ref` iterates keys only → unpack error),
  - run under `torch.inference_mode()`,
  - `feat = self.descriptor(tv_rgba.unsqueeze(0).to(self.descriptor_device)).squeeze().cpu()` — **`.cpu()` is required** (serializer calls `.numpy()`; also frees VRAM),
  - then `self.descriptor.to('cpu'); del self.descriptor`.
  - Delete the backtick float-question line (answered: descriptor handles float conversion internally).
- **`write()`:**
  - Remove the two backtick comment lines after the `raise` (lines 92-93) — invalid syntax + unreachable. Keep the existing `#` comment block above the `id_to_name` comprehension that already documents the instance-filter.
  - **Workbench-alpha fix (chosen approach):** build the observation alpha from valid instance ids, not `seg > 0`. Change `composite_rgba`/`alpha_from_instance_seg` *in this file only* to take the valid id list and use `np.isin(seg, valid_ids)`; call it as `composite_rgba(rgb_hw3, seg_hw, list(id_to_name))`. This excludes the workbench (and any unlabeled scenery) structurally. (Leave `stereo_writer.py`'s copies of these helpers untouched.)
  - Drop dead locals `B = len(unique_ids)` and `obs_1chw = obs_tensor.unsqueeze(0)` (proposer-era leftovers).
  - The per-object body stays: `ref_features = self.names_to_feature[id_to_name[uid]]` (already CPU), `seg_mask = (seg_tensor == uid).bool()`, build `ProtoReferenceSegSample(rgb, ref_rgb, seg_mask, reference_features)`, `serialize`, increment `_frame_id`.
- **Module docstring** (lines 1-6): update to "precomputes reference DIFT features once; serializes one `ProtoReferenceSegSample` per (frame, object) with NN-free per-frame writes" — current text still says it runs a proposal network per frame.

### 3. `src/isaac_datagen/clean_datagen.py`
- Line 73 import: `ReferenceSegWriter` → `ProtoReferenceSegWriter`.
- Call site (line 79) already matches the corrected signature — no change.
- Minor (flag, optional): lines 96-97 still dump `proposer.yaml` from `runtime.proposer_config_path`, now unused by this writer. Not run-blocking (the attr still exists). Recommend dropping the `proposer.yaml` dump for honesty, but leaving `runtime` fields alone. Will only remove it if you want.

## Notes / non-goals
- Batching the ~7 reference forwards into one call is a possible optimization but is **out of scope** (keeps your one-forward-per-object pseudocode).
- This makes `ProtoReferenceSegWriter` the NN-free capture writer; running the proposer over the serialized output is the separate phase-2 script you decided to write.

## Verification
1. **Static:** `python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['src/isaac_datagen/clean_datagen.py','src/isaac_datagen/reference_seg_writer.py','src/isaac_datagen/objects.py']]"` — confirms no backtick/syntax errors remain. Optionally pyright on the three buffers.
2. **Import smoke test (no Isaac):** `python -c "from isaac_datagen.objects import ProtoReferenceSegSample; print('ok')"` — catches the missing `torch`/`tv_tensors` imports and annotation-eval errors.
3. **End-to-end:** run `isaac-datagen <config.yaml> idx=<n>` (entry `reference_segmentation`). Expect `render<idx>/` to contain per-field subdirs `rgb/ ref_rgb/ seg_mask/ reference_features/` with `*_0000.*` files, count = num_frames × labeled-objects-per-frame, plus `runtime.yaml`/`descriptor.yaml`. Confirm `reference_features_0000.npy` loads as a `(N, C)` float32 array, and a `rgb_*.png` alpha channel shows the graspable objects only (no workbench).
4. **Sanity:** first run logs a one-time descriptor load/forward at startup; per-frame steps after that should be markedly faster than the prior 6 s/frame (no per-frame NN).

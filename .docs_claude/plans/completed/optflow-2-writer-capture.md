# Plan 2/3 — Optical-flow capture: `OptFlowWriter` + clean_datagen wiring

Part of a 3-plan series for synthetic dense optical-flow data (fine-tune RoMa/UFM/DKM/LoFTR).
- **Plan 1 (COMPLETE)** — `../completed/optflow-1-reference-dataset.md`: each isolated object → grasp-anchored
  reference RGB-D + `ref_pose` + `ref_intrinsics` → `PreOptFlowObject` dataset (`datasets/ycb_preoptflow/`).
- **Plan 2 (this):** place those objects in clutter, capture per-frame observation RGB-D + camera pose, plus a
  once-per-render-dir constant catalog (one `OptFlowMetadata`).
- **Plan 3** — `optflow-3-trainer-adapter.md`: torch adapter → each matcher's `__getitem__`.

## Implementation status — COMPLETE (2026-06-16), verified end-to-end on a real Isaac render

Code implemented and verified:
- `objects.py` — `OptFlowSample`, `OptFlowMetadata` + `OptFlowSample.visualize(md)` (RoMa `get_gt_warp`).
- `scene.py` — `SceneHandle.object_prim_paths`; `build_scene` returns it.
- `optflow_writer.py` — NEW `OptFlowWriter` (3 annotators).
- `clean_datagen.py` — `collect_preoptflow`, `optflow_generation()`, `main()` mode dispatch.
- `runtime_config.py` — `mode` + `optflow_objects_path`; pyproject console entry → `clean_datagen:main`.
- `pyproject.toml` — `romatch` editable dev dep (`uv add --editable ../RoMa --group dev`).

**Real render PASSED.** `mode=optflow num_targets=1 num_frames=2` over `datasets/ycb_preoptflow` →
`datasets/ycb_optflow/render000`: 2 lit frames (rgb_mean ~193, metric depth ~0.8–1.2 m, finite OpenCV
`cam2world`) + one `OptFlowMetadata` cataloging all 7 objects. `OptFlowSample.visualize(md)` (saved as
`optflow_corr_zoom.png` / `optflow_corr_viz.png`) shows every reference object's numbered Xs landing on the
matching physical feature of that object in the clutter (CHEEZ-IT logo, Domino/SUGAR text, StarKist/SPAM
labels), with occluded/out-of-view objects (master-chef-can, mustard) producing no obs correspondences — i.e.
the GL↔CV convention and the `T_ref→obs` composition are correct (a wrong convention would mirror/scatter).

## Context

All four trainers compute the GT warp on-the-fly from `(im_A, im_B, depth_A, depth_B, K_A, K_B, T_A→B)`, so we
only emit those raw fields. **The only new isaacsim extraction is observation depth**
(`distance_to_image_plane`): `ObsMaskWriter` captures `rgb`/`instance_segmentation_fast`/`occlusion` only
(reference_seg_writer.py:94-103), while `StereoSampleWriter` already pulls `distance_to_image_plane` +
`camera_params` (stereo_writer.py:51-52) — the writer below unions those.

**UFM fine-tune — verified compatible (these fields suffice as-is).** UFM-train ships the trainer
(Hydra `scripts/train.py` → Lightning `train_pl`) + dataloaders (`BaseStereoViewDataset`), and computes flow +
covisibility **on-the-fly at GPU collate** (`flow_postprocessing.py:544`, via `train_pl.py:on_after_batch_transfer`)
from per-view `depthmap` + `camera_pose` (cam2world) + `camera_intrinsics` — **no precomputed flow on disk**.
Same OpenCV +Z convention; `depthmap` = metric **Z-depth** = our `distance_to_image_plane` (planar, not range).
Plan 3's adapter maps each view: reference `camera_pose = local2world @ ref_pose`, obs `camera_pose = cam2world`
(UFM composes `inv(other.camera_pose)` itself), plus UFM's per-view `covisible_rendering_parameters` knob.

### ⚠️ Convention rule (the one high-risk error)
The warp (Plan 3 / `warp_kpts`) needs **OpenCV** (+Z-fwd) poses, but `world_poses` are **OpenGL/USD** (−Z-fwd).
Read the obs extrinsic from the `camera_params` annotator via the repo's single converter
`camera_params_to_world2cam = GL2CV @ cameraViewTransform.reshape(4,4).T` (stereo_writer.py:21,33) — **never
hand-invert `world_poses`.** `local2world` from `get_target2world` is a rigid USD transform, convention-agnostic.

**No instance segmentation.** Obs depth is stored **full-frame** (`distance_to_image_plane`, unmasked):
`warp_kpts` masks invalid pixels off the *source* depth (`depth_A==0`, already done in Plan 1); the obs depth is
only sampled for the consistency/occlusion check, so workbench/background depth is correct context. And
object↔frame pairing is Plan 3's covisibility filter (`get_gt_warp` `prob.mean() < τ`), not an obs id-map — so
we need neither the seg mask nor `iid_to_name`. Writer = **3 annotators**: `rgb`, `distance_to_image_plane`,
`camera_params`.

---

## Stage 1 — Datastructs (`isaac_datagen/objects.py`, beside `PreOptFlowObject`)

`OptFlowSample` (one per frame) + `OptFlowMetadata` (per-render constants, written once at idx 0). The metadata
follows the `ObsMaskMetadata` paradigm (datastructs.py:277): bundle the per-render constants as name-keyed dicts
and `torch.save` them once via `_DICT_PT_SERIALIZER` (imported from `vision_core.datastructs`) — so the
reference depth is stored once, not duplicated into every frame.

```python
@dataclass
class OptFlowSample(SerializableSample):      # ONE per frame
    observation: tv_tensors.Image    # obs RGB (3,H,W)                                 → base .png
    observation_depth: np.ndarray    # (H,W) f32 metric z, FULL frame (unmasked)       → base .npy
    cam2world: np.ndarray            # (4,4) obs camera pose, OpenCV (from camera_params) → base .npy
    _serializers = SerializableSample._serializers
    def visualize(self, md, *, name=None, points=None, n_points=12, rel=0.05, title=None) -> np.ndarray: ...

@dataclass
class OptFlowMetadata(SerializableSample):    # per-render constants, written ONCE at idx 0
    obs_intrinsics: np.ndarray       # (3,3) obs K                                     → base .npy
    name_to_reference: dict          # {name → tv_tensors.Image} reference RGB          ┐
    name_to_reference_depth: dict    # {name → torch.Tensor (H,W)} metric z, 0 off      ├ dict → .pt
    name_to_ref_intrinsics: dict     # {name → torch.Tensor (3,3)}                      │ (_DICT_PT_SERIALIZER)
    name_to_ref_pose: dict           # {name → torch.Tensor (4,4)} camera2local CV      │
    name_to_local2world: dict        # {name → torch.Tensor (4,4)} placement            ┘
    _serializers = {**SerializableSample._serializers, **_DICT_PT_SERIALIZER}
```

### `OptFlowSample.visualize(md)` — labeled correspondence check (numbered Xs)
Marks a few candidate reference pixels (**sampled only where `reference_depth>0`, on the object**) with numbered
colored Xs (left), warps each through the **exact warp Plan 3 hands the trainer** — directly importing RoMa's
`get_gt_warp` (no port/vendor) — and stamps the **same numbered X** where it lands in the observation (right).
Read it directly: *X #3 on a box corner in the reference must sit on that corner in the clutter.* Covisible
candidates get a matched obs X; on-object-but-occluded ones show muted in the reference only. Pass
`points=[(x,y),…]` (ref pixels, e.g. the grasp pixel) for specific coordinates. Figure via
`vision_core.viz.panel_grid` + `figure_to_ndarray`; `romatch`/matplotlib imported lazily so importing `objects`
stays light.

`get_gt_warp(depth1, depth2, T_1to2, K1, K2)` (utils.py:325) → `x2 (B,H,W,2)` warped coords (normalized [-1,1])
+ `prob (B,H,W)` covisibility/consistency mask, fed `T_ref→obs = inv(cam2world) @ local2world @ ref_pose` as
`(B,3,4)`, all OpenCV.

---

## Stage 2 — `SceneHandle`: expose placed-object prim paths

`build_scene` has `objects_paths` (aligned to `objects`, scene.py:416) but dropped it; the orchestrator needs it
for each object's `local2world`. Added field `object_prim_paths: List[str] = None` to the frozen dataclass and
passed `object_prim_paths=objects_paths` in the `build_scene` return. `add_object`/`build_scene` read only
`obj.usd_path` + `obj.meta`, so they run unchanged on `PreOptFlowObject`s.

---

## Stage 3 — `OptFlowWriter` (`optflow_writer.py`)

`ObsMaskWriter`-shaped; `attach` keeps the **left** RP only (`capture_session` passes both ZED RPs). Per frame:
`OptFlowSample(observation=rgb[:3], observation_depth=distance_to_image_plane (full),
cam2world=inv(camera_params_to_world2cam(camera_params)))`. `finalize_metadata` builds the name-keyed dicts from
the `PreOptFlowObject`s + their `local2world` and serializes one `OptFlowMetadata` at idx 0.

---

## Stage 4 — Orchestrator `optflow_generation()` in `clean_datagen.py`

Mirrors `reference_segmentation()`; `plan_capture` reused as the camera-pose generator. `collect_preoptflow`
deserializes `PreOptFlowObject`s; `main()` dispatches on `runtime.mode` (`reference_segmentation` | `optflow`).
`runtime.optflow_objects_path` added to `RuntimeConfig` (required when `mode=optflow`). The `dry_run`
debug-export branch is preserved.

Run: `uv run clean_datagen.py <config.yaml> mode=optflow optflow_objects_path=[datasets/ycb_preoptflow] ...`

## Dependency: RoMa (`romatch`) — for `visualize()` only
Added via `uv add --editable ../RoMa --group dev` → `[tool.uv.sources] romatch = {path="../RoMa", editable=true}`
+ `[dependency-groups] dev = ["romatch"]`. UFM has no RoMa install instructions to borrow (READMEs only
acknowledge it). The core writer pipeline does not import `romatch`; only `visualize()` does.

## Verification
1. **local2world:** `get_target2world(scene.object_prim_paths)` returns one SE3 per object, order-aligned to
   `scene.objects`. ✅ (exercised by the real render — per-object `name_to_local2world` correct)
2. **Two-frame capture** (`mode=optflow num_frames=2 num_targets=1`): per-frame `OptFlowSample` (lit obs RGB,
   metric full-frame depth, finite `cam2world`) + one `OptFlowMetadata` covering all 7 objects. ✅
3. **Convention guard:** implied by #4 — correspondences land correctly, so the OpenCV `cam2world` (from
   `camera_params`) and `T_ref→obs` composition are right (a GL/CV slip would mirror/scatter them). ✅
4. **GT correspondence (decisive):** `OptFlowSample.visualize(md)` — numbered Xs land on the same physical
   feature in both panels; occluded objects yield no covisible candidates. ✅ on the real render
   (`optflow_corr_zoom.png`)
5. **Round-trip:** `OptFlowMetadata.deserialize(0, render_dir)` keys == placed-object names; Image + pose tensors
   survive the `.pt` round-trip. ✅

## Caveats
- `SceneHandle` is `frozen=True` — field add + the single `build_scene` return edit.
- ZED is stereo — attach the **left** RP only.
- `cam2world` stored OpenCV via `camera_params` — never hand-inverted.
- `rgb` buffer is a view — `.copy()` before `torch.from_numpy`.
- `OptFlowSample.observation` reads back RGBA (base tv_tensors.Image serializer forces `RGB_ALPHA`); downstream
  slices `[:3]`.

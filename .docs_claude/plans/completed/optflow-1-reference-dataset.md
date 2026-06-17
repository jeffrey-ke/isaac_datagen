# Plan 1/3 — Optical-flow reference dataset (`GraspableObject` → `PreOptFlowObject`)

> **Status: COMPLETE (2026-06-16).** All stages implemented, run, and verified end-to-end.
> - Stage 0 `src/isaac_datagen/patch_grasp_frames.py` — run, 6/6 grasp frames patched (det=1, `+X==R_z·old`; pristine backup at `datasets/ycb_dataset/grasp_point.orig.bak/`).
> - Stage 1 `PreOptFlowObject` in `src/isaac_datagen/objects.py` — added.
> - Stage 2 `src/isaac_datagen/optflow_render.py` — run on all 7 YCB objects → `datasets/ycb_preoptflow/`; RGB lit, depth metric + masked to the object.
> - Viz tool `correspondence/viz_preoptflow.py`; panel at `datasets/ycb_preoptflow/preoptflow_viz.png`.
> - **Known wart:** `optflow_render` loads a full `RuntimeConfig` (validates `dataset_dir`/proposer/descriptor paths it doesn't use) → needs a `dataset_dir=` override; a minimal config would clean this up.

Part of a 3-plan series for synthetic dense optical-flow correspondence data (fine-tune RoMa/UFM/DKM/LoFTR).
- **Plan 1 (this):** patch the stale grasp frames, then render each isolated object as RGB-D from a
  grasp-anchored viewpoint → a `PreOptFlowObject` dataset.
- Plan 2 — `optflow-2-writer-capture.md`: place those objects in clutter, capture per-frame observation RGB-D + the static catalog.
- Plan 3 — `optflow-3-trainer-adapter.md`: torch dataset adapter → each matcher's `__getitem__`.

## Context

Parallel experiment (does **not** replace the proposer→verifier→SAM→grasp pipeline). We want to fine-tune a
dense matcher that warps pixels from a canonical **reference** image of an isolated object to its location in
an **observation** image. The reference is the per-object viewpoint; this plan produces it.

**Why grasp-anchored:** a fixed −Y face doesn't generalize (YCB meshes aren't authored along −Y). Each
object's `grasp_point` SE3 was *designed* to "define the reference viewpoint" (`mesh-convert-ycb.md:85`):
its convention (`face_grasp_frames`, `mesh_convert.py:289`) is **+X = outward face normal, +Z = world up,
origin = bbox face center.** So the reference camera sits out along grasp `+X`, looking back at the origin.

---

## Stage 0 — Patch the 6 stale YCB grasp frames (prerequisite) — DONE

`rotate-graspable-meshes-z.md` rotated the *meshes* of 6 YCB objects about Z but **left `grasp_point`
untouched**, so grasp `+X` now points at a different physical face — a real face for the 90°/180° objects, a
**corner** for the −120° mustard. Fix: apply `grasp_point ← R_z(angle) @ grasp_point` with the same angle the
mesh was rotated by (the rotate plan: "would need `R_z(-90)·grasp_point` to track the same face").

| object | name | idx | angle |
|---|---|---|---|
| cheezit | `ycb_003_cracker_box` | 0001 | −90° |
| sugar | `ycb_004_sugar_box` | 0002 | −90° |
| soup | `ycb_005_tomato_soup_can` | 0003 | +180° |
| mustard | `ycb_006_mustard_bottle` | 0004 | −120° |
| tuna | `ycb_007_tuna_fish_can` | 0005 | −90° |
| spam | `ycb_010_potted_meat_can` | 0006 | +180° |

`patch_grasp_frames.py`: target by `meta` name (not bare idx); back up `grasp_point/` once to
`grasp_point.orig.bak/`; per run restore pristine then `grasp_point ← R_z(angle) @ grasp_point` (4×4 rotation —
axes and origin both rotate about mesh Z, tracking the same physical face); rewrite only `grasp_point/` via
`serialize(idx, dir, only={"grasp_point"})`. Idempotent (re-derive from the backup → no compounding). In-script
asserts: `det(R)=1` and patched `+X == R_z(angle)·old_+X`.

Output: a trustworthy `GraspableObject` dataset (`datasets/ycb_dataset`) that Stage 2 consumes.

---

## Stage 1 — `PreOptFlowObject` datastruct (in `objects.py`) — DONE

Defined beside `GraspableObject` (it needs `UsdPath`; the Plan-3 adapter can still import
`isaac_datagen.objects` without booting isaacsim — those imports are lazy, `objects.py:1-19`). Reuses
`GraspableObject._serializers` (`UsdPath→.usdz`, PIL→`.png`, dict→`.yaml`, np→`.npy` base).

```python
@dataclass
class PreOptFlowObject(SerializableSample):   # GraspableObject minus grasp_point, plus ref camera + depth
    usd_path: UsdPath
    meta: dict                       # {"name","class"}
    reference_image: PILImage.Image  # RGB only
    reference_depth: np.ndarray      # (H,W) float32 metric z; 0 OUTSIDE object (Plan 3's warp drops bg via nonzero_mask)
    ref_intrinsics: np.ndarray       # (3,3) K
    ref_pose: np.ndarray             # (4,4) camera2local SE3, OpenCV (+Z-forward) convention
    _serializers = GraspableObject._serializers
```

---

## Stage 2 — Offline reference render `optflow_render.py` (Isaac)

Decoupled-offline Isaac script (boot once, render each isolated object). Isaac, not Blender, so the reference
uses the same `setup_camera` pinhole + `distance_to_image_plane` annotator as the observation renderer →
exact intrinsics/depth parity with Plan 2.

### How the reference camera pose is obtained

The patched `grasp_point` encodes the chosen face: **origin** = face center, **+X column** = outward face
normal (mesh-local). So we read the pose off the grasp frame and only size the *standoff* from K:

```python
origin = grasp_point[:3, 3]                       # face center (mesh-local)
normal = grasp_point[:3, 0] / norm(...)           # outward +X normal
up     = [0,0,1]
horiz_axis = abs(cross(up, normal))               # in-plane horizontal
half_w = 0.5 * dot(bbox_hi - bbox_lo, horiz_axis)
half_h = 0.5 * (bbox_hi[2] - bbox_lo[2])
hfov, vfov = 2*atan(W/(2*fx)), 2*atan(H/(2*fy))
d   = margin * max(half_w/tan(hfov/2), half_h/tan(vfov/2))
eye = origin + d * normal
ref_pose_cv = look_at(at=origin, from=eye)        # OpenCV (+Z-fwd) camera2local — THIS is the STORED ref_pose
ref_pose_gl = cv2opengl(ref_pose_cv)              # GL (−Z-fwd), TRANSIENT — only to position the Isaac camera prim
```

**Two poses, one stored:** `ref_pose` is stored in **OpenCV** (the `look_at` output, which *is* camera2local
because the object sits at the origin → local == world). The **GL** pose is transient — Isaac/USD cameras look
down −Z, so authoring the camera prim needs `cv2opengl(...)`. Plan 3 composes `ref_pose` into the warp, which
is OpenCV-only. `look_at`/`cv2opengl` from `pose_utils.py:62/16` (same convention as `LookAtPoser`,
`posers.py:55`), so the face frames un-mirrored.

- ⚠️ **±Z-face degeneracy:** `look_at` builds `x = cross(z,[0,0,1])` (`pose_utils.py:64`), singular if the view
  dir ∥ world up. Grasp faces are side faces (±X/±Y normals) → safe; guard if top/bottom faces ever appear.

### The script — `src/isaac_datagen/optflow_render.py`

Boot Isaac once; per object: fresh stage → dome light (else PT renders black) → load asset → grasp-anchored
camera → read `rgb`/`distance_to_image_plane`/`instance_segmentation_fast` → mask depth bg→0 → serialize.
Every Isaac call is an existing repo helper (`boot_sim`, `make_dome_light` `scene.py:207`, `setup_camera`
`isaac_utils.py:161`, `load_asset`, `local_bbox_range`, `set_prim_pose` `capture.py:67`, `warmup_render`,
`add_labels`). `ref_pose_from_grasp` is the unit-tested pose helper above. CLI:

```
uv run src/isaac_datagen/optflow_render.py <config.yaml> <graspable_dataset> <out_dataset> [key=val ...]
```

Key in-script choices: `REF_DOME_INTENSITY=1000` (TUNE on first render — boot_sim exposure is fixed),
`MARGIN=1.1`, depth masked to the instance (`where(seg!=0, depth, 0)`), `ref_pose` stored as `ref_pose_cv`,
the GL pose authored on the prim via `set_prim_pose(cv2opengl(ref_pose_cv))`.

**Run-blocking details to validate at the Isaac render (not redesigns):** (a) exact
`instance_segmentation_fast` `get_data()` shape (dict-with-`"data"` vs bare array — handled both ways);
(b) `REF_DOME_INTENSITY` brightness on the first render (empirical lighting tune); (c) whether one
`orchestrator.step()` converges the PT frame or needs `wait_until_complete()`; (d) per-object render-product
re-creation across the dataset — release if GPU memory climbs.

---

## Reuse map (file:line)
- grasp-frame convention: `mesh_convert.py:289` (`face_grasp_frames`), `:33` (FACE_NORMALS); staleness source: `rotate-graspable-meshes-z.md`
- serialization API: `vision_core/datastructs.py:99` (`serialize`, supports `only=`), `:120` (`deserialize`), `:68` (`_serializers`); `GraspableObject` serializers `objects.py:34-51`
- camera from K: `isaac_utils.setup_camera:161`; bbox `local_bbox_range:254`; asset load `load_asset:67`; labels `scene.py:51`
- look_at/cv2opengl: `vision_core/pose_utils.py:62/16`; in-plane axis trick `mesh_blender.py:192`; pose-on-prim `capture.set_prim_pose:67`
- dome light: `scene.make_dome_light:207`; boot/warmup: `scene.boot_sim:254`, `scene.warmup_render:330`
- depth annotator: `stereo_writer.py:51`

## Verification
1. **Stage 0 unit (no Isaac) — PASS:** for each of the 6, `det(R)=+1` and patched `+X == R_z(angle)·old_+X` (asserted in `patch_grasp_frames.py`).
2. **ref_pose unit (no Isaac) — PASS:** synthetic bbox + grasp frame + K → camera out along +X, optical axis back at the object, the 4 face corners inside `[0,W]×[0,H]` filling ~`1/margin` of the frame.
3. **End-to-end (Isaac) — PASS:** rendered all 7 YCB → `reference_image` shows the curated grasp face upright + lit (rgb mean 168–213), `reference_depth` metric + 0 outside the object (fg 16–44%, z∈[0.07, 0.34] m), `ref_pose` finite SE3, round-trips via `deserialize`. Panel: `datasets/ycb_preoptflow/preoptflow_viz.png`.
4. **Cross-check vs old reference:** for a non-rotated object, the new grasp-anchored reference should resemble the existing `reference_image`.

## Caveats
- ±Z faces break `look_at` (side faces only today).
- `ref_pose` is stored **OpenCV** (the `look_at` output), NOT the GL pose authored on the prim — Plan 3 depends on it.
- Background depth must be exactly 0 (mask by instance seg) on the reference; Plan 2 masks the observation likewise.
</content>

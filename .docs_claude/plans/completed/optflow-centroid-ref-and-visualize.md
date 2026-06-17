# Centroid-aligned reference pose + `OptFlowObject.visualize()`

**Status:** completed 2026-06-17. Verified end-to-end; the amazon `*_preoptflow` dataset at
`assets/optflow_objects/amazon` was re-rendered and confirms the fix.

## Context

The offline reference renderer (`graspableobj_to_optflow_obj.py`) anchored the reference camera on
the **grasp point**: it both *looked at* `grasp_point[:3, 3]` and sat at a standoff along the
grasp-face normal from that same point. But the FOV-fit math (`half_w`/`half_h`) assumes the look-at
target is the **center** of the bbox extents. The amazon box's grasp point is at the *bottom*, so the
optical axis centered on the bottom and the top of the box was cropped out of the reference image.

Two complementary changes: (1) fix the framing by aiming at the object centroid; (2) add a per-object
QA visualizer so the framing (and the stored ref pose) can be eyeballed.

## Deliverable 1 — centroid-aligned ref pose

**File:** `src/isaac_datagen/graspableobj_to_optflow_obj.py`, `ref_pose_from_grasp`.

The grasp frame still supplies only the **view direction** (its outward +X face normal). The look-at
target and the camera-position base moved from the grasp origin to the bbox centroid
`centroid = 0.5*(lo+hi)` — same mesh-local frame as `grasp_point`, since the object is loaded at the
origin in the isolated render (`lo`/`hi` from `local_bbox_range(geo)`, `isaac_utils.py:254`).

```python
# BEFORE
    origin = grasp_point[:3, 3]
    return look_at(at_coord=origin,   from_coord=origin   + d * normal)
# AFTER
    centroid = 0.5 * (lo + hi)
    return look_at(at_coord=centroid, from_coord=centroid + d * normal)
```

`half_w`/`half_h`/`d` are unchanged — they are already the full object extents, so with the optical
axis through the centroid the object fills the frame symmetrically (`MARGIN=1.1` absorbs the slight
perspective growth of the near face, now half-depth closer than the look-at point). Function +
module docstrings updated: grasp = view direction, centroid = framing center.

## Deliverable 2 — `OptFlowObject.visualize()`

### 2a. Reusable 3D primitives → `vision_core/src/vision_core/viz.py`

No camera-frustum / 3D-frame helper existed in either repo (only `pose_utils` touched 3D). Per the
"shared primitives live in vision_core" rule, the generic drawing lives in `viz.py` (new section
"2b. 3D pose / camera primitives") and `OptFlowObject.visualize` stays a thin composer — mirroring
how `OptFlowSample.visualize` already imports `panel_grid`/`figure_to_ndarray` from `viz.py`:

- `draw_frame_3d(ax, pose=None, scale, alpha)` — RGB (x,y,z) axes gizmo for an SE3 pose.
- `draw_camera_3d(ax, pose, K, width, height, scale, color)` — wireframe frustum for an OpenCV
  (+Z-forward) cam2world pose: back-projects the 4 image corners through K to depth `scale`, draws
  center→corner rays + the image-plane rectangle, plus the camera's own axes gizmo.
- `set_3d_equal(ax, pts)` — equal-aspect cube around an (N,3) point set (mpl 3D has no `'equal'`).
- `draw_mesh_3d(ax, points, faces, *, facecolors=None, color, alpha, edge, lw, max_faces)` —
  triangulated-surface mesh. With `facecolors` (M,3) RGB each triangle is flat-shaded in its own
  colour (no edges) — matplotlib **cannot UV-texture-map** a 3D surface, so per-face
  texture-sampled colour is how appearance is approximated; without it, a translucent grey surface.
  Strides down to `max_faces` triangles for responsiveness (returns the count drawn).

### 2b. The method → `src/isaac_datagen/objects.py`, on `OptFlowObject`

```python
def visualize(self, *, depth_cmap="turbo", cam_scale=None, show_mesh=True,
              mesh_alpha=0.3, max_faces=6000, title=None) -> np.ndarray
```

Keyword-only; matplotlib + viz imports lazy (matches `OptFlowSample.visualize`). Three panels in a
row — reference RGB | colormapped reference depth (`np.ma.masked_equal(depth, 0)` so off-object
zeros don't dominate the range, + colorbar) | a 3D panel with the object-local frame gizmo at the
origin, the **actual `usd_path` mesh in its local-frame position**, and `ref_pose` as a wireframe
camera. Caption (`fig.suptitle`) lists every `meta` field. Returns an `(H,W,3)` uint8 array via
`figure_to_ndarray`.

The mesh is read by a sibling `OptFlowObject._load_mesh()` that opens the `.usdz` via `pxr`, bakes
each mesh prim's points through its local-to-world transform, and fan-triangulates n-gons. It also
captures the faceVarying `st` UVs and the bound diffuse texture (from the usdz zip) and returns a
**per-face RGB** (`face_rgb`) — each triangle's mean UV sampled from the texture (`st` is
bottom-left, image array top-left, so `v → (1-v)`; verified on the mustard-bottle atlas texture).
`visualize()` draws that as flat per-face colour for an approximate textured appearance (falls back
to translucent grey when the mesh has no UVs/texture). Same usdz-reading recipe as
`correspondence/extract_mesh.py`.

`pxr` is **not** importable in the plain project venv — it needs the `usd-core` overlay
(`uv run --with usd-core ...`) or a booted Isaac kit; `_load_mesh` raises a clear, actionable
`ImportError` otherwise, and `show_mesh=False` skips it entirely. For *true* UV-textured rendering
(beyond mpl's per-face approximation) use the open3d path in `correspondence/` (`plyview`,
`build_obj_axes.py`).

## Verification (done)

- One-object re-render via `debug_scripts/verify_centroid_ref.py` (reuses production `render_one` →
  the fixed `ref_pose_from_grasp`) on amazon_32 (`combined_dataset` idx 33): full box framed, no
  top-crop. Before/after compared directly.
- After the user re-rendered `assets/optflow_objects/amazon`, all 44 objects dumped via
  `debug_scripts/viz_optflow_objects.py` — every box fully framed; depth + 3D ref-pose panels and
  the meta caption render correctly.
- Downstream sample viz (`debug_scripts/viz_optflow.py`) unaffected (only the offline ref render and
  two additive viz surfaces changed).

## Debug scripts added (`src/isaac_datagen/debug_scripts/`)

- `viz_optflow_objects.py` — loop `OptFlowObject.deserialize → visualize → save` over a preoptflow
  dataset.
- `verify_centroid_ref.py` — boot Isaac once, re-render a single GraspableObject, dump its
  `visualize()` panel (spot-check the ref pose without rendering the whole dataset).

## Operational note

Existing `*_preoptflow` datasets rendered before this change are still top-cropped; re-run
`graspableobj_to_optflow_obj.py` per dataset to pick up the centroid fix (the amazon set is done).

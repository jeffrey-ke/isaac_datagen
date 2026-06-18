"""Scene objects for Isaac Sim warehouse simulation."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import tv_tensors

from vision_core.datastructs import SerializableSample, _DICT_PT_SERIALIZER
# ObsMask / PreReferenceSegSample / ObsMaskMetadata live in vision_core.datastructs
# (the shared package) so the sibling `segmentation-train` env — which can't take an
# isaacsim dependency — can import them too. Re-exported here for convenience.
from vision_core.datastructs import ObsMask, PreReferenceSegSample, ObsMaskMetadata

from isaac_datagen.isaac_utils import load_asset, set_transform, create_empty


class UsdPath(str):
    """Str subclass marking a USD file path; signals copy-on-serialize semantics."""
    pass


@dataclass
class GraspableObject(SerializableSample):
    usd_path: UsdPath       # path to a .usdz file; copied into dataset on serialize
    meta: dict              # must contain keys "name" and "class"
    reference_image: PILImage.Image
    grasp_point: np.ndarray  # SE3 (4, 4)

    _serializers = {
        **SerializableSample._serializers,
        UsdPath: (
            '.usdz',
            lambda p, v: shutil.copy(v, p),
            lambda p: UsdPath(str(p)),
        ),
        PILImage.Image: (
            '.png',
            lambda p, v: v.save(str(p)),
            lambda p: PILImage.open(str(p)).copy(),
        ),
        dict: (
            '.yaml',
            lambda p, v: yaml.dump(v, open(p, 'w')),
            lambda p: yaml.safe_load(open(p)),
        ),
    }


@dataclass
class OptFlowObject(SerializableSample):
    """A GraspableObject re-cast as a dense-optical-flow reference.

    The canonical RGB-D view of the object rendered from a grasp-anchored virtual camera,
    plus that camera's intrinsics and pose. ``grasp_point`` is dropped: its sole job —
    defining the reference viewpoint — is now baked into ``ref_pose``. Produced offline by
    ``optflow_render.py``; consumed by the capture writer (Plan 2) and the trainer adapter
    (Plan 3), which compose ``ref_pose`` into the ref->obs warp, so it is stored in OpenCV
    (+Z-forward) convention — NOT the OpenGL pose authored onto the Isaac camera prim.
    """
    usd_path: UsdPath                # path to a .usdz file; copied into dataset on serialize
    meta: dict                       # must contain keys "name" and "class"
    reference_image: PILImage.Image  # RGB of the canonical reference view
    reference_depth: np.ndarray      # (H, W) float32 metric z-depth; 0 OUTSIDE the object
    ref_intrinsics: np.ndarray       # (3, 3) reference-camera K
    ref_pose: np.ndarray             # (4, 4) camera2local SE3, OpenCV (+Z-forward) convention

    # Same field types as GraspableObject (UsdPath/PIL/dict + base np.ndarray) → reuse its table.
    _serializers = GraspableObject._serializers

    def visualize(self, *, depth_cmap="turbo", cam_scale=None, show_mesh=True,
                  mesh_alpha=None, max_faces=20000, title=None) -> np.ndarray:
        """Multi-panel QA figure: reference RGB | colormapped reference depth | 3D ref-pose.

        The 3D panel draws the object-local coordinate frame as an axes gizmo at the origin, the
        actual ``usd_path`` mesh in its local-frame position (``show_mesh``), and ``ref_pose`` as a
        wireframe camera (OpenCV +Z-forward). Caption lists all ``meta`` fields. Returns an
        (H, W, 3) uint8 RGB array. matplotlib imported lazily.

        ``show_mesh`` reads the .usdz geometry via ``pxr`` — run under ``uv run --with usd-core``
        or a booted Isaac kit (pass ``show_mesh=False`` to skip)."""
        import matplotlib.pyplot as plt
        from vision_core.viz import (figure_to_ndarray, draw_frame_3d, draw_camera_3d,
                                     draw_mesh_3d, set_3d_equal)

        rgb = np.asarray(self.reference_image)[..., :3]
        depth = np.asarray(self.reference_depth, dtype=np.float32)
        H, W = depth.shape

        fig = plt.figure(figsize=(15, 5))
        ax_rgb = fig.add_subplot(1, 3, 1)
        ax_rgb.imshow(rgb); ax_rgb.set_title("reference_image"); ax_rgb.axis("off")

        ax_d = fig.add_subplot(1, 3, 2)
        im = ax_d.imshow(np.ma.masked_equal(depth, 0.0), cmap=depth_cmap)   # 0 = off-object → masked
        fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04, label="metric depth (m)")
        ax_d.set_title("reference_depth"); ax_d.axis("off")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        cam_C = self.ref_pose[:3, 3]
        extent_pts = [np.zeros(3), cam_C]                                   # origin + camera center
        if show_mesh:
            pts, faces, face_rgb = self._load_mesh()                       # baked to object-local frame
            a = mesh_alpha if mesh_alpha is not None else (1.0 if face_rgb is not None else 0.35)
            draw_mesh_3d(ax3, pts, faces, facecolors=face_rgb, alpha=a, max_faces=max_faces)
            extent_pts = [pts.min(0), pts.max(0), cam_C]                   # frame the mesh + camera
        s = cam_scale if cam_scale is not None else 0.25 * float(np.linalg.norm(cam_C))
        draw_frame_3d(ax3, scale=s)                                         # object-local frame at origin
        draw_camera_3d(ax3, self.ref_pose, self.ref_intrinsics, W, H, scale=s)
        set_3d_equal(ax3, np.vstack(extent_pts))
        ax3.set_title("ref_pose (cam2local, OpenCV)"); ax3.set_xlabel("x"); ax3.set_ylabel("y")

        caption = "   ".join(f"{k}: {v}" for k, v in self.meta.items())
        fig.suptitle(f"{title}\n{caption}" if title else caption, fontsize=9)
        fig.tight_layout()
        return figure_to_ndarray(fig)

    def _load_mesh(self):
        """(points (N,3), faces (M,3), face_rgb (M,3) in [0,1] or None) for the ``usd_path`` mesh.

        Geometry baked into the object-local frame (each mesh prim's points through its
        local-to-world transform, n-gons fan-triangulated). ``face_rgb`` is per-FACE colour sampled
        from the bound diffuse texture via the mesh's faceVarying ``st`` UVs — matplotlib can't
        UV-texture a 3D surface, so appearance is approximated as one flat colour per triangle
        (None if the mesh has no UVs or no bound texture). Same usdz-reading recipe as
        ``correspondence/extract_mesh.py``. Reads the .usdz via ``pxr`` (needs
        ``uv run --with usd-core ...`` or a booted Isaac kit)."""
        import zipfile
        from io import BytesIO
        try:
            from pxr import Usd, UsdGeom, UsdShade
        except ImportError as e:
            raise ImportError(
                "OptFlowObject.visualize(show_mesh=True) reads the .usdz mesh and needs pxr — run "
                "under `uv run --with usd-core ...` or a booted Isaac kit, or pass show_mesh=False."
            ) from e

        stage = Usd.Stage.Open(str(self.usd_path))
        verts, tris, tri_uvs, voff = [], [], [], 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(prim)
            l2w = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            P = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
            P = (l2w @ np.c_[P, np.ones(len(P))].T).T[:, :3]               # → object-local frame
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
            fidx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            st_vals = np.asarray(st.ComputeFlattened(), dtype=np.float64) if st else None  # faceVarying
            o = 0
            for c in counts:                                              # fan-triangulate n-gons
                f = fidx[o:o + c]
                for k in range(1, c - 1):
                    tris.append([f[0] + voff, f[k] + voff, f[k + 1] + voff])
                    if st_vals is not None:
                        tri_uvs.append([st_vals[o], st_vals[o + k], st_vals[o + k + 1]])
                o += c
            verts.append(P); voff += len(P)
        points, faces = np.vstack(verts), np.asarray(tris, np.int64)

        face_rgb = None
        if tri_uvs:                                                        # sample the bound diffuse texture
            tex = None
            for prim in stage.Traverse():
                sh = UsdShade.Shader(prim) if prim.IsA(UsdShade.Shader) else None
                if sh and "UVTexture" in (sh.GetShaderId() or ""):
                    rel = sh.GetInput("file").Get().path.lstrip("@").lstrip("./")
                    try:
                        with zipfile.ZipFile(str(self.usd_path)) as z:
                            tex = np.asarray(PILImage.open(BytesIO(z.read(rel))).convert("RGB"))
                    except KeyError:
                        tex = None
                    break
            if tex is not None:
                uv = np.asarray(tri_uvs).mean(1)                          # (M, 2) mean UV per triangle
                Ht, Wt = tex.shape[:2]
                u = np.clip(uv[:, 0], 0, 1) * (Wt - 1)
                v = (1.0 - np.clip(uv[:, 1], 0, 1)) * (Ht - 1)            # st is bottom-left; image is top-left
                face_rgb = tex[v.astype(int), u.astype(int)].astype(np.float64) / 255.0
        return points, faces, face_rgb


@dataclass
class OptFlowSample(SerializableSample):
    """One rendered observation frame of the dense-optical-flow dataset.

    The per-frame-unique payload: an ``ObsMask`` (the RGBA observation + cid/iid masks +
    per-instance occlusion — the seg pipeline's exact per-frame datastruct, serialized FLAT so
    ``ObsMask.deserialize(idx, render_dir)`` reads it directly), its FULL-frame metric depth (NOT
    masked — the warp samples it only for the consistency/occlusion check, so workbench and
    background depth are correct context, not noise), and the observation camera pose in OpenCV
    (+Z-forward) convention. The per-instance ``obsmask.iid_mask`` lets the UFM adapter isolate one
    instance (mask out same-class siblings) for 1-to-1 flow. The per-render constants (each
    object's reference RGB-D, pose, intrinsics, placement) live once-per-render-dir in
    ``OptFlowMetadata``. ``obsmask.obs`` (RGBA) replaces the old ``observation`` RGB — read
    ``obsmask.obs[:3]`` for the 3-channel frame (requires ``full_alpha=True`` at capture so it is
    the full, unmasked frame).
    """
    obsmask: ObsMask                 # RGBA obs + cid/iid masks + occlusion; serialized FLAT (nested sample)
    observation_depth: np.ndarray    # (H, W) float32 metric z-depth, full frame (distance_to_image_plane)
    cam2world: np.ndarray            # (4, 4) obs camera2world SE3, OpenCV (+Z-forward)

    # obsmask serializes its own subdirs flat (datastructs nested-sample rule); the two np.ndarray
    # fields use the base serializer.
    _serializers = SerializableSample._serializers

    def visualize(self, md, *, cls_name=None, points=None, n_points=12, rel=0.05, title=None) -> np.ndarray:
        """GT reference→observation correspondence as labeled points, for eyeballing the warp.

        A top ``[cid_mask | iid_mask]`` row shows the per-pixel id masks with class/instance legends.
        One ``[ref | obs]`` row per class (pass ``cls_name`` to restrict to one). The class's single
        canonical reference is warped — via RoMa's ``get_gt_warp`` (the exact warp the trainer uses)
        — into EVERY instance of that class (1-to-many): reference candidates are numbered neutral
        Xs on the left, and each instance fans out into the single obs panel in its own color, with
        connection lines back to the shared reference point. Candidates occluded / out-of-view for a
        given instance simply don't draw for it.

        Candidates are sampled ONLY where the reference has valid depth (``reference_depth>0``, on
        the object): a coarse grid over the valid-depth region by default, or pass
        ``points=[(x, y), ...]`` (reference pixels, e.g. the grasp pixel) for specific coordinates.

        ``md`` is this render dir's ``OptFlowMetadata``. Requires ``romatch`` importable
        (dev/optional dep); matplotlib + romatch are imported lazily so importing this module stays
        light. Returns an (H, W, 3) uint8 RGB array.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch, Patch
        from romatch.utils.utils import get_gt_warp          # authoritative warp, directly imported
        from vision_core.viz import panel_grid, figure_to_ndarray

        obs = self.obsmask.obs.permute(1, 2, 0).numpy()[..., :3]
        Hb, Wb = obs.shape[:2]
        dB = torch.as_tensor(self.observation_depth, dtype=torch.float32)
        K_B = torch.as_tensor(md.obs_intrinsics, dtype=torch.float32)
        rows = []
        for cls in ([cls_name] if cls_name else list(md.class_to_name)):     # iterate classes, 1-many
            dA = md.class_to_reference_depth[cls].float()                    # (Ha, Wa) canonical ref depth
            L = md.class_to_l2w[cls].float()                                 # (N, 4, 4) this class's placements
            N = L.shape[0]
            inv_c2w = torch.as_tensor(np.linalg.inv(self.cam2world), dtype=torch.float32)
            T = torch.einsum('ij,njx,xy->niy', inv_c2w, L,                   # (N, 4, 4) ref-cam → obs-cam, per instance
                             md.class_to_ref_pose[cls].float())
            K_A = md.class_to_ref_intrinsics[cls].float()
            x2, prob = get_gt_warp(                                          # batch = N: expand singletons to match T
                dA[None].expand(N, -1, -1), dB[None].expand(N, -1, -1), 
                T[:, :3],
                K_A[None].expand(N, -1, -1), K_B[None].expand(N, -1, -1),
                relative_depth_error_threshold=rel,
            )
            x2, prob = x2.numpy(), prob.numpy()                             # x2 (N,Ha,Wa,2) normalized, prob (N,Ha,Wa)
            valid_ref = dA.numpy() > 0                                       # ref pixels on the object
            Ha, Wa = valid_ref.shape
            if points is not None:
                pts = [(int(round(y)), int(round(x))) for x, y in points]
                cand = [(yy, xx) for yy, xx in pts
                        if 0 <= yy < Ha and 0 <= xx < Wa and valid_ref[yy, xx]]
            else:
                vy, vx = np.nonzero(valid_ref)
                if not len(vy):
                    continue
                gy = np.linspace(vy.min(), vy.max(), 5).astype(int)
                gx = np.linspace(vx.min(), vx.max(), 5).astype(int)
                cand = [(y, x) for y in gy for x in gx if valid_ref[y, x]][:n_points]
            if cand:
                rows.append((cls, x2, prob, cand))

        def draw_id_mask(ax, mask, id_to_label, name):                   # discrete id mask + legend
            ids = [int(i) for i in np.unique(mask) if i != 0]            # 0 = background
            rgb = np.zeros((*mask.shape, 3))
            handles = []
            for k, i in enumerate(ids):
                c = plt.cm.tab20(k % 20)[:3]
                rgb[mask == i] = c
                handles.append(Patch(color=c, label=f"{i}: {id_to_label.get(i, '?')}"))
            ax.imshow(rgb)
            ax.set_title(name, fontsize=8)
            ax.axis("off")
            if handles:
                ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)

        fig, axes = panel_grid(2 + 2 * len(rows), cols=2, panel_w=5.0, panel_h=4.0)   # [cid|iid] masks, then [ref|obs] per class
        draw_id_mask(axes[0], self.obsmask.cid_mask.numpy(), md.obsmaskmeta.cid_to_class, "cid_mask")
        draw_id_mask(axes[1], self.obsmask.iid_mask.numpy(), md.obsmaskmeta.iid_to_name, "iid_mask")
        for r, (cls, x2, prob, cand) in enumerate(rows):
            ax_ref, ax_obs = axes[2 + 2 * r], axes[2 + 2 * r + 1]
            ax_ref.imshow(md.class_to_reference[cls].permute(1, 2, 0).numpy()[..., :3])
            ax_obs.imshow(obs)
            ax_ref.set_title(f"{cls} ref", fontsize=8)
            ax_obs.set_title(f"obs · {x2.shape[0]} instances", fontsize=8)
            for i, (y, x) in enumerate(cand):                              # shared reference candidates (neutral)
                ax_ref.plot(x, y, "x", ms=9, mew=2, color="k")
                ax_ref.text(x + 3, y - 3, str(i), color="k", fontsize=8)
            for n in range(x2.shape[0]):                                   # one color per instance → fan-out
                c = plt.cm.turbo(n / max(x2.shape[0] - 1, 1))
                for i, (y, x) in enumerate(cand):
                    if prob[n, y, x] <= 0:                                 # occluded / out-of-view for this instance
                        continue
                    ub, vb = (x2[n, y, x, 0] + 1) * Wb / 2, (x2[n, y, x, 1] + 1) * Hb / 2
                    ax_obs.plot(ub, vb, "x", ms=8, mew=2, color=c)
                    ax_obs.text(ub + 3, vb - 3, str(i), color=c, fontsize=7)
                    fig.add_artist(ConnectionPatch((ub, vb), (x, y), "data", "data",
                                                   axesA=ax_obs, axesB=ax_ref, color=c, lw=0.5, alpha=0.4))
            ax_ref.axis("off")
            ax_obs.axis("off")
        if title:
            fig.suptitle(title, fontsize=10)
        return figure_to_ndarray(fig)


@dataclass
class OptFlowMetadata(SerializableSample):
    """Per-render-dir catalog of the optical-flow dataset, serialized once (at idx=0).

    Keyed BY CLASS, not instance. Each class owns one canonical reference (RGB, metric depth,
    intrinsics, camera2local pose) plus the world placements of ALL its instances in this render
    dir — the 1-to-many contract: one reference depth map warps into every same-class instance.
    Mirrors ``ObsMaskMetadata``: every per-class collection is a plain dict ``torch.save``d once
    via ``_DICT_PT_SERIALIZER`` (preserves str keys + Image/Tensor values), so the reference
    depth is stored once rather than duplicated into every frame.

    The trainer warps the one reference into each instance ``n`` of class ``cls`` via
    ``T_ref→obs[n] = inv(cam2world) @ class_to_l2w[cls][n] @ class_to_ref_pose[cls]`` (all OpenCV).
    """
    obsmaskmeta: ObsMaskMetadata     # the seg-pipeline catalog (cid_to_class/iid_to_name/name_to_class/
                                     # class_to_ref/class_to_descriptors/principal_components), serialized
                                     # FLAT; supplies cid_to_class (pairs with obsmask.cid_mask) + iid_to_name
    obs_intrinsics: np.ndarray       # (3, 3) observation K (shared)
    class_to_name: dict              # {class → list[str]} instance names, aligned to class_to_l2w rows
    class_to_reference: dict         # {class → tv_tensors.Image (3, H, W)} canonical reference RGB
    class_to_reference_depth: dict   # {class → torch.Tensor (Ha, Wa)} metric z-depth, 0 off-object
    class_to_ref_intrinsics: dict    # {class → torch.Tensor (3, 3)} reference K
    class_to_ref_pose: dict          # {class → torch.Tensor (4, 4)} camera2local SE3, OpenCV
    class_to_l2w: dict               # {class → torch.Tensor (N, 4, 4)} the class's N instance placements

    # obsmaskmeta serializes its own subdirs flat (nested-sample rule); obs_intrinsics → .npy
    # (base np.ndarray); every class_to_* dict → one .pt (torch.save).
    _serializers = {**SerializableSample._serializers, **_DICT_PT_SERIALIZER}


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(SCRIPT_PATH, "resources")

class Workbench:
    """Workbench environment with table and structure.
    
    Note: This expects USD assets converted from the original Blender files.
    The workbench_frame.blend should be converted to workbench_frame.usd
    """
    
    def __init__(self, parent_path="/World"):
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.prims import create_prim
        self.prim_path = f"{parent_path}/Workbench"
        stage = get_current_stage()
        self.prim = create_prim(self.prim_path, "Xform")
        
        # Load workbench frame asset (converted from workbench_frame.blend to USD)
        workbench_usd_path = os.path.join(RESOURCE_PATH, "workbench_world.usd")
        
        if os.path.exists(workbench_usd_path):
            # Load the workbench structure
            self.workbench_prim = load_asset(
                f"{self.prim_path}/WorkbenchFrame",
                workbench_usd_path
            )
                
            print(f"Loaded workbench frame from {workbench_usd_path}")
        else:
            raise FileNotFoundError(
                f"Workbench asset not found at {workbench_usd_path}. "
                "Please convert the Blender asset to USD and place it in the resources folder."
            )
class Warehouse:
    """Warehouse environment with racks and structure.
    
    Note: This expects USD assets converted from the original Blender files.
    The rack_frame.blend should be converted to rack_frame.usd
    """
    
    def __init__(self, parent_path="/World"):
        from pxr import UsdGeom
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.prims import create_prim
        self.prim_path = f"{parent_path}/Warehouse"
        stage = get_current_stage()
        self.prim = create_prim(self.prim_path, "Xform")
        rack_usd_path = os.path.join(RESOURCE_PATH, "rack_frame.usdc")
        if os.path.exists(rack_usd_path):
            self.rack_prim = load_asset(
                f"{self.prim_path}/RackFrame",
                rack_usd_path
            )
            floor_prim = UsdGeom.Mesh.Get(stage, f"{self.prim_path}/RackFrame/Floor")
            set_transform(floor_prim.GetPrim(), scale=(10.0, 10.0, 1.0))
                
            print(f"Loaded warehouse rack from {rack_usd_path}")
        else:
            raise FileNotFoundError(
                f"Rack asset not found at {rack_usd_path}. "
                "Please convert the Blender asset to USD and place it in the resources folder."
            )

class Floor:
    """Ground floor for the warehouse."""
    
    def __init__(self, parent_path="/World", size=10.0):
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.prims import create_prim
        self.prim_path = f"{parent_path}/Floor"
        stage = get_current_stage()
        self.prim = create_prim(self.prim_path, "Xform")
        plane_path = f"{self.prim_path}/Plane"
        plane = UsdGeom.Mesh.Define(stage, plane_path)
        half_size = size / 2.0
        points = [
            Gf.Vec3f(-half_size, -half_size, 0.0),
            Gf.Vec3f(half_size, -half_size, 0.0),
            Gf.Vec3f(half_size, half_size, 0.0),
            Gf.Vec3f(-half_size, half_size, 0.0)
        ]
        plane.GetPointsAttr().Set(points)
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]
        plane.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        plane.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        collision_api = UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(plane.GetPrim())
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(plane.GetPrim())
        physx_rigid_body_api.CreateKinematicEnabledAttr(True)


class OccupancyGrid:
    """Spatial model of object presence on a 3-D grid, usable as a placement policy.

    Grid coordinates (i, j, k): i is left-right, j is front-back
    (j=0 is front), k is vertical.

    As a placement policy, call it sequentially once per occupied slot;
    it returns (translation, rotation) for that slot.
    """

    EPSILON = 0.002

    def __init__(self, grid_dims, object_bbox_dims):
        self.grid = np.ones(grid_dims, dtype=np.int8)
        self.object_bbox_dims = object_bbox_dims
        self._idx = 0
        self._placed = {}  # prim_path -> (i, j, k), recorded as __call__ drives placement

    def is_occupied(self, i, j, k):
        return bool(self.grid[i, j, k])

    def is_top(self, i, j, k):
        return k == self.grid.shape[2] - 1 or not any(
            self.grid[i, j, z] for z in range(k + 1, self.grid.shape[2])
        )

    def is_front(self, i, j, k):
        return j == 0 or not any(
            self.grid[i, y, k] for y in range(j)
        )

    def is_bottom(self, i, j, k):
        return k == 0

    def is_restricted(self, i, j, k):
        return self.is_bottom(i, j, k) or (j > 0 and bool(self.grid[i, j - 1, k - 1]))

    def select_boxes(self, *predicates):
        return [
            (int(i), int(j), int(k))
            for i, j, k in zip(*self.grid.nonzero())
            if all(p(i, j, k) for p in predicates)
        ]

    def remove(self, i, j, k):
        self.grid[i, j, k] = 0

    def slot_position(self, i, j, k):
        bx, by, bz = self.object_bbox_dims
        gx, gy, _ = self.grid.shape
        eps = self.EPSILON
        span_x = gx * bx + (gx - 1) * eps
        span_y = gy * by + (gy - 1) * eps
        return (
            -span_x / 2.0 + bx / 2.0 + i * (bx + eps),
            -span_y / 2.0 + by / 2.0 + j * (by + eps),
            bz / 2.0 + k * (bz + eps),
        )

    @property
    def sequence(self):
        return [
            (int(i), int(j), int(k))
            for i, j, k in zip(*self.grid.nonzero())
        ]

    def __call__(self, prim_path):
        coords = self.sequence
        if self._idx >= len(coords):
            raise IndexError(
                f"OccupancyGrid exhausted after {self._idx} placements "
                f"({len(coords)} occupied slots)"
            )
        i, j, k = coords[self._idx]
        self._idx += 1
        self._placed[prim_path] = (i, j, k)
        return self.slot_position(i, j, k), (0.0, 0.0, 0.0)

    def graspability(self):
        """Per-prim-path graspability: is_front AND is_top for each placed slot.

        Meaningful only after every slot has been driven via __call__ (the
        full-wall contract). Folds in what the caller used to compute inline."""
        return {
            path: (self.is_front(*coord) and self.is_top(*coord))
            for path, coord in self._placed.items()
        }


class LoadedPallet:
    """A pallet loaded with boxes.
    
    Matches the Blender datagen pallet randomisation by starting from a
    full 3x3x4 stack of boxes and randomly removing picks while
    respecting visibility constraints (no occluder above or in front).
    """

    # Pallet layout constants chosen to mirror the Blender setup
    EPSILON = 0.002  # spacing to avoid z-fighting
    CONTACT_ROTATION = (0.0, 0.0, 180.0)  # gripper approaches negative X

    def __init__(self,
                 parent_path,
                 num_picks,
                 rng,
                 uid=0,
                 has_pallet_base=True,
                 grid_dims=(3,3,4),
                 box_file_path= "crunch_box.usdc",
                 box_size=(0.402,0.341,0.266),
                 outer_full_grastability=True,
                 randomize_texture_prefix=None):
        """Initialize a loaded pallet.

        Args:
            parent_path: Parent prim path
            pallet_template: Unused placeholder for API compatibility
            num_picks: Number of boxes removed from the pallet (0 == full)
            rng: Random number generator
        """
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.prims import create_prim
        stage = get_current_stage()

        self.prim_path = f"{parent_path}/LoadedPallet_{uid}"
        self.root = create_prim(self.prim_path, "Xform")

        # Load pallet base
        self.PALLET_TOP_OFFSET = 0.0
        if has_pallet_base:
            self.PALLET_TOP_OFFSET = 0.06  # metres from pallet origin to first layer centre
            pallet_usd_path = os.path.join(RESOURCE_PATH, "pallet.usdc")
            if os.path.exists(pallet_usd_path):
                self.pallet_prim = load_asset(
                    f"{self.prim_path}/Pallet",
                    pallet_usd_path
                )
            else:
                print(f"Warning: Pallet asset not found at {pallet_usd_path}")
        
        self.randomize_texture = False
        if randomize_texture_prefix is not None:
            texture_folder_path = os.path.join(RESOURCE_PATH, "boxes", "textures")
            self.texture_files = [f for f in os.listdir(texture_folder_path)
                             if f.startswith(randomize_texture_prefix)]
            self.randomize_texture = True

        # Tracking containers
        self.boxes = []
        self.grasp_contact_points = []
        self.GRID_DIMS = grid_dims
        self.BOX_SIZE = box_size
        self.box_file_path = box_file_path

        occupancy = OccupancyGrid(self.GRID_DIMS, self.BOX_SIZE)
        max_picks = int(occupancy.grid.sum())
        picks_to_remove = min(num_picks, max_picks)

        for _ in range(picks_to_remove):
            candidates = occupancy.select_boxes(occupancy.is_top, occupancy.is_front)
            if not candidates:
                break
            i, j, k = candidates[rng.randrange(len(candidates))]
            occupancy.remove(i, j, k)

        for i, j, k in occupancy.select_boxes():
            box_path = f"{self.prim_path}/Box_{i}_{j}_{k}"
            box_prim = self._create_box(box_path, 1.0, 1.0, 1.0)
            set_transform(box_prim, translation=self._slot_position(i, j, k))
            self.boxes.append(box_prim)

        self._populate_grasps(occupancy)

    def _slot_position(self, i, j, k):
        box_x, box_y, box_height = self.BOX_SIZE
        grid_x, grid_y, _ = self.GRID_DIMS
        eps = self.EPSILON
        stride_x, stride_y, stride_z = box_x + eps, box_y + eps, box_height + eps
        span_x = grid_x * box_x + (grid_x - 1) * eps
        span_y = grid_y * box_y + (grid_y - 1) * eps
        first_center_x = -span_x / 2.0 + box_x / 2.0
        first_center_y = -span_y / 2.0 + box_y / 2.0
        return (
            first_center_x + i * stride_x,
            first_center_y + j * stride_y,
            self.PALLET_TOP_OFFSET + box_height / 2.0 + k * stride_z,
        )

    def _populate_grasps(self, occupancy):
        _, box_y, box_height = self.BOX_SIZE
        for i, j, k in occupancy.select_boxes(occupancy.is_top, occupancy.is_front, lambda i, j, k: not occupancy.is_restricted(i, j, k)):
            box_path = f"{self.prim_path}/Box_{i}_{j}_{k}"
            grasp_prim = create_empty(f"GraspPoint", box_path)
            set_transform(
                grasp_prim,
                translation=(0.0, -box_y / 2, -box_height / 2.0),
                rotation=self.CONTACT_ROTATION,
            )
            self.grasp_contact_points.append(grasp_prim)

    def _create_box(self, prim_path, width, depth, height):
        """Create a box with physics.
        
        Args:
            prim_path: Path for the box prim
            width: Box width (x)
            depth: Box depth (y)
            height: Box height (z)
            
        Returns:
            Box prim
        """
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.utils.semantics import add_labels
        stage = get_current_stage()

        box_usd_path = os.path.join(RESOURCE_PATH, self.box_file_path)
        if os.path.exists(box_usd_path):
            box_prim = load_asset(prim_path, box_usd_path)
            set_transform(box_prim, scale=(width, depth, height))
            if self.randomize_texture:
                texture_file = self.texture_files[np.random.randint(len(self.texture_files))]
                texture_path = os.path.join(RESOURCE_PATH, "boxes", "textures", texture_file)
                material_path = f"{prim_path}/_materials/Material"
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim.IsValid():
                    for prim in Usd.PrimRange(material_prim):
                        if prim.IsA(UsdShade.Shader):
                            shader = UsdShade.Shader(prim)
                            file_input = shader.GetInput("file")
                            if file_input:
                                file_input.Set(Sdf.AssetPath(texture_path))
                                print(f"Assigned texture {texture_file} to box at {prim_path}")
                                break
                else:
                    print(f"Warning: Material prim not found at {material_path}")
        else:
            box_prim = create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube.Get(stage, prim_path)
            cube.GetSizeAttr().Set(1.0)
            set_transform(box_prim, scale=(width, depth, height))
            collision_api = UsdPhysics.CollisionAPI.Apply(box_prim)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(box_prim)
            rigid_body_api.CreateRigidBodyEnabledAttr(True)
            mass_api = UsdPhysics.MassAPI.Apply(box_prim)
            mass_api.GetMassAttr().Set(1.0)

        add_labels(box_prim, labels=["box"], instance_name="class")
        return box_prim

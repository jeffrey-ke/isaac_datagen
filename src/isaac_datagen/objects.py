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

from vision_core.datastructs import SerializableSample
# ObsMask / PreReferenceSegSample / ObsMaskDescriptorMetadata / OptFlowSample / OptFlowMetadata live in
# vision_core.datastructs (the shared package) so sibling envs — `segmentation-train` and the
# UFM trainer, which can't take an isaacsim dependency — can import them too. The OptFlow* pair is
# the optflow dataset contract; re-exported here for convenience (e.g. optflow_writer.py).
from vision_core.datastructs import (
    ObsMask, PreReferenceSegSample, ObsMaskDescriptorMetadata, OptFlowSample, OptFlowMetadata,
)

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
    plus that camera's intrinsics and pose. ``grasp_point`` is carried through from the
    GraspableObject: ``ref_pose`` bakes the reference viewpoint for the ref->obs warp,
    while ``grasp_point`` stays the object-local grasp/aim frame — store-mode capture
    (store_scene.add_catalog_grasp_frame) authors it as the GraspPoint camera-aim anchor,
    and it keeps the reference face inspectable on disk. Produced offline by
    ``graspableobj_to_optflow_obj.py``; consumed by the capture writer (Plan 2) and the
    trainer adapter (Plan 3), which compose ``ref_pose`` into the ref->obs warp, so it is
    stored in OpenCV (+Z-forward) convention — NOT the OpenGL pose authored onto the Isaac
    camera prim.
    """
    usd_path: UsdPath                # path to a .usdz file; copied into dataset on serialize
    meta: dict                       # must contain keys "name" and "class"
    reference_image: PILImage.Image  # RGB of the canonical reference view
    reference_depth: np.ndarray      # (H, W) float32 metric z-depth; 0 OUTSIDE the object
    ref_intrinsics: np.ndarray       # (3, 3) reference-camera K
    ref_pose: np.ndarray             # (4, 4) camera2local SE3, OpenCV (+Z-forward) convention
    grasp_point: np.ndarray          # (4, 4) SE3, usdz-local frame; +X = reference face normal

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

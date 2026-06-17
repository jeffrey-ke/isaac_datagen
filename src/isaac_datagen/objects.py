"""Scene objects for Isaac Sim warehouse simulation."""

import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import tv_tensors

from vision_core.datastructs import SerializableSample, ReferenceSegSample, _DICT_PT_SERIALIZER
# ObsMask / PreReferenceSegSample / ObsMaskMetadata live in vision_core.datastructs
# (the shared package) so the sibling `segmentation-train` env — which can't take an
# isaacsim dependency — can import them too. Re-exported here for convenience.
from vision_core.datastructs import ObsMask, PreReferenceSegSample, ObsMaskMetadata

from isaac_datagen.isaac_utils import load_asset, set_transform, create_empty, local_bbox_range


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
class PreOptFlowObject(SerializableSample):
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


@dataclass
class OptFlowSample(SerializableSample):
    """One rendered observation frame of the dense-optical-flow dataset.

    The only per-frame-unique payload: the observation RGB, its FULL-frame metric depth (NOT
    masked — the warp samples it only for the consistency/occlusion check, so workbench and
    background depth are correct context, not noise), and the observation camera pose in OpenCV
    (+Z-forward) convention. The per-render constants (each object's reference RGB-D, pose,
    intrinsics, placement) live once-per-render-dir in ``OptFlowMetadata``.
    """
    observation: tv_tensors.Image    # (3, H, W) obs RGB
    observation_depth: np.ndarray    # (H, W) float32 metric z-depth, full frame (distance_to_image_plane)
    cam2world: np.ndarray            # (4, 4) obs camera2world SE3, OpenCV (+Z-forward)

    # observation → .png (base tv_tensors.Image serializer); depth / cam2world → .npy (base np.ndarray).
    _serializers = SerializableSample._serializers

    def visualize(self, md, *, name=None, points=None, n_points=12, rel=0.05, title=None) -> np.ndarray:
        """GT reference→observation correspondence as labeled points, for eyeballing the warp.

        Candidates are sampled ONLY where the reference has valid depth (``reference_depth>0``, on
        the object) — numbered colored Xs in the reference (left), warped through RoMa's
        ``get_gt_warp`` (the exact warp the trainer uses), and stamped with the SAME numbered X
        where they land in the observation (right). Covisible candidates get a matched obs X;
        candidates on the object but occluded / out-of-view show muted in the reference only.
        Default candidates are a coarse grid over the valid-depth region; pass
        ``points=[(x, y), ...]`` (reference pixels, e.g. the grasp pixel) for specific coordinates.

        ``md`` is this render dir's ``OptFlowMetadata``. Requires ``romatch`` importable
        (dev/optional dep); matplotlib + romatch are imported lazily so importing this module stays
        light. Returns an (H, W, 3) uint8 RGB array.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch
        from romatch.utils.utils import get_gt_warp          # authoritative warp, directly imported
        from vision_core.viz import panel_grid, figure_to_ndarray

        obs = self.observation.permute(1, 2, 0).numpy()[..., :3]
        Hb, Wb = obs.shape[:2]
        dB = torch.as_tensor(self.observation_depth, dtype=torch.float32)
        K_B = torch.as_tensor(md.obs_intrinsics, dtype=torch.float32)
        rows = []
        for nm in ([name] if name else list(md.name_to_reference)):
            dA = md.name_to_reference_depth[nm].float()                               # (Ha, Wa)
            T = (torch.as_tensor(np.linalg.inv(self.cam2world), dtype=torch.float32)
                 @ md.name_to_local2world[nm].float() @ md.name_to_ref_pose[nm].float())  # ref-cam → obs-cam
            x2, prob = get_gt_warp(dA[None], dB[None], T[None, :3],
                                   md.name_to_ref_intrinsics[nm].float()[None], K_B[None],
                                   relative_depth_error_threshold=rel)
            x2 = x2[0].numpy()                                                         # (Ha, Wa, 2) warped coords
            valid_ref = dA.numpy() > 0                                                 # ref pixels on the object
            covis = (prob[0] > 0).numpy()                                             # subset also covisible in obs
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
                rows.append((nm, x2, covis, cand))

        fig, axes = panel_grid(2 * max(len(rows), 1), cols=2, panel_w=5.0, panel_h=4.0)   # [ref, obs] per object
        for r, (nm, x2, covis, cand) in enumerate(rows):
            ax_ref, ax_obs = axes[2 * r], axes[2 * r + 1]
            ax_ref.imshow(md.name_to_reference[nm].permute(1, 2, 0).numpy()[..., :3])
            ax_obs.imshow(obs)
            ax_ref.set_title(f"{nm} ref", fontsize=8)
            ax_obs.set_title("obs", fontsize=8)
            for i, (y, x) in enumerate(cand):
                c = plt.cm.turbo(i / max(len(cand) - 1, 1))
                seen = bool(covis[y, x])                                               # covisible → valid obs landing
                ax_ref.plot(x, y, "x", ms=9, mew=2, color=c, alpha=1.0 if seen else 0.3)
                ax_ref.text(x + 3, y - 3, str(i), color=c, fontsize=8, alpha=1.0 if seen else 0.3)
                if not seen:                                                           # on object but occluded/out-of-view
                    continue
                ub, vb = (x2[y, x, 0] + 1) * Wb / 2, (x2[y, x, 1] + 1) * Hb / 2         # denormalize → obs px
                ax_obs.plot(ub, vb, "x", ms=9, mew=2, color=c)
                ax_obs.text(ub + 3, vb - 3, str(i), color=c, fontsize=8)
                fig.add_artist(ConnectionPatch((ub, vb), (x, y), "data", "data",
                                               axesA=ax_obs, axesB=ax_ref, color=c, lw=0.5, alpha=0.5))
            ax_ref.axis("off")
            ax_obs.axis("off")
        if title:
            fig.suptitle(title, fontsize=10)
        return figure_to_ndarray(fig)


@dataclass
class OptFlowMetadata(SerializableSample):
    """Per-render-dir catalog of the optical-flow dataset, serialized once (at idx=0).

    The constants shared across every frame of a render dir: the observation intrinsics and,
    per placed object (keyed by ``meta["name"]``), its reference RGB, reference metric depth,
    reference intrinsics, reference camera2local pose (OpenCV), and its world placement
    (``local2world``). Mirrors ``ObsMaskMetadata``: every per-object collection is a plain dict,
    ``torch.save``d once via ``_DICT_PT_SERIALIZER`` (preserves str keys + Image/Tensor values),
    so the reference depth is stored once rather than duplicated into every frame.

    The trainer adapter (Plan 3) composes each (reference, observation) pair's relative pose as
    ``T_ref→obs = inv(cam2world) @ local2world @ ref_pose`` (all OpenCV).
    """
    obs_intrinsics: np.ndarray       # (3, 3) observation K (shared)
    name_to_reference: dict          # {name → tv_tensors.Image (3, H, W)} reference RGB
    name_to_reference_depth: dict    # {name → torch.Tensor (H, W)} metric z-depth, 0 off-object
    name_to_ref_intrinsics: dict     # {name → torch.Tensor (3, 3)} reference K
    name_to_ref_pose: dict           # {name → torch.Tensor (4, 4)} camera2local SE3, OpenCV
    name_to_local2world: dict        # {name → torch.Tensor (4, 4)} placed-object world pose

    # obs_intrinsics → .npy (base np.ndarray); every name_to_* dict → one .pt (torch.save).
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


class UntilExhaustedStacker:
    """Until-exhausted column stacker for HETEROGENEOUS object bounding boxes.

    Chunks prim_paths into columns of <= column_height; each column stacks its
    members base-to-base (bottom seated on the ground, each next object's base on
    the previous object's top), centroids aligned on the column center-line and on
    y=0. Column footprint width = the widest member's x-extent; columns abut
    left->right with EPSILON gaps; the whole wall is centered on x=0. No physics
    (a wider object may overhang a narrower one below). The last column may be
    partial.

    Unlike OccupancyGrid this MEASURES each loaded prim's bbox size AND center and
    corrects for prim-origin-!=-bbox-center. The full layout is precomputed in
    __init__; __call__ is a lookup.

    Graspability: only the top (last-stacked) object of each column.
    """

    EPSILON = 0.002

    def __init__(self, prim_paths, column_height):
        if column_height < 1:
            raise ValueError(f"column_height must be >= 1, got {column_height}")
        if len(prim_paths) < 1:
            raise ValueError("UntilExhaustedStacker needs >= 1 object")

        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()

        # Columns of prim paths (deques so the "top" is unambiguous: last pushed).
        self.columns = [
            deque(prim_paths[s:s + column_height])
            for s in range(0, len(prim_paths), column_height)
        ]

        # Measure size + center per prim once, from the loaded stage prims.
        size, center = {}, {}
        for p in prim_paths:
            rng = local_bbox_range(stage.GetPrimAtPath(p))
            sz, mid = rng.GetSize(), rng.GetMidpoint()
            size[p] = (sz[0], sz[1], sz[2])
            center[p] = (mid[0], mid[1], mid[2])

        # Column footprint width = widest member's x-extent; center the wall on x=0.
        col_widths = [max(size[p][0] for p in col) for col in self.columns]
        total_w = sum(col_widths) + (len(self.columns) - 1) * self.EPSILON
        left_edge = -total_w / 2.0

        self._placements = {}  # prim_path -> (translation, rotation)
        for col, col_w in zip(self.columns, col_widths):
            col_x = left_edge + col_w / 2.0
            floor_z = 0.0
            for p in col:  # bottom -> top
                sx, sy, sz = size[p]
                cx, cy, cz = center[p]
                # set_transform places the prim ORIGIN; the bbox center lands at
                # origin + (cx,cy,cz), so subtract the midpoint per axis to seat
                # the centroid on (col_x, 0) and the bbox base at floor_z.
                translation = (col_x - cx, -cy, floor_z - cz + sz / 2.0)
                self._placements[p] = (translation, (0.0, 0.0, 0.0))
                floor_z += sz + self.EPSILON
            left_edge += col_w + self.EPSILON

    def __call__(self, prim_path):
        return self._placements[prim_path]

    def graspability(self):
        """Per-prim-path graspability: only the top object of each column."""
        tops = {col[-1] for col in self.columns if col}
        return {p: (p in tops) for p in self._placements}


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

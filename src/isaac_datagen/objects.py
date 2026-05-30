"""Scene objects for Isaac Sim warehouse simulation."""

import os
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_core.datastructs import SerializableSample

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
        return self.slot_position(i, j, k), (0.0, 0.0, 0.0)

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

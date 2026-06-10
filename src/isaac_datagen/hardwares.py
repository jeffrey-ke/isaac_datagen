"""Hardware definitions for gripper and camera setup in Isaac Sim."""

import math
import numpy as np
from dataclasses import dataclass

from isaac_datagen.isaac_utils import create_empty, setup_camera, set_transform, setup_render_product
class ZedMini:
    """Zed Mini stereo camera simulation.

    Specs:
    RGB:
        1920x1080, 1280x720, 720x404
        Aperture: ƒ/2.0
        Focal Length: 2.8mm (0.11")
        Field of View: 102° (H) x 57° (V) x 118° (D) max.
        Baseline: 63 mm
    """
    BASELINE = 0.063  # 63 mm
    FOCAL_LENGTH_MM = 2.8  # arbitrary, only the ratio to sensor_width matters
    # Per-camera translation on the rig — the single source of truth for both the
    # real rig geometry (placed in __init__) and the dry-run debug offset (left2rig).
    LEFT_CAM_OFFSET = (-BASELINE * 0.5, 0.0, 0.0)
    RIGHT_CAM_OFFSET = (BASELINE * 0.5, 0.0, 0.0)

    def __init__(self, name, parent_path, intrinsics, width=1920, height=1080):
        self.name = name
        self.width = width
        self.height = height
        self._intrinsics = intrinsics

        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)

        self.left_camera_path = f"{self.prim_path}/{name}_left"
        self.left_camera = self._create_camera(f"{name}_left", self.left_camera_path)
        set_transform(
            self.left_camera,
            translation=self.LEFT_CAM_OFFSET,
        )

        self.right_camera_path = f"{self.prim_path}/{name}_right"
        self.right_camera = self._create_camera(f"{name}_right", self.right_camera_path)
        set_transform(
            self.right_camera,
            translation=self.RIGHT_CAM_OFFSET,
        )

        self.left_rp = setup_render_product(
            self.left_camera_path, (width, height), f"{name}_left"
        )
        self.right_rp = setup_render_product(
            self.right_camera_path, (width, height), f"{name}_right"
        )

    @property
    def rps(self):
        return (self.left_rp, self.right_rp)

    @property
    def intrinsics(self) -> np.ndarray:
        """The 3x3 OpenCV K this rig was built with (no disk reload needed)."""
        return self._intrinsics

    @property
    def left2rig(self) -> np.ndarray:
        """SE3 of the left (dataset) camera relative to the moved rig — the
        transform that maps a rig world pose to the camera whose RGB the dataset uses."""
        T = np.eye(4)
        T[:3, 3] = self.LEFT_CAM_OFFSET
        return T

    def _create_camera(self, name, prim_path):
        return setup_camera(
            name, prim_path,
            width=self.width, height=self.height,
            intrinsics=self._intrinsics,
            focal_length_mm=self.FOCAL_LENGTH_MM,
        )

class Gripper:
    """Dual camera gripper setup."""
    
    def __init__(self, name, parent_path, args):
        """Initialize the gripper with dual Orbbec Gemini 2 cameras.
        
        Args:
            name: Name of the gripper
            parent_path: Parent prim path
            args: OrbbecGemini2Args configuration
        """
        self.name = name
        self.args = args
        
        # Gripper geometry
        # Measured offset: 0.2764 * 0.5 = 0.1382m (half baseline between cameras)
        self.offset = 0.2764 * 0.5
        self.camera_rotation = 18  # 18° tilt from model
        
        # Create gripper empty transform
        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        
        # Create left camera
        self.orbbec_left = OrbbecGemini2(f"{name}_left", self.prim_path, args)
        set_transform(
            self.orbbec_left.prim,
            translation=(self.offset, 0.0, 0.0),
            rotation=(0.0, self.camera_rotation, 180)
        )
        
        # Create right camera
        self.orbbec_right = OrbbecGemini2(f"{name}_right", self.prim_path, args)
        set_transform(
            self.orbbec_right.prim,
            translation=(-self.offset, 0.0, 0.0),
            rotation=(0.0, -self.camera_rotation, 180)
        )
        
        # Set gripper orientation
        set_transform(
            self.prim,
            translation=(0.0, 0.0, 0.0),
            rotation=(90.0, 0.0, 180.0)
        )
        
        self.current_parent = None
    
    def get_all_render_products(self):
        """Get all render products from both camera rigs.
        
        Returns:
            List of all render products
        """
        return self.orbbec_left.render_products + self.orbbec_right.render_products

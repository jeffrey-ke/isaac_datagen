
import math
import numpy as np
from dataclasses import dataclass

from isaac_datagen.isaac_utils import create_empty, setup_camera, set_transform, setup_render_product
class ZedMini:
    BASELINE = 0.063
    FOCAL_LENGTH_MM = 2.8
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
        return self._intrinsics

    @property
    def left2rig(self) -> np.ndarray:
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
    
    def __init__(self, name, parent_path, args):
        self.name = name
        self.args = args
        
        self.offset = 0.2764 * 0.5
        self.camera_rotation = 18
        
        self.prim_path = f"{parent_path}/{name}"
        self.prim = create_empty(name, parent_path)
        
        self.orbbec_left = OrbbecGemini2(f"{name}_left", self.prim_path, args)
        set_transform(
            self.orbbec_left.prim,
            translation=(self.offset, 0.0, 0.0),
            rotation=(0.0, self.camera_rotation, 180)
        )
        
        self.orbbec_right = OrbbecGemini2(f"{name}_right", self.prim_path, args)
        set_transform(
            self.orbbec_right.prim,
            translation=(-self.offset, 0.0, 0.0),
            rotation=(0.0, -self.camera_rotation, 180)
        )
        
        set_transform(
            self.prim,
            translation=(0.0, 0.0, 0.0),
            rotation=(90.0, 0.0, 180.0)
        )
        
        self.current_parent = None
    
    def get_all_render_products(self):
        return self.orbbec_left.render_products + self.orbbec_right.render_products

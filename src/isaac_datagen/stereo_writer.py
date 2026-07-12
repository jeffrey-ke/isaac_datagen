
from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer
import omni.replicator.core as rep

from vision_core.datastructs import StereoSample
from vision_core.pose_utils import intrinsics_from_camera_params


GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def alpha_from_instance_seg(seg: np.ndarray) -> np.ndarray:
    return (seg > 0).astype(np.uint8) * 255


def composite_rgba(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    alpha = alpha_from_instance_seg(seg)
    return np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)


def camera_params_to_world2cam(cam: dict) -> np.ndarray:
    return GL2CV @ cam["cameraViewTransform"].reshape(4, 4).T


def camera_params_to_intrinsics(cam: dict) -> np.ndarray:
    return intrinsics_from_camera_params(
        cam["cameraFocalLength"],
        cam["cameraAperture"],
        cam["cameraApertureOffset"],
        cam["renderProductResolution"],
    )


class StereoSampleWriter(Writer):
    def __init__(self, output_dir: str, offsets: list, target2world: np.ndarray):
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            AnnotatorRegistry.get_annotator("camera_params"),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast",
                init_params={"colorize": False},
            ),
        ]
        self._output_dir = output_dir
        self._offsets = offsets
        self._target2world = target2world.astype(np.float32)
        self._frame_id = 0

    def attach(self, left_rp, right_rp):
        self._left_rp_key = left_rp.path.rsplit("/", 1)[-1]
        self._right_rp_key = right_rp.path.rsplit("/", 1)[-1]
        super().attach([left_rp, right_rp])

    def write(self, data: dict):
        rps = data["renderProducts"]
        left = rps[self._left_rp_key]
        right = rps[self._right_rp_key]

        left_rgb = left['rgb']['data']
        right_rgb = right['rgb']['data']
        left_depth = left['distance_to_image_plane']['data']
        right_depth = right['distance_to_image_plane']['data']
        left_seg = left['instance_segmentation_fast']['data']
        right_seg = right['instance_segmentation_fast']['data']

        left_cam = left["camera_params"]
        right_cam = right["camera_params"]

        sample = StereoSample(
            left_img=tv_tensors.Image(torch.from_numpy(left_rgb).permute(2, 0, 1)),
            right_img=tv_tensors.Image(torch.from_numpy(right_rgb).permute(2, 0, 1)),
            left_depth=torch.from_numpy(left_depth),
            right_depth=torch.from_numpy(right_depth),
            offset=torch.tensor(self._offsets[self._frame_id], dtype=torch.float32),
            original_image_path=("", ""),
            left_world2cam=torch.from_numpy(camera_params_to_world2cam(left_cam)),
            right_world2cam=torch.from_numpy(camera_params_to_world2cam(right_cam)),
            target2world=torch.from_numpy(self._target2world),
            left_intrinsics=torch.from_numpy(camera_params_to_intrinsics(left_cam)),
            right_intrinsics=torch.from_numpy(camera_params_to_intrinsics(right_cam)),
            left_seg=torch.from_numpy(left_seg),
            right_seg=torch.from_numpy(right_seg),
        )
        sample.serialize(self._frame_id, Path(self._output_dir))
        self._frame_id += 1

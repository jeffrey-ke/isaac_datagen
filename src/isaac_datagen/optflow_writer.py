"""Replicator writer for the dense-optical-flow dataset.

Per rendered frame it serializes one ``OptFlowSample`` — the observation RGB, its full-frame
metric depth (``distance_to_image_plane``, NOT masked: the warp samples obs depth only for the
consistency/occlusion check, so workbench/background depth is correct context), and the
observation camera2world pose in OpenCV convention (from the ``camera_params`` annotator via the
repo's single ``GL2CV`` converter — never hand-inverted from the GL ``world_poses``).

The per-render constants (each placed object's reference RGB-D, pose, intrinsics, and its world
placement) are written once via ``finalize_metadata`` as one ``OptFlowMetadata`` — the
``ObsMaskMetadata`` paradigm (``reference_seg_writer.ObsMaskWriter.finalize_metadata``), so the
reference depth is stored once per render dir rather than duplicated into every frame.

No instance-segmentation annotator: object↔frame pairing and the off-object mask are recovered
downstream by the trainer's covisibility warp, not an obs id-map. Annotators: ``rgb``,
``distance_to_image_plane``, ``camera_params``.
"""

from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from isaac_datagen.stereo_writer import camera_params_to_world2cam
from isaac_datagen.objects import OptFlowSample, OptFlowMetadata


class OptFlowWriter(Writer):
    def __init__(self, objects, local2worlds, obs_intrinsics, render_dir):
        """objects: list[PreOptFlowObject] (== scene.objects). local2worlds: (M, 4, 4) world
        poses aligned to ``objects`` (from get_target2world). obs_intrinsics: (3, 3) obs K."""
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            AnnotatorRegistry.get_annotator("camera_params"),
        ]
        self._objects = objects
        self._l2w = local2worlds
        self._obs_K = obs_intrinsics
        self._render_dir = Path(render_dir)
        self._frame_id = 0

    def attach(self, *rps):
        # ZED is stereo — capture_session passes both RPs; keep the LEFT one only
        # (mirrors ObsMaskWriter.attach).
        self._rp_key = rps[0].path.rsplit("/", 1)[-1]
        super().attach([rps[0]])

    def write(self, data: dict):
        rp = data["renderProducts"][self._rp_key]
        obs = tv_tensors.Image(torch.from_numpy(rp["rgb"]["data"][:, :, :3].copy()).permute(2, 0, 1))
        depth = np.asarray(rp["distance_to_image_plane"]["data"], dtype=np.float32)        # full frame
        cam2world = np.linalg.inv(camera_params_to_world2cam(rp["camera_params"]))         # OpenCV cam2world
        OptFlowSample(
            observation=obs,
            observation_depth=depth,
            cam2world=cam2world.astype(np.float32),
        ).serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        """Write the per-render-dir constants once (at idx=0). Call after capture."""
        directory = Path(directory) if directory is not None else self._render_dir
        nm = lambda o: o.meta["name"]
        OptFlowMetadata(
            obs_intrinsics=np.asarray(self._obs_K, dtype=np.float32),
            name_to_reference={
                nm(o): tv_tensors.Image(torch.from_numpy(np.array(o.reference_image)).permute(2, 0, 1))
                for o in self._objects
            },
            name_to_reference_depth={nm(o): torch.from_numpy(o.reference_depth).float() for o in self._objects},
            name_to_ref_intrinsics={nm(o): torch.from_numpy(o.ref_intrinsics).float() for o in self._objects},
            name_to_ref_pose={nm(o): torch.from_numpy(o.ref_pose).float() for o in self._objects},
            name_to_local2world={nm(o): torch.from_numpy(L).float() for o, L in zip(self._objects, self._l2w)},
        ).serialize(0, directory)

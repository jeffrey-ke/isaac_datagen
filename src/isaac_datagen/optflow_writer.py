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

Annotators: ``rgb``, ``distance_to_image_plane``, ``camera_params``, and
``instance_segmentation_fast`` — the per-frame cid/iid masks (same encoding as ``ObsMask``) let the
downstream UFM adapter split the 1-to-many warp into per-instance 1-to-1 pairs.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from isaac_datagen.stereo_writer import camera_params_to_world2cam
from isaac_datagen.isaac_utils import cid_iid_masks
from isaac_datagen.objects import OptFlowSample, OptFlowMetadata


class OptFlowWriter(Writer):
    def __init__(self, objects, local2worlds, obs_intrinsics, render_dir):
        """objects: list[OptFlowObject] (== scene.objects). local2worlds: (M, 4, 4) world
        poses aligned to ``objects`` (from get_target2world). obs_intrinsics: (3, 3) obs K."""
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
        self._objects = objects
        self._l2w = local2worlds
        self._obs_K = obs_intrinsics
        self._render_dir = Path(render_dir)
        self._frame_id = 0

        # Deterministic cids from the SORTED class set, starting at 2 (matches ObsMaskWriter).
        classes = sorted({o.meta["class"] for o in objects})
        self.class_to_cid = {cls: cid for cid, cls in enumerate(classes, start=2)}
        self.cid_to_class = {cid: cls for cls, cid in self.class_to_cid.items()}
        self.iid_to_name: dict[int, str] = {}   # accumulated per frame (iids are session-local)

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

        seg_hw = rp["instance_segmentation_fast"]["data"]
        labels = rp["instance_segmentation_fast"]["idToSemantics"]
        iid_mask, cid_mask, frame_iid_to_name = cid_iid_masks(seg_hw, labels, self.class_to_cid)
        if not frame_iid_to_name:
            raise ValueError("write() called with no labeled instances — expected ≥1")
        self.iid_to_name.update(frame_iid_to_name)

        OptFlowSample(
            observation=obs,
            cid_mask=cid_mask,
            iid_mask=iid_mask,
            observation_depth=depth,
            cam2world=cam2world.astype(np.float32),
        ).serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        """Write the per-render-dir constants once (at idx=0). Call after capture."""
        directory = Path(directory) if directory is not None else self._render_dir
        by_class = defaultdict(list)                                   # class → [(object, l2w), ...]
        for o, L in zip(self._objects, self._l2w):
            by_class[o.meta["class"]].append((o, L))
        rep = {c: members[0][0] for c, members in by_class.items()}    # representative object per class
        OptFlowMetadata(
            obs_intrinsics=np.asarray(self._obs_K, dtype=np.float32),
            class_to_name={c: [o.meta["name"] for o, _ in members] for c, members in by_class.items()},
            class_to_reference={
                c: tv_tensors.Image(torch.from_numpy(np.array(o.reference_image)).permute(2, 0, 1))
                for c, o in rep.items()
            },
            class_to_reference_depth={c: torch.from_numpy(o.reference_depth).float() for c, o in rep.items()},
            class_to_ref_intrinsics={c: torch.from_numpy(o.ref_intrinsics).float() for c, o in rep.items()},
            class_to_ref_pose={c: torch.from_numpy(o.ref_pose).float() for c, o in rep.items()},
            class_to_l2w={
                c: torch.from_numpy(np.stack([L for _, L in members])).float()
                for c, members in by_class.items()
            },
            cid_to_class=self.cid_to_class,
            iid_to_name=self.iid_to_name,
        ).serialize(0, directory)

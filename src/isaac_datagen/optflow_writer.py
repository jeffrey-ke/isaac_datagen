"""Replicator writer for the dense-optical-flow dataset.

Per rendered frame it serializes one ``OptFlowSample``: a nested ``ObsMask`` (RGBA observation +
cid/iid masks + per-instance occlusion — built by the shared ``reference_seg_writer`` helpers and
serialized FLAT, so the render dir doubles as a reference-seg render dir consumable by
``run_pipeline`` phases 2 & 3), its full-frame metric depth (``distance_to_image_plane``, NOT
masked: the warp samples obs depth only for the consistency/occlusion check, so workbench/background
depth is correct context), and the observation camera2world pose in OpenCV convention (from the
``camera_params`` annotator via the repo's single ``GL2CV`` converter — never hand-inverted from the
GL ``world_poses``).

The per-render constants are written once via ``finalize_metadata`` as one ``OptFlowMetadata`` that
NESTS the ``ObsMaskDescriptorMetadata`` (also serialized flat): the optflow per-class catalog (each object's
reference RGB-D, pose, intrinsics, world placement) plus the seg catalog (id maps, per-class
reference DIFT descriptors, PCA basis) that the seg trainers require.

Annotators: ``rgb``, ``distance_to_image_plane``, ``camera_params``, ``instance_segmentation_fast``,
and ``occlusion`` — the last two feed the nested ``ObsMask``.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from isaac_datagen.stereo_writer import camera_params_to_world2cam
from isaac_datagen.reference_seg_writer import reference_catalog, obsmask_from_data, obsmask_metadata
from isaac_datagen.objects import OptFlowSample, OptFlowMetadata


class OptFlowWriter(Writer):
    def __init__(self, objects, local2worlds, obs_intrinsics, render_dir,
                 descriptor_config_path, descriptor_device, full_alpha=False):
        """objects: list[OptFlowObject] (== scene.objects). local2worlds: (M, 4, 4) world poses
        aligned to ``objects`` (from get_target2world). obs_intrinsics: (3, 3) obs K. ``obsmask.obs`` is
        RGBA whose alpha carries the instance foreground (full_alpha=False); the RGB channels are the full
        frame regardless, and UFM reads RGB[:3], so the alpha never reaches the warp."""
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            AnnotatorRegistry.get_annotator("camera_params"),
            AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": False}),
            # Per-leaf-prim occlusion ratio → ObsMask.iid_to_occlusion (see _occlusion_by_iid).
            AnnotatorRegistry.get_annotator("occlusion"),
        ]
        self._objects = objects
        self._l2w = local2worlds
        self._obs_K = obs_intrinsics
        self._render_dir = Path(render_dir)
        self._frame_id = 0
        self._full_alpha = full_alpha
        self.iid_to_name: dict[int, str] = {}   # accumulated per frame (iids are session-local)
        # The same per-class catalog ObsMaskWriter builds (cids, refs, DIFT descriptors): the only NN forward.
        (self.class_to_cid, self.name_to_class,
         self.class_to_ref, self.class_to_descriptors, self.backbone) = reference_catalog(
            objects, descriptor_config_path, descriptor_device)

    def attach(self, *rps):
        # ZED is stereo — capture_session passes both RPs; keep the LEFT one only
        # (mirrors ObsMaskWriter.attach).
        self._rp_key = rps[0].path.rsplit("/", 1)[-1]
        super().attach([rps[0]])

    def write(self, data: dict):
        obsmask, frame_iid_to_name = obsmask_from_data(
            data, self._rp_key, self.class_to_cid, full_alpha=self._full_alpha)
        self.iid_to_name.update(frame_iid_to_name)

        rp = data["renderProducts"][self._rp_key]
        depth = np.asarray(rp["distance_to_image_plane"]["data"], dtype=np.float32)        # full frame
        cam2world = np.linalg.inv(camera_params_to_world2cam(rp["camera_params"]))         # OpenCV cam2world

        OptFlowSample(
            obsmask=obsmask,                 # serialized FLAT → obs/ iid_mask/ cid_mask/ iid_to_occlusion/
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
            obsmaskmeta=obsmask_metadata(self.class_to_cid, self.name_to_class,
                                         self.class_to_ref, self.class_to_descriptors,
                                         self.iid_to_name, self.backbone),
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
        ).serialize(0, directory)

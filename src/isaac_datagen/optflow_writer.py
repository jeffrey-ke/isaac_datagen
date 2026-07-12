
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from vision_core.pose_utils import instance_visibility

from isaac_datagen.stereo_writer import camera_params_to_world2cam
from isaac_datagen.reference_seg_writer import reference_catalog, obsmask_from_data, obsmask_metadata
from isaac_datagen.objects import OptFlowSample, OptFlowMetadata


class OptFlowWriter(Writer):
    def __init__(self, objects, local2worlds, obs_intrinsics, render_dir,
                 descriptor_config_path, descriptor_device, full_alpha=False):
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            AnnotatorRegistry.get_annotator("camera_params"),
            AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": False}),
            AnnotatorRegistry.get_annotator("occlusion"),
        ]
        self._objects = objects
        self._l2w = local2worlds
        self._obs_K = obs_intrinsics
        self._render_dir = Path(render_dir)
        self._frame_id = 0
        self._full_alpha = full_alpha
        self.iid_to_name: dict[int, str] = {}
        self._md = None
        self._ref_cache = {}
        (self.class_to_cid, self.name_to_class,
         self.class_to_ref, self.class_to_descriptors, self.backbone) = reference_catalog(
            objects, descriptor_config_path, descriptor_device)

    def attach(self, *rps):
        self._rp_key = rps[0].path.rsplit("/", 1)[-1]
        super().attach([rps[0]])

    def write(self, data: dict):
        obsmask, frame_iid_to_name = obsmask_from_data(
            data, self._rp_key, self.class_to_cid, full_alpha=self._full_alpha)
        self.iid_to_name.update(frame_iid_to_name)

        rp = data["renderProducts"][self._rp_key]
        depth = np.asarray(rp["distance_to_image_plane"]["data"], dtype=np.float32)
        cam2world = np.linalg.inv(camera_params_to_world2cam(rp["camera_params"]))

        sample = OptFlowSample(
            obsmask=obsmask,
            observation_depth=depth,
            cam2world=cam2world.astype(np.float32),
            iid_to_visibility={},
        )
        sample.iid_to_visibility = instance_visibility(
            sample, self._optflow_metadata(), ref_cache=self._ref_cache)
        sample.serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def _optflow_metadata(self) -> OptFlowMetadata:
        if self._md is None:
            by_class = defaultdict(list)
            for o, L in zip(self._objects, self._l2w):
                by_class[o.meta["class"]].append((o, L))
            rep = {c: members[0][0] for c, members in by_class.items()}
            self._md = OptFlowMetadata(
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
            )
        self._md.obsmaskmeta.iid_to_name = dict(self.iid_to_name)
        return self._md

    def finalize_metadata(self, directory: str | Path | None = None):
        directory = Path(directory) if directory is not None else self._render_dir
        self._optflow_metadata().serialize(0, directory)

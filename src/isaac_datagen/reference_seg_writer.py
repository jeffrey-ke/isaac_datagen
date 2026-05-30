"""Replicator writer that generates ReferenceSegSamples from a mono camera render.

For each frame: identifies labeled objects by instance ID, runs a descriptor and
proposal network against each object's canonical reference image, and serializes
one ReferenceSegSample per (frame, object) to disk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from vision_core.datastructs import ReferenceSegSample


def _pil_to_tv_rgba(pil_img: PILImage.Image) -> tv_tensors.Image:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    return tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))


def alpha_from_instance_seg(seg: np.ndarray) -> np.ndarray:
    return (seg > 0).astype(np.uint8) * 255


def composite_rgba(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    alpha = alpha_from_instance_seg(seg)
    return np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)


class ReferenceSegWriter(Writer):
    def __init__(
        self,
        proposal_config_path: str,
        descriptor_config_path: str,
        object_specs,
        render_dir: str | Path,
        descriptor_device: str,
        proposer_device: str,
    ):
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast",
                init_params={"colorize": False},
            ),
        ]
        self._render_dir = Path(render_dir)
        self._frame_id = 0
        self.names_to_ref = {
            obj.meta["name"]: _pil_to_tv_rgba(obj.reference_image)
            for obj in object_specs
        }
        from reference_matching import proposal as proposal_module
        from reference_matching import descriptor as descriptor_module
        self.proposer = proposal_module.from_config(proposal_config_path).to(proposer_device)
        self.descriptor = descriptor_module.from_config(descriptor_config_path).to(descriptor_device)
        self.descriptor_device = descriptor_device
        self.proposer_device = proposer_device

    def attach(self, *rps):
        rp = rps[0]
        self._rp_key = rp.path.rsplit("/", 1)[-1]
        super().attach([rp])

    def write(self, data: dict):
        rp = data["renderProducts"][self._rp_key]
        rgb_hw3 = rp["rgb"]["data"][:, :, :3]
        seg_hw = rp["instance_segmentation_fast"]["data"]
        labels = rp["instance_segmentation_fast"]["idToSemantics"]

        # Only instances carrying an "instance" semantic are graspable objects;
        # BACKGROUND and UNLABELLED scenery (e.g. the workbench) carry no such
        # key and are skipped.
        id_to_name = {int(k): v["instance"] for k, v in labels.items() if "instance" in v}

        if not id_to_name:
            raise ValueError("write() called with no labeled instances — expected ≥1")

        obs_rgba = composite_rgba(rgb_hw3, seg_hw)
        obs_tensor = tv_tensors.Image(torch.from_numpy(obs_rgba).permute(2, 0, 1))

        seg_tensor = torch.from_numpy(seg_hw)

        unique_ids = list(id_to_name.keys())
        B = len(unique_ids)

        obs_1chw = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            for uid in unique_ids:
                ref_img = self.names_to_ref[id_to_name[uid]].unsqueeze(0)
                seg_mask = (seg_tensor == uid).bool()

                ref_features = self.descriptor(ref_img.to(self.descriptor_device))
                (xy, scores), = self.proposer(obs_1chw.to(self.proposer_device), ref_img.to(self.proposer_device))

                sample = ReferenceSegSample(
                    rgb=tv_tensors.Image(obs_tensor),
                    ref_rgb=tv_tensors.Image(ref_img[0]),
                    seg_mask=tv_tensors.Mask(seg_mask),
                    proposal_coordinates=xy.cpu(),
                    reference_features=ref_features[0].cpu(),
                )
                sample.serialize(self._frame_id, self._render_dir)
                self._frame_id += 1

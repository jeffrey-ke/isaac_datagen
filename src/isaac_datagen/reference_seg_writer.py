"""Replicator writer that emits one ObsMask per rendered frame (NN-free).

Per frame it serializes only the genuinely per-frame-unique payload — the RGBA
observation and the (H, W) instance-id mask. The constant object catalog
(reference images, precomputed DIFT descriptors, instance-id → name) is written
once per render dir via ``finalize_metadata`` as an ``ObsMaskMetadata``. The
expensive proposer forward is deferred to a phase-2 pass (see ``add_proposals``).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from vision_core.datastructs import ObsMask, ObsMaskMetadata


def _pil_to_tv_rgba(pil_img: PILImage.Image) -> tv_tensors.Image:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    return tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))


def alpha_from_instance_seg(seg: np.ndarray, valid_ids) -> np.ndarray:
    """Opaque only where ``seg`` is a graspable instance id; the workbench and
    background carry ids outside ``valid_ids`` and become transparent."""
    return np.isin(seg, list(valid_ids)).astype(np.uint8) * 255


def composite_rgba(rgb: np.ndarray, seg: np.ndarray, valid_ids) -> np.ndarray:
    alpha = alpha_from_instance_seg(seg, valid_ids)
    return np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)


class ObsMaskWriter(Writer):
    def __init__(
        self,
        descriptor_config_path: str,
        descriptor_device: str,
        object_specs,
        render_dir: str | Path,
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
        self.id_to_name: dict[int, str] = {}

        self.names_to_ref = {
            obj.meta["name"]: _pil_to_tv_rgba(obj.reference_image)
            for obj in object_specs
        }

        # Precompute the constant reference DIFT features once (only ~11 objects),
        # then drop the descriptor so per-frame write() stays NN-free.
        from reference_matching import descriptor as descriptor_module
        descriptor = descriptor_module.from_config(descriptor_config_path).to(descriptor_device)
        with torch.inference_mode():
            self.names_to_descriptors = {
                name: descriptor(tv_rgba.unsqueeze(0).to(descriptor_device)).squeeze().cpu()
                for name, tv_rgba in self.names_to_ref.items()
            }
        del descriptor

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
        # BACKGROUND and UNLABELLED scenery (e.g. the workbench) carry no such key.
        frame_id_to_name = {
            int(k): v["instance"] for k, v in labels.items() if "instance" in v
        }
        if not frame_id_to_name:
            raise ValueError("write() called with no labeled instances — expected ≥1")
        self.id_to_name.update(frame_id_to_name)

        obs_rgba = composite_rgba(rgb_hw3, seg_hw, frame_id_to_name.keys())
        obs = tv_tensors.Image(torch.from_numpy(obs_rgba).permute(2, 0, 1))
        id_mask = tv_tensors.Mask(torch.from_numpy(seg_hw.astype(np.int32)))

        ObsMask(obs=obs, id_mask=id_mask).serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        """Serialize the per-render-dir catalog once (at idx=0). Call after capture."""
        directory = Path(directory) if directory is not None else self._render_dir
        ObsMaskMetadata(
            id_to_name=self.id_to_name,
            name_to_ref=self.names_to_ref,
            name_to_descriptors=self.names_to_descriptors,
        ).serialize(0, directory)

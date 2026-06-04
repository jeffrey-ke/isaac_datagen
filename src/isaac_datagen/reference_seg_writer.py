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


def _occlusion_by_mask_id(occ, id_to_labels, instance_mappings, present_ids) -> dict[int, float]:
    """Map per-frame occlusion ratios onto ``instance_segmentation_fast`` mask ids.

    The ``occlusion`` annotator keys ratios by *leaf-prim* id, while the seg mask keys by
    *semantic-instance* id — different, non-aligned id spaces (verified in a kit probe). Both
    sides expose the geo prim path, so we bridge ``leaf id → path`` via ``instance_mappings``
    (its ``instanceIds`` are the leaf ids occlusion keys on, ``name`` is the path) and
    ``path → mask id`` via the seg payload's ``idToLabels``.

    Returns ``{mask_id: ratio}`` for every id in ``present_ids``; an id with no occlusion
    measurement (e.g. fully outside the frustum → NaN row) maps to ``float('nan')`` so the
    caller can tell "unknown" apart from "unoccluded". Leaf ratios are averaged per object so
    a (hypothetical) multi-mesh asset still yields one number; our assets are single-mesh.
    """
    occ_by_leaf = {
        int(r["instanceId"]): float(r["occlusionRatio"])
        for r in occ
        if not np.isnan(r["occlusionRatio"])
    }
    path_to_occ: dict[str, float] = {}
    for row in instance_mappings:
        vals = [occ_by_leaf[int(l)] for l in row["instanceIds"] if int(l) in occ_by_leaf]
        if vals:
            path_to_occ[row["name"]] = float(np.mean(vals))

    out = {i: float("nan") for i in present_ids}
    for mask_id in present_ids:
        path = id_to_labels.get(mask_id, id_to_labels.get(str(mask_id)))
        if path in path_to_occ:
            out[mask_id] = path_to_occ[path]
    return out


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
            # Per-leaf-prim occlusion ratio (0=unoccluded … 1=fully occluded). Keyed by
            # leaf-prim id, NOT the mask's semantic-instance id — see _occlusion_by_mask_id.
            AnnotatorRegistry.get_annotator("occlusion"),
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

        # Per-instance occlusion, keyed by the same ids as id_mask (graspable ids actually
        # present this frame — the set add_proposals later filters on).
        from omni.syntheticdata.scripts import helpers
        present_ids = {int(i) for i in np.unique(seg_hw)} & set(frame_id_to_name)
        id_to_occlusion = _occlusion_by_mask_id(
            rp["occlusion"]["data"],
            rp["instance_segmentation_fast"]["idToLabels"],
            helpers.get_instance_mappings(),
            present_ids,
        )

        obs_rgba = composite_rgba(rgb_hw3, seg_hw, frame_id_to_name.keys())
        obs = tv_tensors.Image(torch.from_numpy(obs_rgba).permute(2, 0, 1))
        id_mask = tv_tensors.Mask(torch.from_numpy(seg_hw.astype(np.int32)))

        ObsMask(obs=obs, id_mask=id_mask, id_to_occlusion=id_to_occlusion) \
            .serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        """Serialize the per-render-dir catalog once (at idx=0). Call after capture."""
        directory = Path(directory) if directory is not None else self._render_dir
        ObsMaskMetadata(
            id_to_name=self.id_to_name,
            name_to_ref=self.names_to_ref,
            name_to_descriptors=self.names_to_descriptors,
        ).serialize(0, directory)

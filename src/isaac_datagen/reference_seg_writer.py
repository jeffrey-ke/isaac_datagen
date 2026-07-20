
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

import yaml

from vision_core.datastructs import ObsMask, build_obsmask_metadata

from isaac_datagen.isaac_utils import IidCanonicalizer, cid_iid_masks


def _pil_to_tv_rgba(pil_img: PILImage.Image) -> tv_tensors.Image:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    return tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))


def alpha_from_instance_seg(seg: np.ndarray, valid_ids) -> np.ndarray:
    return np.isin(seg, list(valid_ids)).astype(np.uint8) * 255


def composite_rgba(rgb: np.ndarray, seg: np.ndarray, valid_ids, full_alpha: bool = False) -> np.ndarray:
    alpha = (np.full(seg.shape, 255, np.uint8) if full_alpha
             else alpha_from_instance_seg(seg, valid_ids))
    return np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)


def _occlusion_by_iid(occ, iid_to_labels, instance_mappings, present_iids) -> dict[int, float]:
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

    out = {i: float("nan") for i in present_iids}
    for iid in present_iids:
        path = iid_to_labels.get(iid, iid_to_labels.get(str(iid)))
        if path in path_to_occ:
            out[iid] = path_to_occ[path]
    return out


def reference_catalog(object_specs, descriptor_config_path, descriptor_device):
    classes = sorted({obj.meta["class"] for obj in object_specs})
    class_to_cid = {cls: cid for cid, cls in enumerate(classes, start=2)}
    name_to_class = {obj.meta["name"]: obj.meta["class"] for obj in object_specs}
    class_to_ref: dict[str, tv_tensors.Image] = {}
    for obj in sorted(object_specs, key=lambda o: o.meta["name"]):
        class_to_ref.setdefault(obj.meta["class"], _pil_to_tv_rgba(obj.reference_image))

    backbone = yaml.safe_load(Path(descriptor_config_path).read_text())["name"]
    from reference_matching import descriptor as descriptor_module
    descriptor = descriptor_module.from_config(descriptor_config_path).to(descriptor_device)
    with torch.inference_mode():
        class_to_descriptors = {
            cls: descriptor.to_leaf(descriptor(descriptor.prep(tv_rgba).unsqueeze(0).to(descriptor_device)))
            for cls, tv_rgba in class_to_ref.items()
        }
    del descriptor
    return class_to_cid, name_to_class, class_to_ref, class_to_descriptors, backbone


def obsmask_from_data(data, rp_key, class_to_cid, *, canon, full_alpha):   # canon REQUIRED — no
    rp = data["renderProducts"][rp_key]
    seg_hw = rp["instance_segmentation_fast"]["data"]
    labels = rp["instance_segmentation_fast"]["idToSemantics"]

    iid_mask, cid_mask, frame_iid_to_name = cid_iid_masks(seg_hw, labels, class_to_cid)
    if not frame_iid_to_name:
        raise ValueError("write() called with no labeled instances — expected ≥1")

    from omni.syntheticdata.scripts import helpers
    present_iids = {int(i) for i in np.unique(seg_hw)} & set(frame_iid_to_name)
    iid_to_occlusion = _occlusion_by_iid(
        rp["occlusion"]["data"],
        rp["instance_segmentation_fast"]["idToLabels"],
        helpers.get_instance_mappings(),
        present_iids,
    )

    obs_rgba = composite_rgba(rp["rgb"]["data"][:, :, :3], seg_hw,
                              frame_iid_to_name.keys(), full_alpha=full_alpha)  # raw keys: pixel set
    obs = tv_tensors.Image(torch.from_numpy(obs_rgba).permute(2, 0, 1))         # identical either way
    iid_mask, frame_iid_to_name, iid_to_occlusion = canon.canonicalize(    # last step: annotator
        iid_mask, frame_iid_to_name, iid_to_occlusion)                     # tables above are raw-keyed
    return ObsMask(obs=obs, iid_mask=iid_mask, cid_mask=cid_mask,
                   iid_to_occlusion=iid_to_occlusion), frame_iid_to_name


def obsmask_metadata(class_to_cid, name_to_class, class_to_ref, class_to_descriptors, iid_to_name,
                     backbone):
    return build_obsmask_metadata(class_to_cid, name_to_class, class_to_ref, class_to_descriptors,
                                  iid_to_name, backbone)


class ObsMaskWriter(Writer):
    def __init__(
        self,
        descriptor_config_path: str,
        descriptor_device: str,
        object_specs,
        render_dir: str | Path,
        full_alpha: bool = False,
    ):
        self.data_structure = "renderProduct"
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast",
                init_params={"colorize": False},
            ),
            AnnotatorRegistry.get_annotator("occlusion"),
        ]
        self._render_dir = Path(render_dir)
        self._frame_id = 0
        self._full_alpha = full_alpha
        self.iid_to_name: dict[int, str] = {}
        self._canon = IidCanonicalizer()          # one per render: same lifetime as iid_to_name
        from isaac_datagen import cid_iid_trace
        if not cid_iid_trace.enabled():
            cid_iid_trace.init(self._render_dir)
        (self.class_to_cid, self.name_to_class,
         self.class_to_ref, self.class_to_descriptors, self.backbone) = reference_catalog(
            object_specs, descriptor_config_path, descriptor_device)
        self.cid_to_class = {cid: cls for cls, cid in self.class_to_cid.items()}

    def attach(self, *rps):
        rp = rps[0]
        self._rp_key = rp.path.rsplit("/", 1)[-1]
        super().attach([rp])

    def write(self, data: dict):
        obsmask, frame_iid_to_name = obsmask_from_data(
            data, self._rp_key, self.class_to_cid,
            canon=self._canon, full_alpha=self._full_alpha)   # thread the shared remap state
        self.iid_to_name.update(frame_iid_to_name)
        obsmask.serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        assert len(set(self.iid_to_name.values())) == len(self.iid_to_name), \
            "writer contract violated: iid_to_name not 1:1"   # documents the new contract at the source
        directory = Path(directory) if directory is not None else self._render_dir
        obsmask_metadata(self.class_to_cid, self.name_to_class, self.class_to_ref,
                         self.class_to_descriptors, self.iid_to_name,
                         self.backbone).serialize(0, directory)

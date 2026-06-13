"""Replicator writer that emits one ObsMask per rendered frame (NN-free).

Per frame it serializes only the genuinely per-frame-unique payload — the RGBA
observation and (H, W) masks in BOTH id spaces: the raw instance-id mask (iid_mask,
pairs with per-instance occlusion) and a class-id mask (cid_mask) derived from it by
a LUT over ``idToSemantics``'s class labels, so all same-class boxes share one value.
The constant object catalog (canonical per-class reference images, precomputed DIFT
descriptors, id-space maps) is written once per render dir via ``finalize_metadata``
as an ``ObsMaskMetadata``. The expensive proposer forward is deferred to a phase-2
pass (see ``add_proposals``).

Why not Isaac's ``semantic_segmentation`` annotator for the class mask: it assigns ids
by each prim's FULL semantic-set string, and our prims carry both ``class`` and
``instance`` labels (scene.py), so every box's string is unique and same-class boxes
would NOT merge (see OgnSemanticSegmentation.py:150-159 — exact-string ``_get_ids``).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import tv_tensors

from omni.replicator.core import AnnotatorRegistry, Writer

from vision_core.datastructs import ObsMask, ObsMaskMetadata
from vision_core.viz import fit_pca_basis


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


def _occlusion_by_iid(occ, iid_to_labels, instance_mappings, present_iids) -> dict[int, float]:
    """Map per-frame occlusion ratios onto ``instance_segmentation_fast`` mask iids.

    The ``occlusion`` annotator keys ratios by *leaf-prim* id, while the seg mask keys by
    *semantic-instance* id (iid) — different, non-aligned id spaces (verified in a kit
    probe). Both sides expose the geo prim path, so we bridge ``leaf id → path`` via
    ``instance_mappings`` (its ``instanceIds`` are the leaf ids occlusion keys on, ``name``
    is the path) and ``path → iid`` via the seg payload's ``idToLabels``.

    Returns ``{iid: ratio}`` for every iid in ``present_iids``; an iid with no occlusion
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

    out = {i: float("nan") for i in present_iids}
    for iid in present_iids:
        path = iid_to_labels.get(iid, iid_to_labels.get(str(iid)))
        if path in path_to_occ:
            out[iid] = path_to_occ[path]
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
            AnnotatorRegistry.get_annotator("HdrColor"),   # DEBUG probe (black-render investigation)
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast",
                init_params={"colorize": False},
            ),
            # Per-leaf-prim occlusion ratio (0=unoccluded … 1=fully occluded). Keyed by
            # leaf-prim id, NOT the mask's semantic-instance id (iid) — see _occlusion_by_iid.
            AnnotatorRegistry.get_annotator("occlusion"),
        ]
        self._render_dir = Path(render_dir)
        self._frame_id = 0
        self.iid_to_name: dict[int, str] = {}   # accumulated per frame (iids are session-local)

        # Static class catalog. cids are deterministic across render dirs only because
        # they derive from the SORTED class set of the scene's objects — a config that
        # places a different subset shifts them; each render dir stays self-describing
        # via cid_to_class. cids start at 2, mirroring Isaac's 0=BACKGROUND/1=UNLABELLED.
        classes = sorted({obj.meta["class"] for obj in object_specs})
        self.class_to_cid = {cls: cid for cid, cls in enumerate(classes, start=2)}
        self.cid_to_class = {cid: cls for cls, cid in self.class_to_cid.items()}
        self.name_to_class = {obj.meta["name"]: obj.meta["class"] for obj in object_specs}

        # One canonical reference per class: first member by sorted name (same-class
        # members are near-duplicate box fronts by construction of the relabeling).
        self.class_to_ref: dict[str, tv_tensors.Image] = {}
        for obj in sorted(object_specs, key=lambda o: o.meta["name"]):
            self.class_to_ref.setdefault(obj.meta["class"], _pil_to_tv_rgba(obj.reference_image))

        # Precompute the constant reference DIFT features once (one per class),
        # then drop the descriptor so per-frame write() stays NN-free.
        from reference_matching import descriptor as descriptor_module
        descriptor = descriptor_module.from_config(descriptor_config_path).to(descriptor_device)
        with torch.inference_mode():
            self.class_to_descriptors = {
                cls: descriptor(tv_rgba.unsqueeze(0).to(descriptor_device)).squeeze(0).cpu()
                for cls, tv_rgba in self.class_to_ref.items()   # (C, h, w) spatial
            }
        del descriptor

    def attach(self, *rps):
        rp = rps[0]
        self._rp_key = rp.path.rsplit("/", 1)[-1]
        super().attach([rp])

    def write(self, data: dict):
        rp = data["renderProducts"][self._rp_key]
        rgb_hw3 = rp["rgb"]["data"][:, :, :3]

        # DEBUG probe (black-render investigation): does a black frame have zero
        # radiance in the raw HDR buffer (→ scene/lighting/beauty-pass), or only
        # in the post-tonemap LDR `rgb` (→ capture/tonemap)? Also log the dome
        # light's actual intensity at capture time.
        import omni.usd
        from pxr import UsdLux
        _hdr = rp["HdrColor"]["data"] if "HdrColor" in rp else None
        _hmax = float(np.asarray(_hdr).max()) if _hdr is not None else None
        _dome = UsdLux.DomeLight.Get(omni.usd.get_context().get_stage(), "/World/DomeLight")
        _di = _dome.GetIntensityAttr().Get() if _dome and _dome.GetPrim().IsValid() else None
        print(f"[PROBE] f={self._frame_id} ldr_max={int(rgb_hw3.max())} hdr_max={_hmax} dome_I={_di}", flush=True)

        seg_hw = rp["instance_segmentation_fast"]["data"]
        labels = rp["instance_segmentation_fast"]["idToSemantics"]

        # Only instances carrying an "instance" semantic are graspable objects;
        # BACKGROUND and UNLABELLED scenery (e.g. the workbench) carry no such key.
        frame_iid_to_name = {
            int(k): v["instance"] for k, v in labels.items() if "instance" in v
        }
        if not frame_iid_to_name:
            raise ValueError("write() called with no labeled instances — expected ≥1")
        self.iid_to_name.update(frame_iid_to_name)

        # Per-instance occlusion, keyed by the same iids as iid_mask (graspable iids
        # actually present this frame).
        from omni.syntheticdata.scripts import helpers
        present_iids = {int(i) for i in np.unique(seg_hw)} & set(frame_iid_to_name)
        iid_to_occlusion = _occlusion_by_iid(
            rp["occlusion"]["data"],
            rp["instance_segmentation_fast"]["idToLabels"],
            helpers.get_instance_mappings(),
            present_iids,
        )

        # Class-id mask: LUT remap iid → cid via the class half of idToSemantics, so all
        # same-class boxes share one value. Unknown iids (background, UNLABELLED, scenery)
        # default to 0.
        frame_iid_to_cid = {
            int(k): self.class_to_cid[v["class"]]
            for k, v in labels.items()
            if "class" in v and v["class"] in self.class_to_cid
        }
        lut = np.zeros(max(int(seg_hw.max()), max(frame_iid_to_cid, default=0)) + 1, dtype=np.uint8)
        for iid, cid in frame_iid_to_cid.items():
            lut[iid] = cid
        cid_mask = tv_tensors.Mask(torch.from_numpy(lut[seg_hw]))

        obs_rgba = composite_rgba(rgb_hw3, seg_hw, frame_iid_to_name.keys())
        obs = tv_tensors.Image(torch.from_numpy(obs_rgba).permute(2, 0, 1))
        iid_mask = tv_tensors.Mask(torch.from_numpy(seg_hw.astype(np.int32)))

        ObsMask(obs=obs, iid_mask=iid_mask, cid_mask=cid_mask, iid_to_occlusion=iid_to_occlusion) \
            .serialize(self._frame_id, self._render_dir)
        self._frame_id += 1

    def finalize_metadata(self, directory: str | Path | None = None):
        """Serialize the per-render-dir catalog once (at idx=0). Call after capture."""
        directory = Path(directory) if directory is not None else self._render_dir
        # Shared PCA→RGB basis over ALL classes' tokens (each (C,h,w) → (h*w, C),
        # stacked — the same flatten(1).T tokenization consumers read), so every
        # class projects into comparable colors. Mandatory ObsMaskMetadata field.
        tokens = torch.cat(
            [d.flatten(1).T for d in self.class_to_descriptors.values()], dim=0
        )
        ObsMaskMetadata(
            iid_to_name=self.iid_to_name,
            cid_to_class=self.cid_to_class,
            name_to_class=self.name_to_class,
            class_to_ref=self.class_to_ref,
            class_to_descriptors=self.class_to_descriptors,
            principal_components=fit_pca_basis(tokens, n=3),
        ).serialize(0, directory)

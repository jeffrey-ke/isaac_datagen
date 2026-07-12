
from __future__ import annotations

from vision_core.pose_utils import instance_visibility


def gate_classes(om, md, min_visible_px: int) -> dict[str, int]:
    present_iids = {int(i) for i in om.iid_mask.unique().tolist()} & set(md.iid_to_name)
    out: dict[str, int] = {}
    for iid in present_iids:
        area = int((om.iid_mask == iid).sum())
        if area > min_visible_px:
            cls = md.name_to_class[md.iid_to_name[iid]]
            out[cls] = max(out.get(cls, 0), area)
    return out


def gate_classes_reproj(
    sample, md, min_visible_ratio: float, tau_d: float = 0.001, tau_r: float = 0.005,
    ref_cache: dict | None = None,
) -> dict[str, float]:
    mm = md.obsmaskmeta
    iid_to_name = {int(k): v for k, v in mm.iid_to_name.items()}
    vis = instance_visibility(sample, md, tau_d, tau_r, ref_cache)
    out: dict[str, float] = {}
    for iid, ratio in vis.items():
        if ratio > min_visible_ratio:
            cls = mm.name_to_class[iid_to_name[iid]]
            out[cls] = max(out.get(cls, 0.0), ratio)
    return out

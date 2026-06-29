"""The proposer's class gate, single-sourced for the phase-2 pass and the gate-decision viz.

Current gate — **reprojection coverage** (``instance_visibility`` / ``gate_classes_reproj``): a class is
proposed on a frame iff ANY of its member instances has more than ``min_visible_ratio`` of its reference
texture visible in the observation. Per instance we back-project the class's canonical reference RGB-D
into the object-local frame, push it through that instance's ``class_to_l2w`` placement into the
observation camera, and take the fraction of reference-surface points that are NOT occluded / off-frame
(``vision_core.pose_utils.reproject_local_to_obs`` → ``reprojection_occlusion``). This is the true notion
of occlusion w.r.t. the visible reference texture, and — because the denominator is reference points, not
observation pixels — it is invariant to how far the camera is (a small-but-fully-visible object scores
~1.0). It needs ``OptFlowSample``/``OptFlowMetadata`` (obs depth, ``cam2world``, per-instance
``class_to_l2w``, perspective reference depth/pose/K).

Superseded gates (kept harmless for back-compat): ``gate_classes`` gated on raw visible pixel area
(``iid_mask``), which fixed the older occlusion-ratio gate but dropped fully-visible objects when the
camera was far; the occlusion-ratio gate before that measured something similar-but-unrelated.
"""

from __future__ import annotations

from vision_core.pose_utils import instance_visibility  # moved here; re-exported for back-compat importers


def gate_classes(om, md, min_visible_px: int) -> dict[str, int]:
    """Classes admitted by the visible-pixel gate on one frame → ``{class: best_member_px}``.

    A class is admitted iff some member instance has ``(iid_mask == iid).sum() > min_visible_px``; the
    value is that best (largest) member's pixel count. ``om`` is an ``ObsMask`` (uses ``iid_mask``),
    ``md`` an ``ObsMaskDescriptorMetadata`` (uses ``iid_to_name`` + ``name_to_class``). iids are
    session-local instance ids, valid only within this render dir.
    """
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
    """Classes admitted by the reprojection-coverage gate on one frame → ``{class: best_member_ratio}``.

    A class is admitted iff some member instance's reference-texture visibility (``instance_visibility``)
    exceeds ``min_visible_ratio``; the value is that best member's ratio — the per-instance "best-visible
    member" structure of the old px gate, swapping pixel area for visibility ratio."""
    mm = md.obsmaskmeta
    iid_to_name = {int(k): v for k, v in mm.iid_to_name.items()}
    vis = instance_visibility(sample, md, tau_d, tau_r, ref_cache)
    out: dict[str, float] = {}
    for iid, ratio in vis.items():
        if ratio > min_visible_ratio:
            cls = mm.name_to_class[iid_to_name[iid]]
            out[cls] = max(out.get(cls, 0.0), ratio)
    return out

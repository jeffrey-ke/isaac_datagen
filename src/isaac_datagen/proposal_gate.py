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

import numpy as np
import torch

from vision_core.pose_utils import (
    depthmap_to_pointmap,
    reproject_local_to_obs,
    transform_pointmap,
)


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


def _ref_points(md, cls: str) -> torch.Tensor:
    """The class's canonical reference RGB-D back-projected into object-LOCAL surface points.

    ``(N, 3)`` for the ``N`` on-object reference pixels (``ref_depth > 0``). Depends only on the class,
    not the frame — callers cache it per render dir (see ``ref_cache``)."""
    ref_depth = md.class_to_reference_depth[cls].float()
    ref_K = md.class_to_ref_intrinsics[cls].float()
    ref_pose = md.class_to_ref_pose[cls].float()                 # camera2local, OpenCV
    pm_local = transform_pointmap(depthmap_to_pointmap(ref_depth, ref_K), ref_pose)
    return pm_local[ref_depth > 0]                               # (N, 3) dense — no FPS


def instance_visibility(
    sample, md, tau_d: float = 0.001, tau_r: float = 0.005, ref_cache: dict | None = None,
) -> dict[int, float]:
    """Per-instance reference-texture visibility on one frame → ``{iid: visible_ratio in [0, 1]}``.

    For every present iid that maps to a (class, ``class_to_l2w`` row) and whose class has a reference,
    reproject the class's dense reference surface through that instance's placement into the observation
    camera and take ``1 - (occluded fraction)``. ``sample`` is an ``OptFlowSample`` (``obsmask.iid_mask``,
    ``observation_depth``, ``cam2world``); ``md`` an ``OptFlowMetadata``. Pass a persistent ``ref_cache``
    dict across frames of one render dir to compute each class's reference points once."""
    mm = md.obsmaskmeta
    ref_cache = {} if ref_cache is None else ref_cache
    name_to_iid = {nm: int(i) for i, nm in mm.iid_to_name.items()}
    cn = {                                                       # iid -> (class, l2w row)
        name_to_iid[nm]: (c, n)
        for c, names in md.class_to_name.items()
        for n, nm in enumerate(names)
        if nm in name_to_iid
    }
    iid_mask = sample.obsmask.iid_mask
    present = {int(i) for i in iid_mask.unique().tolist()} & set(cn)
    if not present:
        return {}

    obs_K = torch.as_tensor(np.asarray(md.obs_intrinsics), dtype=torch.float32)
    obs_depth = torch.as_tensor(np.asarray(sample.observation_depth), dtype=torch.float32)
    w2c = torch.linalg.inv(torch.as_tensor(np.asarray(sample.cam2world), dtype=torch.float32))

    class_members: dict[str, list[tuple[int, int]]] = {}        # class -> [(iid, l2w row), ...]
    for iid in present:
        c, row = cn[iid]
        class_members.setdefault(c, []).append((iid, row))

    out: dict[int, float] = {}
    for c, members in class_members.items():
        if c not in md.class_to_reference_depth:
            continue
        ref_pts = ref_cache.get(c)
        if ref_pts is None:
            ref_pts = _ref_points(md, c)
            ref_cache[c] = ref_pts
        if ref_pts.shape[0] == 0:
            continue
        rows = torch.as_tensor([row for _, row in members], dtype=torch.long)
        L = md.class_to_l2w[c][rows].float()                    # (M, 4, 4)
        occluded, _, _ = reproject_local_to_obs(ref_pts, L, w2c, obs_K, obs_depth, tau_d, tau_r)
        ratios = (1.0 - occluded.float().mean(1)).tolist()      # (M,) visible fraction per member
        for (iid, _), r in zip(members, ratios):
            out[iid] = float(r)
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

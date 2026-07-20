
from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import torch
from torchvision import tv_tensors


def cid_iid_masks(seg_hw, labels, class_to_cid):
    frame_iid_to_name = {int(k): v["instance"] for k, v in labels.items() if "instance" in v}
    frame_iid_to_cid = {
        int(k): class_to_cid[v["class"]]
        for k, v in labels.items()
        if "class" in v and v["class"] in class_to_cid
    }
    lut = np.zeros(max(int(seg_hw.max()), max(frame_iid_to_cid, default=0)) + 1, dtype=np.uint8)
    for iid, cid in frame_iid_to_cid.items():
        lut[iid] = cid
    cid_mask = tv_tensors.Mask(torch.from_numpy(lut[seg_hw]))
    iid_mask = tv_tensors.Mask(torch.from_numpy(seg_hw.astype(np.int32)))
    return iid_mask, cid_mask, frame_iid_to_name


class IidCanonicalizer:
    """One id per physical object: remaps sibling component ids to the first-seen id per name."""

    def __init__(self):
        self._name_to_canon: dict[str, int] = {}   # name -> first-seen id (the canonical id)
        self._iid_to_name: dict[int, str] = {}     # raw id -> name, for the reuse guard

    def canonicalize(self, iid_mask, frame_iid_to_name, iid_to_occlusion):
        for iid, name in frame_iid_to_name.items():
            seen = self._iid_to_name.setdefault(iid, name)
            if seen != name:                                        # annotator id re-used for a
                raise ValueError(                                   # different object: fail loud,
                    f"annotator id {iid} renamed {seen!r} -> {name!r} mid-render")  # never remap
            self._name_to_canon.setdefault(name, iid)               # first-seen id wins, forever
        remap = {i: self._name_to_canon[n] for i, n in frame_iid_to_name.items()
                 if self._name_to_canon[n] != i}                    # only sibling ids need moving
        if not remap:
            return iid_mask, frame_iid_to_name, iid_to_occlusion    # common case: nothing to do
        t = iid_mask.as_subclass(torch.Tensor)
        for raw, canon in remap.items():
            t[t == raw] = canon                                     # collapse pixels onto canonical id
        canon_names = {self._name_to_canon[n]: n
                       for n in frame_iid_to_name.values()}         # 1:1 per-frame map
        canon_occ = {}
        for raw, v in iid_to_occlusion.items():
            canon = self._name_to_canon[frame_iid_to_name[raw]]
            if canon == raw or canon not in canon_occ:              # canonical id's own row wins;
                canon_occ[canon] = v                                # a sibling row only fills a gap
        return iid_mask, canon_names, canon_occ                     # occ values viz-only (spec §3)


@dataclass
class WriterSpec:
    render_product: object
    output_dir: str
    rgb: bool = False
    distance_to_camera: bool = False
    camera_params: bool = False
    bounding_box_3d: bool = False
    instance_segmentation: bool = False

    def writer_kwargs(self):
        return {
            'output_dir': self.output_dir,
            'rgb': self.rgb,
            'distance_to_camera': self.distance_to_camera,
            'camera_params': self.camera_params,
            'bounding_box_3d': self.bounding_box_3d,
            'instance_segmentation': self.instance_segmentation,
        }


def setup_replicator_writers(specs: list[WriterSpec], rep):
    writers = []
    for spec in specs:
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(**spec.writer_kwargs())
        writer.attach([spec.render_product])
        writers.append(writer)
    return writers


def create_empty(name, parent_prim=None):
    from pxr import Usd, UsdGeom
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    if parent_prim:
        prim_path = f"{parent_prim}/{name}"
    else:
        prim_path = f"/World/{name}"

    with Usd.EditContext(stage, stage.GetRootLayer()):
        xform = UsdGeom.Xform.Define(stage, prim_path)
    return xform.GetPrim()


def load_asset(prim_path, usd_path, ref_prim_path=None):
    from pxr import Usd
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        prim = stage.DefinePrim(prim_path)
        if ref_prim_path is None:
            prim.GetReferences().AddReference(usd_path)
        else:
            prim.GetReferences().AddReference(usd_path, ref_prim_path)
    return prim


def set_transform(prim, translation=None, rotation=None, scale=None):
    from pxr import Gf, Usd, UsdGeom
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        xform = UsdGeom.Xformable(prim)

        def _get_existing_op(op_type_token):
            for op in xform.GetOrderedXformOps():
                if op.GetOpName() == op_type_token:
                    return op
            return None

        if scale is not None:
            scale_op = _get_existing_op('xformOp:scale')
            if scale_op is None:
                scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3f(*scale))

        if translation is not None:
            translate_op = _get_existing_op('xformOp:translate')
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*translation))

        if rotation is not None:
            rotate_op = _get_existing_op('xformOp:rotateXYZ')
            if rotate_op is None:
                rotate_op = xform.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(*rotation))


def get_transform(prim):
    from pxr import Gf, Usd, UsdGeom
    xform = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotation()
    scale = Gf.Vec3f(1, 1, 1)
    return {
        'translation': (translation[0], translation[1], translation[2]),
        'rotation': rotation,
        'scale': (scale[0], scale[1], scale[2])
    }


def setup_camera(name, prim_path, width, height, intrinsics, focal_length_mm=2.8):
    from pxr import Gf, Sdf, UsdGeom
    from isaacsim.core.utils.stage import get_current_stage

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    sensor_width_mm = width * focal_length_mm / fx
    sensor_height_mm = height * focal_length_mm / fy

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.GetHorizontalApertureAttr().Set(sensor_width_mm)
    camera.GetVerticalApertureAttr().Set(sensor_height_mm)
    camera.GetFocalLengthAttr().Set(focal_length_mm)
    camera.GetHorizontalApertureOffsetAttr().Set((cx - width / 2) / fx)
    camera.GetVerticalApertureOffsetAttr().Set((cy - height / 2) / fy)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

    prim = camera.GetPrim()
    prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
    prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx").Set(fx)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:fy").Set(fy)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:cx").Set(cx)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:cy").Set(cy)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Set(Gf.Vec2i(width, height))

    return prim


def setup_render_product(camera_path, resolution, output_name):
    import omni.replicator.core as rep
    return rep.create.render_product(camera_path, resolution)


def find_prims(root, pattern="*", action=None, action_mode=";"):
    from pxr import Usd
    from isaacsim.core.utils.stage import get_current_stage
    if isinstance(root, str):
        root = get_current_stage().GetPrimAtPath(root)
    matches = [
        p.GetPath().pathString
        for p in Usd.PrimRange(root)
        if fnmatch(p.GetName(), pattern)
    ]
    if not matches:
        raise ValueError(f"No prims matching '{pattern}' under {root.GetPath()}")
    if action is None:
        return matches
    if action_mode == ";":
        return [action(path) for path in matches]
    elif action_mode == "+":
        return action(*matches)


def bounding_half_extents(prim):
    from pxr import Usd, UsdGeom
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    extent = bbox.ComputeLocalBound(prim).GetRange().GetSize()
    return (extent[0] / 2, extent[1] / 2, extent[2] / 2)


def local_bbox_range(prim):
    from pxr import Usd, UsdGeom
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return bbox.ComputeLocalBound(prim).GetRange()


def untransformed_bbox_range(prim):
    from pxr import Usd, UsdGeom
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()


def bottom_face_center(prim):
    bbox_range = local_bbox_range(prim)
    center = bbox_range.GetMidpoint()
    return (center[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2])


def deactivate_prim(prim):
    from pxr import Usd
    stage = prim.GetStage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        prim.SetActive(False)


def disable_rigid_body(prim) -> bool:
    from pxr import Usd, UsdPhysics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    stage = prim.GetStage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    return True


def _localize_remote_assets(layer_path):
    import os
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(layer_path)
    dep_dir = os.path.dirname(layer_path)
    downloaded = {}

    def localize(asset_path):
        if not asset_path.startswith(("http://", "https://")):
            return asset_path
        if asset_path not in downloaded:
            import omni.client
            name = f"{len(downloaded):03d}_{os.path.basename(asset_path).split('?')[0]}"
            result = omni.client.copy(asset_path, os.path.join(dep_dir, name))
            if result != omni.client.Result.OK:
                raise RuntimeError(f"remote asset download failed ({result}): {asset_path}")
            downloaded[asset_path] = f"./{name}"
        return downloaded[asset_path]

    UsdUtils.ModifyAssetPaths(layer, localize)
    if downloaded:
        layer.Save()


def export_subtree_usdz(stage, subtree_path, output_dir, base_name="scene",
                        root_prim=None, neutralize_root_xform=False):
    import os
    import tempfile
    from pxr import Usd, UsdUtils

    src_prim = stage.GetPrimAtPath(subtree_path)
    if not src_prim or not src_prim.IsValid():
        raise ValueError(f"No valid prim at {subtree_path!r} on the given stage")

    solo = Usd.Stage.CreateInMemory()

    export_prim = solo.DefinePrim(root_prim or f"/{src_prim.GetName()}")
    export_prim.GetReferences().AddReference(
        assetPath=stage.GetRootLayer().identifier,
        primPath=subtree_path,
    )
    if neutralize_root_xform:
        from pxr import UsdGeom, Vt
        UsdGeom.Xformable(export_prim).CreateXformOpOrderAttr().Set(Vt.TokenArray())

    solo.SetDefaultPrim(export_prim)

    os.makedirs(output_dir, exist_ok=True)
    usdz_path = os.path.join(output_dir, f"{base_name}.usdz")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_usd = os.path.join(temp_dir, f"{base_name}.usdc")
        if not solo.Export(temp_usd):
            raise RuntimeError(f"solo.Export({temp_usd}) returned False")
        _localize_remote_assets(temp_usd)
        if not UsdUtils.CreateNewUsdzPackage(temp_usd, usdz_path):
            raise RuntimeError("UsdUtils.CreateNewUsdzPackage returned False")

    if not os.path.exists(usdz_path) or os.path.getsize(usdz_path) == 0:
        raise RuntimeError(f"USDZ packaging produced no/empty file at {usdz_path}")

    return usdz_path


def export_flattened_usdz(stage, output_dir, base_name="scene"):
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError("Stage has no valid defaultPrim to export")
    return export_subtree_usdz(
        stage, default_prim.GetPath().pathString, output_dir, base_name=base_name
    )

def class_label(prim_path):
    from isaacsim.core.utils.semantics import get_labels
    stage = get_current_stage()
    geo = stage.GetPrimAtPath(f"{prim_path}/geo")
    labels = get_labels(geo)
    if not labels.get("class"):
        raise ValueError(f"ShelfPlacer: no 'class' label on {prim_path}/geo")
    return labels["class"][0]

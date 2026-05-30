"""Utility functions for Isaac Sim Replicator data generation."""

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np


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
    """Creates an empty transform (Xform) in the USD stage and authors it in the root layer.

    Why: When called inside replicator's rep.new_layer(), the current edit target can be a
    transient in-memory layer. Authoring directly to the root ensures the prim appears in the
    exported USD file.

    Args:
        name: Name of the empty transform
        parent_prim: Optional parent prim path

    Returns:
        The created Xform prim
    """
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
    """Loads a USD asset into the stage, referencing it from the root layer.

    Args:
        prim_path: Path where the asset should be loaded
        usd_path: Path to the USD file
        ref_prim_path: Optional prim path inside usd_path to reference. Pass this
            for assets that lack a defaultPrim (e.g. the dataset .usdz files,
            whose content lives under "/World"); omit to use the defaultPrim.

    Returns:
        The loaded prim
    """
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
    """Sets (or updates) the transform ops on a prim and authors them in the root layer.

    This function is safe to call repeatedly; it will reuse existing
    xformOps (translate, rotateXYZ, scale) if they already exist instead of
    authoring duplicate ops, preventing Tf.ErrorException about existing ops.

    Authoring in the root layer avoids losing transforms when called from within
    rep.new_layer(), whose edit target is a transient layer not written by stage.Export().

    Args:
        prim: The prim to transform
        translation: Optional translation as (x, y, z) tuple
        rotation: Optional rotation as Euler angles (x, y, z) in degrees
        scale: Optional scale as (x, y, z) tuple
    """
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
    """Gets the transform of a prim.
    
    Args:
        prim: The prim to query
        
    Returns:
        Dictionary with 'translation', 'rotation', and 'scale'
    """
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
    """Creates and configures a camera in Isaac Sim from a K matrix.

    Args:
        name: Name of the camera
        prim_path: Path where to create the camera
        width: Image width in pixels
        height: Image height in pixels
        intrinsics: 3x3 OpenCV intrinsics matrix (fx, fy, cx, cy)
        focal_length_mm: Focal length in mm (arbitrary — only the ratio to
            sensor_width matters for the USD physical camera model)

    Returns:
        The camera prim
    """
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
    """Sets up a render product for a camera using Replicator.
    
    Args:
        camera_path: Path to the camera prim
        resolution: Tuple of (width, height)
        output_name: Base name for output files
        
    Returns:
        Render product
    """
    import omni.replicator.core as rep
    return rep.create.render_product(camera_path, resolution)


def find_prims(root, pattern="*", action=None, action_mode=";"):
    """Find prims by name pattern under root, like `find -name`.

    Without action: returns matched prim paths.
    With action: runs it on matches and returns results (swallows path output).
        action_mode=";" → action(path) per match, returns list of results
        action_mode="+" → action(*paths) once, returns its result
    """
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


def bottom_face_center(prim):
    bbox_range = local_bbox_range(prim)
    center = bbox_range.GetMidpoint()
    return (center[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2])

"""Utility functions for Isaac Sim Replicator data generation."""

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import torch
from torchvision import tv_tensors


def cid_iid_masks(seg_hw, labels, class_to_cid):
    """Class-id and instance-id masks from an instance_segmentation_fast payload.

    seg_hw: (H, W) raw iid array. labels: idToSemantics {iid → {"class", "instance"}}.
    class_to_cid: {class name → cid}. Returns (iid_mask, cid_mask, frame_iid_to_name) —
    two (H, W) tv_tensors.Mask + the graspable {iid → instance name} this frame
    (background/scenery carry no "instance" key).
    """
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


def untransformed_bbox_range(prim):
    """Aligned bbox range EXCLUDING the prim's own xformOps (ComputeUntransformedBound)
    — the usdz-frame bbox of a subtree exported with neutralize_root_xform=True.
    Same double-count rationale documented on scene.add_grasp_frame."""
    from pxr import Usd, UsdGeom
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()


def bottom_face_center(prim):
    bbox_range = local_bbox_range(prim)
    center = bbox_range.GetMidpoint()
    return (center[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2])


def _localize_remote_assets(layer_path):
    """Download a flattened layer's http(s) asset dependencies next to it and
    rewrite the layer's asset paths to the local copies.

    ``UsdUtils.CreateNewUsdzPackage`` can only map LOCAL files into the package;
    a stage composed from a remote subLayer (store001's synthesis-multiverse
    https layer) leaves texture paths as URLs in the flattened export. Downloads
    go through ``omni.client`` (the same resolver that composed the stage), so
    this only works inside a booted kit — which is where export_subtree_usdz
    runs anyway. No-op when the layer has no remote asset paths.
    """
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
    """Export one prim subtree of a live Isaac Sim stage to a standalone .usdz.

    The exported package inlines all geometry, materials, and textures of every
    descendant of ``subtree_path`` and is fully self-contained (no external
    references). Safe to call from inside a running Isaac Sim process.

    Strategy (the "reference-and-flatten" idiom):
      1. Build a brand-new in-memory stage with ``Usd.Stage.CreateInMemory()``.
         We never call ``Usd.Stage.Open(file)`` — inside Isaac Sim that call is
         intercepted and returns the active sim stage, which would silently make
         us operate on (and corrupt) the live scene.
      2. On that solo stage, define a single root prim and add a *reference* to
         the live stage's root layer, targeting ``subtree_path``. A reference
         composes the ENTIRE named subtree as one arc, so all children (and their
         own nested references to per-object usdz files) come along — this avoids
         the "flatten drops all but the first sibling" failure.
      3. Set the solo stage's defaultPrim — a standalone layer without one
         exports as an invalid/near-empty package.
      4. ``solo.Export(temp.usdc)`` flattens the single reference arc into a
         concrete layer on disk, resolving every nested reference and material.
      5. ``UsdUtils.CreateNewUsdzPackage`` walks that layer's asset dependencies
         (textures/HDRs) and zips them into the .usdz with rewritten in-package
         relative paths. (The ``@N/foo.ext@`` resolve warnings it prints are a
         red herring — the files ARE bundled, often under a numbered subfolder.)

    Args:
        stage: The live USD stage to export from.
        subtree_path: Prim path of the subtree to isolate, e.g. "/World".
        output_dir: Directory to write the .usdz into.
        base_name: Stem for the output file.
        root_prim: Exported root prim path (e.g. "/World" so the package satisfies
            the ``load_asset(..., ref_prim_path="/World")`` contract every catalog
            loader assumes). None → "/<subtree prim name>" (legacy behavior).
        neutralize_root_xform: Author an EMPTY xformOpOrder on the export root (a
            local opinion, stronger than the reference arc) so the flattened
            package holds the subtree in the source prim's OWN local frame —
            required for a catalog ``ref_pose`` (usdz-local) and a capture-time
            ``get_target2world(P)`` to compose without double-counting P's
            placement/scale.

    Returns:
        str: Absolute path to the created .usdz.
    """
    import os
    import tempfile
    from pxr import Usd, UsdUtils

    src_prim = stage.GetPrimAtPath(subtree_path)
    if not src_prim or not src_prim.IsValid():
        raise ValueError(f"No valid prim at {subtree_path!r} on the given stage")

    # 1. Fresh in-memory stage — never Usd.Stage.Open(file) inside Isaac Sim.
    solo = Usd.Stage.CreateInMemory()

    # 2. One root prim that *references* the subtree of the live root layer.
    export_prim = solo.DefinePrim(root_prim or f"/{src_prim.GetName()}")
    export_prim.GetReferences().AddReference(
        assetPath=stage.GetRootLayer().identifier,
        primPath=subtree_path,
    )
    if neutralize_root_xform:
        from pxr import UsdGeom, Vt
        UsdGeom.Xformable(export_prim).CreateXformOpOrderAttr().Set(Vt.TokenArray())

    # 3. A standalone layer MUST name a defaultPrim or it exports invalid/tiny.
    solo.SetDefaultPrim(export_prim)

    os.makedirs(output_dir, exist_ok=True)
    usdz_path = os.path.join(output_dir, f"{base_name}.usdz")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 4. Export flattens the reference arc into a concrete on-disk layer.
        temp_usd = os.path.join(temp_dir, f"{base_name}.usdc")
        if not solo.Export(temp_usd):
            raise RuntimeError(f"solo.Export({temp_usd}) returned False")
        # 4b. Localize http(s) asset dependencies: scenes composed from a remote
        # subLayer (store001) reference textures by URL, which the zip writer
        # cannot map ("Failed to map 'https://...': No such file or directory").
        _localize_remote_assets(temp_usd)
        # 5. Package the layer + discovered asset dependencies into .usdz.
        if not UsdUtils.CreateNewUsdzPackage(temp_usd, usdz_path):
            raise RuntimeError("UsdUtils.CreateNewUsdzPackage returned False")

    if not os.path.exists(usdz_path) or os.path.getsize(usdz_path) == 0:
        raise RuntimeError(f"USDZ packaging produced no/empty file at {usdz_path}")

    return usdz_path


def export_flattened_usdz(stage, output_dir, base_name="scene"):
    """Export the stage's defaultPrim subtree (for these scenes, "/World")."""
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError("Stage has no valid defaultPrim to export")
    return export_subtree_usdz(
        stage, default_prim.GetPath().pathString, output_dir, base_name=base_name
    )

def class_label(prim_path):
    from isaacsim.core.utils.semantics import get_labels
    stage = get_current_stage()
    geo = stage.GetPrimAtPath(f"{prim_path}/geo")  # add_object labels geo as "class"
    labels = get_labels(geo)
    if not labels.get("class"):
        raise ValueError(f"ShelfPlacer: no 'class' label on {prim_path}/geo")
    return labels["class"][0]

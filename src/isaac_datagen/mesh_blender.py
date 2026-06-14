"""Blender worker for mesh_convert: import a mesh, export a self-contained .usdz, and
render the 4 side-face ortho reference tiles. Run via Blender 4.2, NOT a normal python:

    blender --background --python mesh_blender.py -- <mesh> <out_usdz> <tiles_dir> <cameras.json>

Generalizes the original build_object_dataset.py render_front_face (-Y face only) to all
4 side faces, orienting each ortho camera with cv2opengl(look_at(...)) so the tile is
upright (image-up = world +Z) and not mirrored.

The per-face ortho camera ROTATIONS are computed in the DRIVER (vision_core's
cv2opengl(look_at(...)), which is bbox-independent) and passed in via cameras.json, so this
worker does NO SE3 convention math and nothing is duplicated -- it only composes matrix_world
from a given R plus a bbox-derived translation. (We can't import vision_core here: Blender ships
its own Python, and vision_core.pose_utils does `import torch`/`scipy`/`matplotlib` at module
load -- none exist in Blender's interpreter -- so even fixing sys.path raises ImportError.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import bpy, mathutils

# format -> importer. Axis kwargs preserve each file's native up where the importer exposes
# it; YCB google_16k obj is Z-up. VERIFY per-format in Blender 4.2.
IMPORTERS = {
    ".obj":  (lambda p: bpy.ops.wm.obj_import(filepath=p, up_axis='Z', forward_axis='Y')),
    ".ply":  (lambda p: bpy.ops.wm.ply_import(filepath=p)),
    ".stl":  (lambda p: bpy.ops.wm.stl_import(filepath=p)),     # untextured -> gray tile
    ".glb":  (lambda p: bpy.ops.import_scene.gltf(filepath=p)),
    ".gltf": (lambda p: bpy.ops.import_scene.gltf(filepath=p)),
    ".fbx":  (lambda p: bpy.ops.import_scene.fbx(filepath=p)),
    ".dae":  (lambda p: bpy.ops.wm.collada_import(filepath=p)),
    ".usd":  (lambda p: bpy.ops.wm.usd_import(filepath=p)),
    ".usdc": (lambda p: bpy.ops.wm.usd_import(filepath=p)),
    ".usda": (lambda p: bpy.ops.wm.usd_import(filepath=p)),
    ".usdz": (lambda p: bpy.ops.wm.usd_import(filepath=p)),
}


def argv_after_dashes():
    return sys.argv[sys.argv.index('--') + 1:]


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: Path):
    ext = path.suffix.lower()
    if ext not in IMPORTERS:
        raise SystemExit(f"unsupported mesh format: {ext}")
    IMPORTERS[ext](str(path))


def export_usdz(out_usdz: Path):
    out_usdz.parent.mkdir(parents=True, exist_ok=True)
    # /World root -> loads via scene.py add_object's load_asset(..., ref_prim_path="/World").
    # convert_orientation=False keeps Blender's native Z-up (Isaac convention). .usdz filepath
    # auto-packages; export_textures bundles the texture maps. VERIFY param names in 4.2.
    bpy.ops.wm.usd_export(filepath=str(out_usdz), root_prim_path="/World",
                          convert_orientation=False, export_textures=True,
                          overwrite_textures=True, use_instancing=False)


def world_bbox():
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    corners = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    lo = np.array([min(c[i] for c in corners) for i in range(3)])
    hi = np.array([max(c[i] for c in corners) for i in range(3)])
    return lo, hi


def add_lighting(scene, world_strength=1.0):
    """Uniform world fill so curved/side surfaces aren't black, plus 'Standard' view transform
    (not Blender's default AgX, which darkens/desaturates) for true bright texture colors. The
    KEY light is a per-face sun added in render_faces (front-lights whichever face the camera
    views), so every tile is evenly bright regardless of camera direction."""
    if scene.world is None:
        scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = world_strength
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0


def render_faces(scene, lo, hi, faces, tiles_dir: Path, base=512, margin=1.3):
    """`faces`: [{name, normal:[3], R:[[3]x3]}] from the driver (cameras.json). Per face the
    ortho camera ROTATION is the given R (= cv2opengl(look_at)); its translation + ortho_scale
    come from the bbox. No SE3 convention math here -- only matrix assembly + bbox arithmetic."""
    c = (lo + hi) / 2.0
    standoff = float(np.linalg.norm(hi - lo)) + 1.0
    up = np.array([0.0, 0.0, 1.0])
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True                    # clean object-only reference
    tiles_dir.mkdir(parents=True, exist_ok=True)
    # One key sun, re-aimed per face: its local -Z runs along -normal (camera->object), so it
    # front-lights whichever face the camera views -> every tile is evenly, brightly lit.
    sun = bpy.data.lights.new("Key", 'SUN'); sun.energy = 3.0
    sun_obj = bpy.data.objects.new("Key", sun); scene.collection.objects.link(sun_obj)
    for f in faces:
        n = np.array(f["normal"], float)
        horiz_axis = np.abs(np.cross(up, n))                # in-plane horizontal unit (+/-X or +/-Y)
        horiz = float(np.dot(hi - lo, horiz_axis))          # bbox extent across that axis
        vert = float(hi[2] - lo[2])
        scene.render.resolution_x = base
        scene.render.resolution_y = max(1, round(base * vert / horiz))
        cam_data = bpy.data.cameras.new(f["name"]); cam_data.type = 'ORTHO'
        cam_data.sensor_fit = 'HORIZONTAL'                  # ortho_scale maps to WIDTH (not the larger axis)
        cam_data.ortho_scale = horiz * margin; cam_data.clip_end = 10000.0
        cam = bpy.data.objects.new(f["name"], cam_data); scene.collection.objects.link(cam)
        M = np.eye(4); M[:3, :3] = np.array(f["R"]); M[:3, 3] = c + n * standoff
        cam.matrix_world = mathutils.Matrix(M.tolist())     # R from driver; Blender cam == OpenGL
        sun_obj.matrix_world = mathutils.Matrix(M.tolist())  # sun shares cam orientation -> front-lights face
        scene.camera = cam
        scene.render.filepath = str(tiles_dir / f"face_{f['name']}.png")
        bpy.ops.render.render(write_still=True)


def main():
    mesh, out_usdz, tiles_dir, cameras = (Path(a) for a in argv_after_dashes()[:4])
    faces = json.loads(cameras.read_text())["faces"]        # driver-computed per-face rotations
    clear(); import_mesh(mesh); export_usdz(out_usdz)        # convert
    clear(); bpy.ops.wm.usd_import(filepath=str(out_usdz))   # render what we actually stored
    scene = bpy.context.scene
    lo, hi = world_bbox()
    add_lighting(scene)
    render_faces(scene, lo, hi, faces, tiles_dir)
    (tiles_dir / "meta.json").write_text(json.dumps(
        {"bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
         "faces": [f["name"] for f in faces]}, indent=2))


if __name__ == "__main__":
    main()

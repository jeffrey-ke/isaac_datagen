from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import bpy, mathutils

IMPORTERS = {
    ".obj":  (lambda p: bpy.ops.wm.obj_import(filepath=p, up_axis='Z', forward_axis='Y')),
    ".ply":  (lambda p: bpy.ops.wm.ply_import(filepath=p)),
    ".stl":  (lambda p: bpy.ops.wm.stl_import(filepath=p)),
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
    if scene.world is None:
        scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = world_strength
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0


def render_faces(scene, lo, hi, faces, tiles_dir: Path, base=512, margin=1.3):
    c = (lo + hi) / 2.0
    standoff = float(np.linalg.norm(hi - lo)) + 1.0
    up = np.array([0.0, 0.0, 1.0])
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    tiles_dir.mkdir(parents=True, exist_ok=True)
    sun = bpy.data.lights.new("Key", 'SUN'); sun.energy = 3.0
    sun_obj = bpy.data.objects.new("Key", sun); scene.collection.objects.link(sun_obj)
    for f in faces:
        n = np.array(f["normal"], float)
        horiz_axis = np.abs(np.cross(up, n))
        horiz = float(np.dot(hi - lo, horiz_axis))
        vert = float(hi[2] - lo[2])
        scene.render.resolution_x = base
        scene.render.resolution_y = max(1, round(base * vert / horiz))
        cam_data = bpy.data.cameras.new(f["name"]); cam_data.type = 'ORTHO'
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.ortho_scale = horiz * margin; cam_data.clip_end = 10000.0
        cam = bpy.data.objects.new(f["name"], cam_data); scene.collection.objects.link(cam)
        M = np.eye(4); M[:3, :3] = np.array(f["R"]); M[:3, 3] = c + n * standoff
        cam.matrix_world = mathutils.Matrix(M.tolist())
        sun_obj.matrix_world = mathutils.Matrix(M.tolist())
        scene.camera = cam
        scene.render.filepath = str(tiles_dir / f"face_{f['name']}.png")
        bpy.ops.render.render(write_still=True)


def main():
    mesh, out_usdz, tiles_dir, cameras = (Path(a) for a in argv_after_dashes()[:4])
    faces = json.loads(cameras.read_text())["faces"]
    clear(); import_mesh(mesh); export_usdz(out_usdz)
    clear(); bpy.ops.wm.usd_import(filepath=str(out_usdz))
    scene = bpy.context.scene
    lo, hi = world_bbox()
    add_lighting(scene)
    render_faces(scene, lo, hi, faces, tiles_dir)
    (tiles_dir / "meta.json").write_text(json.dumps(
        {"bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
         "faces": [f["name"] for f in faces]}, indent=2))


if __name__ == "__main__":
    main()

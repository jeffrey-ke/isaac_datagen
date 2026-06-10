"""Headless Blender renderer for the dry-run debug bundle (NO Isaac Sim deps).

Consumes the bundle written by debug_export.export_debug_bundle:
    <debug_dir>/scene.usdz     scene geometry + baked debug cameras + grasp-frame axes
    <debug_dir>/dryrun.npz     planned poses + the dataset intrinsics (K, width, height)

and writes two render modes (the export is unlit, so we add a sun + ambient world):

    <debug_dir>/poses/pose_NNNN.png   one frame per planned pose, rendered from each
                                      baked left-camera (the RGB the dataset uses), with
                                      the dataset's intrinsics/resolution — a sanity check
                                      that pose-planning aims the camera where we expect.
    <debug_dir>/orbit/orbit_NNN.png   a turntable orbiting the scene centroid, to inspect
                                      the geometry without opening Isaac.

Run it through Blender (4.2 LTS), NOT a normal python interpreter:

    blender --background --python blender_render.py -- <debug_dir> [--orbit-frames N]

Args after the `--` separator are read via sys.argv[sys.argv.index('--')+1:], the
same idiom the original build_object_dataset.py Blender script used.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


def _argv_after_dashes():
    return sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def parse_args():
    p = argparse.ArgumentParser(prog="blender_render.py")
    p.add_argument("debug_dir", type=Path, help="dir holding scene.usdz + dryrun.npz")
    p.add_argument("--orbit-frames", type=int, default=36, help="number of turntable views")
    p.add_argument("--orbit-fps", type=int, default=12, help="orbit.gif playback fps")
    p.add_argument("--sun-energy", type=float, default=1.0)
    p.add_argument("--ambient", type=float, default=0.25, help="world background strength (1.0 washes out)")
    # AgX has a long highlight latitude, so the near-white boxes need a big negative
    # exposure before they stop glaring; -5 reads cleanly, go lower for richer color.
    p.add_argument("--exposure", type=float, default=-5.0, help="view-transform exposure stops (lower = darker)")
    return p.parse_args(_argv_after_dashes())


def reset_and_import(usdz_path: Path):
    """Blank scene + USD import (geometry AND the baked USD cameras)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(usdz_path))
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    return scene


def add_lighting(scene, sun_energy: float, ambient: float, exposure: float):
    """The USDZ exports unlit — add a sun + ambient world so anything shows up.

    The near-white box cardboard blows out easily, so keep the sun/ambient modest
    and pull the view-transform exposure down (a clean global stop control)."""
    sun = bpy.data.lights.new("DebugSun", 'SUN')
    sun.energy = sun_energy
    sun_obj = bpy.data.objects.new("DebugSun", sun)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), 0.0, math.radians(30))
    if scene.world is None:
        scene.world = bpy.data.worlds.new("DebugWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = ambient
    scene.view_settings.exposure = exposure


def color_grasp_axes(emission: float = 2.5):
    """Recolor the baked grasp-frame axis cylinders X=red / Y=green / Z=blue.

    UsdShade bindings don't survive the usdz flatten, so we assign Blender
    materials here by the cylinders' prim names (x/y/z, possibly with .NNN import
    suffixes). Emission makes the thin axes readable against the bright boxes."""
    rgb = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    mats = {}
    for k, c in rgb.items():
        m = bpy.data.materials.new(f"axis_{k}")
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*c, 1.0)
        bsdf.inputs["Emission Color"].default_value = (*c, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
        mats[k] = m
    n = 0
    for o in bpy.data.objects:
        if o.type == 'MESH':
            key = o.name.split('.')[0]          # 'x', 'x.001', ...
            if key in mats:
                o.data.materials.clear()
                o.data.materials.append(mats[key])
                n += 1
    print(f"colored {n} grasp-axis cylinders")


def scene_bounds():
    """World-space (centroid, bbox-diagonal-length) over imported MESH objects only
    (excludes our camera/gizmo empties)."""
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    corners = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    if not corners:
        return mathutils.Vector((0.0, 0.0, 0.0)), 1.0
    lo = mathutils.Vector(map(min, zip(*corners)))
    hi = mathutils.Vector(map(max, zip(*corners)))
    return (lo + hi) / 2, (hi - lo).length


def set_intrinsics(cam_data, K, width, height):
    """Match a Blender camera to a 3x3 OpenCV K (fx,fy,cx,cy) at width x height.

    sensor_fit=HORIZONTAL ties focal length to fx via the sensor width; aperture
    offset (cx,cy) maps to lens shift, normalized by the larger sensor dimension
    (= width here, since the fit is horizontal)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.lens = fx * cam_data.sensor_width / width
    cam_data.shift_x = (width / 2 - cx) / width
    cam_data.shift_y = (cy - height / 2) / width


def render_pose_sanity(scene, data, out_dir: Path):
    """One render per baked debug camera (named cam_NNNN), with the dataset K/res."""
    W, H = int(data['width']), int(data['height'])
    scene.render.resolution_x, scene.render.resolution_y = W, H
    cams = sorted((o for o in bpy.data.objects
                   if o.type == 'CAMERA' and o.name.startswith("cam_")),
                  key=lambda o: o.name)
    if not cams:
        print("WARNING: no baked 'cam_*' cameras found in the USDZ; skipping pose-sanity.")
        return
    for i, cam in enumerate(cams):
        set_intrinsics(cam.data, data['K'], W, H)
        scene.camera = cam
        scene.render.filepath = str(out_dir / f"pose_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"wrote {len(cams)} pose-sanity renders to {out_dir}")


def _frames_to_gif(frames_dir: Path, out_gif: Path, fps: int) -> bool:
    """Assemble orbit_*.png into a single GIF. ffmpeg's gif encoder is built-in
    (no GPL codec needed); two-stage palette in one filter graph for clean colors.
    Falls back to imageio. Returns True on success."""
    pat = str(frames_dir / "orbit_*.png")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = (f"fps={fps},scale=720:-1:flags=lanczos,split[s0][s1];"
              f"[s0]palettegen[p];[s1][p]paletteuse")
        r = subprocess.run([ffmpeg, "-y", "-pattern_type", "glob", "-i", pat, "-vf", vf, str(out_gif)])
        if r.returncode == 0:
            return True
        print("ffmpeg gif failed; trying imageio", flush=True)
    try:
        import imageio.v2 as imageio
        frames = sorted(frames_dir.glob("orbit_*.png"))
        imageio.mimsave(out_gif, [imageio.imread(f) for f in frames], duration=1.0 / fps, loop=0)
        return True
    except Exception as e:
        print(f"could not assemble gif ({e})", flush=True)
        return False


def render_orbit(scene, centroid, radius, debug_dir: Path, n_frames: int, fps: int):
    """Turntable -> a single orbit.gif. Frames render to a temp folder, get muxed
    into the GIF, then the folder is removed (the GIF is the deliverable)."""
    frames_dir = debug_dir / "orbit_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    cam_data = bpy.data.cameras.new("Orbit")
    cam_data.lens = 35.0
    cam = bpy.data.objects.new("Orbit", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    dist = max(radius, 1e-3) * 1.3
    for i in range(n_frames):
        th = 2 * math.pi * i / n_frames
        cam.location = centroid + mathutils.Vector(
            (dist * math.cos(th), dist * math.sin(th), dist * 0.45))
        direction = centroid - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()  # look_at centroid
        scene.render.filepath = str(frames_dir / f"orbit_{i:03d}.png")
        bpy.ops.render.render(write_still=True)

    out_gif = debug_dir / "orbit.gif"
    if _frames_to_gif(frames_dir, out_gif, fps):
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"wrote {out_gif}", flush=True)
    else:
        print(f"kept {n_frames} orbit frames in {frames_dir} (gif encode unavailable)", flush=True)


def main():
    args = parse_args()
    debug_dir = args.debug_dir
    data = np.load(debug_dir / "dryrun.npz")

    scene = reset_and_import(debug_dir / "scene.usdz")
    add_lighting(scene, args.sun_energy, args.ambient, args.exposure)
    color_grasp_axes()
    centroid, diag = scene_bounds()

    (debug_dir / "poses").mkdir(parents=True, exist_ok=True)
    render_pose_sanity(scene, data, debug_dir / "poses")
    render_orbit(scene, centroid, diag, debug_dir, args.orbit_frames, args.orbit_fps)


if __name__ == "__main__":
    main()


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
    p.add_argument("--exposure", type=float, default=-5.0, help="view-transform exposure stops (lower = darker)")
    return p.parse_args(_argv_after_dashes())


def reset_and_import(usdz_path: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(usdz_path))
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    return scene


def add_lighting(scene, sun_energy: float, ambient: float, exposure: float):
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
            key = o.name.split('.')[0]
            if key in mats:
                o.data.materials.clear()
                o.data.materials.append(mats[key])
                n += 1
    print(f"colored {n} grasp-axis cylinders")


def scene_bounds():
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    corners = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    if not corners:
        return mathutils.Vector((0.0, 0.0, 0.0)), 1.0
    lo = mathutils.Vector(map(min, zip(*corners)))
    hi = mathutils.Vector(map(max, zip(*corners)))
    return (lo + hi) / 2, (hi - lo).length


def set_intrinsics(cam_data, K, width, height):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.lens = fx * cam_data.sensor_width / width
    cam_data.shift_x = (width / 2 - cx) / width
    cam_data.shift_y = (cy - height / 2) / width


def render_pose_sanity(scene, data, out_dir: Path):
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
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
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

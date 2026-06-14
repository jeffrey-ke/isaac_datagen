## Keywords / Tags
- datagen
- plan-completed
- mesh-convert
- ycb
- blender
- usdz
- graspable-object
- reference-image
- grasp-frame

# Plan: `mesh_convert` — arbitrary meshes → GraspableObject dataset (Blender-only)

**Status:** Complete (2026-06-13) — see **"Implementation outcome & discoveries"** at the end for the
authoritative final state (verification + the four real-data fixes that diverged from the plan body).
**Entry points:** `uv run src/isaac_datagen/mesh_convert.py {ycb,stage,finalize}` (run from repo root)
**New modules:** `mesh_convert.py`, `mesh_blender.py`. **Touched:** `CLAUDE.md`.

## Context

We need to turn a folder of meshes (`.obj/.ply/.stl/.glb/.gltf/.fbx/.dae`, e.g. the YCB
`google_16k` set) into serialized `GraspableObject` dataset entries that drop into the existing
reference-seg pipeline (`collect_objects` → `build_scene` → `ObsMaskWriter`). This implements the
pseudocode in `src/isaac_datagen/mesh_convert` (the spec file). Two public functions: `convert(...)`
and `ycb_download(...)`.

A `GraspableObject` (objects.py:27) has four fields: `usd_path` (.usdz, copied on serialize), `meta`
(`{"name","class"}`), `reference_image` (PIL), `grasp_point` (SE3 4×4). The pseudocode only covers
`usd_path` + `meta`; this plan fills `reference_image` and `grasp_point` by **generalizing the
original `build_object_dataset.py`** (preserved at
`visual_servoing/datagen2_isaacsim/.build_object_dataset.py.bak`), which already headless-rendered
each object's reference image in Blender (ortho cam at the −Y face) and set a face-center grasp SE3.

The whole tool is **Blender-only — no Isaac Sim**. A headless Blender subprocess per mesh imports it,
exports the `.usdz` (`/World` root, Z-up, textures), and renders the 4 side-face ortho tiles; a
plain project-env driver (where `vision_core` imports cleanly) computes the SE3s, *stages* the
labeled renders to disk, and a later `finalize` step serializes once you've judged the winning face.
All SE3/convention math lives in the driver (single source = `vision_core`); the Blender worker is
handed precomputed per-face camera rotations and does none. This mirrors `debug_render.py`'s
driver→Blender hand-off and reuses the proven recipe from `blender_render.py` / `.build_object_dataset.py.bak`.

## Decisions (forks you answered)

| Fork | Decision |
|---|---|
| Reference image source | **Find + reuse** the original Blender renderer (`render_front_face` in `.build_object_dataset.py.bak`); generalize from 1 face (−Y) to 4. Add an explicit pointer in `CLAUDE.md`. |
| `reference_image` populate | **Render in Blender** (ortho, per face). |
| `grasp_point` default | **Guess-and-check**: render 4 candidate side-face frames; **you judge** the winning face. Winner → both `grasp_point` and `reference_image`. |
| Candidate frames | **4 side faces** (±X, ±Y); up = mesh +Z. |
| Reference camera | **Ortho canonical** (port `render_front_face`'s ortho framing). |
| Convert backend | **Blender `wm.usd_export`** (no Isaac). |
| Grasp-frame axes | **+X = outward normal, +Z = world up, +Y = Z×X**; built via `vision_core` (`make_se3`); render cam via `cv2opengl(look_at(...))`. |
| Judging | **Decoupled / offline.** `convert` *stages* uniquely-labeled candidate renders + per-object `candidate.json` (usdz, bbox, all 4 grasp SE3s) to disk — no inline prompt. A later `finalize(stage, output, winners)` applies your winning-face picks and serializes. Robust for batch (full YCB). |

## Architecture (two phases — render, then judge offline)

```
ycb_download(out)         # urllib + tarfile: google_16k .tgz → <out>/ycb/<obj>/google_16k/textured.obj
        │ returns mesh folder
        ▼
PHASE 1  convert(input, stage, names=None, classes=None)        # driver, project env (vision_core + PIL)
  write_camera_spec(stage/cameras.json)                 # constant per-face cv2opengl(look_at) R; once
  for idx, mesh in enumerate(sorted recursive mesh glob):
    ├─ subprocess: blender … mesh_blender.py -- <mesh> <stage>/<label>/model.usdz <stage>/<label> <stage>/cameras.json
    │     Blender: import mesh → export usdz(/World, Z-up, textures)
    │              → re-import usdz → measure bbox → render face_{-Y,+Y,-X,+X}.png (ortho, R from cameras.json)
    │              → dump meta.json {bbox_min, bbox_max}
    ├─ frames = face_grasp_frames(bbox)                 # vision_core make_se3; X=normal, Z=up
    └─ write <stage>/<label>/ : model.usdz, face_<±X,±Y>.png, grid.png, candidate.json
            #   label = f"{idx:04d}_{mesh.stem}"  (unique per object)
            #   candidate.json = {label,name,class,bbox,usdz, grasp_frames:{face:4×4}}
            #   NO GraspableObject serialized yet — winning face unknown.

        … you eyeball each <stage>/<label>/grid.png and record the winning face …

PHASE 2  finalize(stage, output, winners)                       # later script / invocation
  winners = {label: "+X"  or  label: {"face":"+X","class":"can"}}   # e.g. from winners.yaml
  for idx, label in enumerate(sorted(winners)):
    c = candidate.json[label]
    GraspableObject(usd_path=<usdz>, meta={name,class},
                    reference_image=face_<winner>.png, grasp_point=c.grasp_frames[winner])
        .serialize(idx, output)        # writes meta/ usd_path/ reference_image/ grasp_point/
```

Note `GraspableObject.grasp_point` is currently **read nowhere** in `src/` (live grasp frames come
from pallet placement). Its job in this tool is to *define the reference viewpoint* and populate the
dataset contract — not consumed by capture yet.

This is a deliberate divergence from the pseudocode's single-shot `convert` (which serialized inline):
the human-judged guess-and-check is decoupled so renders persist with unique labels and a later step
applies the winners — mirroring `relabel_classes.py`'s "write grid, then relabel offline" pattern.

## New file 1 — `src/isaac_datagen/mesh_blender.py` (runs in Blender's python; no project deps)

```python
"""Blender worker for mesh_convert: import a mesh, export a self-contained .usdz, and
render the 4 side-face ortho reference tiles. Run via Blender 4.2, NOT a normal python:

    blender --background --python mesh_blender.py -- <mesh> <out_usdz> <tiles_dir> <cameras.json>

Generalizes the original build_object_dataset.py render_front_face (−Y face only) to all
4 side faces, orienting each ortho camera with cv2opengl(look_at(...)) so the tile is
upright (image-up = world +Z) and not mirrored.

The per-face ortho camera ROTATIONS are computed in the DRIVER (vision_core's
cv2opengl(look_at(...)), which is bbox-independent) and passed in via cameras.json, so this
worker does NO SE3 convention math and nothing is duplicated — it only composes matrix_world
from a given R plus a bbox-derived translation. (We can't import vision_core here: Blender ships
its own Python, and vision_core.pose_utils does `import torch`/`scipy`/`matplotlib` at module
load — none exist in Blender's interpreter — so even fixing sys.path raises ImportError.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import bpy, mathutils

# ── format → (operator, kwargs). Axis kwargs preserve each file's native up where the
#    importer exposes it; YCB google_16k obj is Z-up. VERIFY per-format in Blender 4.2.
IMPORTERS = {
    ".obj":  (lambda p: bpy.ops.wm.obj_import(filepath=p, up_axis='Z', forward_axis='Y')),
    ".ply":  (lambda p: bpy.ops.wm.ply_import(filepath=p)),
    ".stl":  (lambda p: bpy.ops.wm.stl_import(filepath=p)),     # untextured → gray tile
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
    # /World root → loads via scene.py add_object's load_asset(..., ref_prim_path="/World").
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


def add_lighting(scene, sun_energy=3.0, exposure=-5.0):
    import math
    sun = bpy.data.lights.new("Sun", 'SUN'); sun.energy = sun_energy
    obj = bpy.data.objects.new("Sun", sun); scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(45), 0.0, math.radians(30))
    if scene.world is None:
        scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    scene.view_settings.exposure = exposure                 # tame AgX highlight latitude


def render_faces(scene, lo, hi, faces, tiles_dir: Path, base=512, margin=1.005):
    """`faces`: [{name, normal:[3], R:[[3]×3]}] from the driver (cameras.json). Per face the
    ortho camera ROTATION is the given R (= cv2opengl(look_at)); its translation + ortho_scale
    come from the bbox. No SE3 convention math here — only matrix assembly + bbox arithmetic."""
    c = (lo + hi) / 2.0
    standoff = float(np.linalg.norm(hi - lo)) + 1.0
    up = np.array([0.0, 0.0, 1.0])
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True                    # clean object-only reference
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for f in faces:
        n = np.array(f["normal"], float)
        horiz_axis = np.abs(np.cross(up, n))                # in-plane horizontal unit (±X or ±Y)
        horiz = float(np.dot(hi - lo, horiz_axis))          # bbox extent across that axis
        vert = float(hi[2] - lo[2])
        scene.render.resolution_x = base
        scene.render.resolution_y = max(1, round(base * vert / horiz))
        cam_data = bpy.data.cameras.new(f["name"]); cam_data.type = 'ORTHO'
        cam_data.ortho_scale = horiz * margin; cam_data.clip_end = 10000.0
        cam = bpy.data.objects.new(f["name"], cam_data); scene.collection.objects.link(cam)
        M = np.eye(4); M[:3, :3] = np.array(f["R"]); M[:3, 3] = c + n * standoff
        cam.matrix_world = mathutils.Matrix(M.tolist())     # R from driver; Blender cam == OpenGL
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
```

## New file 2 — `src/isaac_datagen/mesh_convert.py` (driver; project env)

```python
"""Build a GraspableObject dataset from arbitrary meshes (+ YCB download), in two phases.

    uv run src/isaac_datagen/mesh_convert.py ycb       <download_dir>            # fetch YCB
    uv run src/isaac_datagen/mesh_convert.py stage      <input_dir> <stage_dir>  # phase 1: render candidates
    uv run src/isaac_datagen/mesh_convert.py finalize   <stage_dir> <output_dir> <winners.yaml>   # phase 2

Phase 1 (`convert`/stage): a headless Blender subprocess (mesh_blender.py) exports each .usdz and
renders the 4 side-face ortho tiles; everything is written to <stage>/<label>/ with unique labels,
plus a candidate.json holding the 4 grasp SE3s. NO GraspableObject is serialized — you pick the
winning face offline. Phase 2 (`finalize`): apply the winners → serialize the dataset.
Generalizes visual_servoing/datagen2_isaacsim/.build_object_dataset.py.bak.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, tarfile, urllib.request
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from vision_core.pose_utils import make_se3, look_at, cv2opengl
from isaac_datagen.objects import GraspableObject, UsdPath

BLENDER = shutil.which("blender") or "/usr/local/bin/blender"
MESH_BLENDER = Path(__file__).with_name("mesh_blender.py")
MESH_EXTS = (".obj", ".ply", ".stl", ".glb", ".gltf", ".fbx", ".dae",
             ".usd", ".usdc", ".usda", ".usdz")
# Outward unit normals of the 4 side faces (mesh/world frame, Z-up).
FACE_NORMALS = {"-Y": [0., -1, 0], "+Y": [0., 1, 0], "-X": [-1., 0, 0], "+X": [1., 0, 0]}


def find_meshes(input_path: Path) -> list[Path]:
    """Recursive (YCB nests <obj>/google_16k/textured.obj); skip our own usdz outputs."""
    return sorted(p for p in input_path.rglob("*")
                  if p.suffix.lower() in MESH_EXTS and p.suffix.lower() != ".usdz")


def write_camera_spec(out_json: Path):
    """Per-face ortho camera ROTATIONS — constant (bbox-independent), so computed once here in
    the project env via vision_core's cv2opengl(look_at(...)) and handed to the Blender worker.
    look_at(at=0, from=n): camera sits on the +normal side looking back at the object; cv2opengl
    maps the +Z-forward (OpenCV) frame to Blender/OpenGL (−Z-forward), so tiles aren't mirrored."""
    faces = [{"name": k, "normal": n,
              "R": cv2opengl(look_at(np.zeros(3), np.array(n, float)))[:3, :3].tolist()}
             for k, n in FACE_NORMALS.items()]
    out_json.write_text(json.dumps({"faces": faces}, indent=2))
    return out_json


def run_blender(mesh: Path, work: Path, cameras: Path) -> tuple[Path, dict, list[Path]]:
    """Blender: mesh → work/model.usdz + work/face_*.png + work/meta.json (bbox)."""
    work.mkdir(parents=True, exist_ok=True)
    usdz = work / "model.usdz"
    r = subprocess.run([BLENDER, "--background", "--python", str(MESH_BLENDER), "--",
                        str(mesh), str(usdz), str(work), str(cameras)], capture_output=True, text=True)
    if r.returncode != 0 or not usdz.exists():
        raise RuntimeError(f"Blender failed for {mesh.name}:\n{r.stderr[-2000:]}")
    meta = json.loads((work / "meta.json").read_text())
    return usdz, meta, [work / f"face_{n}.png" for n in meta["faces"]]


def face_grasp_frames(bbox_min, bbox_max) -> dict[str, np.ndarray]:
    """4 side-face grasp frames in mesh/world coords. Convention (user):
    +X = outward face normal, +Z = world up, +Y = Z×X. Origin = measured bbox face center
    (robust to where the mesh's local origin sits — no origin==centroid assumption)."""
    lo, hi = np.asarray(bbox_min, float), np.asarray(bbox_max, float)
    c = (lo + hi) / 2.0
    up = np.array([0.0, 0.0, 1.0])
    faces = {
        "-Y": (np.array([0., -1, 0]), np.array([c[0], lo[1], c[2]])),
        "+Y": (np.array([0., 1, 0]), np.array([c[0], hi[1], c[2]])),
        "-X": (np.array([-1., 0, 0]), np.array([lo[0], c[1], c[2]])),
        "+X": (np.array([1., 0, 0]), np.array([hi[0], c[1], c[2]])),
    }
    out = {}
    for name, (n, origin) in faces.items():
        x = n / np.linalg.norm(n)
        z = up
        y = np.cross(z, x)                         # |y| = 1 since z ⟂ x for side faces
        out[name] = make_se3(origin, np.column_stack([x, y, z]))
    return out


def write_grid(tiles: list[Path], faces: list[str], label: str, out_png: Path):
    """Per-object 1×4 contact sheet (faces labeled) for offline review (relabel_classes UX)."""
    fig, axes = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 3.6))
    for ax, t, f in zip(np.atleast_1d(axes), tiles, faces):
        ax.imshow(PILImage.open(t)); ax.set_title(f); ax.axis("off")
    fig.suptitle(label); fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def convert(input_path, stage_path, names=None, classes=None):
    """PHASE 1 (stage): render uniquely-labeled candidate face tiles + per-object candidate.json
    (usdz, bbox, the 4 grasp SE3s, name/class) to <stage>/<label>/. Does NOT serialize the final
    GraspableObject — pick the winning face offline, then run finalize()."""
    input_path, stage_path = Path(input_path), Path(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)
    cameras = write_camera_spec(stage_path / "cameras.json")   # constant; written once, reused
    meshes = find_meshes(input_path)
    if names is not None or classes is not None:
        assert names is not None and classes is not None, "supply both names and classes"
        assert len(names) == len(classes) == len(meshes), \
            f"need 1-1 name,class↔mesh: {len(names)}/{len(classes)} vs {len(meshes)} meshes"
    for idx, mesh in enumerate(meshes):
        label = f"{idx:04d}_{mesh.stem}"             # unique per object
        work = stage_path / label
        usdz, meta, tiles = run_blender(mesh, work, cameras)
        frames = face_grasp_frames(meta["bbox_min"], meta["bbox_max"])
        write_grid(tiles, meta["faces"], label, work / "grid.png")
        (work / "candidate.json").write_text(json.dumps({
            "label": label, "mesh": str(mesh),
            "name": names[idx] if names else mesh.stem,
            "class": classes[idx] if classes else "",
            "usdz": str(usdz), "faces": meta["faces"],
            "bbox_min": meta["bbox_min"], "bbox_max": meta["bbox_max"],
            "grasp_frames": {f: frames[f].tolist() for f in meta["faces"]},
        }, indent=2))
        print(f"  staged [{idx:04d}] {label}  → review {work/'grid.png'}")
    print(f"\nStaged {len(meshes)} objects in {stage_path}. Record the winning face per object "
          f"(winners.yaml: '<label>: +X'), then: finalize(stage, output, winners.yaml)")


def finalize(stage_path, output_path, winners):
    """PHASE 2 (select): `winners` maps object label → winning face name ("+X"), or →
    {"face": "+X", "class": "can"} to also set/override class. Serializes the chosen reference
    tile + grasp frame as a GraspableObject dataset at output_path. `winners` may be a dict or a
    path to a winners.yaml (so a later script can just call finalize(...))."""
    stage_path, output_path = Path(stage_path), Path(output_path)
    if isinstance(winners, (str, Path)):
        winners = yaml.safe_load(Path(winners).read_text())
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, label in enumerate(sorted(winners)):
        sel = winners[label]
        face = sel if isinstance(sel, str) else sel["face"]
        c = json.loads((stage_path / label / "candidate.json").read_text())
        cls = (sel.get("class") if isinstance(sel, dict) else None) or c["class"]
        GraspableObject(
            usd_path=UsdPath(str(stage_path / label / "model.usdz")),
            meta={"name": c["name"], "class": cls},
            reference_image=PILImage.open(stage_path / label / f"face_{face}.png").convert("RGB"),
            grasp_point=np.array(c["grasp_frames"][face]),
        ).serialize(idx, output_path)
        print(f"  [{idx:04d}] {label} face={face} class={cls}")
    print(f"{len(winners)} objects → {output_path}")


def ycb_download(output_path):
    """Download YCB google_16k textured meshes; return the folder of extracted meshes.
    Each object unpacks to <out>/ycb/<obj>/google_16k/{textured.obj,.mtl,texture_map.png}.
    Objects lacking a google_16k variant 404 and are skipped (try/except)."""
    base = "https://ycb-benchmarks.s3.amazonaws.com/data"
    meshes_dir = Path(output_path) / "ycb"; meshes_dir.mkdir(parents=True, exist_ok=True)
    objects = json.loads(urllib.request.urlopen(f"{base}/objects.json").read())["objects"]
    for obj in objects:
        try:
            tgz, _ = urllib.request.urlretrieve(f"{base}/google/{obj}_google_16k.tgz")
            with tarfile.open(tgz) as t:
                t.extractall(meshes_dir)
            print(f"  ok   {obj}")
        except Exception as e:
            print(f"  skip {obj}: {e}")
    return meshes_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_y = sub.add_parser("ycb");      p_y.add_argument("download_dir")
    p_s = sub.add_parser("stage");    p_s.add_argument("input_dir"); p_s.add_argument("stage_dir")
    p_f = sub.add_parser("finalize"); p_f.add_argument("stage_dir"); p_f.add_argument("output_dir")
    p_f.add_argument("winners")
    a = ap.parse_args()
    if a.cmd == "ycb":
        print(ycb_download(a.download_dir))
    elif a.cmd == "stage":
        convert(a.input_dir, a.stage_dir)
    else:
        finalize(a.stage_dir, a.output_dir, a.winners)
```

## Edit — `CLAUDE.md` (your ask: explicit reference to the original Blender renderer)

Module index — add two rows:

```
| `mesh_convert.py` | Build a GraspableObject dataset from arbitrary meshes (+ YCB download): stage candidate renders, then finalize winners | `convert`, `finalize`, `ycb_download` |
| `mesh_blender.py` | Blender worker: mesh → /World usdz + 4 side-face ortho reference tiles | (run via `blender --background`) |
```

"Where to look next" — add:

```
- The reference-image render recipe generalizes the original `build_object_dataset.py`
  (ortho cam at the −Y face), preserved verbatim at
  `visual_servoing/datagen2_isaacsim/.build_object_dataset.py.bak`. `mesh_blender.py`
  extends it to all 4 side faces and orients each ortho camera with `cv2opengl(look_at)`.
```

## Called library code this reuses (for review)

- `GraspableObject` + serializers — `objects.py:27-51`; `SerializableSample.serialize(idx, dir, only=None)`
  writes `directory/{field}/{field}_{idx:04d}{ext}` (`vision_core/datastructs.py:99`). `UsdPath`
  copies the `.usdz` into the dataset; PIL→`.png`; `meta`→`.yaml`.
- `vision_core.pose_utils.look_at`/`cv2opengl`/`make_se3` — pose_utils.py:62 / :16 / :103.
- Original `render_front_face` + `grasp_se3` — `.build_object_dataset.py.bak:34-124` (the −Y ortho recipe).
- `collect_objects` round-trips the output: `len(sorted((path/"meta").glob("meta_*.yaml")))` then
  `GraspableObject.deserialize(i, path)` (clean_datagen.py:66).

## Verification

1. **Unit (no Blender):** `face_grasp_frames` on a unit cube `[-.5]^3..[.5]^3` → for `-Y`,
   translation `[0,-0.5,0]`, columns `X=[0,-1,0]`, `Z=[0,0,1]`, `Y=Z×X=[1,0,0]`; assert `det(R)=+1`,
   `R@Rᵀ=I`.
2. **One mesh end-to-end:** download one YCB obj (`002_master_chef_can`), run
   `blender --background --python mesh_blender.py -- <obj> /tmp/x.usdz /tmp/tiles` → confirm
   `/tmp/x.usdz` exists, re-imports, and 4 upright textured `face_*.png` tiles + `meta.json` written.
3. **Two-phase round-trip:** `convert(<1–2 mesh folder>, stage)` → confirm `stage/<0000_…>/`
   holds `model.usdz`, 4 `face_*.png`, `grid.png`, `candidate.json`. Hand-write `winners.yaml`
   (`0000_…: +X`), run `finalize(stage, output, "winners.yaml")` → confirm `meta/ usd_path/
   reference_image/ grasp_point/` dirs with `*_0000.*`; then `collect_objects(output)` returns the
   objects and `add_object` loads each `usd_path` via `load_asset(ref_prim_path="/World")` without error.
4. **Pipeline smoke:** point a config's `graspable_objects_path` at the new dataset and run a short
   `reference_segmentation` (or `relabel_classes.py --grid-only`) to confirm the reference grid renders.

## Run-blocking details to validate at implementation (not redesigns)

- Exact Blender 4.2 importer kwargs per format (esp. `obj_import` `up_axis`/`forward_axis` to keep
  YCB Z-up) and `wm.usd_export` param names (`root_prim_path`, `convert_orientation`,
  `export_textures`, usdz packaging).
- That a Blender `/World`-root usdz re-imports and loads under Isaac's `ref_prim_path="/World"`.
- `urlretrieve` over the YCB S3 endpoint (http vs https); swap to the `http://…s3-website…` host if needed.

---

## Implementation outcome & discoveries (2026-06-13)

Built and verified end-to-end on real hardware (RTX 4090) against the **live YCB S3 bucket** and an
**Isaac boot**. Shipped: `src/isaac_datagen/mesh_convert.py` (driver), `src/isaac_datagen/mesh_blender.py`
(Blender worker); `CLAUDE.md` updated. Full YCB staged: **78/78 objects** → `datasets/ycb_stage/`.

### Verified
- **Geometry** (no Blender): `face_grasp_frames` (+X=normal, +Z=up, +Y=Z×X, origin=measured bbox face
  center) and the `cv2opengl(look_at)` camera rotations (R[:,1]=up, R[:,2]=normal, det +1) on a unit cube.
- **Blender worker**: import keeps **Z-up** through obj→usdz→reimport (a 0.2×0.1×0.3 box returns bbox
  [0.2,0.1,0.3]); `wm.usd_export` to a `/World`-root .usdz with textures; ortho face renders.
- **Isaac load**: the Blender `/World` usdz loads via `load_asset(ref_prim_path="/World")` exactly like
  dataset assets (mesh present, scale intact) — output is drop-in for the reference-seg scene.
- **Two-phase round-trip**: `convert`(stage) → `finalize`(winner) → `GraspableObject.deserialize`.
- **`ycb_download`** against live S3: **79 ok / 24 skipped** (objects lacking a google_16k variant 404
  and are skipped by the try/except); 855 MB; layout `<obj>/google_16k/textured.obj`.

### Discoveries / fixes found only by running on real YCB data
1. **YCB bundles 4 mesh reps per folder** (textured.{obj,dae} + nontextured.{ply,stl}) → the naive
   recursive glob returned **312 files for 78 objects**. `find_meshes(one_per_dir=True)` now keeps ONE
   mesh per leaf dir (prefers a "textured" name + `.obj`), logging every skip; `one_per_dir=False`
   for a flat folder of distinct meshes.
2. **Every YCB file is named `textured.obj`** → `mesh.stem` made every label `00NN_textured` and every
   `name="textured"`. Added `object_name(mesh, input_path)`: the id is the TOP-LEVEL folder under input
   (`002_master_chef_can`); flat folders fall back to the stem.
3. **Renders came out black / "only some faces visible"**: a single fixed SUN left camera-opposite
   faces in shadow (black), and the dry-run's `exposure=-5.0` crushed everything. Fix: a KEY sun
   **re-aimed per face** (its −Z along −normal → front-lights the viewed face) + uniform world fill +
   **`Standard`** view transform (not AgX) + exposure 0. Mean brightness 27 → ~90–170; all faces evenly
   lit; no blow-out on white objects.
4. **Framing too tight / clipped**: Blender's default `sensor_fit='AUTO'` maps `ortho_scale` to the
   LARGER sensor axis, so on portrait objects (tall cans/boxes) the width-based framing hit the wrong
   axis and the object overflowed. Fix: **`sensor_fit='HORIZONTAL'`** + `margin 1.005 → 1.3`. Object now
   fills ~50–58% of the frame with clean padding.

### Notes / divergences from the approved plan
- The Blender worker can't import `vision_core` (Blender's python lacks torch/scipy/matplotlib), so ALL
  SE3 math stays in the driver and the bbox-independent per-face camera rotations are passed via
  `cameras.json` — nothing is duplicated. (Memory: `blender-subprocess-cant-import-vision-core`.)
- `GraspableObject.grasp_point` is still read nowhere in `src/` (live grasp frames come from pallet
  placement); in this tool it defines the reference viewpoint + populates the dataset contract.
- Judging is decoupled/offline (stage → review `grid.png` → `finalize` with `winners.yaml`), a divergence
  from the pseudocode's single-shot `convert`. (Memory: `datagen-decoupled-offline-judging`.)
```

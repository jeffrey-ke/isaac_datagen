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
_EXT_PRIORITY = (".obj", ".glb", ".gltf", ".dae", ".fbx", ".ply", ".stl", ".usd", ".usdc", ".usda")
FACE_NORMALS = {"-Y": [0., -1, 0], "+Y": [0., 1, 0], "-X": [-1., 0, 0], "+X": [1., 0, 0]}


def find_meshes(input_path: Path, one_per_dir: bool = True) -> list[Path]:
    found = [p for p in input_path.rglob("*")
             if p.suffix.lower() in MESH_EXTS and p.suffix.lower() != ".usdz"]
    if not one_per_dir:
        return sorted(found)
    by_dir: dict[Path, list[Path]] = {}
    for p in found:
        by_dir.setdefault(p.parent, []).append(p)
    rank = lambda p: ("nontextured" in p.name.lower(), _EXT_PRIORITY.index(p.suffix.lower()), p.name)
    picked = []
    for d, group in by_dir.items():
        best = min(group, key=rank)
        if len(group) > 1:
            print(f"  {d}: using {best.name}, skipped {sorted(q.name for q in group if q != best)}")
        picked.append(best)
    return sorted(picked)


def object_name(mesh: Path, input_path: Path) -> str:
    rel = mesh.relative_to(input_path)
    return rel.parts[0] if len(rel.parts) > 1 else mesh.stem


def write_camera_spec(out_json: Path):
    faces = [{"name": k, "normal": n,
              "R": cv2opengl(look_at(np.zeros(3), np.array(n, float)))[:3, :3].tolist()}
             for k, n in FACE_NORMALS.items()]
    out_json.write_text(json.dumps({"faces": faces}, indent=2))
    return out_json


def run_blender(mesh: Path, work: Path, cameras: Path) -> tuple[Path, dict, list[Path]]:
    work.mkdir(parents=True, exist_ok=True)
    usdz = work / "model.usdz"
    r = subprocess.run([BLENDER, "--background", "--python", str(MESH_BLENDER), "--",
                        str(mesh), str(usdz), str(work), str(cameras)], capture_output=True, text=True)
    if r.returncode != 0 or not usdz.exists():
        raise RuntimeError(f"Blender failed for {mesh.name}:\n{r.stderr[-2000:]}")
    meta = json.loads((work / "meta.json").read_text())
    return usdz, meta, [work / f"face_{n}.png" for n in meta["faces"]]


def face_grasp_frames(bbox_min, bbox_max) -> dict[str, np.ndarray]:
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
        y = np.cross(z, x)
        out[name] = make_se3(origin, np.column_stack([x, y, z]))
    return out


def write_grid(tiles: list[Path], faces: list[str], label: str, out_png: Path):
    fig, axes = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 3.6))
    for ax, t, f in zip(np.atleast_1d(axes), tiles, faces):
        ax.imshow(PILImage.open(t)); ax.set_title(f); ax.axis("off")
    fig.suptitle(label); fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def convert(input_path, stage_path, names=None, classes=None):
    input_path, stage_path = Path(input_path), Path(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)
    cameras = write_camera_spec(stage_path / "cameras.json")
    meshes = find_meshes(input_path)
    if names is not None or classes is not None:
        assert names is not None and classes is not None, "supply both names and classes"
        assert len(names) == len(classes) == len(meshes), \
            f"need 1-1 name,class<->mesh: {len(names)}/{len(classes)} vs {len(meshes)} meshes"
    for idx, mesh in enumerate(meshes):
        name = names[idx] if names else object_name(mesh, input_path)
        label = f"{idx:04d}_{name}"
        work = stage_path / label
        usdz, meta, tiles = run_blender(mesh, work, cameras)
        frames = face_grasp_frames(meta["bbox_min"], meta["bbox_max"])
        write_grid(tiles, meta["faces"], label, work / "grid.png")
        (work / "candidate.json").write_text(json.dumps({
            "label": label, "mesh": str(mesh),
            "name": name,
            "class": classes[idx] if classes else "",
            "usdz": str(usdz), "faces": meta["faces"],
            "bbox_min": meta["bbox_min"], "bbox_max": meta["bbox_max"],
            "grasp_frames": {f: frames[f].tolist() for f in meta["faces"]},
        }, indent=2))
        print(f"  staged [{idx:04d}] {label}  -> review {work/'grid.png'}")
    print(f"\nStaged {len(meshes)} objects in {stage_path}. Record the winning face per object "
          f"(winners.yaml: '<label>: +X'), then: finalize(stage, output, winners.yaml)")


def staged_labels(stage_path: Path) -> list[str]:
    return sorted(d.name for d in Path(stage_path).iterdir()
                  if d.is_dir() and (d / "candidate.json").exists())


def open_grid(grid: Path):
    try:
        subprocess.Popen(["xdg-open", str(grid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"    (couldn't open {grid}: {e})")


def prompt_winners(stage_path, winners_path, show=True) -> dict:
    stage_path, winners_path = Path(stage_path), Path(winners_path)
    winners = yaml.safe_load(winners_path.read_text()) if winners_path.exists() else {}
    labels = staged_labels(stage_path)
    for i, label in enumerate(labels):
        c = json.loads((stage_path / label / "candidate.json").read_text())
        faces = c["faces"]
        prev = winners.get(label)
        prev_face = prev["face"] if isinstance(prev, dict) else prev
        prev_cls = (prev.get("class") if isinstance(prev, dict) else None) or c["class"]
        if show:
            open_grid(stage_path / label / "grid.png")
        print(f"\n[{i + 1}/{len(labels)}] {label}   faces: {' '.join(faces)}"
              f"\n  grid: {stage_path / label / 'grid.png'}")
        while True:
            ans = input(f"  winning face{f' [{prev_face}]' if prev_face else ''} "
                        f"(s=skip, q=save+quit): ").strip()
            if ans == "q":
                save_winners(winners, winners_path)
                print(f"\nsaved {len(winners)} picks -> {winners_path} (stopped before {label})")
                return winners
            if ans == "s":
                face = None; break
            if not ans and prev_face:
                face = prev_face; break
            if ans in faces:
                face = ans; break
            print(f"    enter one of {faces}, or s/q")
        if face is None:
            continue
        cls = input(f"  class{f' [{prev_cls}]' if prev_cls else ''}: ").strip() or prev_cls
        winners[label] = {"face": face, "class": cls}
        save_winners(winners, winners_path)
    print(f"\nsaved {len(winners)} picks -> {winners_path}")
    return winners


def save_winners(winners, winners_path):
    Path(winners_path).write_text(yaml.safe_dump(winners, sort_keys=True))


def finalize(stage_path, output_path, winners=None, show=True):
    stage_path, output_path = Path(stage_path), Path(output_path)
    if winners is None:
        winners = prompt_winners(stage_path, stage_path / "winners.yaml", show=show)
    elif isinstance(winners, (str, Path)):
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
    print(f"{len(winners)} objects -> {output_path}")


def ycb_download(output_path):
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
    p_f.add_argument("winners", nargs="?", default=None,
                     help="winners.yaml; omit to pick interactively")
    p_f.add_argument("--no-open", action="store_true", help="don't auto-open each grid.png")
    a = ap.parse_args()
    if a.cmd == "ycb":
        print(ycb_download(a.download_dir))
    elif a.cmd == "stage":
        convert(a.input_dir, a.stage_dir)
    else:
        finalize(a.stage_dir, a.output_dir, a.winners, show=not a.no_open)

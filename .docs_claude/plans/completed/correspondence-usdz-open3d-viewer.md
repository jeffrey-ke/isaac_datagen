# Viewing GraspableObject `.usdz` assets in open3d, with coordinate axes (`correspondence/`)

## Context

Needed a way to eyeball a `GraspableObject`'s mesh together with its coordinate
frames — the object's local origin and its `grasp_point` SE3 — in open3d, to
sanity-check the assets that train the grasp-pose stage. The blocker:
**open3d cannot read USD/USDZ** (`read_triangle_mesh` on a `.usdz` →
"unknown file extension"; its IO is ASSIMP/native — obj/ply/stl/gltf/glb/off).
And the GraspableObject `.usdz` assets bundle their appearance as a
`UsdPreviewSurface` + `UsdUVTexture`, which a naive geometry dump drops.

Datasets visualized: `object_dataset_amazon` (the original single example,
`amazon_0`), then batched over `kleenex_dataset` (7 boxes) and
`datasets/ycb_dataset` (7 full-resolution YCB meshes — cans/boxes/bottles).

## Approach: two-stage, two-env pipeline with an env-agnostic npz bridge

open3d isn't in the project venv, and `pxr` isn't importable under plain
`uv run` (isaacsim only exposes it after a full kit boot). So the work splits
across two ephemeral envs, bridged by a plain `.npz`:

- **Stage A — `correspondence/extract_mesh.py <dataset_dir> <data_dir> [idx]`**,
  run `uv run --with usd-core python …` (`vision_core` + standalone `pxr` from
  the `usd-core` overlay). Batch-`GraspableObject.deserialize(idx, dataset_dir)`
  over every sample (counted from `usd_path/`) via the real routine
  (`SerializableSample.deserialize`, `vision_core/datastructs.py:120`). For each:
  pxr reads the usdz, bakes points to the object frame
  (`ComputeLocalToWorldTransform`), fan-triangulates n-gons, captures faceVarying
  `st` UVs aligned to those triangles, and pulls the bound `UsdUVTexture` PNG
  straight out of the `.usdz` **zip** → dumps `_<name>.npz` + `<name>_texture.png`
  (names from `meta["name"]`) into `<data_dir>`.

- **Stage B**, run `uvx --from open3d python …`. Two builders:
  - **`build_obj_axes.py <data_dir> <out_dir>`** — the per-object viewer artifact:
    one self-contained **multi-material** `<name>.obj` (+`.mtl`+`<name>_tex.png`)
    per npz, carrying the textured object mesh (`map_Kd`) *and* baked origin +
    grasp coordinate axes as flat-color materials (`Kd`). Axis geometry comes from
    `create_coordinate_frame`, split into colored parts by vertex color. Written
    as raw OBJ/MTL text (open3d's writer is single-material).
  - **`build_ply.py <npz>`** — single-object alternative: `<name>.ply` (grey mesh
    + origin triad + grasp-pose triad, vertex-colored) **and** a single-material
    textured `<name>.obj`, plus an offscreen `<name>_preview.png`.

- **`correspondence/view.py`**, installed on PATH as **`plyview`** — a PEP-723
  self-contained `uv run --script` (shebang `#!/usr/bin/env -S uv run --script`;
  `uv` provisions open3d, no project venv). Symlinked `~/.local/bin/plyview →
  correspondence/view.py`. `plyview FILE [--frame] [--grasp] [--wire] [--save PNG]`.

## Usage

```
# batch a dataset → per-object OBJs with axes, organized per folder
uv run --with usd-core python correspondence/extract_mesh.py <dataset> correspondence/<name>/.data
uvx --from open3d  python correspondence/build_obj_axes.py    correspondence/<name>/.data correspondence/<name>

# view one (textured object + colored origin & grasp axes)
plyview correspondence/ycb_dataset/ycb_006_mustard_bottle.obj
```

## Outcome (2026-06-16)

- **14 per-object OBJs** delivered, organized per source folder, named by
  `meta["name"]`: `correspondence/kleenex_dataset/kleenex_{0..6}.obj` and
  `correspondence/ycb_dataset/ycb_*.obj` (each a self-contained
  `.obj`+`.mtl`+`_tex.png` with relative `map_Kd`). Plus the `amazon_0` example
  at the `correspondence/` root.
- **`plyview`** works as a global command and renders the multi-material OBJs
  faithfully (object texture + colored axes).
- Verified texture placement against the datasets' own `reference_image` (French's
  mustard label correct & upright; Cheez-It correct).
- The mechanism is registered in `CLAUDE.md` (section "Viewing GraspableObject
  assets in open3d (`correspondence/`)") and the `vision_core` (de)serialization
  routine pointers were added to the external-deps line.

## Decisions & gotchas (the hard-won bits)

- **NO UV flip.** USD `st`, OBJ `vt`, and open3d `triangle_uvs` *all* use a
  **bottom-left** origin → pass `st` through verbatim. An earlier `v → 1−v` flip
  (mis-justified as "open3d is top-left") shipped silently: on **box** geometry it
  only mirrors within each face and looks fine, but on YCB **atlas** textures it
  samples the wrong region → labels scrambled onto the wrong faces ("textures at
  weird positions"). Caught by comparing the mustard render to its reference.
  **Lesson: verify texture orientation against an atlas-textured non-box mesh,
  never a box.** Fixed in both `build_obj_axes.py` and `build_ply.py`.
- **Appearance IS recoverable.** The "geometry only" limit was a property of the
  naive pxr point/face dump, not the asset: the usdz bundles diffuse texture + UVs
  + a `UsdPreviewSurface`. Reading the texture bytes directly off the bound
  `UsdUVTexture` (`@textures/…@`, read from the usdz zip) works — even though a USD
  *flatten* drops UsdShade bindings (the `blender-usdz-import-gotchas` note).
- **`pxr` via the `usd-core` overlay**, not isaacsim — plain `uv run` has no
  top-level `pxr` and booting a full kit just to read geometry is overkill.
- **Multi-material OBJ** is what lets a single file show texture *and* colored
  axes (object = textured material, each axis = flat `Kd`). open3d's own OBJ
  writer is single-material, so the combined OBJ is written as raw text.
- **`plyview` must use `read_triangle_model`** for `.obj`/`.glb`/`.gltf` —
  `read_triangle_mesh` flattens a multi-material OBJ to one untextured material
  (renders everything black). Verified: the same can rendered black via
  `read_triangle_mesh`, correct via `read_triangle_model`.
- **Frames as large *separate* geometries** (axis len ~0.7×bbox-diag) so they
  clear the surface — a triad baked at the centroid hides *inside* the opaque
  mesh (the original "I can't see the axes"). `--wire` renders the mesh as a
  `LineSet` so an *interior* frame (the grasp pose) shows through.
- **YCB origin triad sits beside the mesh, not at its centroid** — faithful, not a
  bug: it's the asset's authored (scanner-origin) local frame, the frame in which
  `grasp_point` and all rendered poses are defined. The amazon/kleenex boxes
  happen to be centered, so their origin triad lands inside the box.
- **Offscreen rendering:** reuse ONE `OffscreenRenderer` per process and
  `clear_geometry()` between shots — constructing a second EGL context in the same
  process crashes. EGL headless works on this box.

## Files

```
correspondence/
  extract_mesh.py        # Stage A: deserialize + pxr extract (geometry/UVs/texture) → npz
  build_obj_axes.py      # Stage B: multi-material OBJ (textured + baked colored axes)  ← main deliverable
  build_ply.py           # Stage B: single-object PLY (frames) + single-material textured OBJ
  view.py                # plyview CLI (PEP-723), symlinked to ~/.local/bin/plyview
  kleenex_dataset/  ycb_dataset/   # 7 OBJs each (+ .mtl + _tex.png); intermediates in .data/
```

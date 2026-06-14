# Surgical −90° Z mesh rotation for select YCB GraspableObjects

## Context

Four objects in `~/repo/isaac_datagen/datasets/ycb_dataset` had their meshes authored in the
wrong canonical yaw and needed a **−90° rotation about Z** applied to the geometry only:

| object | class | name | serialization idx | usdz |
|---|---|---|---|---|
| cheezit | `cheezit` | `ycb_003_cracker_box` | 0001 | `usd_path_0001.usdz` |
| sugar | `sugar` | `ycb_004_sugar_box` | 0002 | `usd_path_0002.usdz` |
| mustard | `mustard` | `ycb_006_mustard_bottle` | 0004 | `usd_path_0004.usdz` |
| tuna can | `fish can` | `ycb_007_tuna_fish_can` | 0005 | `usd_path_0005.usdz` |

(Targets located by `meta/meta_XXXX.yaml` `class`/`name`.) Each `.usdz` is a 2-file package:
`model.usdc` + `textures/texture_map.png`, with prim tree
`/World` (Xform) → `/World/textured` (Xform, **no ops**) → `/World/textured/textured` (Mesh).

## Approach: rotateZ op on the intermediate Xform (non-destructive)

Scene loading references the usdz at `ref_prim_path="/World"` (`scene.py` →
`load_asset(..., ref_prim_path="/World")`), then **placement** (`set_transform`) authors
translate/rotateXYZ/scale on that `/World` content root. So putting the rotation on the
`/World` root itself would tangle with placement's `xformOpOrder`. Instead, author it one level
down on the op-free intermediate `/World/textured`:

```
/World/textured  ops:  []  →  ['xformOp:rotateZ' = -90.0]
```

This composes *beneath* the placed `/World` root, so:
- placement's `set_transform` never touches it (it only edits `/World`);
- `UntilExhaustedStacker` / `local_bbox_range` (`ComputeLocalBound("/World")`) picks up the
  rotated extent automatically — x/y extents swap, confirming the rotation took.
- It is non-destructive — no point/normal baking, fully reversible.

Per-file procedure (run with `uv run --with usd-core`, no Isaac boot needed):
1. Back up the pristine usdz to `usd_path.orig.bak/` (skip if already present → idempotent;
   always edit *from* the backup).
2. Unzip, `Usd.Stage.Open(model.usdc)`, assert `/World/textured` has no existing ops,
   `UsdGeom.Xformable(prim).AddRotateZOp().Set(-90.0)`, `layer.Save()`.
3. Repackage with `UsdUtils.CreateNewUsdzPackage(model.usdc, out.usdz)` — re-bundles the
   texture and rewrites the in-package path; assert the member set is unchanged.
4. Move `out.usdz` over the original.

## Scope decision (asked, answered "Mesh only")

`grasp_point` for these sits with identity rotation on the box **+X face** (e.g. cheezit
`t=(0.023,-0.014,0.104)`, x = bbox max). A −90° Z maps +X→−Y, so a mesh-only rotation moves
the grasp onto a different physical face. User chose **mesh only**:
- `grasp_point` — **left untouched** (would need `R_z(-90) · grasp_point` to track the same face).
- `reference_image` — **left untouched**; it's a standalone 2D match target, not re-derived
  from the mesh frame, so mesh yaw doesn't invalidate it.

## Outcome (2026-06-14)

All four rotated in place, verified in a **fresh process** (see gotcha below):

| object | idx | x/y extent before → after | op |
|---|---|---|---|
| cheezit | 0001 | `0.0718 × 0.164` → `0.164 × 0.0718` | `rotateZ = -90` |
| sugar | 0002 | `0.0495 × 0.0942` → `0.0942 × 0.0495` | `rotateZ = -90` |
| mustard | 0004 | `0.0972 × 0.0666` → `0.0666 × 0.0972` | `rotateZ = -90` |
| tuna can | 0005 | `0.0856 × 0.0855` → `0.0855 × 0.0856` (cylinder; ~unchanged silhouette) | `rotateZ = -90` |

Texture (`@./textures/texture_map.png@`, 9.6 MB) and 2-file member layout preserved in every
package. Pristine originals in `ycb_dataset/usd_path.orig.bak/` (idx 0001/0002/0004/0005).

## Gotcha: USD layer registry is stale after overwrite

The mustard apply-run readback wrongly showed `ops=[]` / unchanged bbox. Cause: an up-front
`Usd.Stage.Open(src)` sanity check pinned the *original* root layer in USD's `Sdf.Layer`
registry; after overwriting `src`, re-opening the same path returned the cached old layer.
The write was correct — a clean separate process confirmed all four. **Verify in-place
usdz/usd edits in a fresh process (or `layer.Reload()`); don't trust a same-process readback
that opened the path before the overwrite.**

## Follow-up: +180° Z for soup + spam (2026-06-14)

Same non-destructive `rotateZ` op on the op-free `/World/textured`, this time **180°**, for two
more YCB objects (located via `meta/meta_XXXX.yaml` `class`):

| object | class | name | idx | usdz |
|---|---|---|---|---|
| soup can | `soup` | `ycb_005_tomato_soup_can` | 0003 | `usd_path_0003.usdz` |
| spam | `spam` | `ycb_010_potted_meat_can` | 0006 | `usd_path_0006.usdz` |

Both confirmed `/World/textured` op-free before editing; both edited from a fresh
`usd_path.orig.bak/` backup; 2-file member set (`model.usdc` + `textures/texture_map.png`)
preserved.

**Verification differs from the −90° case:** a 180° Z rotation *preserves the axis-aligned x/y
extent*, so the extent-swap check used above doesn't apply. Instead verified (fresh process) that
(a) `xformOp:rotateZ = 180.0` is authored, and (b) the bbox **min/max corners flip sign about Z**.
Both meshes are off-center in xy, so the flip is unambiguous (a true no-op would leave corners
unchanged):

| object | idx | bbox min→ / max→ before → after | op |
|---|---|---|---|
| soup | 0003 | min `(-0.043,+0.050)`→`(-0.025,-0.118)`, max `(+0.025,+0.118)`→`(+0.043,-0.050)` | `rotateZ = 180` |
| spam | 0006 | min `(-0.084,-0.057)`→`(-0.018,-0.004)`, max `(+0.018,+0.004)`→`(+0.084,+0.057)` | `rotateZ = 180` |

Same scope decision as before: **mesh only** — `grasp_point` and `reference_image` left untouched.

## Follow-up: mustard −30° more → −120° total (2026-06-14)

User asked for "−30° more" on the mustard (idx 0004), already at `rotateZ = -90`. Per the
idempotency rule, re-derived from the pristine `usd_path.orig.bak/usd_path_0004.usdz` and
authored a single `rotateZ = -120` (= −90 + −30), **not** a second compounding op. Members
preserved.

**Verification gotcha for non-90° angles:** the extent-swap / corner-flip checks don't apply, and
worse, `UsdGeom.BBoxCache.ComputeLocalBound` transforms the mesh's authored **`extent`-box
corners**, so for a non-axis-aligned angle it rotates 8 corners and takes their AABB — inflating
past the true rotated-points AABB. A points-based bbox comparison therefore *falsely* fails. The
unambiguous check is the **mesh local-to-world matrix identity**: the only delta from pristine is
the added ancestor op, so `M_live == M_pristine · R_z(-120)` must hold exactly. It does
(`M_pristine` = identity; `M_live` top-left = `[[-0.5,-0.866],[0.866,-0.5]]` = cos/sin(−120°)).
Confirmed in a fresh process. `grasp_point` / `reference_image` left untouched as before.

## Decisions

- **rotateZ op on `/World/textured`, not point-baking and not on the `/World` root** — survives
  placement, bbox-correct, reversible.
- **Always edit from the backup, skip-if-exists** — re-running is idempotent and re-derives
  from pristine input rather than compounding rotations.
- **`usd_path.orig.bak/` sibling dir** (not inline `.orig.bak` files) mirrors the existing
  `meta.orig.bak/` precedent and keeps the `{field}/{field}_NNNN.usdz` field dir clean so the
  `SerializableSample` glob isn't polluted.

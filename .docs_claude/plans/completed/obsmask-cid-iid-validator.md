# ObsMask cid/iid consistency validator

## Problem

`mixed-persp` has frames where a graspable instance is segmented in `iid_mask` but
gets **no class label** in `cid_mask` (tuna: 38k iid pixels, 0 cid pixels). Phase-3
inlier GT uses `cid_mask == cid`, so all proposals become outliers. This plan adds a
**per-frame detector** and a **run_pipeline gate** after phase 1.

Root cause fix (`cid_iid_masks` / Isaac semantics) is out of scope here — **resolved in
[`tuna-fish-can-cid-orphan-root-cause.md`](tuna-fish-can-cid-orphan-root-cause.md)** (2026-06-19):
Isaac `instance_segmentation_fast` tokenizes `'fish can'` → `'fish'` at capture; writer catalog
uses `'fish can'` → LUT miss → orphans.

## Design constraint: per-sample, not cross-frame

Isaac **iids are session-local** — the same numeric iid on frame 7 vs frame 50 is not
guaranteed to name the same object. The render-dir catalog `ObsMaskMetadata.iid_to_name`
accumulates mappings across the session and must **not** drive validation membership.

**Per-frame graspable iids** come from the sample itself:

```python
graspable_iids = obs.iid_to_occlusion.keys()
```

This is exactly what [`obsmask_from_data`](src/isaac_datagen/reference_seg_writer.py)
writes: `present_iids = unique(seg) ∩ frame_iid_to_name` — graspable boxes visible
this frame only.

## Check (one ObsMask sample)

Constants: `MIN_CLASS_CID = 2` (0=BACKGROUND, 1=UNLABELLED; classes start at 2).

For each `iid` in `obs.iid_to_occlusion`:

```python
pixels = iid_mask == iid
cids = cid_mask[pixels]
orphan if (cids < MIN_CLASS_CID).any()
```

Report row (`CidOrphan`):

| field | source |
|-------|--------|
| `frame` | sample index |
| `iid` | orphan iid |
| `name` | `md.iid_to_name.get(iid, "?")` — **display only**, not membership |
| `n_pixels` | `pixels.sum()` |
| `cids_seen` | `tuple(sorted(unique(cids)))` |

## Module: `src/isaac_datagen/validate_obsmask.py`

Composable routines:

### 1. `load_obsmasks(render_dir: Path) -> list[ObsMask]`

- `n = count_samples(render_dir, "obs")`
- Deserialize each frame; IO loads mask fields only (`iid_mask`, `cid_mask`,
  `iid_to_occlusion`) — skip RGBA for speed

### 2. `graspable_iids(obs: ObsMask) -> set[int]`

- `return set(obs.iid_to_occlusion)` — per-sample helper, no cross-frame state

### 3. `check_obsmask(obs: ObsMask, frame: int, *, min_class_cid: int = 2) -> list[CidOrphan]`

- Run the pixel check above for each iid from `graspable_iids(obs)`
- Returns orphans **for this sample only**

### 4. `validate_render_dir(render_dir: Path) -> list[CidOrphan]`

```python
masks = load_obsmasks(render_dir)
md = ObsMaskMetadata.deserialize(0, render_dir)
orphans = []
for f, obs in enumerate(masks):
    for o in check_obsmask(obs, f):
        orphans.append(replace(o, name=md.iid_to_name.get(o.iid, "?")))
return orphans
```

### CLI

```
isaac-datagen-validate-obsmask <render_dir>
```

- Print per-orphan lines; summary count
- Exit 1 if any orphans, 0 if clean

Console script in `pyproject.toml`:

```toml
isaac-datagen-validate-obsmask = "isaac_datagen.validate_obsmask:main"
```

## Pipeline: `run_pipeline.py`

After phase 1, **before** luminance prompt and phase 2 (runs on resume too):

```python
fresh = not n_obs
if not n_obs:
    _run("isaac-datagen", ...)
    n_obs = ...

orphans = validate_render_dir(render_dir)
if orphans:
    sys.exit(...)

if fresh:
    _confirm_or_abort(render_dir, n_obs)
```

In-process call (no subprocess); fail-fast like other phases.

## Out of scope

- Fixing `cid_iid_masks` writer bug
- Cross-frame iid catalog validation

---

## Outcome (2026-06-19)

Shipped as planned:

- [`src/isaac_datagen/validate_obsmask.py`](src/isaac_datagen/validate_obsmask.py) —
  `load_obsmasks`, `graspable_iids`, `check_obsmask`, `validate_render_dir`, CLI
- [`pyproject.toml`](pyproject.toml) — `isaac-datagen-validate-obsmask` console script
- [`run_pipeline.py`](src/isaac_datagen/run_pipeline.py) — in-process gate after phase 1,
  before luminance prompt; prints up to 20 orphan rows then exits

Verification:

- `datasets/mixed-persp/render005` → exit 1, 125 tuna orphan rows
- `datasets/expanded-refseg/render000` → exit 0, 1000 frames clean

---

## Results: `datasets/mixed-persp` (2026-06-19)

Ran `isaac-datagen-validate-obsmask` on all six render dirs (125 frames each).

**Every orphan is the same instance:** `ycb_007_tuna_fish_can` (class `'fish can'`).
On affected frames, tuna pixels in `iid_mask` all map to `cid_mask=0` (`cids_seen=(0,)`).
No other graspable instances flagged.

**553 orphan rows total** across 750 frames.

| Render dir | Orphan frames | Clean frames | Tuna session iid |
|------------|---------------|--------------|------------------|
| render000 | 73 / 125 | 52 | 9 |
| render001 | 115 / 125 | 10 | 15 |
| render002 | 32 / 125 | 93 | 13 |
| render003 | **125 / 125** | 0 | 15 |
| render004 | 83 / 125 | 42 | 4 |
| render005 | **125 / 125** | 0 | 11 |

### Affected frame indices

**render000** (73): 1, 2, 4, 5, 6, 7, 8, 9, 15, 16, 18, 19, 20, 21, 23, 27, 29, 30,
31, 32, 33, 34, 37, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 56, 57, 58, 59,
65, 68, 69, 70, 71, 76, 77, 79, 80, 81, 82, 83, 84, 90, 93, 94, 95, 96, 102, 104, 105,
106, 107, 108, 109, 112, 115, 116, 118, 119, 120, 122, 123, 124

**render001** (115): all except **clean** 1, 8, 12, 13, 14, 76, 83, 87, 88, 89

**render002** (32): 14, 16, 23, 41, 46, 66, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 116, 121

**render003** (125): **frames 0–124 (all)**

**render004** (83): 0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 21, 23, 25, 27,
28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 44, 45, 46, 48, 49, 50, 52, 53, 55,
56, 57, 58, 59, 60, 61, 62, 63, 64, 70, 71, 73, 77, 78, 80, 81, 82, 83, 84, 85, 86,
87, 93, 95, 99, 100, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117,
119, 120, 121, 123, 124

**render005** (125): **frames 0–124 (all)**

### Interpretation

This is not “tuna off-screen” — on orphan frames the instance **is** present in
`iid_mask` (e.g. render005 frame 50: 38,780 iid pixels) but **`cid_mask` never
assigns class `'fish can'`** (always background 0). Phase-3 inlier labels for `'fish can'`
are all-outlier on those frames. `run_pipeline` correctly rejects all six `mixed-persp`
render dirs until `cid_iid_masks` is fixed and data re-rendered.

Example orphan line:

```
frame 0050  iid 11  ycb_007_tuna_fish_can  38780 px  cids_seen=(0,)
```

Proof viz: `/tmp/mixed_persp_tuna_mask_proof.png` (render005 frame 50).

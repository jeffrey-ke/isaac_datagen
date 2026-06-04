# Port `coords_in_mask` → phase-3 inlier-labeling pass (`add_inlier_data`)

## Context

We want a **verifier** that, given a proposer point on an observation, predicts whether
that point actually lands on the target object. To train it we need labeled data: every
phase-2 proposal point tagged **inlier** (inside the object's instance mask) or **outlier**
(outside it). This is *labeling*, not filtering — we keep all points and attach a boolean.

The labeling primitive already exists: `coords_in_mask` (currently
`segmentation/utils.py:246`) returns the per-proposal `(N,)` bool we need. It just lives in
a package `isaac_datagen` doesn't depend on, so the port is (a) move that helper to a shared
home and (b) add a phase-3 pass + datastruct that consume it.

Mapping invariant (verified this session): within a render dir, `id_to_name` is a
**bijection** — each object has a unique name, stamped as its `"instance"` semantic
(`scene.py:47`) and used as its name-derived prim path (`scene.py:40`), so two objects can't
share a name. Therefore `name → id` inverts cleanly and `(id_mask == id)` is exactly that
object's instance mask.

**Pipeline order:** phase-1 capture (`obs/`, `id_mask/`, catalog) → phase-2 `add_proposals`
(`proposals/`) → **phase-3 `add_inlier_data` (`labels/`)**. Phase-3 requires `proposals/` to
already exist.

## Residual write = superset datastruct (the core idea)

`ImageInlierSample` is a **superset** of `PreReferenceSegSample`: identical `obs`/`id_mask`/
`proposals` fields **plus** `labels`. Identical field names ⇒ identical serialize subdirs. So
phase-3 reads a `PreReferenceSegSample`, computes `labels`, and writes **only** `labels/` via
`serialize(idx, dir, only={"labels"})` — `obs/`, `id_mask/`, `proposals/` are never rewritten.
A later full `ImageInlierSample.deserialize(idx, dir)` reassembles all four fields from the
shared subdirs. This is the same residual pattern phase-2 used to add `proposals/`.

## Changes by file

### 1. NEW `vision_core/src/vision_core/mask_utils.py` — shared helper
Move `coords_in_mask` here verbatim (signature `coords_in_mask(mask, coords)`, returns `(N,)`
bool, xy→`mask[y,x]`, clamps OOB). `vision_core` is a dependency of both repos, so both can
import it. Needs `numpy` + `torch` (already available).

### 2. `segmentation-train/src/segmentation/utils.py` — re-export
Delete the local `coords_in_mask` def (lines ~246-258); add
`from vision_core.mask_utils import coords_in_mask`. Existing callers
(`from segmentation.utils import coords_in_mask`, e.g. `preprocess.py`) keep working.

### 3. `vision_core/src/vision_core/datastructs.py` — add `ImageInlierSample`
Mirror `PreReferenceSegSample` (lines 177-199) with one extra field. Place after it:
```python
@dataclass
class ImageInlierSample(SerializableSample):
    """PreReferenceSegSample + per-proposal inlier labels (phase-3, residual).

    Superset of PreReferenceSegSample: shares obs/id_mask/proposals subdirs, so phase-3
    writes ONLY ``labels`` via serialize(idx, dir, only={"labels"}). ``labels`` maps object
    name → (N,) bool, True where that proposal lands inside the object's instance mask.
    """
    obs: tv_tensors.Image
    id_mask: tv_tensors.Mask
    proposals: dict            # {name: torch.Tensor (N, 2)}
    labels: dict               # {name: torch.Tensor (N,) bool}

    _serializers = {
        **SerializableSample._serializers,
        **ReferenceSegSample._serializers,   # tv_tensors.Mask → .npy
        **_DICT_PT_SERIALIZER,               # dict → .pt (proposals + labels)
    }

    def __post_init__(self):
        if not isinstance(self.obs, tv_tensors.Image):
            self.obs = tv_tensors.Image(self.obs)
        if not isinstance(self.id_mask, tv_tensors.Mask):
            self.id_mask = tv_tensors.Mask(self.id_mask)
```
`_DICT_PT_SERIALIZER` (line 144) already covers both dict fields; bool tensors round-trip fine.

Also add a tiny per-render-dir stats catalog (written once at `idx=0`, like `ObsMaskMetadata`):
```python
@dataclass
class ImageInlierMetadata(SerializableSample):
    """Per-render-dir inlier/outlier counts from the phase-3 labeling pass."""
    stats: dict        # {"n_inliers": int, "n_total": int}

    _serializers = {
        **SerializableSample._serializers,
        **ReferenceSegSample._serializers,   # dict → .json (human-readable, plain-int stats)
    }
```
The base has **no `int` serializer**, which is why the counts live inside a `dict`; plain ints ⇒
use the JSON dict serializer (NOT `_DICT_PT_SERIALIZER`). Writes `stats/stats_0000.json` — a fresh
subdir, no collision with `obs/`/`id_mask/`/`proposals/`/`labels/` or the `ObsMaskMetadata` catalog.

### 4. NEW `isaac_datagen/src/isaac_datagen/add_inlier_data.py` — phase-3 pass
Follows the user's pseudocode; CLI takes the **render dir** directly (metadata lives there),
no config/NN/device needed.
```python
"""Phase-3 pass: label each phase-2 proposal inlier/outlier for verifier training.

NN-free. For each frame, tag every proposer point True iff it lands inside its object's
instance mask, then write residually as ImageInlierSample.serialize(idx, dir, only={"labels"})
— obs/, id_mask/, proposals/ are never rewritten.

Usage: isaac-datagen-inliers <render_dir>
"""
import sys
from pathlib import Path

from vision_core.datastructs import (
    ObsMaskMetadata, PreReferenceSegSample, ImageInlierSample, ImageInlierMetadata,
)
from vision_core.mask_utils import coords_in_mask


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-inliers <render_dir>", file=sys.stderr)
        sys.exit(1)
    render_dir = Path(sys.argv[1])

    md = ObsMaskMetadata.deserialize(0, render_dir)
    name_to_id = {name: i for i, name in md.id_to_name.items()}
    if len(name_to_id) != len(md.id_to_name):          # bijection guard (pseudocode req)
        raise ValueError("id_to_name is not 1-to-1; cannot invert to name->id")

    n_frames = len(list((render_dir / "obs").iterdir()))
    n_inliers = n_total = 0
    for idx in range(n_frames):
        pre = PreReferenceSegSample.deserialize(idx, render_dir)
        labels = {
            name: coords_in_mask(pre.id_mask == name_to_id[name], coords)   # (mask, coords) order
            for name, coords in pre.proposals.items()
        }
        ImageInlierSample(
            obs=pre.obs, id_mask=pre.id_mask, proposals=pre.proposals, labels=labels,
        ).serialize(idx, render_dir, only={"labels"})
        n_in = sum(int(v.sum()) for v in labels.values())
        n_tot = sum(int(v.numel()) for v in labels.values())
        n_inliers += n_in
        n_total += n_tot
        print(f"[{idx + 1}/{n_frames}] {render_dir.name}: {n_in}/{n_tot} inliers, {len(labels)} object(s)")

    # Per-render-dir stats catalog (written once, like ObsMaskMetadata).
    ImageInlierMetadata(stats={"n_inliers": n_inliers, "n_total": n_total}).serialize(0, render_dir)
    print(f"{render_dir.name}: {n_inliers}/{n_total} inliers total → stats/stats_0000.json")


if __name__ == "__main__":
    main()
```

### 5. `isaac_datagen/pyproject.toml` — console script
Under `[project.scripts]`, add:
`isaac-datagen-inliers = "isaac_datagen.add_inlier_data:main"`

## Run-blocker polish (vs. the raw pseudocode)
- **Arg order**: real signature is `coords_in_mask(mask, coords)` — pseudocode had `(coords, mask)`. Pass mask first.
- **`only` is a set**: `only={"labels"}`, not the string `"labels"` (the impl does `f.name not in only`).
- **Helper relocation**: `coords_in_mask` is in `vision_core.mask_utils` now (decision: one source of truth; `segmentation` re-exports).
- **`labels` keys == `proposals` keys** by construction (iterating `pre.proposals.items()`), so they align field-for-field.
- **Empty proposals**: if a name has `N=0`, `coords_in_mask` returns a `(0,)` bool — no crash (this is labeling, not the segmenter's ≥1-point path).

## Verification
1. **Static / import**: `python -c "from vision_core.mask_utils import coords_in_mask; from vision_core.datastructs import ImageInlierSample"` and back-compat `python -c "from segmentation.utils import coords_in_mask"`.
2. **End-to-end** on the existing render dir (run phase-2 first — `proposals/` isn't there yet):
   ```
   uv run isaac-datagen-proposals src/isaac_datagen/configs/randomized.yaml      # writes proposals/
   uv run isaac-datagen-inliers   src/isaac_datagen/expanded-refseg/render000    # writes labels/
   ```
3. **Residual check**: `labels/` has one `.pt` per frame (== `obs/` count); `obs/`, `id_mask/`,
   `proposals/` mtimes unchanged.
4. **Round-trip**: load a `labels_*.pt` → `dict[str, bool tensor]`; for each name assert
   `labels[name].shape[0] == proposals[name].shape[0]`. Spot-check one point: its label equals
   whether its `(x,y)` indexes True in `(id_mask == name_to_id[name])`.
5. **Full deserialize**: `ImageInlierSample.deserialize(0, render_dir)` returns all four fields
   populated (confirms the superset/shared-subdir reassembly).
6. **Stats metadata**: `stats/stats_0000.json` exists; `ImageInlierMetadata.deserialize(0, render_dir).stats`
   gives `{"n_inliers", "n_total"}` with `0 ≤ n_inliers ≤ n_total`, and `n_total` equals the summed
   `N` over every frame's `proposals/`.

## Out of scope (downstream)
The verifier's training `Dataset`/model that consumes `ImageInlierSample` (obs + proposals +
labels) — separate effort, mirrors how the seg `Dataset` consumed the phase-2 output.

## Staged checklist
1. `vision_core/mask_utils.py` (move `coords_in_mask`) + `segmentation/utils.py` re-export. Verify (1).
2. `vision_core/datastructs.py`: add `ImageInlierSample` + `ImageInlierMetadata`. Import smoke.
3. `add_inlier_data.py` + `pyproject.toml` entry. End-to-end + round-trip + stats (2-6).

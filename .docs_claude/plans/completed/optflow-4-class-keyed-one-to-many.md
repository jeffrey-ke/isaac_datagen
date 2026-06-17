# OptFlow dataset: 1-to-many dense correspondence (per-class references)

## Context

The optflow dataset (`optflow_writer.py` + `OptFlowObject`/`OptFlowSample`/`OptFlowMetadata` in
`objects.py`) renders labeled ref→obs dense-correspondence pairs for fine-tuning dense matchers
(RoMa/UFM/DKM/LoFTR). It was originally keyed **by instance name** (`name_to_*`): each placed
object had its own reference, warped 1-to-1 against each observation frame.

We're re-casting it as **1-to-many**: a **class** owns one canonical reference (RGB, depth, pose,
intrinsics), and the scene places **N instances** of that class. From that single reference depth
map + reference pose we compute **N relative transforms** — one to each instance's `local2world` —
and warp the one reference into all N locations of the observation. This is what reference-prompted
instance segmentation needs ("a point on ANY same-class instance is an inlier"): one reference,
many same-class targets.

Both files are **mid-refactor and currently un-importable**: `OptFlowSample.visualize` declares
`*, class=None` (a Python syntax error — `class` is reserved), the metadata is half-renamed
(`name_to_*` ↔ `class_to_*` mixed), and the warp body has leftover `nm` references plus a dangling
prose line `the multiplication with einsum:`. This plan completes the conversion.

The einsum the user wrote is **correct**: `'ij,njx,xy->niy'` over `inv(cam2world)` (4,4),
`class_to_l2w` (N,4,4), `ref_pose` (4,4) yields `inv(cam2world) @ l2w[n] @ ref_pose` for each
instance `n` → `(N,4,4)` ref-cam→obs-cam transforms. The remaining work is data-model plumbing +
polishing the run-blocking warp details.

**Decisions (from clarifying Qs):** scope = isaac_datagen only (writer + metadata + viz; downstream
trainer adapter / `optflow-3` doc left as a noted follow-up); viz = one `[ref | obs]` row per class
with per-instance-colored fan-out; reference dedup = pick the first instance's reference per class
(trust caller to duplicate the same asset), stack all `l2w`.

## The 1-to-many mechanic (confirmed against RoMa)

`get_gt_warp(depth1, depth2, T_1to2, K1, K2)` (`RoMa/romatch/utils/utils.py:325`) sets its batch
`B` from `depth1.shape[0]`, then calls `warp_kpts` which requires **all** of
`depth1/depth2/T/K1/K2` to share batch `B` (`utils.py:357`). To warp the single class reference into
N instances we pass `B = N`: the einsum already gives `T` as `(N,3,4)`; the single reference depth,
reference K, obs depth, and obs K must each be `.expand(N, …)`. Safe without `.contiguous()` because
`get_gt_warp` `.double()`s every argument internally (materializes contiguous float64 copies before
`warp_kpts`).

## Changes

### 1. `objects.py` — `OptFlowMetadata` schema (currently lines 178-200)

Replace the six `name_to_*` fields with class-keyed fields; add `class_to_name`. `_DICT_PT_SERIALIZER`
(serializes each `dict` field to one `.pt`) is unchanged and handles `list[str]` / Image / Tensor
values.

```python
@dataclass
class OptFlowMetadata(SerializableSample):
    """Per-render-dir catalog of the optical-flow dataset, serialized once (at idx=0).

    Keyed BY CLASS, not instance. Each class owns one canonical reference (RGB, metric depth,
    intrinsics, camera2local pose) plus the world placements of ALL its instances in this render
    dir. The trainer warps the one reference into each instance: per instance n,
    ``T_ref→obs[n] = inv(cam2world) @ class_to_l2w[cls][n] @ class_to_ref_pose[cls]`` (OpenCV).
    Mirrors ``ObsMaskMetadata``: every per-class collection is a plain dict torch.save'd once.
    """
    obs_intrinsics: np.ndarray       # (3, 3) observation K (shared)
    class_to_name: dict              # {class → list[str]} instance names, aligned to class_to_l2w rows
    class_to_reference: dict         # {class → tv_tensors.Image (3, H, W)} canonical reference RGB
    class_to_reference_depth: dict   # {class → torch.Tensor (Ha, Wa)} metric z-depth, 0 off-object
    class_to_ref_intrinsics: dict    # {class → torch.Tensor (3, 3)} reference K
    class_to_ref_pose: dict          # {class → torch.Tensor (4, 4)} camera2local SE3, OpenCV
    class_to_l2w: dict               # {class → torch.Tensor (N, 4, 4)} the class's N instance placements

    _serializers = {**SerializableSample._serializers, **_DICT_PT_SERIALIZER}
```

Also update the `OptFlowObject` docstring (line 57-65) line that says references are "INDEXED BY
NAME" → by class (the object itself is unchanged; only the metadata keying changed).

### 2. `optflow_writer.py` — `finalize_metadata` (currently lines 65-81)

Group placements by class, take a representative reference per class, stack each class's `l2w`.
Add `from collections import defaultdict` at the top.

```python
def finalize_metadata(self, directory: str | Path | None = None):
    """Write the per-render-dir constants once (at idx=0). Call after capture."""
    directory = Path(directory) if directory is not None else self._render_dir
    by_class = defaultdict(list)                                   # class → [(object, l2w), ...]
    for o, L in zip(self._objects, self._l2w):
        by_class[o.meta["class"]].append((o, L))
    rep = {c: members[0][0] for c, members in by_class.items()}    # representative object per class
    OptFlowMetadata(
        obs_intrinsics=np.asarray(self._obs_K, dtype=np.float32),
        class_to_name={c: [o.meta["name"] for o, _ in members] for c, members in by_class.items()},
        class_to_reference={
            c: tv_tensors.Image(torch.from_numpy(np.array(o.reference_image)).permute(2, 0, 1))
            for c, o in rep.items()
        },
        class_to_reference_depth={c: torch.from_numpy(o.reference_depth).float() for c, o in rep.items()},
        class_to_ref_intrinsics={c: torch.from_numpy(o.ref_intrinsics).float() for c, o in rep.items()},
        class_to_ref_pose={c: torch.from_numpy(o.ref_pose).float() for c, o in rep.items()},
        class_to_l2w={
            c: torch.from_numpy(np.stack([L for _, L in members])).float()
            for c, members in by_class.items()
        },
    ).serialize(0, directory)
```

This drops the leftover `nm`/`cls` lambdas, the `class_to_name={...}` placeholder, and the stray
`name_to_reference` entry. Also update the `__init__` docstring `list[PreOptFlowObject]` →
`list[OptFlowObject]` if not already (the rename is `OptFlowObject`).

### 3. `objects.py` — `OptFlowSample.visualize` (currently lines 94-175)

**Signature/docstring:** `*, class=None` → `*, cls_name=None` (syntax-error fix); update the
docstring's `points` / param wording to say "class" filter.

**Warp loop body** — replace the broken block (lines 119-150). Per class: einsum → N transforms,
`.expand` the singletons to N, one batched `get_gt_warp`, keep the full `(N, …)` outputs:

```python
for cls in ([cls_name] if cls_name else list(md.class_to_name)):     # iterate classes, 1-many
    dA = md.class_to_reference_depth[cls].float()                    # (Ha, Wa) canonical ref depth
    L = md.class_to_l2w[cls].float()                                 # (N, 4, 4) this class's placements
    N = L.shape[0]
    inv_c2w = torch.as_tensor(np.linalg.inv(self.cam2world), dtype=torch.float32)
    T = torch.einsum('ij,njx,xy->niy', inv_c2w, L,                   # (N, 4, 4) ref-cam → obs-cam, per instance
                     md.class_to_ref_pose[cls].float())
    K_A = md.class_to_ref_intrinsics[cls].float()
    x2, prob = get_gt_warp(                                          # batch = N: expand singletons to match T
        dA[None].expand(N, -1, -1), dB[None].expand(N, -1, -1), T[:, :3],
        K_A[None].expand(N, -1, -1), K_B[None].expand(N, -1, -1),
        relative_depth_error_threshold=rel,
    )
    x2, prob = x2.numpy(), prob.numpy()                              # x2 (N,Ha,Wa,2) normalized, prob (N,Ha,Wa)
    valid_ref = dA.numpy() > 0
    Ha, Wa = valid_ref.shape
    if points is not None:
        pts = [(int(round(y)), int(round(x))) for x, y in points]
        cand = [(yy, xx) for yy, xx in pts if 0 <= yy < Ha and 0 <= xx < Wa and valid_ref[yy, xx]]
    else:
        vy, vx = np.nonzero(valid_ref)
        if not len(vy):
            continue
        gy = np.linspace(vy.min(), vy.max(), 5).astype(int)
        gx = np.linspace(vx.min(), vx.max(), 5).astype(int)
        cand = [(y, x) for y in gy for x in gx if valid_ref[y, x]][:n_points]
    if cand:
        rows.append((cls, x2, prob, cand))
```

**Plot loop** — replace lines 152-172. One `[ref | obs]` per class; reference candidates drawn once
(neutral, numbered); each instance fans out into the single obs panel, colored per-instance, with
connection lines back to the shared reference point:

```python
fig, axes = panel_grid(2 * max(len(rows), 1), cols=2, panel_w=5.0, panel_h=4.0)   # [ref | obs] per class
for r, (cls, x2, prob, cand) in enumerate(rows):
    ax_ref, ax_obs = axes[2 * r], axes[2 * r + 1]
    ax_ref.imshow(md.class_to_reference[cls].permute(1, 2, 0).numpy()[..., :3])
    ax_obs.imshow(obs)
    ax_ref.set_title(f"{cls} ref", fontsize=8)
    ax_obs.set_title(f"obs · {x2.shape[0]} instances", fontsize=8)
    for i, (y, x) in enumerate(cand):                               # shared reference candidates (neutral)
        ax_ref.plot(x, y, "x", ms=9, mew=2, color="k")
        ax_ref.text(x + 3, y - 3, str(i), color="k", fontsize=8)
    for n in range(x2.shape[0]):                                    # one color per instance → fan-out
        c = plt.cm.turbo(n / max(x2.shape[0] - 1, 1))
        for i, (y, x) in enumerate(cand):
            if prob[n, y, x] <= 0:                                  # occluded / out-of-view for this instance
                continue
            ub, vb = (x2[n, y, x, 0] + 1) * Wb / 2, (x2[n, y, x, 1] + 1) * Hb / 2
            ax_obs.plot(ub, vb, "x", ms=8, mew=2, color=c)
            ax_obs.text(ub + 3, vb - 3, str(i), color=c, fontsize=7)
            fig.add_artist(ConnectionPatch((ub, vb), (x, y), "data", "data",
                                           axesA=ax_obs, axesB=ax_ref, color=c, lw=0.5, alpha=0.4))
    ax_ref.axis("off")
    ax_obs.axis("off")
```

## Run-blocking details polished (the gotchas)

- `class` → `cls_name` param (reserved word → the file doesn't even import today).
- einsum dtype: cast `inv(cam2world)` to `float32` (the user's draft dropped `dtype=`; mixing f64
  `inv` with f32 `L`/`ref_pose` raises in `torch.einsum`).
- `md.class_to_ref_pose(cls)` → `md.class_to_ref_pose[cls]` (dict subscript, not call).
- Batch match: `.expand(N, -1, -1)` the single ref depth, ref K, obs depth, obs K so all equal `T`'s
  N (RoMa derives B from `depth1[0]` and `warp_kpts` asserts a shared batch).
- Keep `prob` per-instance (`(N,Ha,Wa)`) for the fan-out occlusion test; the old code collapsed to
  `prob[0]`.
- Drop all dangling `nm` references and the prose `the multiplication with einsum:` line.

## Verification

1. **Imports again** (it currently doesn't): `uv run python -c "import isaac_datagen.objects, isaac_datagen.optflow_writer"`.
2. **Synthetic 1-many warp sanity** (no Isaac boot needed) — `uv run python` scratch: build an
   `OptFlowMetadata` for one class with `class_to_l2w` of shape `(2,4,4)` (two instances, e.g. two
   translations of identity), a flat `class_to_reference_depth` (constant z > 0), identity
   `class_to_ref_pose`, matching `obs_intrinsics`/`class_to_ref_intrinsics`, and an `OptFlowSample`
   with identity `cam2world` + matching obs depth; call `sample.visualize(md)` and assert it returns
   an `(H, W, 3) uint8` array with two distinct instance colors landing at the two offsets. Confirms
   the einsum, the `expand` batch, and the per-instance fan-out without rendering.
3. **End-to-end** (if an optflow capture config exists): `uv run clean_datagen.py <optflow_config> num_frames=4`,
   then deserialize `OptFlowMetadata` + an `OptFlowSample` from the render dir and run `visualize`,
   eyeballing that same-class instances each receive the reference fan-out and occluded instances
   drop out via `prob`.

## Out of scope (noted follow-up)

The on-disk `OptFlowMetadata` contract changed (`name_to_*` → `class_to_*`, `l2w` now `(N,4,4)`).
The trainer adapter and `.docs_claude/plans/active/optflow-3-trainer-adapter.md` still describe the
1-to-1 `name_to_*` schema and must be updated to enumerate per-class instances (N warps per class)
to consume this data — deferred per the "datagen only" scope decision.

# Decentered `-1inst` pools (Track A) + train-time translate aug for m2f-gligen (Track B)

**Status 2026-07-13: PLANNED, not started.** Cross-repo (isaac_datagen + vision_core + segmentation); render-pipeline precedent: `plans/completed/emptyworld-1inst-regeneration.md`.

## Context

**Why.** The 4 solo K-shot fine-tune pools (`datasets/{snack031,snack033,snack034,snack035}-1inst`, 200 frames each, regenerated empty-world 2026-07-12) have a center-of-image bias: `LookAtPoser` (`isaac_datagen/src/isaac_datagen/posers.py:36`) aims the optical axis exactly at the grasp-frame origin, so the single target projects to the principal point (~(922, 544) px) in **every** frame; `look_at` (`vision_core/src/vision_core/pose_utils.py:62`) additionally fixes roll — the horizon is level in every frame. A model fine-tuned on these pools never sees the target off-center or tilted; "target = image center" and "up = image-up" are learnable shortcuts.

**Both mechanisms below were proven with runnable numeric mocks during planning (2026-07-13)**; the numbers cited are from those runs, and the Verification section defines the mocks to re-create as one-off tests. Two independent, composable tracks:

- **Track A (render-time):** NEW `DecenteredLookAtPoser` — identical halo camera *positions* to `LookAtPoser` under the same seed, plus constructed off-center pointing and roll — and regeneration of the 4 pools under NEW dataset names (`<class>-1inst-decentered`). Existing pools untouched.
- **Track B (train-time):** YAML-selectable train-only transform pipeline on the m2f-gligen data path (`LightningInstanceSeg`), used to add stock `v2.RandomAffine` translate. Eval/benchmark still read centered renders; Track A produces decentered *data*, Track B decentered *augmentation* — they compose.

**STOP-AND-ASK clause: if anything unexpected appears mid-execution (assert failures, frame drops, non-zero rsync diffs, RNG mismatch vs the old pools, config validation errors), STOP and ask the user before deviating.**

---

## Track A — `DecenteredLookAtPoser` + 4 new pools

### A1a. NEW frustum-geometry helpers in `vision_core/src/vision_core/pose_utils.py`

Generic camera geometry — reusable, so they live in vision_core beside `look_at`/`cv2opengl` (no existing pixel→ray or frustum helper there; `pixel_to_ndc` is a torch NDC converter, different job). The poser keeps only policy.

```python
def frustum_normals(K: np.ndarray, resolution) -> np.ndarray:   # NEW
    """Inward unit normals of the 4 pinhole frustum face planes (camera frame)."""
    w, h = resolution
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ns = np.array([
        [fx, 0.0, cx],          # u >= 0
        [-fx, 0.0, w - cx],     # u <= W
        [0.0, fy, cy],          # v >= 0
        [0.0, -fy, h - cy],     # v <= H
    ])
    return ns / np.linalg.norm(ns, axis=1, keepdims=True)


def cone_in_frustum(d: np.ndarray, half_angle: float, normals: np.ndarray) -> bool:   # NEW
    """Exact: the cone (unit axis d, half_angle) lies inside all face half-spaces."""
    return bool((normals @ d >= np.sin(half_angle)).all())


def pixel_direction(K: np.ndarray, uv) -> np.ndarray:           # NEW: unit ray through a pixel
    d = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    return d / np.linalg.norm(d)


def erode_frame_rect(K: np.ndarray, resolution, half_angle: float, iters=40):   # NEW
    """Pixel rect (u_lo, u_hi, v_lo, v_hi) whose rays all pass cone_in_frustum(half_angle),
    or None if it collapses. Two-pass bisection, each axis vs its OWN faces with worst-case
    other coordinate: v vs top/bottom at the extreme columns, then u vs left/right at the
    eroded v extremes. Conservative by construction."""
    w, h = resolution
    ns = frustum_normals(K, resolution)
    s = np.sin(half_angle)

    def ok(u, v, faces):
        return (ns[faces] @ pixel_direction(K, (u, v)) >= s).all()

    def bisect(lo, hi, f):
        # f False at lo, True at hi (or nowhere); returns the True point nearest lo
        if not f(hi):
            return None
        if f(lo):
            return lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        return hi

    mid_u, mid_v = w / 2, h / 2
    v_lo = bisect(0, mid_v, lambda v: ok(0, v, [2]) and ok(w, v, [2]))
    v_hi = bisect(h, mid_v, lambda v: ok(0, v, [3]) and ok(w, v, [3]))
    if v_lo is None or v_hi is None or v_lo >= v_hi:
        return None
    u_lo = bisect(0, mid_u, lambda u: ok(u, v_lo, [0]) and ok(u, v_hi, [0]))
    u_hi = bisect(w, mid_u, lambda u: ok(u, v_lo, [1]) and ok(u, v_hi, [1]))
    if u_lo is None or u_hi is None or u_lo >= u_hi:
        return None
    return u_lo, u_hi, v_lo, v_hi
```

Implementation note found by mock: pass 1 must test ONLY the top/bottom faces at the extreme columns — the full 4-face test can never pass at u∈{0,W} (those pixels lie ON the side face planes), which collapses the rect to None everywhere.

### A1b. New poser in `isaac_datagen/src/isaac_datagen/posers.py`

No registration needed — `posers.get()` (`posers.py:11`) resolves classes by `getattr`; `capture.py:31` instantiates by name with `**runtime.pose_generation_policy_args`.

Load-bearing constraint: `OptFlowWriter.write` → `obsmask_from_data` raises `ValueError("write() called with no labeled instances — expected ≥1")` (`reference_seg_writer.py:81-82`) and **nothing catches it** — one target-out-of-frame frame aborts the whole 200-frame render. The design is loop-free, construct-then-assert: sample WHERE the target should land, build the rotation that puts it there. No retry cap, no silent fallback — the exact cone test is a fail-loud assert.

AFTER (append after `LookAtPoser`; plus `from scipy.spatial.transform import Rotation as R` to imports — scipy already a dep via pose_utils):

```python
class DecenteredLookAtPoser:                                # NEW
    """LookAtPoser halo + decentering: the target's grasp origin lands at a pixel sampled
    uniformly over the frame rect eroded so the whole object_radius sphere stays visible,
    plus roll about the target ray (visibility-invariant by construction).

    Offsets are drawn in ONE generate_random_offsets call before any decentering draw, so
    under the same seed camera POSITIONS are identical to LookAtPoser's — clean A/B vs the
    existing pools. Close-ups whose eroded rect collapses stay centered (defined policy,
    matches baseline behavior; ~10% of frames at object_radius 0.25 over the pool halo box).
    """

    def __init__(self, xrange, yrange, zrange, intrinsics_path, resolution,
                 object_radius, margin_deg=1.0, max_roll_deg=15.0):
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange
        self.K = np.load(intrinsics_path)                   # fail-loud: no default intrinsics
        self.resolution = tuple(resolution)
        self.object_radius = float(object_radius)
        self.margin = np.radians(margin_deg)
        self.max_roll = np.radians(max_roll_deg)
        self.normals = frustum_normals(self.K, self.resolution)

    def __call__(self, num_frames: int) -> np.ndarray:
        # ONE call, BEFORE any decentering draw: identical position stream to LookAtPoser
        offsets = generate_random_offsets(self.xrange, self.yrange, self.zrange, num_frames)
        return np.array([self._decentered(off) for off in offsets])

    def _decentered(self, off: np.ndarray) -> np.ndarray:
        pose = look_at(np.zeros(3), off)                     # CV cam2world, z at the target
        ang_r = np.arcsin(min(self.object_radius / np.linalg.norm(off), 1.0)) + self.margin
        rect = erode_frame_rect(self.K, self.resolution, ang_r)
        if rect is None:                                     # close-up: object can't fit off-center
            return cv2opengl(pose)                           # defined policy: centered, as today
        uv = (np.random.uniform(rect[0], rect[1]), np.random.uniform(rect[2], rect[3]))
        d = pixel_direction(self.K, uv)
        assert cone_in_frustum(d, ang_r, self.normals), f"eroded rect violated at {uv}"
        r_point, _ = R.align_vectors([[0.0, 0.0, 1.0]], [d])         # maps d -> optical axis
        r_roll = R.from_rotvec(np.random.uniform(-self.max_roll, self.max_roll) * d)
        pose[:3, :3] = pose[:3, :3] @ (r_point * r_roll).as_matrix() # target ray -> pixel uv
        return cv2opengl(pose)
```

Why this shape (user review of the first draft): an earlier rejection-sampling version (draw jitter angles, project a 64-point sphere, retry up to N_TRIES, silently fall back) was rejected as hacky — arbitrary cap, silent behavior change, tunables as class constants. Constructing the placement directly needs no loop; the sphere-visibility condition `d·n ≥ sin(ang_r)` against the 4 face normals is *exact* for a sphere (tangent half-angle `asin(r/dist)`, not `atan`), so the point cloud disappears; roll about the target ray cannot move the target's pixel or the cone, so it needs no bound beyond `max_roll_deg`.

### A2. 4 NEW configs in `isaac_datagen/src/isaac_datagen/configs/`

`emptyworld-optflow-snacks-kshot-{snack031,snack033,snack034,snack035}-1inst-decentered.yaml` — copies of the existing emptyworld `-1inst` siblings except three deltas (the other 3 classes substitute the class token in `dataset_dir`, the `DisablePhysics` pattern, and the `RegexFilter` value):

```yaml
# ... header comment: decentered variant of <class>-1inst; same seed + halo draw order ->
#     identical camera positions, orientation is the only camera delta (A/B) ...
dataset_dir: datasets/snack031-1inst-decentered   # DELTA (a): NEW dir, MUST pre-exist
pose_generation_policy: DecenteredLookAtPoser     # DELTA (b): was LookAtPoser
pose_generation_policy_args:
  xrange: ${xrange}
  yrange: ${yrange}
  zrange: ${zrange}
  intrinsics_path: ${intrinsics_path}             # DELTA (c): sibling interpolation (top-level key exists)
  resolution: [1920, 1080]                        # DELTA (c): matches ZedMini defaults
  object_radius: 0.25                             # DELTA (c): bounding sphere, per-class tunable
  margin_deg: 1.0                                 # DELTA (c)
  max_roll_deg: 15.0                              # DELTA (c)
# everything else byte-identical: seed 1001, idx 0, num_frames 200, scene_builder build_scene,
# grasp_frames catalog, DisablePhysics(<class>), RegexFilter '^model_<class>/',
# expanded-refseg-v2 lighting + jitter, exposure block, xrange [0.3,2.0] yrange ±2.0 zrange ±0.7
```

(`pose_generation_policy_args` is a free-form dict on `RuntimeConfig` — `runtime_config.py:78` — so new keys pass validation; `RuntimeConfig.__post_init__` asserts `dataset_dir` exists, hence the mkdir stage. Launch CWD is `src/isaac_datagen/` per the prior chains.)

### A3. Staged regen pipeline (tesu, 2-GPU chains — same conventions as `launch-logs/gpu{0,1}-chain.sh` / `bake-gpu{0,1}.sh`)

1. **mkdir** the 4 NEW dirs: `mkdir src/isaac_datagen/datasets/{snack031,snack033,snack034,snack035}-1inst-decentered`
2. **Render ×4** — `launch-logs/gpu{0,1}-chain-decentered.sh` (copy the `gpu0-chain.sh` pattern: cd to `src/isaac_datagen`, `CUDA_VISIBLE_DEVICES`, `env -u PYTHONPATH uv run isaac-datagen configs/<cfg>.yaml idx=0`, nohup logs in `launch-logs/`); 2 configs per GPU.
3. **Verify renders**: 200/200 frames per pool; one `[MUT] DisablePhysics(<class>): disabled 1 rigid body(ies)` per log; zero `no labeled instances` / tracebacks; `runtime.yaml` reads `pose_generation_policy: DecenteredLookAtPoser`.
4. **Bake** `CleanDiftFinetunedFpn` from the `isaac_datagen/` repo root (descriptor-config ckpt path is CWD-relative): per pool `env -u PYTHONPATH uv run python -m isaac_datagen.migrate_descriptors_backbone add-backbone src/isaac_datagen/datasets/<class>-1inst-decentered ../reference_matching/src/reference_matching/configs/fpn_cleandift_finetuned_123.yaml --device cuda:N` (verify exact CLI against `launch-logs/bake-gpu0.sh` before launching).
5. **Squash** from `segmentation/`: `uv run m2f-squash-vis --out datasets/filtered/vis030 --min-visibility 0.30 <the 4 new pools>` — expect 0 instances dropped; `squash_meta.yaml` provenance per dir.
6. **Sync-script edit** — `sync-filtered-to-psc.sh` (workspace root): append 4 lines after the existing solo-pool block:
   ```bash
   	segmentation/datasets/filtered/vis030/snack031-1inst-decentered # DecenteredLookAtPoser variants of the solo pools
   	segmentation/datasets/filtered/vis030/snack033-1inst-decentered
   	segmentation/datasets/filtered/vis030/snack034-1inst-decentered
   	segmentation/datasets/filtered/vis030/snack035-1inst-decentered
   ```
7. **Sync**: `./sync-filtered-to-psc.sh --delete` (needs a live psc-data ControlMaster: `ssh -fN psc-data` first).
8. **Checksum dry-run**: rsync `-n -c` must itemize zero diffs on the 4 new pools; remote `runtime.yaml` sanity-read.

---

## Track B — train-time translate aug, m2f-gligen path only

### B1. `vision_core/src/vision_core/transforms.py` — public `build_pipeline` seam

BEFORE (`transforms.py:235-242`):
```python
def create_transforms(config: dict[str, Any]) -> dict[str, v2.Compose]:
    tcfg = config["transforms"]
    train_specs = tcfg["train_pipeline"]
    eval_specs = tcfg["eval_pipeline"]
    return {
        "train": v2.Compose([_instantiate(spec) for spec in train_specs]),
        "eval": v2.Compose([_instantiate(spec) for spec in eval_specs]),
    }
```
AFTER:
```python
def build_pipeline(specs: list) -> v2.Compose:                 # NEW public seam: [{name,args}] -> Compose
    return v2.Compose([_instantiate(spec) for spec in specs])  # names resolve via _resolve_class:
                                                               # this module, then torchvision v2 —
                                                               # stock RandomAffine is YAML-selectable

def create_transforms(config: dict[str, Any]) -> dict[str, v2.Compose]:
    tcfg = config["transforms"]
    return {
        "train": build_pipeline(tcfg["train_pipeline"]),       # refactor onto the seam
        "eval": build_pipeline(tcfg["eval_pipeline"]),
    }
```
Policy: segmentation reuses the public seam only — never imports `_instantiate`.

### B2. `segmentation/src/segmentation/dataset.py` — `DataConfig` field + fixed `build_instance_transform`

`DataConfig`: add after `resize_max_size` (defaults keep every existing config loading unchanged under struct validation):
```python
    train_pipeline: list = field(default_factory=list)   # NEW: train-only [{name,args}] steps after Resize
```

BEFORE (`dataset.py:330-341` — the `transforms_spec` branch is dead AND buggy: `create_transforms` returns a dict, so `list()` yields the key strings; all three call sites — `train_m2f.py:404`, `train_m2f_closed.py:335`, `segmenter.py:274` — are 2-arg):
```python
    if transforms_spec:
        from vision_core.transforms import create_transforms
        steps += list(create_transforms(transforms_spec))
```
AFTER (param renamed `transforms_spec` → `pipeline_specs`; safe, no 3-arg callers):
```python
    if pipeline_specs:
        from vision_core.transforms import build_pipeline
        steps += build_pipeline(pipeline_specs).transforms   # actual transform instances now
```

### B3. `segmentation/src/segmentation/train_m2f.py` — train_tf vs eval_tf

- Ctor (`train_m2f.py:337`): add defaulted `train_pipeline=()` param; `self.train_pipeline = list(train_pipeline)`.
- `from_config` (`train_m2f.py:370`): add `train_pipeline=cfg.train_pipeline` beside the resize kwargs.
- `setup` — BEFORE (`train_m2f.py:404-410`):
```python
        tf = build_instance_transform(self.rese, self.remax)
        if tf is not None:
            self.train_ds = TransformedInstanceSegDataset(self.train_ds, tf)
            self.val_ds = TransformedInstanceSegDataset(self.val_ds, tf)
        if self.real_test_path is not None:
            rv = RenderDirInstanceSegDataset(self.real_test_path, descriptor=self.descriptor, ref_scales=self.ref_scales)
            self.real_ds = TransformedInstanceSegDataset(rv, tf) if tf is not None else rv
```
  AFTER:
```python
        eval_tf = build_instance_transform(self.rese, self.remax)                        # resize only: val/real stay clean
        train_tf = build_instance_transform(self.rese, self.remax, self.train_pipeline)  # + train-only aug
        if train_tf is not None:
            self.train_ds = TransformedInstanceSegDataset(self.train_ds, train_tf)
        if eval_tf is not None:
            self.val_ds = TransformedInstanceSegDataset(self.val_ds, eval_tf)
        if self.real_test_path is not None:
            rv = RenderDirInstanceSegDataset(self.real_test_path, descriptor=self.descriptor, ref_scales=self.ref_scales)
            self.real_ds = TransformedInstanceSegDataset(rv, eval_tf) if eval_tf is not None else rv
```
Order is already correct: the train-only background compositor (`setup` lines 411-416) wraps `train_ds` AFTER the transform wrapper, so `RandomAffine(fill=0)`'s alpha-0 borders get background-filled. `TransformedInstanceSegDataset.__getitem__` (`dataset.py:308-327`) applies the transform JOINTLY to `(Image(s.rgb), Mask(s.targets["masks"]))` — v2 shares params across the pair (mock-proven). No point/box coords on this path (M2F is not point-prompted). `train_m2f_closed.py` / `segmenter.py` untouched by design.

### B4. NEW fine-tune arm config

`segmentation/src/segmentation/configs/finetune_grid/ftgrid-lwf-f0-n50-shift40.yaml` (NEW thin delta; the `base:` include mechanism is vision_core `load_config`; the `lwf:` block inherited from `_ftgrid-lwf-base.yaml` routes it to `m2flwf`). Existing arms are NOT edited — they're reproducible ablation arms.

```yaml
# Shift-aug arm: ftgrid-lwf-f0-n50 + train-only random translate (Track B of the decentering plan).
# translate 0.4 < 0.5 can never fully evict a centered object (mock: 500 draws, 0 empty masks);
# fill=0 zeroes alpha at revealed borders -> the train-only background compositor fills them.
base: ftgrid-lwf-f0-n50.yaml

data:
  train_pipeline:
    - {name: RandomAffine, args: {degrees: 0, translate: [0.4, 0.4], fill: 0}}
```

A follow-up arm may additionally point `data.paths` at the Track-A `-decentered` pools — the tracks compose.

---

## Verification

**Track A — pre-render.** NEW `isaac_datagen/.docs_claude/one_off_tests/mock_decentered_poser.py` (dir exists; runs sim-free — posers.py never imports Isaac). Re-creates the planning-session numeric mock. With the real `zed_K.npy` and the pool halo box (`x [0.3,2.0], y ±2.0, z ±0.7`), assert:
- seed-matched positions: under the same seed, `DecenteredLookAtPoser(...)(200)[:, :3, 3] == LookAtPoser(...)(200)[:, :3, 3]` exactly;
- pointing exactness: the grasp origin projects onto the sampled pixel within 0.5 px;
- ground truth vs the exact test: a dense 256-point sphere never leaves the frame, **including frames pinned to the eroded-rect corners** (worst case);
- over ~5000 frames: origin-projection std ≈ (432, 156) px, coverage ≈ x[212,1700] y[208,871], centered close-ups ≈ 10% (object_radius 0.25).

**Track A — post-render.** NEW one-off `centroid_spread_1inst.py`: per-frame target-mask centroid from the rendered `cid_mask` over each new pool AND its centered sibling — new std must be hundreds of px, old ≈ 0. Contact sheet per pool (`launch-logs/montage-<class>-1inst-decentered.jpg`, existing convention). Then the A3 checkpoints: 200/200 + `[MUT]` lines → bake clean → squash 0 drops → sync checksum zero diffs.

**Track B.**
- NEW smoke `segmentation/.docs_claude/one_off_tests/smoke_train_pipeline_shift.py`: synthetic RGBA Image + Mask pair with a centered blob through `build_instance_transform(edge, max, [{RandomAffine...}])`; assert identical shift on image/mask (joint params), alpha-0 borders, tv_tensor types preserved, 0 empty masks over 500 draws.
- No-behavior-change guard: 2-arg `build_instance_transform` output unchanged; existing arms load with `train_pipeline` defaulting to `[]`.
- m2f smoke run: `uv run m2flwf .../ftgrid-lwf-f0-n50-shift40.yaml` (commit first or `--allow-dirty`) until the first viz dump — train panels show off-center targets with composited (non-black) borders, val panels stay centered; then kill (kill the python process group, not the nohup wrapper).
- End-to-end A/B readout: the arm's fixed-budget fine-tune vs the non-shift `ftgrid-lwf-f0-n50` arm on the frozen set2 benchmark.

---

## Decision log

- **Seed 1001 reused**: all offsets drawn in one `generate_random_offsets` call before any decentering draw → the 200 halo positions per pool are bit-identical to the centered pools; orientation is the only camera delta (clean A/B). Downstream per-frame RNG consumers (lighting jitter) see a shifted stream from the extra draws — accepted.
- **New dirs (`-1inst-decentered`), not in-place**: user's call; both variants must coexist for A/B, and the existing pools are live inputs of reproducible fine-tune arms + PSC mirrors. No parking needed.
- **Construct, don't reject-sample** (user review of draft 1): sampling jitter angles then retrying against a visibility check needed an arbitrary try cap and a silent centered fallback — hacky. Inverting it — sample the target's pixel placement uniformly over an exactly-eroded rect, build the rotation that realizes it — is loop-free and turns the check into a fail-loud assert. Two analytic facts make it exact: sphere-in-frustum is `d·n_i ≥ sin(ang_r)` per face (tangent cone `asin(r/dist)`, not `atan`), and roll about the target ray leaves both the target pixel and the cone invariant, so roll needs no geometric bound.
- **Sphere-visibility over origin-only**: origin-in-frame only guarantees the writer's ≥1-pixel survival; keeping the whole sphere visible keeps visibility ≈ baseline so vis030 drops nothing and pools stay 200-frame.
- **Frustum helpers in vision_core**: `frustum_normals` / `cone_in_frustum` / `pixel_direction` / `erode_frame_rect` are generic camera geometry (user review: reusable helpers don't belong inside a poser); they sit beside `look_at`/`cv2opengl` in `pose_utils.py`. The poser keeps only policy (halo box, radius, margins, roll range).
- **`object_radius: 0.25`**: conservative bounding sphere at raw catalog scale; a per-class config knob, not a poser default — physical args are explicit, fail-loud; ~10% of halo draws are close-ups that stay centered at this radius.
- **`translate: [0.4, 0.4]`, `fill: 0`**: <0.5 can never fully evict a centered object (0 empty masks / 500 draws); fill 0 zeroes alpha so the existing compositor (already ordered after the transform) fills borders with random backgrounds, not black.
- **Track B on the m2f-gligen path only** (user's choice): closed-baseline and SAM-gligen paths untouched.

## Critical files

- `isaac_datagen/src/isaac_datagen/posers.py` (+4 new configs, chain/bake scripts, sync-filtered-to-psc.sh)
- `vision_core/src/vision_core/pose_utils.py` (NEW frustum helpers), `vision_core/src/vision_core/transforms.py`
- `segmentation/src/segmentation/dataset.py`, `segmentation/src/segmentation/train_m2f.py` (+1 new ftgrid arm)

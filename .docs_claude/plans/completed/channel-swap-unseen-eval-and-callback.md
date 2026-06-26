# Plan: channel-swap "unseen 0-shot" eval set + verifier eval-callback wiring

> **STATUS: COMPLETED 2026-06-26.** All four changes shipped and validated end-to-end on a
> `shelf-optflow/render000` slice (frames [0,2)) with BOTH `DiftDescriptor` and the real
> `CleanDiftFinetunedDescriptor` (fullgrid config): frames renumbered 0..1 with R/B-flipped obs, refs
> flipped + descriptors/PCA refilled via `add-backbone`, grid proposer (576×2 cls) + inlier labels
> written, then `viz_unseen_batch.py` rendered a sane panel (blue obs+ref, full 576 grid, GT scatter,
> recomputed `ref_tok` PCA). Delivered: `vision_core.SwapRedBlue`; `isaac_datagen/make_unseen.py`
> (`isaac-datagen-unseen`); `segmentation` `select_unseen_batches` + parameterized
> `FixedBatchVizCallback.wandb_key` + `VizConfig.unseen_paths/unseen_batch_count` + `train_once` wiring +
> `viz_unseen_batch.py`.
>
> **Deviations from the plan as written:**
> - **`vision_core.sampling.PassThrough` already existed** (live in `verifier-training-cleandift-finetuned-fullgrid.yaml`,
>   576 + PassThrough). Change 1b was dropped; the unseen eval inherits it from the config.
> - **CWD fix (not in the plan; found at test time):** `load_config` resolves all relative paths against
>   the process CWD with no chdir, and the workspace has two bases — isaac configs anchor
>   `descriptor_config_path`/`proposer_config_path` at `src/isaac_datagen/`, but the shared descriptor
>   config's `cleandift_ckpt: ../checkpoints/...` is anchored at the `isaac_datagen/` repo root. So
>   `make_unseen` runs the `add-backbone` subprocess from `REPO_ROOT` (with ABSOLUTE dataset+config args)
>   while phase-2 stays in `src/isaac_datagen`. Run `isaac-datagen-unseen` from `src/isaac_datagen/`.

## Context

The verifier (stage 2) trains/vals on a frozen split over the same render dirs — there is **no held-out,
distribution-shifted set** to watch generalization on. Build the *easiest* unseen 0-shot test: take a
slice of an existing phase-1 render dir, **swap R↔B** (tv2) on both the obs frames and the per-class
references (recomputing the catalog's clean-DIFT descriptors + PCA so both sides live in the swapped
domain), re-run phases 2+3 to get a *proper* labeled dir, and feed one fixed batch of it to the
verifier's `FixedBatchVizCallback` under its own `viz/unseen` gallery — **kept out of `data.paths` / the
split manifest**. First of a transform family → implement the transform as a small named type, not a
flag (`[[no-flag-driven-variants]]`).

## Data flow

```
phase-1 render dir (OptFlowSample frames + OptFlowMetadata catalog)
  │ isaac-datagen-unseen <config> <src_dir> <start> <end>            (NEW)
  │  1+2. (de)serialize frames [start,end) → RENUMBER 0..N-1, flip R/B on obs   [SwapRedBlue]
  │  3.   copy catalog with FLIPPED class_to_ref + EMPTY descriptors → reuse `add-backbone` to refill
  │  4.   isaac-datagen-proposals  (phase 2 — proposer sees flipped obs ↔ flipped ref)
  │  5.   isaac-datagen-inliers     (phase 3 — labels from cid_mask)
  ▼ dst = a normal labeled render dir, NOT in data.paths / manifest
  │ verifier training: select_unseen_batches(dst) → pinned CPU batch
  ▼ FixedBatchVizCallback (wandb_key="viz/unseen") → PNGs + gallery every N steps
    plus viz_unseen_batch.py → ImageInlierSample.visualize() panels (no checkpoint)
```

## Change 1 — `vision_core/transforms.py`: `SwapRedBlue`

Named v2 transform; auto-discovered by `_resolve_class`/`create_transforms`. (No sampling change needed —
`vision_core.sampling.PassThrough` already exists and is what the live full-grid config uses; the unseen
eval inherits it via the config in Change 3.)
```python
class SwapRedBlue(v2.Transform):
    """Swap R and B of an (…,C,H,W) image (C in {3,4}); G + alpha/extras untouched. Easiest unseen shift."""
    def transform(self, inpt, params):
        if not isinstance(inpt, tv_tensors.Image) or inpt.shape[-3] < 3:
            return inpt
        perm = list(range(inpt.shape[-3])); perm[0], perm[2] = perm[2], perm[0]
        return tv_tensors.Image(inpt[..., perm, :, :])
```
Add `SwapRedBlue` to the `transforms.py` row in `vision_core/CLAUDE.md`.

## Change 2 — `isaac_datagen/make_unseen.py` (new) + console script

Thin orchestrator; invokes sibling tools' **public** module CLIs via `python -m`.

```python
"""Build a channel-swapped 'unseen 0-shot' render dir from frames [start,end) of an existing phase-1 dir.
Usage: isaac-datagen-unseen <config.yaml> <source_render_dir> <start> <end> [key=value ...]
  config gives the OUTPUT (dataset_dir+idx → dst), proposer/descriptor configs, devices — the SAME config
  you train the verifier with (descriptor_config_path MUST name the verifier's backbone)."""
import shutil, subprocess, sys
from pathlib import Path
from torchvision.transforms import v2
from vision_core.datastructs import OptFlowSample, OptFlowMetadata, SubfolderDict
from vision_core.transforms import SwapRedBlue
from isaac_datagen.runtime_config import load_config

def _run(*argv):   # this module's OWN helper: run a sibling tool's PUBLIC cli via `python -m` in our venv
    subprocess.run([sys.executable, "-m", *argv], check=True)

def main():
    src = Path(sys.argv[2]); start, end = int(sys.argv[3]), int(sys.argv[4])
    runtime = load_config(sys.argv[1], sys.argv[5:])
    dst = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"
    if dst.exists(): sys.exit(f"dst {dst} exists — pick a fresh idx/dataset_dir")
    dst.mkdir(parents=True)
    swap = v2.Compose([SwapRedBlue()])
    # 1+2. subset+RENUMBER: shutil.copy keeps SOURCE numbering (desyncs a subset), so (de)serialize. Read
    # the FULL per-frame OptFlowSample (every geometry field the reproj gate reads), flip its obs, write
    # at the new contiguous index. Phase-2/3 products NOT copied — they re-run on flipped obs.
    for dst_idx, src_idx in enumerate(range(start, end)):
        s = OptFlowSample.deserialize(src_idx, src)
        s.obsmask.obs = swap(s.obsmask.obs)
        s.serialize(dst_idx, dst)
    # 3. catalog: copy verbatim (id dicts/intrinsics flip-invariant) but FLIP class_to_ref and EMPTY the
    # two descriptor SubfolderDicts, then let the existing `add-backbone` tool re-encode the flipped refs
    # → descriptors + PCA (same path datasets normally get this backbone; no hand-rolled encode/PCA, and
    # emptying drops any stale extra backbones cleanly).
    md = OptFlowMetadata.deserialize(0, src); mm = md.obsmaskmeta
    mm.class_to_ref = {cls: swap(ref) for cls, ref in mm.class_to_ref.items()}
    mm.class_to_descriptors = SubfolderDict(); mm.principal_components = SubfolderDict()
    md.serialize(0, dst)
    for name in ("runtime.yaml", "descriptor.yaml", "lighting_log.json"):
        if (src / name).exists(): shutil.copy(src / name, dst / name)   # copy FIRST: descriptor.yaml is add-backbone's key
    # re-encode flipped refs → descriptors + PCA via the tool's PUBLIC cli (not its private _add_backbone)
    _run("isaac_datagen.migrate_descriptors_backbone", "add-backbone",
         str(runtime.dataset_dir), str(runtime.descriptor_config_path), "--device", runtime.descriptor_device)
    # 4+5. phases 2/3 through their public module entrypoints (same modules behind the console scripts)
    _run("isaac_datagen.add_proposals", sys.argv[1], f"dataset_dir={runtime.dataset_dir}", f"idx={runtime.idx}")
    _run("isaac_datagen.add_inlier_data", str(dst), "--eps", str(runtime.inlier_border_eps))
```

Notes (READ first): all sub-steps go through **public** seams — `python -m` CLIs, never another module's
underscore-privates. The `descriptor.yaml` copy precedes `add-backbone` (its backbone key; must match the
source's). Confirm an empty `SubfolderDict()` serializes a usable empty manifest that `for_each_render_dir`
/`add_backbone_to_subfolder` accept (marker `class_to_descriptors/` must exist with a `[]` manifest); else
keep the copied manifest and delete the backbone key before re-adding. `add-backbone` iterates render dirs
under `dataset_dir` — keep the unseen `dataset_dir` dedicated. Add `isaac-datagen-unseen =
"isaac_datagen.make_unseen:main"` to pyproject `[project.scripts]` + a row to the isaac_datagen CLAUDE.md.

## Change 3 — `segmentation`: select an unseen batch + own wandb gallery

**Downsample policy is read from the verifier config, never hardcoded.** The live full-grid config
(`verifier-training-cleandift-finetuned-fullgrid.yaml`) trains on the **full 576-anchor `GridProposal`
grid**: `uniform_proposals: {name: PassThrough}`, `max_proposals: 576`, posenc off — the verifier
grid-samples the frozen provider as a coarse strided activation map (`RefGridPassthrough` = literal
all-pairs obs↔ref correlation; a planned conv sibling reads the obs as a dense volume + `grid_sample`s the
proposal `xy`). The unseen batch must traverse the *same* policy + K; it inherits them by reading
`cfg.data.{max_proposals,uniform_proposals,descriptor}` and `.deterministic()`-ing the policy (identity
for `PassThrough` → the full grid, train and val alike), exactly as `verifier/eval.py:full_dataloader`.
Holds for any policy the config names, and the same `ImageInlierSample` contract feeds both the current
attention `Verifier` and a future conv sibling. (Plans: `proposer-visible-px-gate-grid-proposals`,
`verifier-conv-reframe-posenc-ablation`.)

`select_unseen_batches` reuses `full_dataloader` (builds the dataset object over the dir):
```python
def select_unseen_batches(paths, k, uniform, descriptor, n_batches=1):
    """Build the dataset over the unseen dir(s) (NOT in the split) and pin the first n_batches as CPU
    batches. Deterministic policy (PassThrough=identity → full grid) — same batch every fire."""
    from segmentation.verifier.eval import full_dataloader
    loader = full_dataloader(paths, k=k, uniform=uniform.deterministic(),
                             batch_size=1, num_workers=0, descriptor=descriptor)
    out = []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        out.append((f"unseen{i}", batch.detach().cpu().clone()))
    return out
```

**Give `FixedBatchVizCallback` its own gallery key** (otherwise unseen + train/val panels merge into the
single hardcoded `viz/fixed`):
```python
# before:  def __init__(self, out_dir, every_n_train_steps, batches, log_wandb=True):
# after:   def __init__(self, out_dir, every_n_train_steps, batches, log_wandb=True, wandb_key="viz/fixed"):
#              self.wandb_key = wandb_key
# in _render: log_images_to_wandb(trainer, self.wandb_key, images, captions)   # was "viz/fixed"
```
Existing call sites keep the default → unchanged behavior.

Add `unseen_paths: list[str] = []` + `unseen_batch_count: int = 1` to the verifier `viz` config dataclass,
and a default `unseen_paths: []` to the `viz:` block of **`_verifier-base.yaml`** (the base every leaf
config includes — `[[verifier-config-base-includes]]`). Then in `train_once`, beside the `cfg.viz.fixed_batch`
block:
```python
if cfg.viz.unseen_paths:
    unseen = select_unseen_batches(cfg.viz.unseen_paths, cfg.data.max_proposals,
                                   build_uniform_policy(cfg.data.uniform_proposals),
                                   cfg.data.descriptor, cfg.viz.unseen_batch_count)
    callbacks.append(FixedBatchVizCallback(str(run_dir / "viz_unseen"), cfg.viz.every_n_train_steps,
                                           unseen, cfg.viz.log_wandb, wandb_key="viz/unseen"))
```
Document that `viz.unseen_paths` must be **outside** `data.paths` + the manifest, and `cfg.data.descriptor`
must match the backbone `isaac-datagen-unseen` recomputed (`CleanDiftFinetunedDescriptor` for fullgrid).

## Change 4 — `viz_unseen_batch.py` (checkpoint-free viz test)

`segmentation/.docs_claude/one_off_tests/viz_unseen_batch.py` — loads the SAME verifier config, builds
the batch, renders each sample via `ImageInlierSample.visualize()`. Proves the dir loads as a proper
datasample and the swap + recomputed catalog are visibly sane; no model needed.
```python
# usage: viz_unseen_batch.py <unseen_render_dir> <verifier_config.yaml> [out_dir]
cfg = load_check_config(sys.argv[2], LightningVerifierConfig)
uniform = build_uniform_policy(cfg.data.uniform_proposals)         # select_unseen_batches() .deterministic()s it
(tag, batch), = select_unseen_batches([sys.argv[1]], cfg.data.max_proposals, uniform,
                                      cfg.data.descriptor, n_batches=1)
for i in range(batch.batch_size[0]):
    plt.imsave(out / f"{tag}_s{i}.png", batch[i].to_sample().visualize())
```
Confirm `visualize()` return type, `[i].to_sample()`, and `load_check_config` signature at impl.

## Verification

1. **Transform**: `SwapRedBlue()` on a known RGBA tensor swaps ch 0↔2, keeps ch 1 + alpha; masks pass through.
2. **Build** frames [5,10) — isaac config must use the SAME proposer (`grid_proposal.yaml`) + descriptor
   (`cleandift_finetuned.yaml`) the fullgrid training uses, so the unseen dir matches:
   `cd isaac_datagen && env -u PYTHONPATH uv run isaac-datagen-unseen <grid+cleandift config> <existing_render_dir> 5 10 dataset_dir=datasets/unseen idx=0`.
   Confirm 5 frames **renumbered 0..4**, flipped `obs/`, recomputed catalog (`CleanDiftFinetunedDescriptor`), fresh `proposals/ labels/ stats/`; eyeball one obs + ref.
3. **Visualize (no ckpt)**: `viz_unseen_batch.py isaac_datagen/datasets/unseen/render000 segmentation/src/segmentation/verifier/configs/verifier-training-cleandift-finetuned-fullgrid.yaml` → panels show flipped obs + GT scatter (full 576 grid, `PassThrough`) ‖ PCA-RGB.
4. **Train wiring**: set `viz.unseen_paths: [.../datasets/unseen/render000]` (in the fullgrid config), short `uv run vtrain verifier-training-cleandift-finetuned-fullgrid.yaml`; confirm `runs/<id>/viz_unseen/*.png` at cadence + a **`viz/unseen`** wandb gallery, dir absent from train/val.

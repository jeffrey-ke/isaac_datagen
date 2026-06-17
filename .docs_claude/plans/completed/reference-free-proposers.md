# Add reference-free proposers: `GridProposal` + `RandomProposal`

## Context

Today every Stage-1 proposer is **reference-conditioned** (ALIKED/LightGlue or GIM ref↔obs
matching). We want two **reference-free baseline proposers** that emit candidate point-prompts
independent of any reference:

- `GridProposal` — a `num_w × num_h` meshgrid of cell-center pixel coordinates over the observation.
- `RandomProposal` — `num_points` uniformly-random pixel coordinates over the observation.

These are useful for the verifier's anchor-box classification (Stage 2): a grid gives **dense,
uniform** anchor coverage and random gives **stochastic** coverage, neither biased by a matcher's
recall — good for generating `PreImageInlierSample` training data where every proposed point is
labeled inlier/outlier against the class union mask.

**Coordinate-convention question (answered):** proposers output **pixel** xy, never normalized.
The whole pipeline carries pixel coords (`proposal.py` docstring line 5 "emits pixel coordinates";
`Features.locations` "xy pixel coords"; `vision_core` `ReferenceSegSample.proposal_coordinates`
"(K, 2) pixel-space xy"; `coords_in_mask` indexes masks directly with `coords[:,0].long()`).
Normalization to `[0,1]`/`[-1,1]` happens at exactly **one inference site each** —
`verifier.forward` (`xy01 = (proposals + 0.5)/[W,H]`) and `segmenter.normalize_points` — both of
which take pixel input. So the new proposers must output **pixel** xy, matching all existing ones.
No downstream consumer wants normalized coords from a proposer.

## Design decisions (from clarification)

- **Dimensions read from the observation**, not `__init__`. Only `num_w`/`num_h` (grid) and
  `num_points` (random) are constructor params; `forward` reads `H×W` from `observation.shape`.
- **Cell-centered** grid placement: `(arange(n)+0.5) * size/n` — no points on the exact image edge.
- **Seeding** for `RandomProposal` rides the process-global RNG (`vision_core.seed_everything`,
  which calls `torch.manual_seed`). The proposer takes **no seed param**. See "Seeding" below.

## Registry: no changes needed

`reference_matching/proposal.py` resolves proposers by **module-namespace lookup** —
`get(name)` (`proposal.py:382`) does `getattr(module, name)`, and `from_config` (`:405`) builds
`get(cfg["name"])(reference_image=ref, **cfg["args"])` with `extractor`/`matcher`/`post_filter`
all optional. A module-level class with an arg-only config drops in with **zero** changes to
`get`/`from_config`/`ProposalConfig`. The only constraint: `__init__` must accept the
`reference_image=` kwarg that `from_config` always passes (`:430`) — both new classes accept and
ignore it, exactly like the `reference_image=None` tail param on every existing proposer.

## Changes

### 1. `reference_matching/src/reference_matching/proposal.py` — two new classes

Add near the other orchestrators (after `KeypointMatchProposal`, before `get`). Both subclass
`nn.Module`, mirror the `forward(observation, reference=None) -> list[(xy (M,2), scores (M,))]`
contract, return **pixel** xy with uniform `ones` scores (the datagen caller discards scores —
`add_proposals.py:94` `xy, _scores = ...`), and produce coords on `observation.device`.

```python
class GridProposal(nn.Module):
    """Reference-free baseline: a num_w×num_h meshgrid of cell-center pixel coords over the
    observation. Ignores the reference. Dense uniform anchor coverage for the verifier."""

    def __init__(self, num_w: int, num_h: int, reference_image: tv_tensors.Image | None = None):
        super().__init__()                 # reference_image ignored (reference-free); accepted
        self.num_w = num_w                 # because from_config always passes reference_image=
        self.num_h = num_h

    def forward(
        self,
        observation: tv_tensors.Image,
        reference: tv_tensors.Image | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        _check_batched_image(observation)
        B, _, H, W = observation.shape
        xs = (torch.arange(self.num_w, device=observation.device) + 0.5) * (W / self.num_w)
        ys = (torch.arange(self.num_h, device=observation.device) + 0.5) * (H / self.num_h)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(torch.float32)  # (num_h*num_w, 2)
        scores = torch.ones(xy.shape[0], device=observation.device)
        return [(xy, scores) for _ in range(B)]    # same grid for every batch element


class RandomProposal(nn.Module):
    """Reference-free baseline: num_points uniform-random pixel coords over the observation,
    fresh per call (rides the global RNG — seed via vision_core.seed_everything). Ignores ref."""

    def __init__(self, num_points: int, reference_image: tv_tensors.Image | None = None):
        super().__init__()
        self.num_points = num_points

    def forward(
        self,
        observation: tv_tensors.Image,
        reference: tv_tensors.Image | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        _check_batched_image(observation)
        B, _, H, W = observation.shape
        wh = observation.new_tensor([W, H], dtype=torch.float32)
        return [
            (torch.rand(self.num_points, 2, device=observation.device) * wh,   # independent draws
             torch.ones(self.num_points, device=observation.device))           # per batch element
            for _ in range(B)
        ]
```

Also extend the module docstring's component list (around `proposal.py:25-31`) with a
"Reference-free baselines" subsection so the registry's contents stay self-documenting:

```
  Reference-free baselines  (observation -> pixel coords; reference ignored)
    GridProposal(num_w, num_h)         # cell-centered meshgrid over the observation
    RandomProposal(num_points)         # uniform-random pixel coords, fresh per call
```

### 2. New configs (mirror the minimal existing ones, e.g. `gim_dkm_proposal.yaml`)

`reference_matching/src/reference_matching/configs/grid_proposal.yaml`:
```yaml
name: GridProposal
args:
  num_w: 32
  num_h: 18
```

`reference_matching/src/reference_matching/configs/random_proposal.yaml`:
```yaml
name: RandomProposal
args:
  num_points: 512
```

No `reference_image_path`, `extractor`, `matcher`, or `post_filter` keys — `from_config` leaves
them `None` and builds `GridProposal(reference_image=None, num_w=32, num_h=18)` directly.

### 3. `isaac_datagen/src/isaac_datagen/add_proposals.py` — seed phase-2

The proposer runs in phase-2 (`add_proposals.py`), which today does **no** seeding — only phase-1
(`clean_datagen.py:82`) calls `seed_everything`. So `RandomProposal` (and any RNG inside matcher
proposers) is currently non-reproducible in phase-2. Because this pass is explicitly **shardable
by frame window** (`start_frame`/`end_frame`, "one per GPU via run_pipeline"), seed **per frame**
so a given frame's points are identical whether the pass runs whole or sharded.

Before — the per-frame loop draws with no seeding:
```python
    for idx in bar:
        if idx in done:
            continue
        om = ObsMask.deserialize(idx, render_dir)
```

After — re-seed from `effective_seed + idx` at the top of each frame (mirrors phase-1's
`seed_everything(runtime.effective_seed)`, the same `vision_core.seed_utils` import):
```python
    from vision_core.seed_utils import seed_everything
    for idx in bar:
        if idx in done:
            continue
        seed_everything(runtime.effective_seed + idx)   # per-frame: sharding-invariant RNG
        om = ObsMask.deserialize(idx, render_dir)
```

This makes phase-2 reproducible and shard-invariant for *all* proposers (a latent gap fix), and
`RandomProposal` needs no seed param. (`runtime.effective_seed` already exists — used at
`clean_datagen.py:82`.)

## What is reused (no new code)

- Registry / config loader: `proposal.get` (`:382`), `proposal.from_config` (`:405`),
  `ProposalConfig` (`:395`) — unchanged.
- `_check_batched_image` (`proposal.py:75`) for the `(B,C,H,W)` guard.
- Datagen caller `add_proposals.py` `proposer(obs_b, ref_b)[0]` (`:94`) — works unchanged
  (`ref_b` is passed and ignored; scores discarded).
- Optional FPS thinning `downsample_proposals.fps_downsample` works on the grid/random `(N,2)`
  output as-is (spatial FPS over pixel coords).
- `vision_core.seed_utils.seed_everything` (`seed_utils.py:9`).

## Verification

1. **Unit (no Isaac, under `uv run` in `reference_matching`)** — build each from its config and
   check shape/range/device on a dummy obs:
   ```
   cd /home/jeffk/repo/reference_matching
   uv run python -c "
   import torch; from torchvision import tv_tensors
   from reference_matching import proposal as P
   obs = tv_tensors.Image(torch.zeros(1,3,720,1280, dtype=torch.uint8))
   for cfg in ['grid_proposal.yaml','random_proposal.yaml']:
       m = P.from_config(f'src/reference_matching/configs/{cfg}')
       xy, s = m(obs)[0]
       assert xy.ndim==2 and xy.shape[1]==2 and s.shape[0]==xy.shape[0]
       assert xy[:,0].min()>=0 and xy[:,0].max()<1280 and xy[:,1].max()<720, 'pixel range'
       print(cfg, tuple(xy.shape), 'x[min,max]=', float(xy[:,0].min()), float(xy[:,0].max()))
   "
   ```
   Expect grid → `(576, 2)` (32×18), x spanning ~20→1260 cell-centers; random → `(512, 2)`.
2. **Determinism** — two `RandomProposal` calls bracketed by `seed_everything(0)` produce
   identical xy; without it they differ.
3. **End-to-end phase-2** — point `runtime.proposer_config_path` at `grid_proposal.yaml`, run
   `isaac-datagen-proposals <config.yaml>` on an existing render dir, confirm `proposals/` is
   written and `add_inlier_data` then labels them against `cid_mask` without range errors.
```

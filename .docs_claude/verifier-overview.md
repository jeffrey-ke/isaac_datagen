# Verifier overview: stage-2 anchor-box classification

**What it is.** Stage 2 of the three-stage reference-prompted segmentation pipeline — a learned
replacement for RANSAC/MAGSAC. Crucially it is *not* match verification: correspondence is
just an expedient to get anchor-box-like candidates, so the verifier is anchor-box
classification — "does this observation point lie on an instance of the reference's class?"
— with no 2-view geometry downstream.

**Architecture** (implemented 2026-06-05; design record at
`segmentation/.docs_claude/plans/completed/implement-verifier.md`, original pseudocode at
`isaac_datagen/verifier` — superseded in places by the as-built design):

1. **`MultiScalePointDescriptor`** (`segmentation/src/segmentation/verifier/point_descriptor.py`)
   — the RoI-Align analog. `grid_sample`s every volume of a frozen FPN provider (M2FFpn/DiftFpn
   from `reference_matching`) at the proposal coords, per-scale Linear downshifts to
   `hidden_dim`, fuses by mean over scales → `(B, N, hidden_dim)` point tokens. Contract:
   proposals arrive as **0–1 cell-center normalized xy** (out-of-range raises); the Verifier is
   the single pixel→normalized site, so one `xy01` tensor feeds both this read-out and the posenc.
2. **`PerceiverResampler`** (`vision_core/src/vision_core/nn_blocks.py`) — compresses the
   reference descriptor tokens into a fixed small set of learned cls/latent tokens
   (`num_cls=64`) via cross-attention over `concat(data_toks, cls_toks)` (Flamingo recipe on
   `nn.MultiheadAttention`/SDPA). This is the **global reference representation** — global
   rather than point-wise because you're not scoring point↔point matches, you're scoring
   point↔class membership.
3. **`Verifier`** (`segmentation/src/segmentation/verifier/verifier.py`) — point tokens are the
   SAM mask-decoder **prompt side**, the resampled ref tokens the **image side**. A learned
   ride-through cls token (SAM MaskGen recipe) is prepended to the point sequence; the stack is
   `num_layers` mask-threaded SAM `CrossAttentionBlock`s (modified-muggled-sam, branch
   `ragged-prompt-masking`: `proposals_valid_mask` threads in as `prompt_key_mask`; the
   always-valid cls slot prevents all-masked-row NaNs; both sides threaded, residuals internal)
   plus one finisher `CrossAttentionNormed` so the point side reads the final ref state
   (mirrors SAM's `final_prompt_crossattn`). Readout: encoded cls token ⋅ encoded point tokens
   + learned scalar bias (inlier-ratio prior) → per-point inlier logits `(B, N)`.

   Positional encoding: one shared `SAMV2CoordinateEncoder` (gaussian randn-initialized and
   frozen — original SAM never trains it) serves the proposal points AND the reference grid;
   the ref grid posenc is derived from `ref_tok`'s own `(h, w)` via
   `get_grid_position_encoding` and enters as the resampler's `pos` input. Decoder-side ref
   posenc is zeros — resampled latents have no spatial identity.

**Data it ingests.** `ImageInlierSample` (`vision_core/datastructs.py`): `obs` image,
`proposals` {class → (N,2) pixel xy} from stage-1 matchers (ALIKED/LightGlue or GIM, >50%
outliers), `labels` {class → (N,) bool} where True means the point lands in the class's union
mask (`cid_mask == cid` — on ANY same-class instance, hence "anchor classification" not
instance matching), plus the per-class reference descriptors via
`ObsMaskMetadata.class_to_descriptors` — **spatial `(C_ref, h, w)` = `(1280, 32, 32)`**, which
stacks directly into the forward's `ref_tok (B, C_ref, h, w)` with zero reshapes. `ref_tok` is
the precomputed frozen *input* to the resampler (the trainable resampler always runs in-graph);
the raw-ref-image path is reserved (`NotImplementedError`).

**Where the data comes from.** `isaac_datagen` renders it with Isaac Sim Replicator in phases:
rendered obs + cid masks → phase-2 `add_proposals.py` runs real stage-1 matchers to get
candidate points → phase-3 `add_inlier_data.py` labels each proposal against the union mask.
`DiftDescriptor` emits spatial `(B, 1280, 32, 32)` natively (datasets migrated 2026-06-05 via
`migrate_descriptors_spatial.py`); the 1280-d channel width is why the resampler needs an
input projection.

**Verification.** `segmentation/.docs_claude/one_off_tests/smoke_verifier.py` (8 assert-tests,
CPU, StubFpn): notably batched-vs-solo equivalence proving the `prompt_key_mask` threading
end-to-end, an invalid-slot-perturbation test with negative control, all-padding NaN guard,
and a same-data/different-grid-shape test proving the ref posenc consumes the tensor's own
shape. Run: `env -u PYTHONPATH uv run python .docs_claude/one_off_tests/smoke_verifier.py`.

**Downstream consumers.**

- **Inference:** verified (inlier) points become the point prompts fed to stage-3's
  `prompt_encoder` of the GLIGEN-wrapped SAM in `segmentation` — SAM is brittle to outlier
  prompts, which is the verifier's whole reason to exist.
- **Training (not yet built):** ImageInlierSample dataloader + masked BCE; the optimizer must
  filter `requires_grad` (frozen FPN inside the extractor, frozen posenc gaussian).

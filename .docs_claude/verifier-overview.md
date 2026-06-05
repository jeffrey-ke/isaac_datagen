# Verifier overview: stage-2 anchor-box classification

**What it is.** Stage 2 of the three-stage reference-prompted segmentation pipeline — a learned
replacement for RANSAC/MAGSAC. Crucially it is *not* match verification: correspondence is
just an expedient to get anchor-box-like candidates, so the verifier is anchor-box
classification — "does this observation point lie on an instance of the reference's class?"
— with no 2-view geometry downstream (`verifier:122-123`).

**Architecture** (pseudocode at `isaac_datagen/verifier`, partial implementation at
`segmentation/src/segmentation/verifier/point_descriptor.py`):

1. **`MultiScalePointDescriptor`** (implemented) — the RoI-Align analog. `grid_sample`s every
   volume of a frozen FPN provider (M2FFpn/DiftFpn from `reference_matching`) at the proposal xy
   coords, per-scale Linear downshifts to `hidden_dim`, fuses by mean over scales → `(B, N,
   hidden_dim)` point tokens.
2. **`PerceiverResampler`** (pseudocode, `verifier:4-15`) — compresses the variable-length
   reference descriptor tokens into a fixed small set of learned cls/latent tokens
   (`num_cls=64`, `hidden_dim=256`) via cross-attention over `concat(cls_toks, data_toks)`.
   This is the **global reference representation** — global rather than point-wise because
   you're not scoring point↔point matches, you're scoring point↔class membership
   (`verifier:115-117`).
3. **`Verifier`** (pseudocode) — point tokens (+ Fourier/SAM-style posenc of the proposal coords)
   cross-attend the resampled reference tokens through N SAM-style decoder blocks; a learned
   `logit_proj_token` einsum + scalar bias (inlier-ratio prior) produces per-point inlier logits
   `(B, N)`.

**Data it ingests.** `ImageInlierSample` (`vision_core/datastructs.py:243`): `obs` image,
`proposals` {class → (N,2)} from stage-1 matchers (ALIKED/LightGlue or GIM, >50% outliers),
`labels` {class → (N,) bool} where True means the point lands in the class's union mask
(`cid_mask == cid` — on ANY same-class instance, hence "anchor classification" not instance
matching), plus the per-class reference descriptors via `ObsMaskMetadata.class_to_descriptors`.
The forward also accepts `ref_tok` as a shortcut since the dataset precomputes reference
features.

**Where the data comes from.** `isaac_datagen` renders it with Isaac Sim Replicator in phases:
rendered obs + cid masks → phase-2 `add_proposals.py` runs real stage-1 matchers to get
candidate points → phase-3 `add_inlier_data.py` labels each proposal against the union mask.
The DIFT reference token shape (`1024,1280 → 32,32,1280`, `verifier:3`) is why the resampler
needs an input projection.

**Downstream consumers.**

- **Inference:** verified (inlier) points become the point prompts fed to stage-3's
  `prompt_encoder` of the GLIGEN-wrapped SAM in `segmentation` — SAM is brittle to outlier
  prompts, which is the verifier's whole reason to exist.
- The verifier implementation lives in `segmentation/src/segmentation/verifier/`; the
  pseudocode file in `isaac_datagen` (`verifier`) is the design artifact.

# MultiScalePointDescriptor — the verifier's point read-out

The verifier (stage 2 of the reference-prompted segmentation pipeline) is anchor-box classification,
not match verification: stage-1 ref↔obs correspondence (ALIKED/LightGlue or GIM) is only an expedient
to get anchor-box-like candidate points on the observation (>50% outliers). The verifier answers, per
point, "does this point lie on an instance of the reference's class?" — a learned replacement for
RANSAC/MAGSAC with no 2-view geometry downstream. Every anchor-box classifier needs a feature read-out
at the anchor locations before it can classify them, and `MultiScalePointDescriptor` is that read-out —
the RoI-Align analog of this system. In one sentence: **it converts (observation image, sparse anchor
points) into per-anchor semantic tokens by reading the frozen multi-scale descriptor pyramid at those
points and learning only how to fuse the scales — the front half of an anchor classifier whose back
half is cross-attention against the reference.**

Implementation: `segmentation/src/segmentation/verifier/point_descriptor.py`. Owning design pseudocode:
`isaac_datagen/verifier` lines 17–51.

## Where it sits in the Verifier

It is the `point_descriptor_extractor` collaborator injected into `Verifier.__init__`
(`verifier:62`) and invoked at the top of `Verifier.forward` (`verifier:87`). Everything downstream
perceives the observation *only* through the tokens it emits.

```
observation (B,C,H,W)   proposals (B,N,2) + valid (B,N)        reference image
        │                        │                                   │
        ▼                        ▼                                   ▼
  ┌─────────────────────────────────────┐                 ┌─────────────────────┐
  │       MultiScalePointDescriptor     │                 │  PerceiverResampler │
  │  frozen FPN pyramid ── grid_sample  │                 │  (global ref repr:  │
  │  at points ── per-scale Linear ──   │                 │   M cls tokens)     │
  │  mean over scales                   │                 └──────────┬──────────┘
  └──────────────────┬──────────────────┘                            │
                     ▼                                               │
        point tokens (B,N,hidden_dim)  ── queries ──┐                │
                     │                              ▼                ▼
                     │                   ┌──────────────────────────────┐
                     └── + residual ──── │ cross-attn decoder × L       │
                                         │ (keys/values = ref tokens)   │
                                         └──────────────┬───────────────┘
                                                        ▼
                                  logit projection token (+ learned inlier-prior bias)
                                                        ▼
                                        per-point inlier logits (B,N)
```

## Why each design choice

- **Frozen FPN backbone (`M2FFpn`/`DiftFpn` from `reference_matching`).** Reuses the same descriptor
  volumes that condition stage 3, so both learned stages see the world through one representation.
  Only the per-scale downshift Linears train.
- **Multi-scale, fused by mean.** A point's class membership often isn't decidable from the finest
  level alone — a point can land on a textureless patch whose identity comes from surrounding object
  context. Coarse levels contribute "what object/part is this region," fine levels "what exactly is
  here." Per-scale `nn.Linear` layers project heterogeneous channel widths (e.g. DIFT 1280/1280/640/320)
  into one embedding space where averaging is a meaningful fusion; being learned, they weight what each
  scale contributes.
- **Sample-then-project, not project-then-sample.** Only N points of each map matter, so grid_sample
  first and run the Linear on `(B,N,C)` instead of projecting entire volumes. Pure efficiency; no
  modeling consequence.
- **No reference-side points.** The module takes only observation points. The descriptor is compared
  against a *global* reference representation because the question is "is this point on the class?",
  not "does this point match that point" — which is why the ref-side halves of stage-1 correspondences
  stay discarded (design notes, `verifier:115-123`).
- **Padded proposals are harmless.** Batching variable-N proposals forces padding; padded points read
  out garbage descriptors, but the fusion mean runs across *scales* per point (axis V), never across
  points — garbage stays confined to its own row. The decoder is cross-attention-only (point tokens
  are queries that never attend each other), so invalid rows can't contaminate valid ones. The validity
  mask's real consumer is the loss; the module zeroes invalid rows on return purely for determinism.

## Interface contract

```
forward(image, proposals, proposals_valid) -> (B, N, hidden_dim)
  image:           tv_tensors.Image (B, C, H, W), raw / un-prepped
  proposals:       (B, N, 2) xy as 0-1 cell-center NORMALIZED coords — the CALLER
                   normalizes once, (xy_px + 0.5) / [W, H]; out-of-range raises ValueError
  proposals_valid: (B, N) bool; padded rows hold harmless values like 0
```

> Contract changed with the Verifier implementation (2026-06-05): the Verifier is the single
> normalization site — the same `xy01` tensor feeds this read-out and the coordinate posenc.
> The module converts to `grid_sample`'s [-1, 1] internally (mechanism identity) and raises
> on un-normalized input instead of silently border-sampling.

FPN provider contract consumed (`reference_matching/src/reference_matching/descriptor.py`):
`.channels` (dict key→width), `.keys` (tuple, fixes volume order), `.strides`, `.prep`
(square-resize Compose), `forward → list` of `(B, C_v, H_v, W_v)` volumes in key order,
frozen / `@torch.no_grad()`. Note the attribute is `channels` — the pseudocode's `fpn.widths`
does not exist.

Coordinate convention: `grid_sample` with `align_corners=False`, cell-center normalization
`x_norm = (x + 0.5)/W·2 − 1` — the 0-1 half (`(x + 0.5)/W`) now lives at the caller (Verifier),
the `·2 − 1` half inside the module; still computed **once against the original image dims** and
valid at every scale because `prep` square-resizes with no pad/crop (convention established in
`reference_matching/tests/smoke_fpn_descriptor.py:51-68`).

## Resolved design questions

- **The fusion mean is over V (scales), per point — `proposals_valid` never enters it.**
  `descriptors[b,n,:] = (1/V)·Σ_v multi_scale[v,b,n,:]`: the sum runs over `v` with `(b,n)` fixed, so
  a padded row appears in no valid point's mean. The pseudocode's `mean_with_mask` (lines 40–51)
  solves a reduction-over-N problem — pooling points into one vector, where padded rows *would*
  corrupt the result and valid counts can hit zero — that this module never performs. No masked-mean
  utility was added anywhere.
- **"Valid descriptors without ragged tensors?"** Keep dense `(B,N,hidden_dim)`; mask at the loss
  (padded logits contribute nothing, including zero gradient to the shared downshifts).
- **The output zeroing (`* proposals_valid`) is hygiene, not correctness** — deterministic zeros for
  padded rows; correctness would survive deleting it.

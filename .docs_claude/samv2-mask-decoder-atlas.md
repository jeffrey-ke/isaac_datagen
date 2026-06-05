# SAMv2 Mask Decoder — Atlas

Multi-level map of `SAMV2MaskDecoder` from the muggled_sam repo
(`/home/jeffk/repo/muggled_sam/muggled_sam/v2_sam/`).
Each plate expands one node from the plate above. Solid edges = tensor flow,
dotted = optional input, line refs point into the source files.

| Plate | Class | File |
|---|---|---|
| 0 | `SAMV2MaskDecoder` | `mask_decoder_model.py:21` |
| 1 | `MaskDecoderTransformer` (TwoWayTransformer) | `components/mask_decoder_transformer.py:20` |
| 2 | `CrossAttentionBlock` (TwoWayAttentionBlock) | `components/mask_decoder_transformer.py:94` |
| 3 | Attention wrappers (`CrossAttentionNormed` / `SelfAttentionNormed` / `SelfAttentionNoPosenc`) | `components/mask_decoder_attention.py:111-159` |
| 4 | `GenericAttention` | `components/mask_decoder_attention.py:19` |
| 5a | `MaskGen` + `MaskUpscalerWithHiresSupport` | `mask_decoder_model.py:272, 348` |
| 5b | `ObjectPointerGen` | `mask_decoder_model.py:475` |

---

## Plate 0 — `SAMV2MaskDecoder` (mask_decoder_model.py:21)

The big idea: each output head owns a **learned query token** (DETR-style) that rides
through the transformer alongside the prompt tokens. The transformer charges those
tokens with content; the heads just decode them.

```mermaid
flowchart TB
    subgraph INPUTS["inputs — forward :85"]
        lowres["lowres img tokens<br/>B×256×64×64"]
        hires2["hires_x2 tokens<br/>B×64×128×128"]
        hires4["hires_x4 tokens<br/>B×32×256×256"]
        prompts["encoded prompts<br/>B×N×256"]
        posenc["image posenc<br/>B×256×64×64"]
        hint["mask_hint (optional)<br/>B×1×256×256"]
    end

    clsparams["learned cls tokens :68-70<br/>obj ×1 + iou ×1 + mask ×4<br/>(DETR-style learned queries)"]

    mhe["MaskHintEncoder :395<br/>conv-downscale hint (1→4→16→256 ch),<br/>else learned no-mask embed;<br/>ADDED onto every image token"]
    cat["concat :138-144<br/>[obj, iou, mask×4, prompts]<br/>→ B×(6+N)×256"]
    xfmr["MaskDecoderTransformer :145<br/>two-way attention — Plate 1"]
    split["split cls tokens back out :148-151"]

    maskgen["MaskGen :154 — Plate 5a"]
    ioumlp["iou_token_mlp :155<br/>MLP3Layers + sigmoid"]
    objgen["ObjectPointerGen :158 — Plate 5b"]

    masks(["mask_preds B×4×256×256"])
    ious(["iou_preds B×4"])
    score(["obj_score B"])
    ptrs(["obj_ptrs B×4×256"])

    lowres --> mhe
    hint -.-> mhe
    clsparams --> cat
    prompts --> cat
    mhe -- "img tokens B×256×64×64" --> xfmr
    cat -- "prompt tokens B×(6+N)×256" --> xfmr
    posenc --> xfmr

    xfmr -- "encoded prompt tokens" --> split
    xfmr -- "encoded img tokens<br/>B×256×64×64" --> maskgen
    hires2 --> maskgen
    hires4 --> maskgen

    split -- "mask_tokens [:,2:,:]" --> maskgen
    split -- "iou_token [:,1,:]" --> ioumlp
    split -- "obj_token [:,0,:]" --> objgen
    split -- "mask_tokens [:,2:,:]" --> objgen

    maskgen --> masks
    ioumlp --> ious
    objgen --> score
    objgen --> ptrs
```

V2-vs-V1 deltas visible at this level: the obj token / `ObjectPointerGen` (video
memory) and the hires_x2/x4 skip inputs into `MaskGen` (hierarchical Hiera encoder).

---

## Plate 1 — `MaskDecoderTransformer` (mask_decoder_transformer.py:20)

The "TwoWayTransformer": depth=2 blocks + one final prompt→image cross-attention.

Two design moves live here:
- image tokens are flattened to a sequence at entry, restored at exit (:74, :87)
- **prompt tokens are their own positional encoding** (:79): captured once, frozen,
  re-injected at *every* attention layer — the prompts' identity never washes out.

```mermaid
flowchart TB
    ptok_in["prompt tokens B×(6+N)×256"]
    itok_in["image tokens B×256×64×64"]
    ipos_in["image posenc B×256×64×64"]

    flatten["flatten :74-75<br/>B×C×H×W → B×4096×256"]
    snapshot["prompt_posenc = prompt_tokens :79<br/>(frozen snapshot, reused every layer)"]

    blk0["CrossAttentionBlock layer 0 :47<br/>skip_selfattn_posenc=True — Plate 2"]
    blk1["CrossAttentionBlock layer 1 :47<br/>— Plate 2"]
    final["final_prompt_crossattn :51, :84<br/>CrossAttentionNormed — prompts get<br/>one last read of the updated image"]
    unflatten["unflatten :87<br/>B×4096×256 → B×256×64×64"]

    ptok_out(["encoded prompt tokens B×(6+N)×256"])
    itok_out(["encoded image tokens B×256×64×64"])

    itok_in --> flatten
    ipos_in --> flatten
    ptok_in --> snapshot
    ptok_in --> blk0

    flatten -- "img tokens, img posenc<br/>(2 separate seq tensors —<br/>NOT summed; posenc re-added<br/>per layer, Q/K only)" --> blk0
    snapshot -. "prompt_posenc (every layer)" .-> blk0
    snapshot -. "prompt_posenc" .-> blk1
    snapshot -. "prompt_posenc" .-> final

    blk0 -- "prompt toks, img toks" --> blk1
    blk1 -- "prompt toks" --> final
    blk1 -- "img toks" --> unflatten
    blk1 -- "img toks (as K,V)" --> final

    final --> ptok_out
    unflatten --> itok_out
```

---

## Plate 2 — `CrossAttentionBlock` (mask_decoder_transformer.py:94)

One "two-way" block = 4 strictly sequential ops (:149-152). Steps 1–3 update the
prompt side; step 4 flips the roles so the **image** queries the prompts — image
tokens leave having attended to the prompt, their embeddings now prompt-conditioned,
which is what makes the dot-product readout in MaskGen work. (Without this step the
readout would be a linear probe over a fixed, prompt-agnostic feature field —
unable to separate identical-looking instances.)

```mermaid
flowchart TB
    pin["prompt tokens"]
    ppos["prompt posenc (frozen snapshot)"]
    iin["image tokens"]
    ipos["image posenc"]

    sa["① prompt_selfattn :149<br/>SelfAttentionNormed — Plate 3<br/>(layer 0: SelfAttentionNoPosenc,<br/>since tokens ≡ posenc there)"]
    ca1["② prompt_crossattn :150<br/>CrossAttentionNormed — Plate 3<br/>Q=prompts, K,V=image<br/>'what is at my location?'"]
    mlp["③ prompt_mlpnorm :151<br/>MLP2LayersNormed :159<br/>256→2048→256, ReLU,<br/>residual + LayerNorm"]
    ca2["④ image_crossattn :152<br/>CrossAttentionNormed — Plate 3<br/>Q=image, K,V=prompts (post step ③!)<br/>image tokens become prompt-conditioned"]

    pout(["prompt tokens out"])
    iout(["image tokens out"])

    pin --> sa
    ppos -.-> sa
    sa --> ca1
    ppos -.-> ca1
    iin -- "K,V" --> ca1
    ipos -.-> ca1
    ca1 --> mlp
    mlp -- "K,V" --> ca2
    mlp --> pout
    ppos -.-> ca2
    iin -- "Q" --> ca2
    ipos -.-> ca2
    ca2 --> iout
```

---

## Plate 3 — Attention wrappers (mask_decoder_attention.py:111-159)

The most distinctive design decision in the module. All wrappers share one pattern:

- **posenc goes into Q and K only, never V** — position decides *who attends to whom*;
  the content that flows is position-free. Re-added fresh at every layer.
- **post-norm residual** taken from the *raw* (pre-posenc) tokens:
  `norm(a_tokens + attn_result)` — unlike the pre-norm ViT convention.

```mermaid
flowchart TB
    subgraph CAN["CrossAttentionNormed :111"]
        a1["a_tokens (query side)"]
        ap1["a_posenc"]
        b1["b_tokens (context side)"]
        bp1["b_posenc"]
        attn1["GenericAttention — Plate 4<br/>Q = a_tokens + a_posenc<br/>K = b_tokens + b_posenc<br/>V = b_tokens  ← RAW, no posenc"]
        res1["residual: a_tokens (raw) + attn_out"]
        norm1["LayerNorm (post-norm)"]
        out1(["encoded a_tokens"])

        a1 --> attn1
        ap1 -.-> attn1
        b1 --> attn1
        bp1 -.-> attn1
        a1 -- "raw skip" --> res1
        attn1 --> res1
        res1 --> norm1 --> out1
    end

    subgraph SAN["SelfAttentionNormed :138"]
        note2["same pattern with b := a<br/>Q = K = a + posenc, V = a raw<br/>norm(a + attn)"]
    end

    subgraph SNP["SelfAttentionNoPosenc :159"]
        note3["layer-0 prompt self-attn only:<br/>norm(attn(x, x, x))<br/>no posenc AND no residual"]
    end
```

---

## Plate 4 — `GenericAttention` (mask_decoder_attention.py:19)

Vanilla multi-head attention with one twist: an **internal feature bottleneck**.
Cross-attention layers run at `internal_features = downsample_dim = 128` (half the
256-d token width) because they span 4096 image tokens; self-attention layers run
full-width 256 since they only span ~10 prompt tokens. Bottleneck applied exactly
where the token count is large.

```mermaid
flowchart TB
    q_in["q: B×Nq×256"]
    k_in["k: B×Nk×256"]
    v_in["v: B×Nk×256"]

    qp["q_proj :60<br/>256 → F' (128 cross / 256 self)"]
    kp["k_proj :61<br/>256 → F'"]
    vp["v_proj :62<br/>256 → F'"]

    reshape["split heads :93-95<br/>B×N×F' → B×H×N×(F'/H)<br/>H=8 heads"]
    sdpa["scaled_dot_product_attention :98<br/>softmax(QKᵀ/√d)·V"]
    merge["merge heads :104<br/>B×H×Nq×f → B×Nq×F'"]
    op["out_proj :65<br/>F' → 256"]

    out(["encoded q tokens B×Nq×256"])

    q_in --> qp --> reshape
    k_in --> kp --> reshape
    v_in --> vp --> reshape
    reshape --> sdpa --> merge --> op --> out
```

---

## Plate 5a — `MaskGen` (mask_decoder_model.py:272) + upscaler (:348)

The readout is a **dot product**: each mask token becomes a dynamic 1×1-conv filter
applied to the upscaled image tokens — `einsum("bnc,bchw->bnhw")` (:343).
The upscaler is where the V2 hires skip connections land.

```mermaid
flowchart TB
    lowtok["encoded img tokens<br/>B×256×64×64"]
    h2["hires_x2 B×64×128×128"]
    h4["hires_x4 B×32×256×256"]
    mtok["cls mask tokens B×4×256"]

    subgraph UPS["MaskUpscalerWithHiresSupport :348"]
        up1["ConvTranspose2d ×2 :366<br/>256ch → 64ch, 64² → 128²"]
        add1["+ hires_x2, LayerNorm2d, GELU :382-384"]
        up2["ConvTranspose2d ×2 :368<br/>64ch → 32ch, 128² → 256²"]
        add2["+ hires_x4, GELU :387-388"]
    end

    mlps["4× per-token MLP3Layers :300<br/>256 → 32 (match upscaler channels)"]
    einsum["einsum bnc,bchw→bnhw :343<br/>dot product over channels"]
    out(["mask_preds B×4×256×256"])

    lowtok --> up1 --> add1 --> up2 --> add2
    h2 --> add1
    h4 --> add2
    mtok --> mlps
    mlps -- "B×4×32" --> einsum
    add2 -- "B×32×256×256" --> einsum
    einsum --> out
```

---

## Plate 5b — `ObjectPointerGen` (mask_decoder_model.py:475) — V2 only

Object score (is anything masked?) from the obj token; per-mask pointers (video
memory representation) from the mask tokens, gated by the score.

```mermaid
flowchart TB
    otok["encoded obj token B×256"]
    mtok["encoded mask tokens B×4×256"]
    noptr["no_ptr_embed :492<br/>learned fallback pointer"]

    smlp["score_mlp :493<br/>MLP3Layers 256 → 1"]
    pmlp["pointer_mlp :494<br/>MLP3Layers 256 → 256"]
    gate["gate :527-528<br/>score > 0 ? pointers : no_ptr_embed"]

    score(["obj_score B<br/>(+5ish = object present, &lt;0 = none)"])
    ptrs(["obj_ptrs B×4×256<br/>→ video memory fusion"])

    otok --> smlp --> score
    smlp --> gate
    mtok --> pmlp --> gate
    noptr -.-> gate
    gate --> ptrs
```

---

## TL;DR

Two (+1 final) asymmetric two-way blocks where prompts self-attend, then prompts
and image cross-attend in *both directions* alternately; posenc is Q/K-only and
re-injected every layer (prompts acting as their own posenc); cross-attention is
bottlenecked to 128-d for the 4096-token image side; and the whole transformer
exists to charge up 6 learned cls tokens whose dot products / MLP decodings *are*
the outputs.

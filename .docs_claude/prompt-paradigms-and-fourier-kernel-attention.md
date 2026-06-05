# Two paradigms for turning a coordinate into a prompt — and the Fourier-kernel mechanics behind SAM's choice

Companion note to `samv2-mask-decoder-atlas.md`. Code references point into `/home/jeffk/repo/muggled_sam/muggled_sam/v2_sam/`.

---

## Part I — The two paradigms

Given a 2D coordinate $p$ (a click, a keypoint, a box corner) and an image feature volume $F \in \mathbb{R}^{C \times H \times W}$, there are two recognized ways to turn $p$ into a vector a network can consume:

**Paradigm A — coordinate-as-address (positional query).**
Encode the coordinate *symbolically* — $\mathrm{PE}(p)$ via Fourier features or learned anchors — plus a type/intent embedding. The token carries **no image content**; content is pulled in later by attention. The point is an *instruction*: "operate at this location."

- SAM / SAM 2 prompt encoder (Kirillov et al., 2023, arXiv:2304.02643; Ravi et al., 2024, arXiv:2408.00714)
- The DETR query lineage: DETR's learned object queries (Carion et al., 2020, arXiv:2005.12872), then explicitly coordinate-derived queries in Conditional DETR (Meng et al., 2021, arXiv:2108.06152), Anchor DETR (Wang et al., 2022, arXiv:2109.07107), and DAB-DETR — literally titled "Dynamic Anchor Boxes are Better Queries for DETR" (Liu et al., 2022, arXiv:2201.12329)
- Perceiver IO's positional output queries (Jaegle et al., 2021, arXiv:2107.14795)
- Coordinate networks: NeRF's positional encoding (Mildenhall et al., 2020, arXiv:2003.08934)

**Paradigm B — coordinate-as-index (feature sampling).**
Use the coordinate to *index the feature volume*: `desc = grid_sample(F, p)`. The token **is** local image content; the precise address is discarded (or carried separately). The point is an *observation*: "this is what the image looks like here."

- Descriptor learning & matching: CAPS (Wang et al., 2020, arXiv:2004.13324), LoFTR's coarse-to-fine sampling (Sun et al., 2021, arXiv:2104.00680), SuperGlue's keypoint descriptors (Sarlin et al., 2020, arXiv:1911.11763), DIFT (Tang et al., 2023, arXiv:2306.03881)
- Detection/segmentation readouts: ROIAlign (He et al., 2017, arXiv:1703.06870), PointRend's point-sampled features (Kirillov et al., 2020, arXiv:1912.08193)
- One-shot / cross-image segmentation: SEEM's visual prompts are *sampled* image features (Zou et al., 2023, arXiv:2304.06718); PerSAM (Zhang et al., 2023, arXiv:2305.03048), Matcher (Liu et al., 2023, arXiv:2305.13310), SegGPT (Wang et al., 2023, arXiv:2304.03284)

**The hybrid.** Deformable DETR (Zhu et al., 2021, arXiv:2010.04159): paradigm-A queries carry a reference coordinate and *learn offsets* to gather multi-point, multi-scale samples — a learned, multi-tap grid_sample. Notably, when the community did commit to sampling, a single bilinear tap was not competitive; it took learned offsets + multiple samples + multiple scales, which is just sparse attention re-derived.

**A third paradigm, for completeness — dense spatial encoding.** Rasterize the prompt into a map and add/concat it to the image stream: click disks in pre-SAM interactive segmentation (RITM, Sofiiuk et al., 2021, arXiv:2102.06583; SimpleClick, Liu et al., 2022, arXiv:2210.11006), and SAM's own mask prompts — `MaskHintEncoder` conv-encodes the mask and *adds it onto the image tokens* (`mask_decoder_model.py:395`). Sparse symbolic prompt → token (A); dense spatial prompt → feature-map addition (this).

### The design principle

> **The prompt representation should carry the variable the task discriminates on.**

- Matching/correspondence discriminates on **appearance** → sample the volume (B). The address is irrelevant downstream; the descriptor must transfer *across images*, and an address means nothing in another image. This is why every cross-image prompting system (SEEM, PerSAM, Matcher, SegGPT) is B-flavored.
- Same-image spatial selection discriminates on **location + intent** → encode the coordinate (A). The appearance at $p$ is already inside the other attention operand; duplicating it into the prompt adds nothing attention can't retrieve (see Part VI), while hard-coding it creates real failure modes:
  - **negative clicks**: under B the representation is background *appearance*, conflating "exclude this place" with "exclude things that look like this";
  - **box corners**: usually lie on background — $F[\text{corner}]$ is meaningless content;
  - **click noise**: a click 5 px off the object yields a *wrong descriptor* (hard representation failure) under B, vs. a slightly-off address that soft attention degrades gracefully under A. Matching papers don't face this — detectors choose repeatable, informative points; users don't;
  - **deliberate ambiguity**: $F[p]$ at one scale biases part-vs-whole; SAM defers that to its 4 mask tokens.

A linguist would call this deictic vs. descriptive reference: "this one *here*" vs. "the one that *looks like this*."

### Tie-in: the refseg pipeline composes both

Stage 2 (verifier, this repo) asks an **appearance** question — "does this point lie on an instance of the reference's class?" — so it grid_samples descriptors from FPN volumes: paradigm B, correctly. Stage 3 asks SAM a **spatial** question — "segment at these verified locations" — paradigm-A point prompts, while reference *appearance* enters through the proper channel (gated cross-attention on reference descriptors, GLIGEN-style), not smuggled into the point tokens.

---

## Part II — The mechanism in SAM: one shared Fourier basis

`SAMV2CoordinateEncoder` (`coordinate_encoder_model.py:20`) owns a single random Gaussian matrix $G \in \mathbb{R}^{2 \times 128}$ (`:61`; a fixed random buffer `scale * randn` in the original SAM code, an `nn.Parameter` holding those checkpoint values here) and uses it for **both sides** of the decoder's cross-attention:

- **prompt points** (`forward`, `:98-100`): with normalized coords mapped to $\tilde{x} = 2x - 1$,

$$
\mathrm{PE}(x) = \big[\, \sin(2\pi\, \tilde{x}^\top G)\;;\; \cos(2\pi\, \tilde{x}^\top G) \,\big] \in \mathbb{R}^{256}
$$

  plus a learned type embedding (fg / bg / box-TL / box-BR / padding) added on top;

- **the dense grid posenc** (`get_grid_position_encoding`, `:140-157`): the *same* `forward` evaluated at every patch-center coordinate of the $64 \times 64$ grid → `image_posenc_bchw`, which the mask decoder re-adds to image keys at every attention layer (`mask_decoder_attention.py:129-132`).

So in cross-attention, with $Q = W_q(c_{\text{prompt}} + \mathrm{PE}(p))$ and $K = W_k(c_{\text{image}} + \mathrm{PE}(q))$, the attention logit contains the term $\langle \mathrm{PE}(p), \mathrm{PE}(q) \rangle$ (through the learned bilinear form — Part IV). Everything below is about why that term behaves like a spatial kernel peaked at the click.

---

## Part III — Why $\langle \mathrm{PE}(p), \mathrm{PE}(q) \rangle$ peaks at $p = q$

Write $g_i$ for the columns of $G$ ($i = 1, \dots, m$, with $m = 128$), and for a coordinate $x$ define the phases $\theta_i(x) = 2\pi\, g_i^\top \tilde{x}$. Then

$$
\mathrm{PE}(x) = \big[\,\sin\theta_1(x),\dots,\sin\theta_m(x),\;\cos\theta_1(x),\dots,\cos\theta_m(x)\,\big].
$$

**Step 1 — the inner product is a sum of cosines of phase *differences*.**
Pair up the sin and cos entries per frequency and use the angle-difference identity $\sin a \sin b + \cos a \cos b = \cos(a - b)$:

$$
\langle \mathrm{PE}(p), \mathrm{PE}(q) \rangle
= \sum_{i=1}^{m} \big[ \sin\theta_i(p)\sin\theta_i(q) + \cos\theta_i(p)\cos\theta_i(q) \big]
= \sum_{i=1}^{m} \cos\!\big( 2\pi\, g_i^\top \tilde\delta \big),
\qquad \tilde\delta = \tilde p - \tilde q .
$$

The absolute positions have vanished: the result depends **only on the displacement** $\delta$. That is *stationarity* (shift-invariance), and it fell out of one trig identity.

**Step 2 — strict global maximum at $\delta = 0$.**
Each term satisfies $\cos(2\pi\, g_i^\top \tilde\delta) \le 1$, with equality iff $g_i^\top \tilde\delta \in \mathbb{Z}$. At $\delta = 0$ every term equals 1 simultaneously, so the sum attains its maximum possible value $m$. For $\delta \ne 0$, the $m = 128$ constraints $g_i^\top \tilde\delta \in \mathbb{Z}$ must hold *simultaneously* for 128 independently-sampled Gaussian directions in $\mathbb{R}^2$ — almost surely only $\delta = 0$ satisfies all of them. So the peak at the click is the unique global max (a.s.), and any miss costs you on many terms at once.

A second, geometric proof: every PE vector has **constant norm** — $\|\mathrm{PE}(x)\|^2 = \sum_i (\sin^2\theta_i + \cos^2\theta_i) = m$ — so all encodings live on the sphere of radius $\sqrt{m}$, and

$$
\langle \mathrm{PE}(p), \mathrm{PE}(q) \rangle = m - \tfrac{1}{2}\,\|\mathrm{PE}(p) - \mathrm{PE}(q)\|^2 .
$$

Maximizing the inner product is *identical* to minimizing Euclidean distance in the lifted space, which is zero exactly at $p = q$.

**Step 3 — the shape of the falloff: Monte-Carlo of a Gaussian (Bochner / Rahimi–Recht).**
The normalized sum $\hat{k}(\delta) = \frac{1}{m} \sum_i \cos(2\pi\, g_i^\top \tilde\delta)$ is an $m$-sample Monte-Carlo estimate of

$$
\mathbb{E}_{g \sim \mathcal{N}(0,\sigma^2 I)}\big[\cos(2\pi g^\top \tilde\delta)\big]
= \mathrm{Re}\,\mathbb{E}\big[e^{\,i 2\pi g^\top \tilde\delta}\big]
= e^{-2\pi^2 \sigma^2 \|\tilde\delta\|^2},
$$

the characteristic function of the Gaussian — i.e. in expectation, the logit term is a **Gaussian RBF kernel** in coordinate space, with spatial bandwidth $1/(2\pi\sigma)$ set reciprocally by the frequency spread $\sigma$ of $G$. The general statement is **Bochner's theorem** (Rudin, *Fourier Analysis on Groups*): a continuous shift-invariant kernel is positive-definite iff it is the Fourier transform of a non-negative spectral measure. **Rahimi & Recht** (Random Features for Large-Scale Kernel Machines, NeurIPS 2007) turned that into an algorithm — sample frequencies from the spectral measure, use sin/cos features, get an unbiased kernel estimate with uniform $O(1/\sqrt{m})$ error — and **Tancik et al.** (Fourier Features Let Networks Learn High Frequency Functions, NeurIPS 2020, arXiv:2006.10739) showed this same lift governs what spatial frequencies coordinate-MLPs can express, which is exactly why SAM/NeRF-era models encode coordinates this way.

So: *peaked at the click because all $m$ cosines align only at $\delta = 0$; Gaussian-shaped because the frequencies were drawn from a Gaussian spectrum; bandwidth chosen by $\sigma$.*

---

## Part IV — "You can't recover summands from a sum" — so how does this work?

Both statements are true, and the resolution is the key insight.

**Vector recovery is impossible.** The map $(c, \rho) \mapsto c + \rho$ from $\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ has a huge kernel: the preimage of any vector $v$ is the $d$-dimensional affine family $\{(c,\; v - c)\}$. Given only $q = c_{\text{prompt}} + \mathrm{PE}(p)$, no operator can return $c_{\text{prompt}}$ and $\mathrm{PE}(p)$ — that information is gone *from the vector*.

**But attention never needs the summands — it needs the score, and the score is bilinear.**
With $M = W_q^\top W_k / \sqrt{d}$ (per layer, per head), the logit is a bilinear form, and bilinearity distributes over the sums *exactly*:

$$
(c_p + \mathrm{PE}(p))^\top M\, (c_x + \mathrm{PE}(q))
= \underbrace{c_p^\top M c_x}_{\text{content–content}}
+ \underbrace{c_p^\top M\, \mathrm{PE}(q)}_{\text{content–position}}
+ \underbrace{\mathrm{PE}(p)^\top M\, c_x}_{\text{position–content}}
+ \underbrace{\mathrm{PE}(p)^\top M\, \mathrm{PE}(q)}_{\text{position–position (the kernel)}} .
$$

Nothing was "recovered" — the four interaction terms are simply *present, additively,* in the scalar score. The spatial kernel of Part III rides into the logit untouched regardless of our inability to separate the vectors. What summation costs you is not the kernel term but the **cross terms**: $c \leftrightarrow \mathrm{PE}$ interference that the model must learn to manage.

**When *can* you recover summands, linear-algebraically?** If you have side structure: when content and position occupy complementary subspaces $U \oplus V = \mathbb{R}^d$ with $U \cap V = \{0\}$, the decomposition of any vector is unique and recoverable by (oblique) projection. In $\mathbb{R}^{256}$ there is plenty of room: PE vectors lie on a specific $m$-torus (Part V), and content embeddings can learn to occupy roughly complementary directions, making the cross terms small. The learned $W_q, W_k$ can also shape $M$ to approximately block-diagonalize over the two subspaces.

**The lineage — the field converged on enforcing this structurally.** Your instinct (summing is lossy, cross terms are suspect) is precisely the concern that drove a decade of position-encoding best practices:

1. **Vaswani et al. 2017** (arXiv:1706.03762): sinusoidal PE *added* to token embeddings — sum and hope; cross terms unmanaged.
2. **Transformer-XL** (Dai et al., 2019, arXiv:1901.02860): writes out exactly the 4-term expansion above and *reparameterizes each term separately* (relative-position keys, learned global biases for the position-only terms).
3. **DeBERTa** (He et al., 2020, arXiv:2006.03654): "disentangled attention" — content and position kept as *separate vectors* with separate projection matrices; the cross terms become deliberate, independently-parameterized content→position and position→content attentions.
4. **Conditional DETR** (arXiv:2108.06152): in cross-attention, **concatenates** content and positional query parts instead of adding. Concatenation is the direct-sum embedding made literal — $[c; \rho]^\top [c'; \rho'] = c^\top c' + \rho^\top \rho'$ — zero cross terms *by construction*. Reported as a major convergence accelerator for DETR.
5. **RoPE** (Su et al., 2021, arXiv:2104.09864): abandons addition entirely — position enters as a *rotation* of the content vector, so the logit depends on relative position exactly and multiplicatively, not approximately and additively.

SAM (2023) sits at the comfortable middle: additive Fourier PE, but re-injected fresh at every layer, Q/K-only (never V), with a shared basis on both sides — enough structure that the kernel term is strong and clean, while $\mathbb{R}^{256}$ gives the cross terms room to be benign.

---

## Part V — The change-of-basis interpretation

There *is* a clean one, in three readings of the same fact.

**1. A lift, not a basis change of $\mathbb{R}^2$.** $\mathrm{PE}: \mathbb{R}^2 \to \mathbb{R}^{2m}$ is a nonlinear embedding of the coordinate plane onto an $m$-torus (each $(\sin\theta_i, \cos\theta_i)$ pair lives on a unit circle; the image is a product of $m$ circles, sitting on the sphere of radius $\sqrt{m}$). It is a *change of representation*: coordinates are re-expressed in the basis of harmonics $\{\sin(2\pi g_i^\top x),\, \cos(2\pi g_i^\top x)\}$.

**2. The basis in which translation is (block-)diagonal.** This is the deepest reading. Translating the input by $t$ shifts every phase by $\alpha_i = 2\pi\, g_i^\top (2t)$, which acts on each $(\sin, \cos)$ pair as a $2 \times 2$ rotation. Collecting blocks:

$$
\mathrm{PE}(x + t) = R(t)\, \mathrm{PE}(x), \qquad
R(t) = \bigoplus_{i=1}^{m}
\begin{pmatrix} \cos\alpha_i & \;\sin\alpha_i \\ -\sin\alpha_i & \;\cos\alpha_i \end{pmatrix},
$$

an **orthogonal** matrix. The messy, nonlinear act of "moving a point" in pixel space becomes an exact linear isometry in the lifted space. Stationarity is then a one-liner:

$$
\langle \mathrm{PE}(p+t), \mathrm{PE}(q+t) \rangle
= \langle R(t)\mathrm{PE}(p),\, R(t)\mathrm{PE}(q) \rangle
= \langle \mathrm{PE}(p), \mathrm{PE}(q) \rangle .
$$

Group-theoretically: the characters $x \mapsto e^{i 2\pi g^\top x}$ are the irreducible unitary representations of the translation group $(\mathbb{R}^2, +)$; the Fourier basis is *exactly* the basis that simultaneously diagonalizes all translations (into these $2 \times 2$ rotation blocks, in real form); Bochner's theorem is the statement that admissible kernels are non-negative mixtures of these characters. And note the punchline: $\mathrm{PE}(p) = R(\delta)\,\mathrm{PE}(q)$ means the per-block inner product is $\cos\alpha_i$ *independent of the absolute phase* — which is the same algebra **RoPE** later adopted deliberately for sequence position. SAM's "random Fourier PE" and RoPE are the same representation-theoretic object, arrived at from the kernel side and the rotation side respectively.

**3. The kernel-trick reading.** $\mathrm{PE}$ is a truncated feature map for the Gaussian RBF kernel's RKHS: a finite-dimensional change of coordinates under which a *nonlinear* similarity on $\mathbb{R}^2$ ("spatial nearness") becomes *linear* geometry (inner products) — the only language the attention machinery speaks. Attention can't evaluate $\exp(-\|p - q\|^2 / 2\ell^2)$; it can only dot vectors. The Fourier lift hands it a basis where dotting *is* (an unbiased estimate of) that kernel.

---

## Part VI — "Soft grid_sample," made precise

Both operations are instances of *kernel-weighted reads of a feature volume at a continuous coordinate*:

$$
\text{read}(p) = \sum_{j} w_j(p)\, F(x_j), \qquad \sum_j w_j = 1 .
$$

- **Bilinear grid_sample** (CAPS/LoFTR-style): $w_j(p) = \mathrm{tent}(p - x_j)$, the separable triangular kernel, supported on the 4 nearest cells. Fixed shape, fixed (sub-pixel) bandwidth, executed once.
- **SAM's first cross-attention layer**: the prompt query is $c_{\text{type}} + \mathrm{PE}(p)$ (content-free), keys are $c_{x_j} + \mathrm{PE}(x_j)$, values are **raw** image tokens (posenc never enters V — `mask_decoder_attention.py:132`). When the position–position term dominates the logit,

$$
w_j(p) = \mathrm{softmax}_j\!\left( \tfrac{1}{\sqrt{d}}\, \mathrm{PE}(p)^\top M\, \mathrm{PE}(x_j) + \dots \right)
\;\approx\; \frac{ e^{\, m\,\hat k(p - x_j)/\sqrt d} }{ \sum_{j'} e^{\, m\,\hat k(p - x_{j'})/\sqrt d} },
$$

a normalized, approximately-Gaussian spatial kernel centered at the click — i.e. **the first layer can implement grid_sample** (retrieve $\approx F[p]$) as a special case. But unlike grid_sample:

| | bilinear grid_sample | attention with shared Fourier PE |
|---|---|---|
| kernel shape | fixed tent | learned via $M = W_q^\top W_k$ per head — reweighting frequency pairs sculpts anisotropic / offset / multi-lobe kernels within the sampled spectrum |
| support | 4 px | global |
| bandwidth | fixed | set by $\sigma$ of $G$, sharpened by logit scale & learned norms |
| content-awareness | none | content terms shift the read toward "what matches," not just "where I clicked" |
| iteration | once | re-executed every layer, per head — layer 1 reads $\approx F[p]$, layer 2+ attends by content similarity to grow over the instance (learned region-growing) |

This is the precise sense in which **paradigm A subsumes paradigm B**: a content-free positional query over posenc'd keys with raw values *is* a grid_sample — soft, learnable, global, iterated — while a hard grid_sample is its one-shot, fixed-kernel special case. The justification for SAM's choice is then mostly structural (Part I's failure modes) plus the DETR-lineage empirics that explicit, disentangled positional queries converge faster and localize better than content-entangled ones.

---

## References

- Kirillov et al., *Segment Anything*, ICCV 2023 — arXiv:2304.02643
- Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024 — arXiv:2408.00714
- Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020 — arXiv:2005.12872
- Meng et al., *Conditional DETR for Fast Training Convergence*, ICCV 2021 — arXiv:2108.06152
- Wang et al., *Anchor DETR*, AAAI 2022 — arXiv:2109.07107
- Liu et al., *DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR*, ICLR 2022 — arXiv:2201.12329
- Zhu et al., *Deformable DETR*, ICLR 2021 — arXiv:2010.04159
- Jaegle et al., *Perceiver IO*, ICLR 2022 — arXiv:2107.14795
- Mildenhall et al., *NeRF*, ECCV 2020 — arXiv:2003.08934
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, NeurIPS 2020 — arXiv:2006.10739
- Rahimi & Recht, *Random Features for Large-Scale Kernel Machines*, NeurIPS 2007
- Rudin, *Fourier Analysis on Groups* (Bochner's theorem)
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — arXiv:1706.03762
- Dai et al., *Transformer-XL*, ACL 2019 — arXiv:1901.02860
- He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*, ICLR 2021 — arXiv:2006.03654
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)*, 2021 — arXiv:2104.09864
- Wang et al., *Learning Feature Descriptors using Camera Pose Supervision (CAPS)*, ECCV 2020 — arXiv:2004.13324
- Sun et al., *LoFTR: Detector-Free Local Feature Matching with Transformers*, CVPR 2021 — arXiv:2104.00680
- Sarlin et al., *SuperGlue*, CVPR 2020 — arXiv:1911.11763
- Tang et al., *Emergent Correspondence from Image Diffusion (DIFT)*, NeurIPS 2023 — arXiv:2306.03881
- He et al., *Mask R-CNN (ROIAlign)*, ICCV 2017 — arXiv:1703.06870
- Kirillov et al., *PointRend*, CVPR 2020 — arXiv:1912.08193
- Zou et al., *Segment Everything Everywhere All at Once (SEEM)*, NeurIPS 2023 — arXiv:2304.06718
- Zhang et al., *Personalize Segment Anything Model with One Shot (PerSAM)*, 2023 — arXiv:2305.03048
- Liu et al., *Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching*, 2023 — arXiv:2305.13310
- Wang et al., *SegGPT: Segmenting Everything in Context*, ICCV 2023 — arXiv:2304.03284
- Sofiiuk et al., *Reviving Iterative Training with Mask Guidance for Interactive Segmentation (RITM)*, 2021 — arXiv:2102.06583
- Liu et al., *SimpleClick*, ICCV 2023 — arXiv:2210.11006

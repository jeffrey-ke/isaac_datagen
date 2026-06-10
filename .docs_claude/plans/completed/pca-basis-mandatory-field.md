# Mandatory `ObsMaskMetadata.principal_components` + PCA→RGB viz primitives

> **Status: completed 2026-06-06.** All verifications (A–E) green. As-built deviation from
> the plan below: **only `expanded-refseg` was migrated** (user decision) — 19 classes,
> 19456 tokens, orthonormality err 8.3e-7. `temp-render/render000` and
> `cid-mask-verify/render900` are left unmigrated: their catalogs fail full
> `ObsMaskMetadata.deserialize` loudly until someone runs
> `env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py <dataset_root>`
> on them. Also as-built: `pca_rgb` needs no torch import (pure tensor methods); the
> migration's render-dir filter is simply "has `class_to_descriptors/`" (no explicit
> `*_viz_clusters` exclusion needed).

**Precursor plan** — the verifier-training-harness plan (segmentation repo) depends on this:
it will join these fields into a `VerifierSample` and call `pca_rgb` to RGB-visualize
reference features in a common basis. This plan stands alone and is implemented first.

## Context

`ObsMaskMetadata` (`vision_core/src/vision_core/datastructs.py:262-281`) is a per-render-dir
catalog serialized once at idx=0. We add a 6th **MANDATORY** field `principal_components`
(no `None`, no default — user directive) holding a shared PCA→RGB basis fit on the
concatenated tokens of ALL classes' DIFT descriptors, so every class's features project into
comparable colors. We also add three viz primitives to `vision_core/viz.py`, wire the basis
computation into the datagen writer, and migrate the three existing on-disk catalogs.

### Verified findings

- **Serialization machinery** (`datastructs.py:99-135`): `serialize` iterates `fields(self)`,
  maps by type, writes `directory/{field}/{field}_{idx:04d}{ext}`. `deserialize` reads every
  field's subdir with NO graceful-missing handling → missing `principal_components/` raises —
  the desired fail-loud. `deserialize_field` (`:128-135`) reads ONE field in isolation — the
  escape hatch for the chicken-and-egg migration.
- **dict → .pt**: `ObsMaskMetadata._serializers` includes `_DICT_PT_SERIALIZER`
  (`datastructs.py:192-198`, `:278-281`) — a new dict-typed field auto-serializes, no
  serializer change.
- **Field ordering safe**: all 5 current fields are no-default; appending another no-default
  field is legal.
- **Exactly ONE constructor site**: `reference_seg_writer.py:185-191`. Mandatory field →
  won't run without the new arg — fail-loud as intended.
- **All other sites are `.deserialize(0, dir)`** and need NO code change — they break only on
  unmigrated dirs (intended): `segmentation/dataset.py:65`; isaac_datagen `viz_inliers.py:50`,
  `viz_occlusion.py:56`, `viz_clusters.py:204`, `viz_sample.py:104`, `add_proposals.py:41`,
  `add_inlier_data.py:29`, `migrate_descriptors_spatial.py:40`. `reference_matching`: zero usage.
- **Three on-disk catalogs to migrate** (each has `class_to_descriptors/..._0000.pt`, ~95 MB):
  `expanded-refseg/render000`, `temp-render/render000`, `cid-mask-verify/render900`.
- **vision_core declares no torch/numpy/matplotlib** (deps = omegaconf, tqdm); viz.py and
  datastructs.py already rely on consumer envs supplying them. `fit_pca_basis` follows the
  existing convention: **lazy torch import inside the function**, declare nothing new.
- Prototype: `joint_pca_project` (`segmentation/.docs_claude/one_off_tests/smoke_dift_spatial.py:43-63`).

## Edits

### 1. `vision_core/src/vision_core/datastructs.py` — the mandatory field

In `ObsMaskMetadata`, after `class_to_descriptors` (`:276-277`):

```python
    class_to_descriptors: dict        # {class: str → torch.Tensor (C, h, w) DIFT features}
    principal_components: dict        # shared PCA→RGB basis fit on ALL classes' tokens:
                                      # {"mean": (C,), "components": (3, C), "scale": (3,)}
                                      # — one projection per render dir; see viz.fit_pca_basis
```

No `_serializers` change. Update the class docstring to mention the field and that it is fit
on the concatenated tokens of all `class_to_descriptors`.

### 2. `vision_core/src/vision_core/viz.py` — three primitives

Level 1 (pure math; lazy `import torch` inside the functions — keeps matplotlib-only levels
importable everywhere). Update the module docstring's level lists.

```python
def fit_pca_basis(tokens, n: int = 3) -> dict:
    """Fit a deterministic PCA→RGB basis on (M, C) tokens.

    M ≈ n_classes × 1024 (each class's (1280,32,32) DIFT map → 1024 tokens, stacked).
    Returns {"mean": (C,), "components": (n, C), "scale": (n,)} — scale is the
    per-component std of the TRAINING projection. Storing scale makes pca_rgb fully
    deterministic: without it each map self-normalizes by its own min/max and identical
    features map to different colors, defeating the shared basis."""
    import torch
    tokens = tokens.float()
    mean = tokens.mean(0)
    centered = tokens - mean
    _, _, V = torch.pca_lowrank(centered, q=n, center=False)   # V: (C, n)
    components = V.T.contiguous()                              # (n, C)
    proj = centered @ components.T                             # (M, n)
    scale = proj.std(0).clamp_min(1e-6)
    return {"mean": mean, "components": components, "scale": scale}


def pca_rgb(feats, basis: dict):
    """(C, h, w) feature map → (h, w, 3) float RGB in [0,1] via a shared basis.
    Deterministic affine map: center by mean, project, scale to std units, ±2.5σ → [0,1]."""
    import torch
    C, h, w = feats.shape
    centered = feats.float().reshape(C, h * w) - basis["mean"][:, None]
    proj = (basis["components"] @ centered) / basis["scale"][:, None]
    rgb = (proj / 5.0 + 0.5).clamp(0, 1)
    return rgb.reshape(3, h, w).permute(1, 2, 0).cpu().numpy()
```

Level 2 (figure plumbing, near `save_figure` `:176-179`):

```python
def figure_to_ndarray(fig) -> np.ndarray:
    """Render a Figure to (H, W, 3) uint8 RGB, then close it."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())   # mpl ≥3.8 API; consumer env has 3.10.3
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb
```

**Decision — store `scale`: yes.** The prototype's per-map min/max normalization makes the
same feature render different colors across maps — fatal for a "common basis". Std over
percentiles: one-liner, symmetric, robust at this M; documented so switching is contained.

### 3. `isaac_datagen/src/isaac_datagen/reference_seg_writer.py` — writer integration

Add `from vision_core.viz import fit_pca_basis`. In `finalize_metadata` (`:182-191`):

```python
        # Shared PCA→RGB basis over ALL classes' tokens (each (C,h,w) → (h*w, C),
        # stacked) so every class projects into comparable colors. Mandatory field.
        tokens = torch.cat(
            [d.flatten(1).T for d in self.class_to_descriptors.values()], dim=0)
        ObsMaskMetadata(
            iid_to_name=self.iid_to_name,
            cid_to_class=self.cid_to_class,
            name_to_class=self.name_to_class,
            class_to_ref=self.class_to_ref,
            class_to_descriptors=self.class_to_descriptors,
            principal_components=fit_pca_basis(tokens, n=3),
        ).serialize(0, directory)
```

`d.flatten(1).T` matches the exact tokenization consumers use (`segmentation/dataset.py:93`),
so the basis is fit in the same token space.

### 4. NEW `isaac_datagen/src/isaac_datagen/migrate_pca_basis.py` — migration

Models `migrate_descriptors_spatial.py` / `downsample_proposals.py` (argparse, tqdm-style
reporting, `--dry-run`, residual write, idempotent). CRITICAL: must NOT call full
`deserialize` (requires the not-yet-written subdir — the exact chicken-and-egg this breaks).
Reads ONLY `class_to_descriptors` via `deserialize_field`, fits, residual-writes ONLY
`principal_components`:

```python
def migrate_render_dir(rd: Path, dry_run: bool) -> int:
    n = 0
    for pt in sorted((rd / "class_to_descriptors").glob("class_to_descriptors_*.pt")):
        idx = int(pt.stem.rsplit("_", 1)[1])
        c2d = ObsMaskMetadata.deserialize_field(idx, rd, "class_to_descriptors")
        tokens = torch.cat([d.flatten(1).T for d in c2d.values()], dim=0)
        basis = fit_pca_basis(tokens, n=3)
        if not dry_run:
            md = ObsMaskMetadata.__new__(ObsMaskMetadata)  # bypass __init__: serialize(only=)
            md.principal_components = basis                # reads only this one attribute
            md.serialize(idx, rd, only={"principal_components"})
        n += 1
    return n
```

`main()` globs `dataset_root/render*` dirs having `class_to_descriptors/` (excluding
`*_viz_clusters`), errors loud when none found. Safe because `serialize(only=...)` skips all
other fields before any `getattr`. (`ObsMaskMetadata` has no `__post_init__`; if one is ever
added, switch to five `deserialize_field` reads + a full constructor.)

## Migration commands (after code edits; from `/home/jeffk/repo/isaac_datagen`)

```bash
# dry-run all three roots, then real runs, then idempotency re-run on one
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/expanded-refseg  --dry-run
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/temp-render      --dry-run
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/cid-mask-verify  --dry-run
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/expanded-refseg
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/temp-render
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/cid-mask-verify
env -u PYTHONPATH uv run python src/isaac_datagen/migrate_pca_basis.py src/isaac_datagen/temp-render   # idempotent
```

## Verification

A. **Loud failure first** (before migrating, on one dir): full
   `ObsMaskMetadata.deserialize(0, <unmigrated>)` raises on the missing
   `principal_components/` subdir.
B. **Post-migration round-trip** (each dir): full `deserialize(0, dir)` succeeds;
   `pc["components"].shape == (3, 1280)`; orthonormality
   `max|components @ components.T − I| < 1e-3`; `scale > 0`.
C. **`pca_rgb` on a real `class_to_descriptors` entry**: `(32,32,3)` in [0,1], per-channel
   std > 1e-3 (non-degenerate).
D. **`figure_to_ndarray` smoke**: tiny Agg figure → `(H,W,3) uint8`.
E. **Downstream consumer**: `RenderDirReferenceSegDataset(temp-render/render000)` constructs
   (segmentation repo, `env -u PYTHONPATH uv run`).

## Risks / open questions

1. **vision_core dependency honesty**: it declares neither torch nor numpy/matplotlib yet
   already uses them (consumer envs supply). `fit_pca_basis` keeps the convention (lazy
   import). Recommended follow-up, deliberately OUT of this plan: declare them in
   vision_core's pyproject.
2. **Basis sign ambiguity**: PCA component signs are arbitrary → color identity is comparable
   WITHIN a render dir's basis (what the verifier viz needs), not across dirs.
3. **Degenerate token sets**: writer always has ≥1 class × 1024 tokens → SVD well-posed;
   verification C catches degeneracy anyway.

## Critical files

| Action | Path |
|---|---|
| EDIT | `vision_core/src/vision_core/datastructs.py` (mandatory field + docstring) |
| EDIT | `vision_core/src/vision_core/viz.py` (`fit_pca_basis`, `pca_rgb`, `figure_to_ndarray`) |
| EDIT | `isaac_datagen/src/isaac_datagen/reference_seg_writer.py` (compute basis at finalize) |
| NEW | `isaac_datagen/src/isaac_datagen/migrate_pca_basis.py` |

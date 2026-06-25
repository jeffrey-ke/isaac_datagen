# Reprojection-coverage proposer gate (Stage A) + ycb_can reference-pose data fix

> **Status (2026-06-25): Stage A DONE + ycb_can data bug fixed.** The proposer's class gate now gates on
> **reprojection coverage** (fraction of a class's reference texture visible in the observation), replacing
> the visible-pixel floor. Implemented + verified non-destructively on the two geometry-carrying datasets
> (`mixed-persp`, `shelf-optflow`). While reviewing the gate viz we found and fixed a reference-pose data
> bug for `ycb_can`. **Stage B (proposal/label regen) and the `expanded-refseg` re-render remain HELD.**
> Full design plan: `~/.claude/plans/jaunty-skipping-widget.md`. Supersedes the gate section of
> `segmentation/.docs_claude/plans/active/proposer-visible-px-gate-grid-proposals.md`.

## Context

The proposer gate decides which classes get point-prompts per frame. Its history: occlusion-ratio →
visible-pixel floor (60k px) → **reprojection coverage**. The px floor's failure: an **absolute** count
drops fully-visible objects when the camera is far. The fix is a **ratio** that measures the *true notion
of occlusion w.r.t. the visible reference texture*: reproject the class's reference RGB-D into the
observation through each instance's `class_to_l2w` placement and take the fraction NOT occluded/off-frame.
Because the denominator is reference points (not obs pixels), it's invariant to camera distance.

## Part 1 — The gate (Stage A code changes)

All six changes byte-compile; the geometry helper is byte-identical to the chain it replaced.

| # | Change | File |
|---|---|---|
| 1 | Drop dead `K` param from `reprojection_occlusion` (it only reads `obs_depth.shape`) | `vision_core/pose_utils.py` |
| 2 | New shared `reproject_local_to_obs(ref_pts, l2w, w2c, obs_K, obs_depth, …)` (local→world→obs-cam + occlusion); `benchmark/convert.py` refactored onto it | `vision_core/pose_utils.py`, `benchmark/convert.py` |
| 3 | `instance_visibility` (iid→ratio) + `gate_classes_reproj` (class→best-member ratio); old px `gate_classes` kept harmless | `isaac_datagen/proposal_gate.py` |
| 4 | Gate swap → deserialize `OptFlowSample`/`OptFlowMetadata`, threaded per-render-dir `ref_cache` | `isaac_datagen/add_proposals.py` |
| 5 | `proposer_min_visible_ratio=0.30` + `proposer_tau_d/tau_r`; px floor deprecated | `isaac_datagen/runtime_config.py` |
| 6 | Viz colors instances by `ratio > min_visible_ratio`, labels `cls\nNN%`, `--min-visible-ratio` | `isaac_datagen/viz_gate.py` |

The metric (dense — whole reference foreground pointmap, no FPS):
```
ref_pts  = transform_pointmap(depthmap_to_pointmap(ref_depth, ref_K), ref_pose)[ref_depth > 0]
occluded = reproject_local_to_obs(ref_pts, class_to_l2w[c][rows], w2c, obs_K, obs_depth)   # (M,N)
visible_ratio = 1 - occluded.float().mean(1)            # per member instance; class = max over members
```

## Part 2 — ycb_can reference-pose data bug (found via the gate viz)

**Symptom:** `ycb_can` scored ~0% everywhere (max 0.23 over 606 instances) while every other class reached
~1.0. The gate viz scatter showed its reprojected reference points landing on the **unseen back** of the
can. **Root cause:** the reference camera (`ref_pose`, derived from the grasp face normal via
`graspableobj_to_optflow_obj.py:ref_pose_from_grasp`) sits ~180° about the can's vertical (local z) axis
from where instances are placed — it views the opposite side. The reference *image* ("Master Chef" label)
is fine; only the *pose* is wrong. Placement applies no per-instance yaw (`placers.py:70` identity
rotation), so one pose flip fixes all instances. `ycb_can` exists only in `mixed-persp`.

**Fix (pose-only, validated):** rotate `ref_pose` 180° about the can's **true** vertical axis (a circle
fit of the front-half surface — NOT the local origin, which is offset ~(−0.017,−0.009) from the axis and
caused a visible misregistration when we first tried `Rz180 @ ref_pose`). Patched two write sites:
- **Source asset** `$assets/optflow_objects/ycb/ref_pose/ref_pose_0000.npy` (backed up to `.orig.bak`) —
  index 0000 = ycb_can (`meta_0000.yaml`); confirmed byte-identical to the baked dataset pose.
- **Baked metadata** `class_to_ref_pose['ycb_can']` in all 6 `mixed-persp/render00*` dirs (residual write).

**Result:** ycb_can ratios `0.02 → 0.77/0.94/1.0/0.98/0.97/0.94`; the gate viz now admits it (green, e.g.
`render005/f0000` ycb_can 66%, was 4% red).

## Throwaway scripts (scratchpad — passages preserved here)

These one-off scripts lived in the session scratchpad (`/tmp/.../scratchpad/`) and are ephemeral; the
durable record is below. Cheapest-first: smoke → regression → diagnostic scatter → distribution → flip
validation → fix.

### `smoke_gate.py` — end-to-end gate smoke on real data
*Why:* first proof the new gate runs against real `OptFlowSample`/`OptFlowMetadata` (catch API/field
mismatches before anything else).
```python
md = OptFlowMetadata.deserialize(0, render); mm = md.obsmaskmeta
vis = instance_visibility(s, md, ref_cache=ref_cache)              # iid -> ratio
gated = gate_classes_reproj(s, md, 0.30, ref_cache=ref_cache)      # {class: best ratio}
# → cheezit 0.93/0.99 admitted, mustard 0.66, amazon_burgundy 0.09 dropped
```

### `regress_helper.py` — extraction is behavior-preserving (exact)
*Why:* the `convert.py` refactor onto `reproject_local_to_obs` must not change benchmark output. Compares
helper vs the old inline einsum chain on real geometry; asserts byte equality.
```python
cw  = torch.einsum("mij,nj->mni", L, homogeneous(ref_pts))[..., :3]      # OLD inline
cc  = torch.einsum("ij,mnj->mni", w2c, homogeneous(cw))[..., :3]
occ_old = reprojection_occlusion(cc.reshape(-1,3), perspective_projection(cc.reshape(-1,3), torch.eye(4), obs_K), obs_depth, 0.001, 0.005)
occ_new, oc_new, cc_new = reproject_local_to_obs(ref_pts, L, w2c, obs_K, obs_depth)   # NEW helper
assert torch.equal(occ_new.reshape(-1), occ_old)        # PASS over 3.3M points
```

### `scatter_refpoints.py` — diagnostic: which reference points are visible
*Why:* the per-instance ratio (4% for ycb_can) was suspicious; this scatters each instance's reprojected
reference points on the obs (green=visible, red=occluded) to SEE where they land. This is what revealed
ycb_can's points on the can's unseen back.
```python
occ, oc, _ = reproject_local_to_obs(ref_pts[sel], L, w2c, obs_K, obs_depth)
ax.scatter(co[~vis,0], co[~vis,1], s=3, c="red",  alpha=0.35)   # occluded
ax.scatter(co[ vis,0], co[ vis,1], s=3, c="lime", alpha=0.55)   # visible
```

### `diag_ycb.py` — systematic vs random, + dump reference photos
*Why:* decide whether ycb_can is a one-frame fluke (random placement) or a systematic reference bug. Dumps
each class's per-instance ratio distribution across all frames.
```python
print(f"{'class':<18}{'min':>7}{'mean':>7}{'max':>7}{'%>0.30':>8}")
# ycb_can  0.00  0.02  0.23   0%   <- never visible; all other classes reach max ~1.0  => systematic
```

### `validate_flip.py` — prove the flip recovers the ratio, pin the convention
*Why:* before any disk edit, confirm a 180° flip fixes it and which multiply order is correct.
```python
Rz180 = torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=torch.float32)
# orig 0.04 | PRE (Rz180 @ pose, about LOCAL z) 0.50 | POST (pose @ Rz180, camera's own axis) 0.04
```

### `scatter_fixed.py` → `scatter_axisfix.py` — origin-rotate (offset) → axis-rotate (correct)
*Why:* `Rz180 @ ref_pose` rotates about the local ORIGIN; since the can's origin is off its centerline,
the surface lands offset (user caught this). Fix: rotate about the **fitted** axis.
```python
# circle (Kåsa) fit of the front-half surface xy → cylinder axis (a, b)
a, b, _ = np.linalg.lstsq(np.c_[2*x, 2*y, np.ones_like(x)], x**2 + y**2, rcond=None)[0]
Tc[0,3],Tc[1,3] = a,b ; Tmc[0,3],Tmc[1,3] = -a,-b
pose = Tc @ Rz180 @ Tmc @ ref_pose          # rotate about the can's TRUE vertical axis  → 76%, on-surface
```

### `batch_axisfix.py` / `batch_axisfix_3panel.py` — review across many frames
*Why:* confirm the axis-fix holds across viewpoints; the 3-panel adds [observation | reference photo |
scatter] for at-a-glance review.

### `fix_ycb_ref_pose.py` — the actual on-disk fix (asset + baked metadata) + verify
*Why:* apply the validated axis-rotation to the source asset and propagate to existing render dirs without
re-rendering.
```python
fixed = (Tc @ Rz @ Tmc @ ref_pose).numpy().astype(np.float32)
shutil.copy2(npy, npy.with_name("ref_pose_0000.orig.bak")); np.save(npy, fixed)   # (1) asset + backup
for rd in rdirs:                                                                    # (2) baked metadata
    c2rp = OptFlowMetadata.deserialize_field(0, rd, "class_to_ref_pose"); c2rp[CLS] = torch.from_numpy(fixed)
    md = OptFlowMetadata.__new__(OptFlowMetadata); md.class_to_ref_pose = c2rp
    md.serialize(0, rd, only={"class_to_ref_pose"})        # residual write — touches nothing else
```

## Artifacts (gitignored)
- `gate_viz/` — gate-decision PNGs (mixed old px-gate + new ratio-gate; see mtimes).
- `gate_viz/_scatter*/`, `gate_viz/_scatter_axisfix/`, `gate_viz/_scatter_3panel/` — ycb_can diagnostics.
- `gate_viz_postfix/mixed-persp/render00{0..5}/` — 24-frame gate check on the **patched** data (ycb_can green).

## Caveats & follow-ups
- **Pose-only fix:** the asset's reference *image/depth* were left as the original-side render (the gate
  uses only depth geometry, and rotating a symmetric cylinder's depth about its axis is valid). The asset
  reference *photo* still shows the original face — fine for the gate; if an image-consumer (e.g. TOTG
  grasp transfer) ever needs ycb_can, re-render the OptFlowObject from the **negated grasp normal** instead.
- **Reversible:** asset `.orig.bak` saved; the dataset metadata rotation is invertible.
- **Not yet done:** formalize `fix_ycb_ref_pose.py` as a tracked `isaac_datagen` migrate script;
  Stage B proposal/label regen; `expanded-refseg` re-render; verifier retrain + fresh split freeze.

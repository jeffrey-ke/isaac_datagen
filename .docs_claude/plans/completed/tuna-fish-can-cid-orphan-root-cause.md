# Tuna `fish can` cid/iid orphan — root cause investigation

**Status: completed 2026-06-19.**

Follow-up to [`obsmask-cid-iid-validator.md`](obsmask-cid-iid-validator.md), which shipped the
detector and `run_pipeline` gate but explicitly scoped out fixing `cid_iid_masks` / Isaac
semantics. This plan added strategic trace logging, ran a tuna-only repro, and identified the
root cause.

## Problem (recap)

`validate_render_dir` flags graspable iids whose `cid_mask` pixels are all background (`cid < 2`).
On `mixed-persp` and `tuna-only-smoke`, **every orphan is tuna** (`ycb_007_tuna_fish_can`, class
`'fish can'`). Random 3-object smoke tests (amazon only) pass cleanly — bug is tuna-specific, not
general graspable-asset breakage.

## Approach

Append-only trace log at `<render_dir>/cid_iid_trace.log`, written by `cid_iid_trace.py` and
instrumentation in:

| Site | Tag | Hypothesis tested |
|------|-----|-------------------|
| `scene.add_object` | `[author]` | Isaac never stores class on prim (H1) |
| `build_scene` end | `[pre_capture]` | Labels lost before capture (H3) |
| `ObsMaskWriter.__init__` | `[catalog]` | Writer catalog missing `'fish can'` (H7) |
| `obsmask_from_data` | `[frame N]` | `idToSemantics` payload shape (H4/H6) |
| `cid_iid_masks` | `[masks TUNA]` / `[orphan]` | LUT lookup failure (H5/H8) |

Init: `cid_iid_trace.init(render_dir)` in `clean_datagen.reference_segmentation` (and optflow) before
`build_scene`, so author/pre-capture lines survive.

## Repro run

```bash
cd src/isaac_datagen
rm -rf datasets/tuna-only-smoke/render000
uv run isaac-datagen configs/tuna_only_smoke.yaml num_frames=3
uv run isaac-datagen-validate-obsmask datasets/tuna-only-smoke/render000
# → exit 1, 30 orphan rows across 6 frames (2 targets × 3 poses; writer frame ids 0–5)
```

Config: [`src/isaac_datagen/configs/tuna_only_smoke.yaml`](../../../src/isaac_datagen/configs/tuna_only_smoke.yaml)
— 6× replicated tuna, `class: fish can`.

Canonical log on disk:
`datasets/tuna-only-smoke/render000/cid_iid_trace.log` (symlinked under
`/data/user/jeffk/datasets/`).

---

## Discovery

**Root cause: Isaac `instance_segmentation_fast` tokenizes the class semantic on whitespace.**

| Stage | `class` string |
|-------|----------------|
| `meta["class"]` / `add_labels` / `get_labels` at `[author]` and `[pre_capture]` | `'fish can'` ✓ |
| Writer `class_to_cid` at `[catalog]` | `{'fish can': 2}` ✓ |
| `idToSemantics` at capture (`[frame N]`, `[masks TUNA]`) | **`'fish'`** ✗ |

`cid_iid_masks` only maps when `v["class"] in class_to_cid`. Lookup key is `'fish'`; catalog key is
`'fish can'` → `frame_iid_to_cid=MISSING` → `lut_cid=0` → validator orphan (`cids_seen=(0,)`).

Isaac is **not** silently dropping the label — it returns `has_class=True` with a **truncated**
value. Instance names (`ycb_007_tuna_fish_can`) are preserved; only the **class** semantic is split.

Smoking-gun line (repeated every tuna iid, every frame):

```text
[masks TUNA] frame=0 iid=2 ... class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
```

Contrast with working classes: single-token names (`cheezit`, `mustard`, `amazon_red`) have no space
and do not exhibit this mismatch (verified by `random3_smoke` — 0 orphans).

### Hypotheses ruled in/out

| ID | Verdict |
|----|---------|
| H1 — label never on prim | **Ruled out** — `get_labels` shows `'fish can'` at author + pre_capture |
| H3 — label lost before capture | **Ruled out** — pre_capture matches author |
| H4 — no `class` key in idToSemantics | **Ruled out** — key present, value wrong |
| H7 — catalog missing fish can | **Ruled out** — `fish_can_in_catalog=True` |
| H2/H5 — string mismatch at LUT | **Confirmed** — `'fish'` vs `'fish can'` |

---

## Fix directions (not implemented here)

1. **Rename class in asset metadata** to a single token (e.g. `fish_can`) in
   `ycb_graspable/meta/meta_0005.yaml` and any filter/catalog strings; re-render affected dirs.
2. **Normalize in `cid_iid_masks`** — fragile; prefer fixing the authored label to match Isaac output
   semantics policy.
3. Audit other graspable classes for spaces in `meta["class"]` before the next large render.

Trace logging (`cid_iid_trace.py` + call sites) can be removed or gated behind an env var once fix is
verified.

---

## Shipped code (investigation)

- [`src/isaac_datagen/cid_iid_trace.py`](../../../src/isaac_datagen/cid_iid_trace.py)
- Instrumentation: `scene.py`, `isaac_utils.py`, `reference_seg_writer.py`, `clean_datagen.py`

---

## Full trace log (copy)

Source: `datasets/tuna-only-smoke/render000/cid_iid_trace.log` — 2026-06-19 repro.

```text
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1 geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup1' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup1']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2 geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup2' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup2']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3 geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup3' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup3']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4 geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup4' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup4']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[author] wrapper=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5 geo=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5/geo meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup5' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup5']}
[author] class_ok=True instance_ok=True meta_matches_get_labels=True
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can meta_class='fish can' meta_name='ycb_007_tuna_fish_can' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can']}
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1 meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup1' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup1']}
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2 meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup2' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup2']}
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3 meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup3' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup3']}
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4 meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup4' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup4']}
[pre_capture] prim=/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5 meta_class='fish can' meta_name='ycb_007_tuna_fish_can_dup5' get_labels={'class': ['fish can'], 'instance': ['ycb_007_tuna_fish_can_dup5']}
[catalog] classes=['fish can']
[catalog] class_to_cid={'fish can': 2}
[catalog] fish_can_cid=2 fish_can_in_catalog=True
[catalog] fish_can_objects=['ycb_007_tuna_fish_can', 'ycb_007_tuna_fish_can_dup1', 'ycb_007_tuna_fish_can_dup2', 'ycb_007_tuna_fish_can_dup3', 'ycb_007_tuna_fish_can_dup4', 'ycb_007_tuna_fish_can_dup5']
[frame 0] seg_unique_iids=[0, 2, 3, 4, 5, 6, 7] n_idToSemantics=8
[frame 0] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 0] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 0] tuna_iid=5 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3/geo'
[frame 0] tuna_iid=6 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4/geo'
[frame 0] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[frame 0] tuna_iid=7 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5/geo'
[masks] UNMAPPED_CLASS frame=0 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=5 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=6 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=3 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=0 iid=7 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=0 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=0 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=2 name='ycb_007_tuna_fish_can' n_px=12827 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=0 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=0 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=12890 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=0 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=0 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=13759 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=0 iid=5 name='ycb_007_tuna_fish_can_dup3' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'}
[masks TUNA] frame=0 iid=5 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=5 name='ycb_007_tuna_fish_can_dup3' n_px=11914 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=0 iid=6 name='ycb_007_tuna_fish_can_dup4' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'}
[masks TUNA] frame=0 iid=6 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=6 name='ycb_007_tuna_fish_can_dup4' n_px=11971 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=0 iid=7 name='ycb_007_tuna_fish_can_dup5' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'}
[masks TUNA] frame=0 iid=7 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=0 iid=7 name='ycb_007_tuna_fish_can_dup5' n_px=12858 lut_cid=0 cids_seen=(0,)
[write] frame=0 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 5: 'ycb_007_tuna_fish_can_dup3', 6: 'ycb_007_tuna_fish_can_dup4', 3: 'ycb_007_tuna_fish_can_dup1', 7: 'ycb_007_tuna_fish_can_dup5'}
[write] frame=0 n_orphans=6 orphan_iids=[2, 4, 5, 6, 3, 7]
[frame 1] seg_unique_iids=[0, 2, 3, 4] n_idToSemantics=5
[frame 1] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 1] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 1] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[masks] UNMAPPED_CLASS frame=1 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=1 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=1 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=1 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=1 iid=3 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=1 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=1 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=1 iid=2 name='ycb_007_tuna_fish_can' n_px=25208 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=1 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=1 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=1 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=27145 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=1 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=1 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=1 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=27264 lut_cid=0 cids_seen=(0,)
[write] frame=1 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 3: 'ycb_007_tuna_fish_can_dup1'}
[write] frame=1 n_orphans=3 orphan_iids=[2, 4, 3]
[frame 2] seg_unique_iids=[0, 2, 3, 4, 5, 6, 7] n_idToSemantics=8
[frame 2] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 2] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 2] tuna_iid=5 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3/geo'
[frame 2] tuna_iid=6 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4/geo'
[frame 2] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[frame 2] tuna_iid=7 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5/geo'
[masks] UNMAPPED_CLASS frame=2 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=5 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=6 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=3 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=2 iid=7 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=2 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=2 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=2 name='ycb_007_tuna_fish_can' n_px=22496 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=2 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=2 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=21516 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=2 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=2 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=20956 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=2 iid=5 name='ycb_007_tuna_fish_can_dup3' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'}
[masks TUNA] frame=2 iid=5 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=5 name='ycb_007_tuna_fish_can_dup3' n_px=22897 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=2 iid=6 name='ycb_007_tuna_fish_can_dup4' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'}
[masks TUNA] frame=2 iid=6 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=6 name='ycb_007_tuna_fish_can_dup4' n_px=21888 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=2 iid=7 name='ycb_007_tuna_fish_can_dup5' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'}
[masks TUNA] frame=2 iid=7 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=2 iid=7 name='ycb_007_tuna_fish_can_dup5' n_px=21323 lut_cid=0 cids_seen=(0,)
[write] frame=2 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 5: 'ycb_007_tuna_fish_can_dup3', 6: 'ycb_007_tuna_fish_can_dup4', 3: 'ycb_007_tuna_fish_can_dup1', 7: 'ycb_007_tuna_fish_can_dup5'}
[write] frame=2 n_orphans=6 orphan_iids=[2, 4, 5, 6, 3, 7]
[frame 3] seg_unique_iids=[0, 2, 3, 4, 5, 6, 7] n_idToSemantics=8
[frame 3] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 3] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 3] tuna_iid=5 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3/geo'
[frame 3] tuna_iid=6 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4/geo'
[frame 3] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[frame 3] tuna_iid=7 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5/geo'
[masks] UNMAPPED_CLASS frame=3 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=5 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=6 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=3 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=3 iid=7 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=3 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=3 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=2 name='ycb_007_tuna_fish_can' n_px=12827 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=3 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=3 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=12890 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=3 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=3 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=13759 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=3 iid=5 name='ycb_007_tuna_fish_can_dup3' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'}
[masks TUNA] frame=3 iid=5 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=5 name='ycb_007_tuna_fish_can_dup3' n_px=11914 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=3 iid=6 name='ycb_007_tuna_fish_can_dup4' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'}
[masks TUNA] frame=3 iid=6 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=6 name='ycb_007_tuna_fish_can_dup4' n_px=11971 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=3 iid=7 name='ycb_007_tuna_fish_can_dup5' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'}
[masks TUNA] frame=3 iid=7 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=3 iid=7 name='ycb_007_tuna_fish_can_dup5' n_px=12858 lut_cid=0 cids_seen=(0,)
[write] frame=3 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 5: 'ycb_007_tuna_fish_can_dup3', 6: 'ycb_007_tuna_fish_can_dup4', 3: 'ycb_007_tuna_fish_can_dup1', 7: 'ycb_007_tuna_fish_can_dup5'}
[write] frame=3 n_orphans=6 orphan_iids=[2, 4, 5, 6, 3, 7]
[frame 4] seg_unique_iids=[0, 2, 3, 4] n_idToSemantics=5
[frame 4] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 4] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 4] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[masks] UNMAPPED_CLASS frame=4 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=4 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=4 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=4 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=4 iid=3 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=4 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=4 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=4 iid=2 name='ycb_007_tuna_fish_can' n_px=25208 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=4 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=4 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=4 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=27145 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=4 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=4 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=4 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=27264 lut_cid=0 cids_seen=(0,)
[write] frame=4 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 3: 'ycb_007_tuna_fish_can_dup1'}
[write] frame=4 n_orphans=3 orphan_iids=[2, 4, 3]
[frame 5] seg_unique_iids=[0, 2, 3, 4, 5, 6, 7] n_idToSemantics=8
[frame 5] tuna_iid=2 idToSemantics={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can/geo'
[frame 5] tuna_iid=4 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup2/geo'
[frame 5] tuna_iid=5 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup3/geo'
[frame 5] tuna_iid=6 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup4/geo'
[frame 5] tuna_iid=3 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup1/geo'
[frame 5] tuna_iid=7 idToSemantics={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'} path='/World/GeneratedPallets/stack/ycb_007_tuna_fish_can_dup5/geo'
[masks] UNMAPPED_CLASS frame=5 iid=0 class_val='BACKGROUND' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=1 class_val='UNLABELLED' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=2 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=4 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=5 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=6 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=3 class_val='fish' in_class_to_cid=False
[masks] UNMAPPED_CLASS frame=5 iid=7 class_val='fish' in_class_to_cid=False
[masks TUNA] frame=5 iid=2 name='ycb_007_tuna_fish_can' raw_labels={'instance': 'ycb_007_tuna_fish_can', 'class': 'fish'}
[masks TUNA] frame=5 iid=2 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=2 name='ycb_007_tuna_fish_can' n_px=22496 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=5 iid=3 name='ycb_007_tuna_fish_can_dup1' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup1', 'class': 'fish'}
[masks TUNA] frame=5 iid=3 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=3 name='ycb_007_tuna_fish_can_dup1' n_px=21516 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=5 iid=4 name='ycb_007_tuna_fish_can_dup2' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup2', 'class': 'fish'}
[masks TUNA] frame=5 iid=4 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=4 name='ycb_007_tuna_fish_can_dup2' n_px=20956 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=5 iid=5 name='ycb_007_tuna_fish_can_dup3' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup3', 'class': 'fish'}
[masks TUNA] frame=5 iid=5 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=5 name='ycb_007_tuna_fish_can_dup3' n_px=22897 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=5 iid=6 name='ycb_007_tuna_fish_can_dup4' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup4', 'class': 'fish'}
[masks TUNA] frame=5 iid=6 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=6 name='ycb_007_tuna_fish_can_dup4' n_px=21888 lut_cid=0 cids_seen=(0,)
[masks TUNA] frame=5 iid=7 name='ycb_007_tuna_fish_can_dup5' raw_labels={'instance': 'ycb_007_tuna_fish_can_dup5', 'class': 'fish'}
[masks TUNA] frame=5 iid=7 has_instance=True has_class=True class_val='fish' in_class_to_cid=False frame_iid_to_cid=MISSING lut_cid=0
[orphan] frame=5 iid=7 name='ycb_007_tuna_fish_can_dup5' n_px=21323 lut_cid=0 cids_seen=(0,)
[write] frame=5 frame_iid_to_name={2: 'ycb_007_tuna_fish_can', 4: 'ycb_007_tuna_fish_can_dup2', 5: 'ycb_007_tuna_fish_can_dup3', 6: 'ycb_007_tuna_fish_can_dup4', 3: 'ycb_007_tuna_fish_can_dup1', 7: 'ycb_007_tuna_fish_can_dup5'}
[write] frame=5 n_orphans=6 orphan_iids=[2, 4, 5, 6, 3, 7]
```

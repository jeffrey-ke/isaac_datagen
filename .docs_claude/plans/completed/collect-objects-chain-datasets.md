# Make `collect_objects` chain multiple graspable-object datasets

## Context

`configs/randomized.yaml` now lists **three** dataset dirs under `graspable_objects_path`
(`object_dataset_amazon`, `kleenex_dataset`, `ycb_dataset`), and the user has begun rewriting
`collect_objects` to deserialize each and concatenate them into one `list[GraspableObject]`.
The in-progress pseudocode (`clean_datagen.py:67-74`) is close but doesn't run, and the config
schema still describes a single path. This plan finishes the wiring and answers whether naively
chaining is safe for downstream consumers.

The three datasets are structurally identical (same `meta/ reference_image/ grasp_point/ usd_path/`
layout, `meta_NNNN.yaml` with `{class, name}` only), so they are compatible to load and chain.

## Does naive chaining break downstream consumers? — Mostly no

Every downstream identity is keyed off **string metadata**, never list position, so concatenation
order is irrelevant:

- **Class ids (cid):** `reference_seg_writer.py:110-112` computes `class_to_cid` from
  `sorted({obj.meta["class"] ...})`. Deterministic and dataset-order-independent; the same class
  name appearing in two datasets correctly collapses to one cid (intended — "any same-class instance
  is an inlier"), distinct classes get distinct cids. The three datasets' class names are disjoint
  anyway (colors / flowers / `ycb_*`). **Safe.**
- **Instance ids (iid), placement, filters:** Isaac assigns iids from the per-prim semantic labels
  (`scene.add_object` labels by `meta["name"]`/`meta["class"]`); `OccupancyGrid`/stacker and
  `filters.py` operate on object *values*. The `i` in `GraspableObject.deserialize(i, path)` is just
  a per-dataset file index — never persisted as an identifier. **Safe.**

**The one latent hazard — `meta["name"]` must be globally unique.** `name` is an identity string in
two last-wins sinks:
- `scene.add_object` builds the USD prim path `f"{parent}/{obj.meta['name']}"` — a duplicate name
  makes two assets fight for one prim path.
- `reference_seg_writer.py:113` `name_to_class = {o.meta["name"]: o.meta["class"] ...}` silently
  overwrites on a clash, corrupting the iid→name→class chain that `add_proposals.py:80` relies on.

For the three configured datasets the names are already disjoint (`amazon_*`, `kleenex_*`, YCB ids
like `002_master_chef_can`), so naive chaining is correct **today**. Recommendation: add a cheap
assertion so a future name-colliding dataset fails loudly instead of mislabeling. (Auto-namespacing
names by dataset is the alternative, but it rewrites identity strings for no current benefit — skip
it.)

## Changes

### 1. `src/isaac_datagen/runtime_config.py:34` — type the field as a list

```python
# before
graspable_objects_path: str
# after
graspable_objects_path: list[str]
```
Required: the structured-config load (`load_config` → OmegaConf) validates against this schema and
would reject the YAML list otherwise. Both call sites (`clean_datagen.py:43`, `:87`) already pass
`runtime.graspable_objects_path` straight through, so they need no change.

### 2. `src/isaac_datagen/clean_datagen.py` — finish `collect_objects`

Add `import itertools` to the stdlib import block (it's used but never imported), then:

```python
def collect_objects(paths: list[str | Path]) -> list[GraspableObject]:
    """Deserialize every GraspableObject from each dataset dir and concatenate.

    Safe to chain across datasets: cids derive from the sorted class-name set and iids
    from per-prim semantic labels, so neither depends on list position. The one
    requirement is globally-unique meta["name"] — it is the USD prim-path component in
    scene.add_object and the name_to_class key in reference_seg_writer, both last-wins
    on a clash.
    """
    list_of_lists = []
    for p in paths:
        path = Path(p)                       # was Path(path) — wrong variable
        n = len(sorted((path / "meta").glob("meta_*.yaml")))
        list_of_lists.append([GraspableObject.deserialize(i, path) for i in range(n)])

    objects = list(itertools.chain.from_iterable(list_of_lists))

    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate GraspableObject names across datasets: {dupes}")
    return objects
```

Fixes vs. current pseudocode: `Path(p)` (was `Path(path)`), the missing `itertools` import, lowercase
`list[...]` hint (matches the file; `List` isn't imported), and the duplicate-name guard.

## Files

- `src/isaac_datagen/runtime_config.py` — line 34 field type.
- `src/isaac_datagen/clean_datagen.py` — `import itertools`; rewrite `collect_objects` (67-74).

## Verification

- `uv run python -c "from isaac_datagen.clean_datagen import collect_objects"` — imports cleanly.
- Load + chain without booting the sim:
  `uv run python -c "from isaac_datagen.clean_datagen import collect_objects; from isaac_datagen.runtime_config import load_config; r=load_config('src/isaac_datagen/configs/randomized.yaml'); objs=collect_objects(r.graspable_objects_path); print(len(objs), len({o.meta['name'] for o in objs}))"`
  — expect ~58 objects (44 amazon + 7 kleenex + 7 ycb) and an equal unique-name count (no
  duplicate-name `ValueError`), confirming the config now accepts the list and the guard passes.
- Optional smoke test of the full path: `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml dry_run=true` builds the scene with the chained objects without rendering.

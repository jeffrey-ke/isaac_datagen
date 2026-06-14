# GraspableObject filtering registry

## Context

`clean_datagen.py:81` already *calls* `filter_objects(collect_objects(...), runtime.filter_specs)`
in the reference-seg path, but neither `filter_objects` nor `runtime.filter_specs` exists — they
live only as pseudocode in the repo-root scratch file `filter.py`. We want a real, config-driven
**registry of object filters**: a runtime list of `{name, args}` specs that select and configure
stateful filter callables, applied in order to the `[GraspableObject]` before scene build. This lets
a config restrict datagen to one class, shuffle the object pool, etc., without code changes.

The design mirrors the existing registry idiom already used twice in these repos:
- **`segmentation/.../optim.py`** — `OptimConfig` (name + args dataclass) + `configure_optimization`
  doing `getattr(torch.optim, cfg.optimizer)(trainable, **cfg.optimizer_args)`.
- **`isaac_datagen/posers.py`** — `get(name)` = `getattr(sys.modules[__name__], name)`, driven by
  `pose_generation_policy` + `..._args` config fields.

The filter registry = posers' "getattr over this module" + OptimConfig's "name + args dataclass",
iterated over a **list** of specs.

Decisions (confirmed with user):
- `ShuffleFilter` takes a **mandatory** `seed` (no default) — datagen is otherwise fully seeded.
- Wire only the **reference-seg** path (`reference_segmentation()`); leave stereo `main()` untouched.
- `ClassFilter`'s arg is named `class_name` (not `class` — a Python keyword that can't be a splatted
  kwarg); it compares against `GraspableObject.meta["class"]`.

## Changes

### 1. New module `src/isaac_datagen/filters.py` (replaces root `filter.py`)

Import-light on purpose (only `numpy`/`sys`/`dataclasses`; `GraspableObject` hint under
`TYPE_CHECKING`) so `runtime_config` can import `FilterSpec` without dragging in `isaacsim`.

```python
"""GraspableObject filtering registry.

A runtime list of filter specs (name + args) selects and configures stateful filter
callables from this module — same idiom as segmentation's OptimConfig and this repo's
posers registry, but iterated over a list. filter_objects applies them in order.

Filters touch only GraspableObject.meta, so this module imports nothing heavy and is
safe to import from runtime_config (which loads before boot_sim).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaac_datagen.objects import GraspableObject


@dataclass
class FilterSpec:
    name: str                                 # filter class name in this module
    args: dict = field(default_factory=dict)  # splatted into that class's ctor


def make_filters(specs: list[FilterSpec]) -> list:
    """Instantiate each spec by name from this module, splatting its args."""
    return [getattr(sys.modules[__name__], spec.name)(**spec.args) for spec in specs]


def filter_objects(objects: list[GraspableObject],
                   specs: list[FilterSpec]) -> list[GraspableObject]:
    """Apply each filter in order; raise if the candidate set is ever empty."""
    for f in make_filters(specs):
        if not objects:
            raise ValueError(f"no GraspableObjects left to feed filter {f!r}")
        objects = f(objects)
    return objects


class ShuffleFilter:
    """Return a deterministic permutation of the objects (seed is mandatory)."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        order = np.random.RandomState(self.seed).permutation(len(objects))
        return [objects[i] for i in order]


class ClassFilter:
    """Keep only objects whose meta['class'] equals class_name."""

    def __init__(self, class_name: str):
        self.class_name = class_name

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        return [o for o in objects if o.meta["class"] == self.class_name]
```

Then **delete the root `filter.py`** scratch file (its content now lives in the package).

### 2. `src/isaac_datagen/runtime_config.py` — add the config field

Add near the `pose_generation_policy` block (the analogous registry-driven field):

```python
from isaac_datagen.filters import FilterSpec   # top of file, with the other imports
```
```python
# Object-filter registry (filters.py): ordered list of {name, args} specs, each
# splatted into the named filter class and applied in order to the GraspableObject
# pool before scene build. Empty = no filtering (default; backward compatible).
filter_specs: list[FilterSpec] = field(default_factory=list)
```
`field` is already imported; no `__post_init__` change needed. OmegaConf (`>=2.3.0`) coerces a YAML
list of `{name, args}` dicts into `FilterSpec` instances via the typed-list annotation.

### 3. `src/isaac_datagen/clean_datagen.py` — import the function

The call at lines 81-84 already exists; only the import is missing:

```python
from isaac_datagen.filters import filter_objects   # with the other isaac_datagen imports
```

`main()` (stereo) is intentionally left calling `collect_objects` directly (no filtering).

### 4. Config usage (example for `configs/randomized.yaml`)

```yaml
filter_specs:
  - name: ClassFilter
    args: {class_name: cup}
  - name: ShuffleFilter
    args: {seed: 1}
```
Order matters: this keeps only `cup` objects, then deterministically shuffles them.

### 5. CLAUDE.md module index — add a row

`| `filters.py` | GraspableObject filter registry (name+args specs) | `filter_objects`, `FilterSpec`, `ClassFilter`, `ShuffleFilter` |`

## Verification

1. **Unit (no sim):**
   ```
   uv run python -c "
   from isaac_datagen.filters import filter_objects, FilterSpec
   O=lambda c:type('O',(),{'meta':{'class':c}})()
   objs=[O('cup'),O('box'),O('cup'),O('can')]
   out=filter_objects(objs,[FilterSpec('ClassFilter',{'class_name':'cup'}),FilterSpec('ShuffleFilter',{'seed':0})])
   print([o.meta['class'] for o in out])   # -> ['cup','cup'] in a seed-0 order
   "
   ```
   Also confirm the empty-set guard raises (`ClassFilter` with a class_name absent from the pool).
2. **Config coercion (no sim, no path asserts):**
   ```
   uv run python -c "
   from omegaconf import OmegaConf
   from isaac_datagen.runtime_config import RuntimeConfig
   s=OmegaConf.structured(RuntimeConfig)
   m=OmegaConf.merge(s,{'filter_specs':[{'name':'ClassFilter','args':{'class_name':'cup'}}]})
   print(OmegaConf.to_container(m.filter_specs))
   "
   ```
   Confirms the typed `list[FilterSpec]` merges a YAML list of dicts.
3. **End-to-end (optional):** add the §4 block to `randomized.yaml` and run
   `uv run clean_datagen.py src/isaac_datagen/configs/randomized.yaml dry_run=true` — the dry-run
   path builds the scene from the filtered pool and exports the usdz without RTX capture.
```
```

## Post-completion: `ClassFilter` → `MetaFilter` (2026-06-14)

`ClassFilter` was generalized in two steps, ending as a renamed `MetaFilter`. `FilterSpec`,
`make_filters`, `filter_objects`, and `ShuffleFilter` are unchanged; the registry resolves filters by
class name (`getattr` over the module), so renaming the class only requires updating its config `name`.

**Step 1 — add a count cap, keep other matches.** A `max` arg caps how many objects of the selected
class survive; objects of *other* classes pass through untouched. (We deliberately chose "cap one
class, keep the rest" over "keep only this class, capped" — the latter would have dropped the pool
below `occupancy_grid`'s full-wall minimum. See §"Capacity interaction" below.)

**Step 2 — match any meta field by glob.** Replace the fixed `meta["class"]` lookup + `==` with an
arbitrary `key` + a Unix find-style glob `value` (`fnmatch.fnmatchcase`: case-sensitive,
platform-independent; supports `*`, `?`, `[seq]`, `[!seq]`). Renamed `ClassFilter` → `MetaFilter`
since it no longer filters on class specifically. `class`-equality is now just `key=class` with a
literal (wildcard-free) `value`.

Before → after:

```python
# before (§1)
class ClassFilter:
    """Keep only objects whose meta['class'] equals class_name."""
    def __init__(self, class_name: str):
        self.class_name = class_name
    def __call__(self, objects):
        return [o for o in objects if o.meta["class"] == self.class_name]

# after (filters.py) — also: `import fnmatch` at module top
class MetaFilter:
    """Cap how many objects matching a meta-field glob survive: walk `objects` in order
    and keep at most `max` whose meta[key] fnmatches the Unix find-style glob `value`
    (e.g. key='name', value='amazon_*'), dropping the overflow among matches. Objects
    that don't match pass through untouched.
    """
    def __init__(self, key: str, value: str, max: int):
        self.key = key
        self.value = value
        self.max = max
    def __call__(self, objects):
        kept = 0
        out = []
        for o in objects:
            if fnmatch.fnmatchcase(str(o.meta[self.key]), self.value):
                if kept >= self.max:
                    continue
                kept += 1
            out.append(o)
        return out
```

Config (`configs/randomized.yaml`) — supersedes the §4 example:

```yaml
filter_specs:
  - name: ShuffleFilter
    args: {seed: 0}
  - name: MetaFilter
    args: {key: name, value: 'amazon_*', max: 44}
```

**Capacity interaction.** Default `placement: occupancy_grid` is full-wall and requires
`len(objects) >= prod(pallet_dims)` (= 44 for `[11,1,4]`); the amazon pool is *exactly* 44, so any
net drop trips the capacity guard in `create_stack_of_objects`. Every object's `meta['name']` is
`amazon_*`, so `max: 44` keeps all 44 → safe but **inert**. Lower `max` (and shrink `pallet_dims` /
switch placement) to actually drop matches.

**Stale doc:** `CLAUDE.md`'s `filters.py` module-index row still lists the old export `ClassFilter`;
it should read `MetaFilter`.

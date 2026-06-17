# Unify `objects_path` (collapse the two per-mode object-dir fields)

**Status:** completed 2026-06-17.

## Problem

`RuntimeConfig` carried two object-dataset path fields, one per `mode`:

```python
optflow_objects_path: list[str] = field(default_factory=list)   # used by mode=optflow
graspable_objects_path: list[str] = field(default_factory=list) # used by mode=reference_segmentation
```

Two footguns:

1. **Redundant with `mode`.** `mode` already selects the orchestrator, and each
   orchestrator reads exactly one of the two fields. Only one is ever live per run.
2. **Asymmetric validation.** `__post_init__` only required `optflow_objects_path`
   (when `mode=optflow`). `reference_segmentation` with an empty
   `graspable_objects_path` passed validation and silently rendered **nothing** —
   `collect_objects([])` → no objects, no error.

## Decision

Collapse to a single `objects_path`, required for **both** modes. The orchestrator
function already knows the mode, so it picks the right collector
(`collect_objects` for `reference_segmentation`, `collect_preoptflow` for `optflow`);
the field's only job is to say *where* the objects live, not *which kind*. This is
knowledge-in-data: `mode` is the single source of truth for the collector, and the
path stops re-encoding it.

Rejected alternative: keep two fields but make validation symmetric via a
`{mode: attr}` table. Still leaves two fields where one is always dead, and setting
the wrong one stays silently ignored.

## Before → after

```python
# runtime_config.py — fields
- optflow_objects_path: list[str] = field(default_factory=list)
- graspable_objects_path: list[str] = field(default_factory=list)
+ objects_path: list[str] = field(default_factory=list)

# runtime_config.py — __post_init__ (now fires for BOTH modes)
- if self.mode == "optflow":
-     assert self.optflow_objects_path, "mode=optflow requires optflow_objects_path"
+ assert self.objects_path, f"mode={self.mode} requires objects_path"

# clean_datagen.py
- collect_objects(runtime.graspable_objects_path)      # reference_segmentation
- collect_preoptflow(runtime.optflow_objects_path)     # optflow
+ collect_objects(runtime.objects_path)
+ collect_preoptflow(runtime.objects_path)
```

## Blast radius (every reader of the renamed fields)

OmegaConf merges the YAML against the structured `RuntimeConfig` schema in struct
mode, so a stale key now fails loudly with `ConfigKeyError` rather than being
ignored — which forced renaming all configs, not just the requested one:

- `configs/randomized.yaml` — `optflow_objects_path:` → `objects_path:`
- `configs/mixed.yaml` — `graspable_objects_path:` → `objects_path:`
- `debug_scripts/debug_scene.py` — `runtime.graspable_objects_path` → `runtime.objects_path`
- `debug_scripts/debug_occupancy.py` — `cfg["graspable_objects_path"]` → `cfg["objects_path"]`

Left untouched: completed-plan docs and serialized `runtime.yaml` dumps under
`datasets/`, `temp-render/`, `debug/` — historical output, never re-read as config input.

## Verified

`OmegaConf.structured(RuntimeConfig)` exposes `objects_path` only (both old names
gone); a merge carrying `graspable_objects_path` raises `ConfigKeyError` (loud
failure on any un-migrated config).

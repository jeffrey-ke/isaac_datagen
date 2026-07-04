# Jagged column heights + per-object jitter — `UntilExhaustedStacker`

## Context

Third in the stacker-variation lineage (`depth-jitter-stacker.md` → `gap-jitter-stacker.md` →
this). `UntilExhaustedStacker` (`isaac_datagen/src/isaac_datagen/placers.py`) packed placed objects
into columns of a **fixed** `column_height` and seated each object exactly on the one below with a
hardcoded `EPSILON = 0.002` gap. Two changes make the synthetic stacks more varied/realistic:

1. **Jagged column heights** — each column's height is drawn uniformly in `[1, max_column_height]`
   (config-specified max), consuming objects greedily until exhausted, instead of uniform chunks.
2. **Per-object (x, y) jitter** — objects in a column get a small lateral perturbation, and the
   jitter/gap magnitude `epsilon` (formerly the hardcoded class constant) is now a config value.

This landed on top of a **reusable-parts refactor** of the same file (the natural precondition: the
jagged swap is a one-line chunker change *once* the chunking is a clean seam). The god-ish
`__init__` was split into independent primitives and a thin calling sequence, so the placement math
is now pure-of-the-stage and unit-testable.

### Decisions
- **Jagged = base stacker only.** `ShelfPlacer` keeps a fixed `column_height` — random heights
  would split its class-sorted runs at arbitrary boundaries and break one-class-per-column.
- **Jitter x, y only** — z is left clean so objects still rest exactly on the floor / on each other.
- **One `epsilon`** drives both the inter-object vertical gap and the jitter σ (matches the old dual
  use of `EPSILON`); default stays `EPSILON = 0.002`.
- **`column_height` → `max_column_height`** on the base only (meaning changed exact→max);
  `ShelfPlacer` keeps `column_height`.

## Approach

### Reusable-parts refactor (`src/isaac_datagen/placers.py`)

Extracted the trapped logic into module-level primitives, each independently callable / testable:

```python
class Vec3(NamedTuple): x: float; y: float; z: float          # size_of/center_of return this (.z reads)

def centroid_at_point(center, target):                         # pure geometry: origin s.t. centroid=target
    return tuple(t - c for t, c in zip(target, center))        # (size is NOT an input — test 5)

def compute_cols_stride(columns, min_gap, max_gap, sizes):     # the col_xs chain (gap-jitter plan), extracted
    ...                                                         # pure numpy; unit-testable on made-up widths

def fixed_columns(prim_paths, height): ...                     # deques of exactly `height`
def jagged_columns(prim_paths, max_height):                    # deques of random height in [1, max_height]
    cols, i = [], 0
    while i < len(prim_paths):
        h = int(np.random.randint(1, max_height + 1))
        cols.append(deque(prim_paths[i:i + h])); i += h
    return cols
```

`__init__` became a three-phase calling sequence over a shared `_seat` method (measure → layout →
stack); `self.columns` is the lone escaping value of the chunking block, so swapping the chunker is
localized:

```python
# UntilExhaustedStacker.__init__  (jagged + epsilon)
self.columns = jagged_columns(prim_paths, max_column_height)
self._seat(min_y, max_y, min_gap, max_gap, epsilon)

def _seat(self, min_y, max_y, min_gap, max_gap, epsilon):
    prims = [p for col in self.columns for p in col]           # columns = single source of truth
    sizes, centers = {p: size_of(p) ...}, {p: center_of(p) ...}   # MEASURE (impure)
    col_xs = compute_cols_stride(self.columns, min_gap, max_gap, sizes)   # LAYOUT (pure)
    col_ys = np.random.uniform(min_y, max_y, size=len(self.columns)).tolist()
    self.columns_xy = list(zip(col_xs, col_ys))
    for col, xy in zip(self.columns, self.columns_xy):         # STACK (pure)
        floor_z = 0.0
        for p in col:
            target = (*xy, floor_z + sizes[p].z / 2.0)
            bx, by, bz = centroid_at_point(centers[p], target)
            jx, jy = np.random.normal(0, epsilon, size=2)      # x,y jitter only; z stays clean
            self._placements[p] = ((bx + jx, by + jy, bz), (0.0, 0.0, 0.0))
            floor_z += sizes[p].z + epsilon                     # gap now configurable
```

`ShelfPlacer` no longer forwards to `super().__init__`; it sorts by class, builds `fixed_columns`,
and reuses `_seat` — fixed height retained on purpose (documented).

### Config (`src/isaac_datagen/`)

- `configs/jagged-expanded-refseg-v2.yaml` **(new)** — derived from `expanded-refseg-v2.yaml`:
  `max_column_height: 3`, `epsilon: 0.01` (up from 0.002 default), `ReplicateFilter count 5→10`,
  `num_frames: 20`/`num_targets: 5` (100 obs), writes to `datasets/jagged-expanded-refseg-v2`.
- `datasets/debug/render996/runtime.yaml` and `configs/expanded-refseg-v2.yaml` —
  `column_height` → `max_column_height`.
- `runtime_config.py` — updated the `placement_args` example comment.

## Critical files
- `isaac_datagen/src/isaac_datagen/placers.py` — refactor + jagged + jitter + epsilon.
- `isaac_datagen/src/isaac_datagen/configs/jagged-expanded-refseg-v2.yaml` — new dataset config.
- `isaac_datagen/src/isaac_datagen/{configs/expanded-refseg-v2.yaml, datasets/debug/render996/runtime.yaml}` — key rename.

## Known follow-up (NOT done)
The `column_height` → `max_column_height` rename was applied only to the two configs above. **Seven
other configs still pass `column_height` under `placement: UntilExhaustedStacker` and will now
raise** (`amazon`, `expanded-refseg`, `mixed`, `random3_smoke`, `shelf`, `staggered`,
`tuna_only_smoke`). None use `ShelfPlacer`, so all seven should migrate to `max_column_height`.

## Verification
- `python3 -m py_compile placers.py` — OK.
- Pure-Python chunker smoke (no stage): `jagged_columns` heights ∈ `[1, max]`, cover every prim once,
  order preserved, variety present; `fixed_columns` exact chunks.
- **End-to-end full pipeline** on `jagged-expanded-refseg-v2` (render → proposals → inliers):
  - Smoke: `test-jagged-obsmask/render000`, 8 obs (`num_frames=8 num_targets=1`) — obs/mask clean.
  - `idx=0..5` ✅ all complete — `render000..005/` each 100 obs + 100 proposals + inliers stats
    (600 obs total; idx=0 inliers 12576/188352). Ran sequentially (GPU-bound; one Isaac at a time).
    Seeds distinct: `effective_seed = seed(1) + idx` → 1..6, one per render dir.

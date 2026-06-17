# Plan: `ShelfPlacer` placement policy

## Context

We want shelf-like columns of a single class (a column of cans next to a column of boxes), instead
of columns filled in arrival order. This is not new layout math: `UntilExhaustedStacker`
(`objects.py:379`) chunks `prim_paths` into consecutive runs of `column_height`, one run per column.
So if we sort `prim_paths` so same-class paths are adjacent, the chunks become same-class columns for
free. `ShelfPlacer` is a thin `UntilExhaustedStacker` subclass that sorts by class label, then
`super().__init__(...)`.

This branch stays **purely additive** (only adds a class to `objects.py`) so it rebases trivially
onto the sibling `placers.py` registry refactor — registering the class there is a post-rebase
one-liner, not part of this diff.

## Why it works

- **Reordering is safe.** The parent keys `self._placements` by prim-path *string* and
  `organize_objects` (scene.py) looks up each original path by key — so permuting the list handed to
  `super().__init__` changes only column *assignment*; `__call__`/`graspability` stay correct.
- **Class is on the stage, not in the path.** `add_object` (scene.py:36-56) names the wrapper prim
  after the *instance* name but labels the child `geo` prim with the class via
  `add_labels(geo, labels=[obj.meta["class"]], instance_name="class")`. Read it back with
  `isaacsim.core.utils.semantics.get_labels(geo) -> {"class": [name], ...}` — same stage access the
  parent already uses in its own ctor.

## Change: `src/isaac_datagen/objects.py` — add after `UntilExhaustedStacker` (line 448)

```python
class ShelfPlacer(UntilExhaustedStacker):
    """UntilExhaustedStacker that groups same-class objects into the same columns.

    Sorts prim_paths by their semantic "class" label (read off the loaded stage — the
    wrapper path encodes only the instance name) so each run of `column_height` adjacent
    paths is one class: a shelf of cans next to a shelf of boxes. Layout/graspability are
    inherited unchanged. Stable sort preserves within-class order; a class count not a
    multiple of `column_height` yields one mixed boundary column, as with plain chunking.
    """

    def __init__(self, prim_paths, column_height):
        from isaacsim.core.utils.semantics import get_labels
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()

        def class_label(prim_path):
            geo = stage.GetPrimAtPath(f"{prim_path}/geo")  # add_object labels geo as "class"
            labels = get_labels(geo)
            if not labels.get("class"):
                raise ValueError(f"ShelfPlacer: no 'class' label on {prim_path}/geo")
            return labels["class"][0]

        super().__init__(sorted(prim_paths, key=class_label), column_height)
```

## Post-rebase (not in this branch's diff)

Once rebased onto the registry branch, register + select it:

- `placers.py`: add `from isaac_datagen.objects import ShelfPlacer  # noqa: F401`
- config: `placement: ShelfPlacer` + `placement_args: {column_height: 5}`

`ShelfPlacer(prim_paths, column_height)` already fits the registry's `(prim_paths, **kwargs)`
contract, so no scene.py/runtime_config.py edits.

## Verification

1. `uv run python -c "from isaac_datagen.objects import ShelfPlacer"` — bare import works (isaacsim
   imports are inside `__init__`).
2. After rebase, a short multi-class run (`placement=ShelfPlacer`,
   `placement_args={column_height:3}`): inspect the exported scene USDZ and confirm each column is a
   single class except at most one boundary column per class, and the layout otherwise matches an
   `UntilExhaustedStacker` run at the same `column_height`.

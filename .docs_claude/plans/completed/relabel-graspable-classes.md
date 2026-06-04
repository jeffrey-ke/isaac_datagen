# Interactive class relabeling for serialized GraspableObjects (`relabel_classes`)

## Context

Every `GraspableObject` in `object_dataset_amazon` (44 samples, in
`~/repo/visual_servoing/datagen2_isaacsim/object_dataset_amazon/`) was serialized with the
same `meta["class"] = "amazon_box"`. The reference images are color-distinct box fronts —
many are near-duplicates of each other, so they should share a class, but there are roughly
a dozen distinct color families (red, kelly green, sea green, magenta, teal, steel blue,
olive, gold, orange, purple, navy, lime, pink) that deserve distinct class names. We needed
a way to (a) see all 44 reference images at once to judge the color clusters, and (b)
assign new one-word class names per sample, rewriting the dataset in place.

## Approach: deserialize → mutate meta → residual serialize

Same pattern as `preprocess.py` from the original visual_servoing repo
(`.segmentation.bak/preprocess.py`): deserialize each sample, transform it, serialize it
back. The key mechanism is **residual serialization** —
`sample.serialize(idx, dir, only={"meta"})` rewrites only `meta/meta_XXXX.yaml` and leaves
every other field's file untouched. This matters specifically for `GraspableObject`:
the `UsdPath` serializer is copy-on-serialize (`shutil.copy`), and a full re-serialize
would try to copy each dataset usdz onto itself (`SameFileError`). `only={"meta"}` never
touches that path.

## What was built

`src/isaac_datagen/relabel_classes.py` — standalone script (no Isaac boot;
`objects.py` keeps its isaacsim imports function-local, so `GraspableObject` imports clean):

1. **Index discovery** — `collect_indices` globs `meta/meta_*.yaml` and parses the
   4-digit serialization indices (don't assume contiguity).
2. **Grid png** — deserializes all samples, writes an 8-column matplotlib grid of
   `reference_image` fields to `<dataset>/reference_grid.png`, each tile titled
   `"{idx}: {meta['class']}"`. `--grid-only` stops here.
3. **Interactive relabel loop** — shows the grid non-blocking (`plt.show(block=False)`),
   then per sample prompts `[NNNN] {name} class={current!r} -> ` on stdin. Enter keeps the
   current class and skips; any answer overwrites `meta["class"]` and immediately
   residual-serializes that one sample, so quitting mid-run loses nothing.

## Usage

```
uv run src/isaac_datagen/relabel_classes.py <dataset_dir> [--grid-only]
```

The relabel loop needs interactive stdin — run it in a real terminal (or via the `!`
prefix in a Claude Code session), not through a captured-output shell.

## Outcome (2026-06-03)

All 44 samples relabeled in place — no `amazon_box` remains. **19 unique classes**:

| count | classes |
|---|---|
| 6 | pink, cyan |
| 4 | green |
| 3 | yellow, teal, red |
| 2 | purple, pistachio, orange, mustard, burgundy, blue |
| 1 | tea, olive, mauve, indigo, cream, brown, aquamarine |

(`tea` at index 9 is distinct from `teal` — a pale tea-green box.)
The labeled grid lives at `<dataset>/reference_grid.png`.

## Decisions

- **Per-sample serialize inside the loop**, not batch-at-end: each answer is durable
  immediately; an interrupted session resumes by just re-running (already-renamed samples
  show their new class in the prompt and can be enter-skipped).
- **Grid labeled by serialization index** (not name) because the index is what the prompt
  loop and the on-disk filenames key on.
- Grid png written into the dataset dir itself (`reference_grid.png`) — it sits next to
  the field subdirs but doesn't collide with the `{field}/{field}_NNNN` layout.

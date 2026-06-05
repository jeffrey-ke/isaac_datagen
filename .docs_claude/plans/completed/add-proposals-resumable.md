# Make `add_proposals` resumable (skip-if-exists + atomic writes)

## Context

Phase-2 (`add_proposals.py`) is the long-running pass of the pipeline: one proposer
forward per (frame, visible class), over potentially thousands of frames. Today a killed
or crashed run loses everything — the frame loop is `for idx in range(n_frames)` with no
awareness of what's already on disk, and re-running reprocesses and overwrites every frame.

The output format already encodes progress perfectly: one residual
`proposals/proposals_{idx:04d}.pt` per frame. So resume = **skip frames whose proposals
file already exists**. The one hole is the frame being written at kill time: a partial
`.pt` from an interrupted `torch.save` would look "done" and silently survive the skip.
Decision (user): close that hole with **atomic writes in `vision_core`'s
`SerializableSample.serialize`** (tmp + `os.replace`) — an existing file is then
guaranteed complete, and every writer in the pipeline gets crash-safe writes for free
(shared primitives live in vision_core). No frame-range sharding flags (declined —
skip-if-exists is enough); no progress manifest (duplicates state the per-frame files
already encode, can desync).

Force-redo story: delete `proposals/` (or individual `proposals_NNNN.pt` files) and
re-run — no `--force` flag.

## Changes

### 1. `~/repo/vision_core/src/vision_core/datastructs.py` — atomic `serialize`

Current code (`SerializableSample.serialize`, lines 70-84):

```python
def serialize(self, idx: int, directory: Path, only: set[str] | None = None):
    """Serialize fields to ``directory/{field}/{field}_{idx:04d}{ext}``. …"""
    directory = Path(directory)
    for f in fields(self):
        if only is not None and f.name not in only:
            continue
        subdir = directory / f.name
        subdir.mkdir(parents=True, exist_ok=True)
        ext, write_fn, _ = self._get_serializer(f.type)
        write_fn(subdir / f"{f.name}_{idx:04d}{ext}", getattr(self, f.name))
```

becomes (last two lines of the loop body change; docstring gains one sentence; add
`import tempfile` to the top imports):

```python
        ext, write_fn, _ = self._get_serializer(f.type)
        # Atomic write: unique tmp in the SAME dir (os.replace can't cross
        # filesystems), then rename — a file that EXISTS is always COMPLETE
        # (resumable passes skip-if-exists). suffix=ext so np.save doesn't
        # append .npy and PIL picks the format from the suffix.
        fd, tmp = tempfile.mkstemp(dir=subdir, prefix=f".{f.name}_", suffix=ext)
        os.close(fd)  # write lambdas open the path themselves
        try:
            write_fn(Path(tmp), getattr(self, f.name))
            os.replace(tmp, subdir / f"{f.name}_{idx:04d}{ext}")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)  # cleans up on exception/Ctrl-C (not SIGKILL)
            raise
```

Notes:
- `os` is already imported (line 3); add `tempfile`.
- `tempfile.mkstemp` gives a collision-proof unique name; `dir=subdir` is essential —
  the default `/tmp` is often a different filesystem and `os.replace` would raise
  `EXDEV`. Same-dir rename is atomic.
- Cleanup covers Python-level exits (exceptions, `KeyboardInterrupt`). A hard kill
  (SIGKILL/OOM/power) can still leave one stray tmp — no library can run finalizers
  past SIGKILL — but it's a hidden dotfile and harmless: the skip check and frame
  count both glob exact final names (`obs_*.png`, `proposals_{idx:04d}.pt`).
- All four write lambdas (`PIL.Image.save`, `np.save`, `torch.save` via
  `_DICT_PT_SERIALIZER`, `json.dump`) take a path and open it themselves, which is
  why `mkstemp` + close beats `NamedTemporaryFile` (whose open handle we'd discard
  immediately). `suffix=ext` keeps the extension-sensitive ones correct: `np.save`
  won't append another `.npy`, PIL infers PNG.
- Final filenames are unchanged → fully backward compatible with existing render dirs
  and all other repos using `serialize`.

### 2. `src/isaac_datagen/add_proposals.py` — skip-if-exists loop + honest frame count

Current loop head (lines 44-48):

```python
    n_frames = len(list((render_dir / "obs").iterdir()))
    total_pts = 0
    bar = tqdm(range(n_frames), desc=render_dir.name, unit="frame")
    for idx in bar:
        om = ObsMask.deserialize(idx, render_dir)
```

becomes:

```python
    # Glob the real frame files (not iterdir) so a stray .tmp from a killed
    # phase-1 run can't inflate the count.
    n_frames = len(list((render_dir / "obs").glob("obs_*.png")))
    # Resume: a proposals file that exists is complete (serialize is atomic),
    # so skip it. Delete proposals/NNNN.pt (or the whole subdir) to force redo.
    prop_ext = PreReferenceSegSample._get_serializer(dict)[0]   # ".pt"
    done = {idx for idx in range(n_frames)
            if (render_dir / "proposals" / f"proposals_{idx:04d}{prop_ext}").exists()}
    if done:
        print(f"resume: {len(done)}/{n_frames} frames already have proposals — skipping",
              flush=True)

    total_pts = 0
    bar = tqdm(range(n_frames), desc=render_dir.name, unit="frame")
    for idx in bar:
        if idx in done:
            continue
        om = ObsMask.deserialize(idx, render_dir)
```

- `prop_ext` is derived from the struct's own serializer table instead of hardcoding
  `".pt"`, so the skip check can't drift from what `serialize` actually writes.
- Final summary line gains the skip count:
  `print(f"done: {n_frames - len(done)} new + {len(done)} skipped frames, {total_pts} new proposal points → …")`.
- Module docstring: add one sentence — "Resumable: frames whose
  ``proposals/proposals_NNNN.pt`` already exists are skipped (writes are atomic, so an
  existing file is complete); delete files to force redo."
- Note: when the proposer model load (the expensive startup) happens *after* the resume
  scan would be nicer for the fully-done case, but the load (line 38-41) already sits
  after `md` deserialize and before the loop; optionally short-circuit:
  if `len(done) == n_frames`, print and exit before building the proposer at all.
  Cheap win — included: move the `done` scan above the `from reference_matching import
  proposal …` block and early-return when nothing is left to do.

### Untouched

`add_inlier_data.py` (NN-free and fast — resumability not worth it, but it inherits
atomic writes from vision_core anyway), `reference_seg_writer.py` (inherits atomicity),
`runtime_config.py` (no new fields), `pyproject.toml`.

## Plan-file bookkeeping

Copy this plan to `.docs_claude/plans/active/add-proposals-resumable.md` at
implementation start (project convention); move to `plans/completed/` with an Outcome
section when done.

## Verification

1. **Atomicity round-trip (NN-free, no Isaac)**: small script — serialize an
   `ImageInlierSample`-style struct with every serializer type (png/npy/pt/json) to a tmp
   dir; assert final filenames are unchanged, contents round-trip via `deserialize`, and
   no tmp dotfiles remain. Then monkeypatch one write lambda to raise mid-write and
   assert: exception propagates, final file absent, AND the tmp was unlinked by the
   except-cleanup (subdir contains only complete files).
2. **Resume end-to-end** on the existing 4-frame dir
   `src/isaac_datagen/cid-mask-verify/render900` (proposals already present):
   - `uv run isaac-datagen-proposals configs/randomized.yaml idx=900 dataset_dir=cid-mask-verify`
     (from `src/isaac_datagen` — relative config paths resolve against cwd; no
     PYTHONPATH hack: the gim `tools` shadowing was already fixed in
     `reference_matching/proposal.py`)
     → prints `resume: 4/4 … skipping`, exits before loading the proposer, mtimes of all
     `proposals_*.pt` unchanged.
   - `rm proposals/proposals_0002.pt`, re-run → loads proposer, processes exactly frame
     2, other three mtimes unchanged; `proposals_0002.pt` round-trips via
     `PreReferenceSegSample.deserialize(2, …)` and matches phase-3 expectations
     (`isaac-datagen-inliers` still runs clean on the dir).
3. **Interrupt test**: start a fresh run on a larger dir, Ctrl-C mid-run, re-run —
   resume message reports the completed prefix, job continues from the first missing
   frame, and no tmp dotfiles survive in the subdirs (`ls -a` — Ctrl-C path exercises
   the except-cleanup).

## Outcome (2026-06-05)

Shipped, with one refinement over the planned code: the inline mkstemp/replace block
became a reusable context manager **`vision_core.datastructs.atomic_write_path(final)`**
(yield tmp path in final's dir → `os.replace` on success, unlink on exception/Ctrl-C),
and `serialize` is a 2-line `with` over it. Discussed and declined: stdlib
`NamedTemporaryFile` (manages deletion not replacement; hands back an open handle our
path-taking write lambdas can't use; 3.12's `delete_on_close=False` escape hatch absent
on this 3.11 venv) and an `os.fsync` before the rename (power-loss durability — not
needed for the killed-job use case; user chose to keep the original).

Verification, all passed:
- **Atomicity round-trip** (`/tmp/test_atomic_serialize.py`): exact final filenames
  across png/npy/pt/json, content round-trip (incl. NaN occlusion), and a monkeypatched
  mid-write `KeyboardInterrupt` → exception propagates, final absent, tmp unlinked,
  frame-count glob uninflated. (Gotcha: patch `ObsMask._serializers`, not the base —
  subclasses carry their own merged dict.)
- **Resume end-to-end** on `cid-mask-verify/render900`: fully-done dir → `resume: 4/4 …
  skipping`, exits before the proposer load, mtimes untouched; after `rm
  proposals_0002.pt` → `done: 1 new + 3 skipped frames, 29200 new proposal points`,
  other mtimes byte-identical, no tmp dotfiles; regenerated frame round-trips via
  `PreReferenceSegSample.deserialize` and phase-3 (`isaac-datagen-inliers`) runs clean
  (78,344/172,613 inliers).
- **Bonus finding**: the `PYTHONPATH=~/repo/gim` workaround is stale — the gim `tools`
  shadowing was already fixed by the explicit-path loader in
  `reference_matching/proposal.py`; verified by a full proposer run without it.
  Corrected the "Durable fix TBD" note in `plans/completed/cid-mask-dual.md`.

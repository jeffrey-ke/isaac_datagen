# Add a `--help`/tldr cheat-sheet to the isaac_datagen CLI

## Context

`clean_datagen.py` (console script `isaac-datagen`) and `run_pipeline.py`
(`isaac-datagen-pipeline`) currently parse args as raw `sys.argv[1]`/`sys.argv[2:]`
with zero argparse — running either with no arguments raises a bare `IndexError` or a
one-line usage string, and there is no `--help` at all. Meanwhile the pipeline has grown
to 14 console scripts, ~10 config variants, and a ~50-field `RuntimeConfig`, but the only
per-run knowledge that actually gets touched is a handful of dotlist keys (`idx`,
`num_frames`, `num_targets`, `mode`, occasionally a couple of ablation fields) — none of
which is written down anywhere reachable from the command line itself. The user wants a
`clean_datagen.py --help`-style tldr: what commands exist, what to override, what each
config is for, and where the important paths live.

**Can uv facilitate this?** No — `uv run <script>` just execs the console script; any
`--help` output has to come from that script's own argument parsing. uv's only relevant
role is that `[project.scripts]` in `pyproject.toml` already installs every phase script
into `.venv/bin`, so once a script supports `-h`/`--help`, `uv run <script> --help` works
for free. No new uv-level mechanism is needed or available.

Checked `alldocs/PLANS_TOC.md` and `.docs_claude/plans/`: no prior plan covers this.
`isaac-datagen-pipeline` (chains render→proposals→inliers, resumable) already exists and
is the right "just run everything" entry point to fold this into. The `run`/`art`
registry CLIs are unrelated (PSC artifact/run sync, not datagen CLI discoverability).

## Approach

Add a small, hand-maintained tldr text (source of truth for the "cheat sheet"), reached
via `-h`/`--help` on the two front-door commands (`isaac-datagen`,
`isaac-datagen-pipeline`), using the same `argparse` + `parse_known_args` dotlist-passthrough
pattern already established in this package (`make_unseen.py:_parse_args`, `runtime_config.py:275`
merge order: schema < YAML < dotlist).

**New file `src/isaac_datagen/tldr.py`** — one module-level string constant `TLDR`,
assembled from a few local literals (no classes, no logic beyond string formatting):
- **Phases** — one line each for `isaac-datagen`, `isaac-datagen-proposals`,
  `isaac-datagen-downsample-proposals`, `isaac-datagen-inliers`, `isaac-datagen-pipeline`,
  `isaac-datagen-unseen`, mirroring the existing one-line comments already sitting next to
  each entry in `pyproject.toml:32-60` (that table is the accurate source; this just
  restates it where a CLI user will actually see it) plus an example invocation.
- **Common overrides** — dotlist cheat-sheet: `idx`, `num_frames`, `num_targets`, `mode`
  (flag that `amazon.yaml`/`mixed.yaml`/`shelf.yaml` require it on the CLI since it has no
  default), then an "ablation" sub-list: `seed`, `distant_intensity`,
  `distant_light_offset`, `occluders_per_target`, `occluder_scale`,
  `placement_args.max_column_height`. Confirmed by scanning every file in
  `src/isaac_datagen/configs/`.
- **Configs** — a static one-line-per-file table for all 10 files in
  `src/isaac_datagen/configs/*.yaml` (`amazon.yaml`, `expanded-refseg.yaml`,
  `expanded-refseg-v2.yaml`, `jagged-expanded-refseg-v2.yaml`,
  `jagged2-expanded-refseg-v2.yaml`, `mixed.yaml`, `staggered.yaml`, `shelf.yaml`,
  `random3_smoke.yaml`, `tuna_only_smoke.yaml`), each with a one-line purpose already
  captured during exploration (e.g. "amazon.yaml — phase-1 render, amazon-only catalog,
  LookAtPoser halo poses; mode=optflow not set, override on CLI"). **Hand-maintained, not
  scanned from a header-comment convention** — 10 files that change rarely don't justify
  inventing and maintaining a new machine-readable tag across every config; this matches
  how this repo already documents small stable maps (CLAUDE.md's module-index table,
  README's disk-paths table) rather than parsing source for docs.
- **Key paths** — must launch from `src/isaac_datagen/` (config-relative paths depend on
  cwd, per `.docs_claude/psc-isaac-datagen-footguns.md` §5); `dataset_dir` must pre-exist;
  configs live in `src/isaac_datagen/configs/`; footgun notes at
  `.docs_claude/psc-isaac-datagen-footguns.md`; module map in `CLAUDE.md`.

**Edit `src/isaac_datagen/clean_datagen.py`** — replace `main()` (currently
`clean_datagen.py:186-192`, raw `load_config(sys.argv[1], sys.argv[2:])`) with:
- `argparse.ArgumentParser(prog="isaac-datagen", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=TLDR)`,
  one positional `config`.
- Explicit `if len(sys.argv) == 1: parser.print_help(); sys.exit(0)` before parsing —
  today a forgotten config path is a bare `IndexError`; this makes the single most common
  invocation mistake produce the tldr instead.
- `args, overrides = parser.parse_known_args(sys.argv[1:])` so trailing `key=value`
  dotlist tokens pass through untouched (identical mechanism to
  `make_unseen.py:_parse_args`), then `load_config(args.config, overrides)` as before.
- Leave `reference_segmentation(runtime=None)` / `optflow_generation(runtime=None)`'s own
  `sys.argv` fallback branches untouched — out of scope, used only for direct invocation
  outside `main()`.

**Edit `src/isaac_datagen/run_pipeline.py`** — same treatment for `main()`
(`run_pipeline.py:100-104`, currently a manual `len(sys.argv) < 2` guard): swap in the
same argparse + `TLDR` epilog + zero-args pattern, `prog="isaac-datagen-pipeline"`, and
use `args.config`/`overrides` for the `load_config` call. **Leave every other `sys.argv[1:]`
usage in this file unchanged** (`_run("isaac-datagen", *sys.argv[1:])` at line 114,
`_run_proposals_sharded`'s `[*sys.argv[1:], ...]` at line 89, `_run("isaac-datagen-proposals", *sys.argv[1:], *extra)`
at line 136) — those forward the raw dotlist verbatim to sibling console scripts and must
not be rederived from the parsed `Namespace`.

**No `pyproject.toml` changes** — both edited commands already have console-script
entries; no new entry point is being added (a separate `isaac-datagen-tldr` script would
just duplicate the `--help` path for no new capability).

**No changes** to `src/isaac_datagen/configs/*.yaml`, `add_proposals.py`,
`debug_render.py`, or the 6 `viz_*`/tuning/gate scripts (those already have their own
argparse `--help`, or are explicitly deferred as non-front-door tools — fast-follow if
ever wanted, not part of this change).

## Verification

From `src/isaac_datagen/` (the required launch cwd):
1. `uv run isaac-datagen --help` and `uv run isaac-datagen` (no args) both print the full
   tldr (phases, overrides, configs, paths) and exit 0 — no stack trace.
2. `uv run isaac-datagen-pipeline --help` prints the same tldr via its own epilog, exit 0.
3. `uv run isaac-datagen configs/random3_smoke.yaml idx=0 num_frames=1 num_targets=1`
   still boots and renders exactly as before (dotlist passthrough unaffected) — confirms
   the argparse change didn't alter the real invocation path.
4. Read the diff on `run_pipeline.py` to confirm the three raw-`sys.argv[1:]` forwarding
   call sites are byte-for-byte unchanged (they must keep forwarding the original argv,
   not a reconstructed one from `args`/`overrides`).

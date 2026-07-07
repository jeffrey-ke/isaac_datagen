> **COMPLETED (as-built) 2026-07-06.** Extends the `run` training-run registry
> ([`run-registry-psc-sync.md`](run-registry-psc-sync.md)) with a `local` verb that lists runs
> already on disk — no network round-trip — reusing the exact leaf-run-vs-sweep-umbrella KIND
> probe the remote `list` already does (1-level `hparams.yaml` check), just via
> `os.scandir`/`pathlib` instead of `rsync --list-only`. `remote` is optional: when every
> configured remote in `.runs.yaml` resolves to the same local dir (true today — `psc`/
> `psc-mapv4`/`psc-multi` all point at `segmentation/runs/`), `run local` auto-detects it; if
> remotes ever diverge it fails loud naming the distinct dirs instead of guessing. Prior-art scan
> (3 parallel agents over `alldocs/PLANS_TOC.md`) found no other workspace tool does local-only
> run enumeration — the closest precedent was `run`'s own remote-side KIND probe, ported as-is.
> **Caught during implementation:** the initial draft's top-level `len(argv) < 2` guard would have
> rejected a bare `run local` (only 1 arg) before ever reaching the new branch — fixed by loosening
> the guard to `not argv` and adding an explicit `if not rest` check before the non-`local` verbs
> extract their `remote` positional. Verified against the live `segmentation/runs/` tree: correctly
> skips the two sibling `.journal` study files, drills into a sweep to show only locally-present
> arms (`m2f-fullgrid-hpo-v3` → just `t40`), computes `--size` from local `stat()` calls (instant,
> vs. the ~20s Lustre round-trip the remote `--size` needs), the multi-dir ambiguity fail-loud path
> (tested via a scratch `--config` pointing at two remotes with different `dir`s), and confirmed
> `run list`/`push`/`pull` behave identically to before the `main()` refactor.

# `run local` — list runs already on disk, no network round-trip

## Context

`~/dotfiles/bin/run` today only lists runs by asking a specific remote (`run list <remote>`),
which always does a live `rsync --list-only` round-trip and merely annotates local presence with
a ✓/– marker. There was no way to just ask "what do I already have in `segmentation/runs/`" —
whether pulled from PSC or trained locally — without picking a remote and paying the network hop.
`.runs.yaml` has three remotes (`psc`, `psc-mapv4`, `psc-multi`) that all resolve to the *same*
local dir (`segmentation/runs/`) against different remote roots, so a local-only listing is
unambiguous today and doesn't need a remote argument at all — it only would if a future remote
used a different `dir`.

## Design

New verb `run local [remote] [name] [--size]` — mirrors `list`'s shape but reads the filesystem
only.

- **`remote` is optional.** If given, use that remote's `local_dir` (same `registry.entry`+
  `resolve_dir` as today). If omitted, resolve `local_dir` for *every* remote in `.runs.yaml` and
  use it if they all agree (true today); fail loud naming the distinct dirs if they ever diverge,
  rather than silently guessing one.
- **`name`** optionally drills into a sweep exactly like `list` does, but reads local arms only.
- **`--size`** sums `checkpoints/**` file sizes locally (fast — no 20s Lustre listing needed).
- Table drops the `LOCAL` column (everything shown is, by definition, local) but keeps
  `RUN · KIND · MTIME · [CKPT]`.

## Key changes

- `+ registry.all_entries` (full remote→spec dict, not just one key); `~ registry.entry` calls it (dedup YAML parsing) — `~/dotfiles/bin/registry.py`
- `+ run._mtime`, `+ run._do_local` (local `os.scandir`/`pathlib` port of `_do_list`'s remote rsync-based leaf/sweep KIND probe + `--size` summation); `~ run.main()` dispatches the new `local` verb before requiring a `remote` positional (loosened the `len(argv)` guard so a bare `run local` works); `~ run.__doc__` documents the verb — `~/dotfiles/bin/run`
- `+ rlocal` alias next to `rls`/`rpull`/`rpush` — `~/dotfiles/.functions.sh`
- `~ README.md` mentions `run local` beside the existing `run` blurb — `/home/jeffk/repo/refseg-workspace/README.md`

## Verification (all passed against the live `segmentation/runs/` tree)

1. `run local` (no remote) — table lists every top-level dir under `segmentation/runs/` (KIND
   run/sweep, MTIME), correctly skips the two `.journal` files
   (`verifier-cleandift-ft-hpo-probe.journal`, `verifier-grid1280-hpo.journal`), no network I/O.
2. `run local psc m2f-fullgrid-hpo-v3` — drills into the sweep, shows only the locally-present arm
   (`t40`), matching what `run list psc m2f-fullgrid-hpo-v3` shows for the `LOCAL=✓` row.
3. `run local --size` — CKPT column populated from local `stat()` sums.
4. Fail-loud: a scratch `.runs.yaml` with two remotes pointing at different `dir`s makes `run
   local` (no remote) exit loud naming both dirs instead of picking one.
5. `run list psc <name>`, `run pull psc` (no name) — unaffected; same stderr echo line and same
   fail-loud messages as before the `main()` refactor.

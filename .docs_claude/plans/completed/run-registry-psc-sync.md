> **COMPLETED (as-built) 2026-07-03.** Sibling of the `art` artifact registry ([`artifact-registry.md`](artifact-registry.md));
> both now share `~/dotfiles/bin/registry.py`. Commits: dotfiles `06572e3` (registry.py + run + art
> refactor + rls/rpull/rpush), refseg-workspace `1070540` (`.runs.yaml` + README; deleted
> `sync-run-{to,from}-psc.sh`). Verified: `art` no-regression (identical HF listing + stderr), `run
> list psc` (connected, remote runs dir empty), rsync filter recurses sweep arms + captures flat-run
> checkpoints while excluding wandb/viz, all fail-loud paths. Live PSC push/pull round-trip left to
> run in an interactively-authed `psc-data` session.
>
> **DTN correction (2026-07-03, after first use):** `psc-data` is a PSC Data Transfer Node that
> allows rsync but **rejects arbitrary ssh** (`Login denied: … is not an allowed command`), and its
> rsync is **3.1.3** (no `--mkpath`). The initial `run list` (`ssh psc-data du`) and push (`ssh
> psc-data mkdir`) both hit this — and `list` silently swallowed the ssh failure, printing "no runs"
> over a PSC runs dir that in fact holds the whole `m2f-fullgrid-hpo-v3` sweep. Fixed by making `run`
> **rsync-only**: `list` enumerates via `rsync --list-only` (instant shallow name+mtime table; ckpt
> size is opt-in `--size`, a checkpoints-filtered recursive list pruning wandb/viz, ~20s over
> Lustre) and **fails loud on a nonzero rsync rc**; push drops the `ssh mkdir`/`--mkpath` entirely
> and relies on rsync creating the leaf under the pre-existing `segmentation/runs/`. Lesson:
> [[verify-deps-before-asserting]] — a data-mover host is not a general shell host; probe the
> transport's real constraints before shelling out.

# `run` — a name-keyed training-run registry over PSC

## Context

Training runs live under `segmentation/runs/<name>/` (config, hparams, metrics, checkpoints;
some are sweep umbrellas like `m2ffix/lr1e4_cos_nowd_s7/`). Today two hardcoded bash scripts,
`sync-run-to-psc.sh` / `sync-run-from-psc.sh`, push/pull one run's history to PSC as a shared
hub, but there is **no `list`**, host/path are baked in, and there is no single "registry" front
door the way `art` gives datasets/checkpoints/assets and `pl` gives path vars.

Goal: a `run` CLI that mirrors the **`art`** design exactly — a self-contained tool on PATH,
verbs `list | pull | push`, a `<remote>` selector (`psc`) resolved from a walked-up config, and a
`pl`-style annotated table for `list`. It supersedes the two `sync-run-*` scripts (just as `art`
superseded `hf_sync.py`). Transport stays rsync/ssh over the existing `psc-data` alias — HF is
wrong for multi-GB checkpoints; rsync is incremental, resumable, and already proven here.

Decisions (confirmed with user): **supersede** the two scripts · **`.runs.yaml`** registry ·
**annotated** `list` table.

## Precedents to mirror (do not reinvent)

- `~/dotfiles/bin/art` (PEP-723 `uv run` script; symlinked `~/.local/bin/art`): `_find_config`
  walks cwd up for `.artifacts.yaml` / `--config` / `$ART_CONFIG`; `_entry` reads a per-namespace
  dict and resolves `dir` **relative to the yaml** (cwd-independent); fail-loud on missing
  file/key; echoes the resolved target to stderr. `run` copies this skeleton verbatim.
- `sync-run-to-psc.sh` / `sync-run-from-psc.sh` — the exact rsync include/exclude filter and the
  `psc-data:/ocean/projects/cis260205p/jke2/refseg-workspace` root. `run` absorbs these, then
  deletes them.
- `pl` (`~/dotfiles/.functions.sh:607`) — the `column -t`-rendered name→value table `list` imitates.

## 1. New config: `.runs.yaml` at the workspace root

One entry per remote (extensible; `psc` is just the first). `dir` is workspace-relative so the
local tree (`<yaml parent>/dir/<name>`) and the remote tree (`<root>/dir/<name>`) share the
`dir/<name>` suffix — the same mirroring the sync scripts rely on.

```yaml
# /home/jeffk/repo/refseg-workspace/.runs.yaml
# Training-run registry for the `run` tool (~/dotfiles/bin/run). One entry per remote;
# `dir` resolves relative to THIS file (cwd-independent), workspace-relative so local==remote suffix.
psc: { host: psc-data, root: /ocean/projects/cis260205p/jke2/refseg-workspace, dir: segmentation/runs }
```

## 2a. Extract the shared mechanism: `~/dotfiles/bin/registry.py`

`art` and `run` share ONE convention — *walk cwd up for a `.<x>.yaml`, look up a keyed entry,
resolve its `dir` relative to the yaml (cwd-independent), fail loud, honor `--config`/env*. Per
review feedback, that must live in a single importable library, not be copy-pasted as `_private`
helpers into each tool. The library is generic (knows nothing about HF or ssh); each tool layers
its target semantics (HF repo vs ssh remote) on top.

```python
# ~/dotfiles/bin/registry.py — walk-up YAML registry mechanism shared by `art` and `run`.
# Imported, not run: deps (pyyaml) come from the importing PEP-723 script's own /// block.
import os, sys
from pathlib import Path
import yaml

def find_config(filename, env_var, argv):
    """Nearest <filename> walking cwd upward; honors --config <path> (consumed from argv)
    or $<env_var>. Fail-loud."""
    override = os.environ.get(env_var)
    if "--config" in argv:
        i = argv.index("--config"); override = argv[i + 1]; del argv[i:i + 2]
    if override:
        p = Path(override).expanduser()
        if not p.is_file(): sys.exit(f"registry: --config/${env_var} not a file: {p}")
        return p
    for base in (Path.cwd(), *Path.cwd().parents):
        if (p := base / filename).is_file(): return p
    sys.exit(f"registry: no {filename} found from cwd upward (set --config or ${env_var})")

def entry(cfg_path, key, label):
    """Return spec[key] dict; fail-loud if absent. (Target/echo formatting stays in the tool.)"""
    spec = yaml.safe_load(cfg_path.read_text()) or {}
    if key not in spec:
        sys.exit(f"{label}: {key!r} not in {cfg_path} (have: {', '.join(spec) or 'none'})")
    return spec[key]

def resolve_dir(cfg_path, e):
    """`dir` resolved relative to the yaml → cwd-independent absolute path."""
    return (cfg_path.parent / e["dir"]).resolve()
```

Both tools import it via a symlink-robust idiom (they're invoked through `~/.local/bin/` symlinks,
so `sys.path[0]` can't be relied on):

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # -> real ~/dotfiles/bin, not the symlink dir
import registry
```

## 2b. Refactor `art` onto the library (behavior-preserving)

Replace `art`'s private `_find_config`/`_entry` with the shared calls, keeping the exact
`[art: <ns> via <path> -> <repo>]` stderr line and all current behavior:

```python
# art, before:  cfg = _find_config(override); repo, local_dir, require_name, ignore = _entry(ns, cfg)
# art, after:
cfg = registry.find_config(".artifacts.yaml", "ART_CONFIG", argv)   # argv already had -y/flags stripped
e = registry.entry(cfg, ns, "art")
repo, local_dir = e["repo"], registry.resolve_dir(cfg, e)
require_name, ignore = e.get("require_name", False), e.get("ignore", [])
print(f"[art: {ns} via {cfg} -> {repo}]", file=sys.stderr)
```

## 2c. New tool: `~/dotfiles/bin/run` (chmod +x; symlink `~/.local/bin/run`)

PEP-723 python — same skeleton as `art` (import `registry`, dispatch verbs), but shells out to
`ssh`/`rsync` instead of the HF SDK. Argv shape `run <verb> <remote> [name...] [-n|rsync args]`.

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""run — name-keyed training-run registry over rsync/ssh remotes.

    run list <remote>                 annotated table of runs on the remote (size + local?)
    run pull <remote> <name...> [-n]  fetch run history (config/hparams/metrics/ckpts) <- remote
    run push <remote> <name...> [-n]  push run history -> remote

Remotes are defined in the nearest .runs.yaml walking UP from cwd (or --config/$RUN_CONFIG):
    <remote>: {host: <ssh alias>, root: <remote abs path>, dir: <ws-relative runs dir>}
Skips wandb/ and viz/ (regenerable / canonical-in-cloud). Trailing -n / rsync flags pass through.
"""
```

`main()`:
- `cfg = registry.find_config(".runs.yaml", "RUN_CONFIG", argv)`; `e = registry.entry(cfg, remote, "run")`;
  `host, root, rel, local_dir = e["host"], e["root"], e["dir"], registry.resolve_dir(cfg, e)`;
  echo `[run: <remote> via <cfg> -> <host>:<root>/<rel>]` to stderr.
- collect passthrough flags (anything starting `-`, e.g. `-n`); dispatch:
  - **push/pull require ≥1 name** (runs are large) — else loud exit, hint `run list <remote>`.
  - **push** mkdirs the remote parent once (`ssh host mkdir -p root/rel/name`, skipped on `-n`),
    then rsyncs `local_dir/name/` → `host:root/rel/name/`; **pull** rsyncs the reverse into a
    `mkdir -p`'d local dir. Both use the §2 filter below.
  - **list** → §3.

The rsync filter — **fixes a real bug** in the current scripts (nested sweep dirs never sync,
because `--exclude='*'` prunes the `lr…/` subdir before rsync descends). Prune wandb/viz first,
allow directory recursion, then match files at any depth:

```
# OLD (sync-run-*.sh) — only top-level files transfer; m2ffix/<sweep>/hparams.yaml is dropped
--include='config.yaml' --include='hparams.yaml' ... --include='checkpoints/' \
--include='checkpoints/**' --exclude='*'

# NEW (run) — recurses into sweep subdirs, still skips wandb/viz
rsync -avP -m <passthrough> \
  --exclude='wandb/' --exclude='viz/' \      # prune BEFORE the '*/' recurse include
  --include='*/' \                            # descend into sweep subdirs (m2ffix/lr…/)
  --include='config.yaml' --include='hparams.yaml' --include='git_commit.txt' \
  --include='metrics.csv' --include='*.patch' --include='checkpoints/**' \
  --exclude='*' \                             # drop everything else
  <src>/ <dst>/     # push: local_dir/name/ -> host:root/rel/name/ ; pull: reversed
```

(`-m`/`--prune-empty-dirs` keeps recursion from leaving empty dir shells; unmatched basenames like
`hparams.yaml` match at any depth since they carry no leading `/`.)

## 3. `run list <remote>` — annotated table (the `pl` analog)

One ssh round-trip for remote names+sizes, then a local `os.scandir` for the presence marker;
render with a plain aligned print (or `column -t`) like `pl`:

```
ssh host 'cd root/rel 2>/dev/null && du -sh -- */ 2>/dev/null'   # NAME/  <size> per remote run
```

Output:
```
RUN                              CKPT     LOCAL
cleandiftfinetuned-jun26-514pm   646M     ✓
m2ffix                           72M      ✓
gligen-jun26-1017pm              1.2G     –
```
`LOCAL` = `✓` if `local_dir/<name>` exists, else `–` (a run pushed from another machine, not yet
pulled) — giving the "registry / sync-status" glance the path registry's `pl` gives.

## 4. Supersede the old scripts + wire-up

- **Delete** `sync-run-to-psc.sh` and `sync-run-from-psc.sh` (fully replaced by `run push/pull psc`).
  **Keep** `sync-datasets-to-psc.sh` and `sync-m2f-checkpoint-to-psc.sh` (datasets/warm-start, not runs).
- **Symlink** `~/.local/bin/run -> ~/dotfiles/bin/run` (matches `art`); `chmod +x` the tool.
- **README** (`/home/jeffk/repo/refseg-workspace/README.md`, near the `art` block ~L33/L91): replace
  the `sync-run-*` mention with `run list|pull|push psc <name>`.
- **Optional** terse aliases in `~/dotfiles/.functions.sh` next to the `art` block, e.g.
  `rls(){ run list psc; }  rpull(){ run pull psc "$@"; }  rpush(){ run push psc "$@"; }` — the
  primary interface stays the literal `run … psc …` the user asked for.
- Commit the two touched repos separately: `dotfiles` (new `bin/run`, `.functions.sh`) and
  `refseg-workspace` (new `.runs.yaml`, README, deleted scripts).

## Files

| Path | Change |
|---|---|
| `~/dotfiles/bin/registry.py` | **new** shared walk-up YAML registry library (imported by `art` + `run`) |
| `~/dotfiles/bin/art` | refactor onto `registry.py` (behavior-preserving; drop private `_find_config`/`_entry`) |
| `~/dotfiles/bin/run` | **new** PEP-723 tool (imports `registry`; mirrors `art`) |
| `~/.local/bin/run` | **new** symlink → the tool |
| `~/repo/refseg-workspace/.runs.yaml` | **new** remote registry |
| `~/repo/refseg-workspace/sync-run-to-psc.sh` | **delete** (→ `run push psc`) |
| `~/repo/refseg-workspace/sync-run-from-psc.sh` | **delete** (→ `run pull psc`) |
| `~/repo/refseg-workspace/README.md` | update sync section |
| `~/dotfiles/.functions.sh` | optional `rls/rpull/rpush` aliases |

## Verification (run from the workspace)

0. **shared lib / no regression** — `art ls dataset`, `art ls checkpoint` still print the same
   trees and the same `[art: … -> …]` stderr line (proves the `registry.py` refactor preserved
   `art`'s behavior); both `art` and `run` resolve their config when invoked via the `~/.local/bin`
   symlinks (the `Path(__file__).resolve()` import idiom works through the symlink).
1. **list** — `run list psc` prints the annotated table; names match `ssh psc-data ls
   /ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/runs`, `LOCAL` markers correct.
2. **nested-sweep filter (the bug fix)** — `run push psc m2ffix -n` (dry-run): the transfer list
   MUST include `m2ffix/lr1e4_cos_nowd_s7/{hparams.yaml,metrics.csv,checkpoints/…}` (the old
   script dropped these). Confirm `wandb/` and `viz/` are absent from the list.
3. **round-trip** — pick a small run: `run push psc <name>` → `ssh psc-data 'ls
   .../segmentation/runs/<name>/checkpoints'` shows the ckpts → move the local dir aside →
   `run pull psc <name>` → `diff -r` the pulled tree against the moved-aside copy (identical
   modulo the intentionally-skipped `wandb/`, `viz/`).
4. **fail-loud** — `run pull psc` (no name) exits non-zero with the `run list psc` hint; `run push
   nope foo` exits loudly (`nope` not in `.runs.yaml`); running outside any `.runs.yaml` tree
   exits with the "no .runs.yaml found" message.

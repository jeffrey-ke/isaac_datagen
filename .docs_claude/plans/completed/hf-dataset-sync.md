> **DEPRECATED — superseded by `artifact-registry.md`** (`isaac_datagen/.docs_claude/plans/completed/`).
> `hf_sync.py` was removed in segmentation commit `c8e8a9a`; the current mechanism is the `art`
> tool + per-repo `.artifacts.yaml`. Kept for history only — do not use this for new work.

# Plan: `hf_sync.py` — preflight-gated HF dataset push/pull for `segmentation/`

> **Status: completed 2026-06-08.** Shipped `segmentation/scripts/hf_sync.py`,
> `segmentation/src/segmentation/configs/hf_dataset.yaml`, and added `huggingface-hub>=0.32`
> to `segmentation/pyproject.toml` (resolved to 1.17.0). Verified: bad/no args → usage + exit 2;
> `push` preflight prints the checklist (repo_id / auth `jeffke613` / 12,619 files, 23.8 GB /
> "will create PRIVATE") and aborts on prompt-EOF with **no upload**; `pull` preflight
> `[fail]`s on a missing repo and aborts. A real full push (creates the repo, uploads ~24 GB)
> was intentionally **not** run — left for the user to trigger.
>
> **As-built deviations from the plan below:**
> - Loads config via `vision_core.configutils` (`load_config` + `to_dataclass`) directly,
>   **not** `segmentation.utils.load_check_config`, to keep this IO script from importing torch.
> - **No `chdir`-to-package** (unlike `inspect_gligen.py`): the config path is taken as given
>   (cwd-relative or absolute), since overriding a CLI path argument via chdir is surprising.
>   Documented invocation uses the explicit `src/segmentation/configs/hf_dataset.yaml` path.
> - `_resolve_dataset_dir` uses `Path(...).resolve()`, which **follows symlinks**:
>   `src/segmentation/datasets/expanded-refseg` is a symlink to `/data/user/jeffk/datasets/...`,
>   so the resolved dataset dir (and upload source) is the real `/data` location.

## Context

The render datasets (e.g. `expanded-refseg`, ~24 GB, ~12.7k files of `.pt`/`.npy`/`.png`)
are produced in `isaac_datagen` and consumed by `segmentation`'s training (`train.py`,
`verifier/`). We want a **tracked, repeatable** way to move a dataset to/from the Hugging
Face Hub instead of ad-hoc `rsync`. HF behaves like "git for big opaque blobs": with the
`huggingface_hub` file API the bytes round-trip identically in the same layout, and our
existing `SerializableSample` deserialization is unchanged — we deliberately avoid the
`datasets.load_dataset` layer, which would impose format/parsing.

The script is a **preflight checklist**: it validates preconditions (auth, paths, repo,
HF structure limits), prints a summary, and only transfers after an interactive `y/N`.
One YAML drives both directions; the action is chosen on the CLI.

## Deliverables

1. **`segmentation/scripts/hf_sync.py`** — the script (new, tracked).
2. **`segmentation/src/segmentation/configs/hf_dataset.yaml`** — example config (new, tracked).
3. **`segmentation/pyproject.toml`** — add `huggingface_hub` as an explicit dependency.

## Config schema + path resolution

YAML has exactly two fields (matches existing dataclass-config style, `configutils.MISSING`):

```yaml
# segmentation/src/segmentation/configs/hf_dataset.yaml
repo_id: jeffk/expanded-refseg          # "owner/name"
path: ../datasets/expanded-refseg       # dataset dir, RELATIVE TO THIS YAML's DIR
```

```python
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class HfDatasetConfig:
    repo_id: str = MISSING   # owner/name on the Hub
    path: str = MISSING      # dataset dir, resolved relative to the config file's dir
```

- Load with the existing helper: `load_check_config(config_path, HfDatasetConfig)`
  (`segmentation/src/segmentation/utils.py:257` → `vision_core.configutils.to_dataclass`).
  This validates required fields (raises on `MISSING`).
- **Resolution rule (the one deliberate departure from repo convention of absolute paths):**
  `dataset_dir = (Path(config_path).resolve().parent / cfg.path).resolve()`.
  Because `path` is relative to the *committed* YAML, the same config lands the dataset in
  the same repo-relative location on any machine — which is exactly why push-source and
  pull-dest can be the one `path` field.

## CLI / invocation

```
env -u PYTHONPATH uv run python scripts/hf_sync.py {push|pull} <config.yaml> [-y]
```

- `argv[1]` ∈ {`push`, `pull`} (else print usage, exit 2). `argv[2]` = config path.
  `-y`/`--yes` skips the confirm prompt (for non-interactive use).
- `env -u PYTHONPATH` per the isaacsim-PYTHONPATH-leak convention (matches
  `scripts/inspect_gligen.py`'s documented run block). Module docstring carries a `Run:` block.
- No argparse — `sys.argv` membership checks, consistent with `train.py` / `inspect_gligen.py`.

## Behavior

Shared client: `api = HfApi()`. **`repo_type="dataset"` on every call** (the #1 footgun —
default is `model`, and `upload_large_folder` re-uploads everything if the type is wrong).

A small `_check(label, ok, detail)` helper prints `[ok]/[warn]/[fail]` lines and tracks a
hard-fail flag; any `[fail]` aborts before the prompt.

### `push`
Preflight checklist:
1. `repo_id` matches `^[\w.-]+/[\w.-]+$`.
2. Auth: `api.whoami()` succeeds → show username (mandatory for push; on failure point to `hf auth login`).
3. `dataset_dir` exists, is a dir, non-empty.
4. Local scan via `os.walk`: file count, total bytes, max files-in-any-folder. Print a
   summary (`12,657 files, 24.0 GB`); **warn** if total files > 100k or any folder > 10k
   (HF repo-structure limits).
5. Repo existence via `api.repo_info(repo_id, repo_type="dataset")`; if missing, note
   "will create **private**".

Then prompt `Proceed? [y/N]`. On yes:
```python
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(
    repo_id=repo_id, repo_type="dataset", folder_path=str(dataset_dir),
    ignore_patterns=_IGNORE,
)
```
`upload_large_folder` (not `upload_folder`): resumable, multi-threaded, retrying — right for
24 GB / thousands of files. `_IGNORE = ["*_viz_clusters*", "*_eps_sweep*", "**/.cache/*"]`
(skips the disposable viz/sweep siblings and the upload progress-cache). Constant in the
script for now; can be lifted to config later if needed.

### `pull`
Preflight checklist:
1. `repo_id` well-formed.
2. `api.repo_info(repo_id, repo_type="dataset", files_metadata=True)` succeeds → repo
   exists/accessible (auth only needed if private; report `whoami` if it fails).
3. Remote summary: sum `siblings[*].size` → file count + total bytes to download.
4. Destination = `dataset_dir`; parent must exist/be writable. If `dataset_dir` exists and
   is non-empty, **warn**: pull merges/updates in place and will **not** delete local extras.

Then prompt. On yes:
```python
snapshot_download(
    repo_id=repo_id, repo_type="dataset",
    local_dir=str(dataset_dir),   # real files / git-like working tree, not the symlink cache
)
```
`local_dir` (vs the default `~/.cache` symlink store) gives a real working tree the training
code can read directly — the git-like behavior we want.

## Reused / existing code

- `load_check_config` — `segmentation/src/segmentation/utils.py:257`.
- `configutils` (`load_config`/`to_dataclass`, `MISSING`) — `vision_core/src/vision_core/configutils.py`.
- Script style (docstring + `Run:` block, `sys.argv`, `def main() -> int`, `sys.exit(main())`)
  — mirror `segmentation/scripts/inspect_gligen.py`.
- `huggingface_hub` API (`HfApi`, `upload_large_folder`, `snapshot_download`, `create_repo`,
  `repo_info`, `whoami`) — already in `uv.lock` (transitive via `transformers`; `hf-xet`
  present too). Add it to `pyproject.toml` `dependencies` so the tracked script's import is
  declared, then `uv lock`.

## Verification

1. **Arg handling:** run with no args / bad action → prints usage, exits non-zero.
2. **Preflight + abort:** `... push configs/hf_dataset.yaml`, answer `N` at the prompt →
   prints the checklist, performs no upload, exits 0. Confirms the gate works without side effects.
3. **Auth check:** temporarily unset the token (`env -u HF_TOKEN ...` with no cached login) →
   push preflight reports `[fail] auth` and aborts before any network write.
4. **End-to-end on a tiny set (HF "start small"):** point a temp YAML's `path` at a single
   `renderNNN/` (or a throwaway dir), push (answer `y`) to a scratch `repo_id`, confirm the
   repo URL is under `/datasets/` and is private. Then `pull` to a temp dir and
   `diff -r`/sha256 a few files to confirm byte-identical layout. Only scale to the full
   `expanded-refseg` once the small round-trip is verified.
5. **Re-pull is incremental:** second `pull` into the same `local_dir` re-downloads nothing
   (hash-checked), confirming git-like update semantics.

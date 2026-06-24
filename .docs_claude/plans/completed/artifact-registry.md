# Artifact registry: dataset / asset / checkpoint over HF

> **Supersedes `hf-dataset-sync.md`** (the `hf_sync.py` plan, completed 2026-06-08; copies in
> `isaac_datagen/.docs_claude/plans/completed/` and `segmentation/.docs_claude/plans/completed/`).
> That single-dataset push/pull script was removed in segmentation commit `c8e8a9a`
> ("Adopt artifact registry: ../checkpoints + drop hf_sync"). The `art` tool + per-repo
> `.artifacts.yaml` here replace `hf_sync.py`, every per-repo `hf_*.yaml`, and the `fetch_*.sh`
> scripts. Read this plan, not the deprecated one, for the current mechanism.

## Goal

Factor all heavy-artifact sync into one **registry-style generic API** over HF. Three
namespaces — **dataset** (renders), **asset** (sim usdz inputs), **checkpoint** (model
weights) — each backed by one HF repo. A single self-contained `art` tool (PEP-723, on
PATH, runs anywhere like `plyview`) + terse `ds*/as*/ck*` aliases (à la `pp`/`pl`). A
per-repo `.artifacts.yaml` says, by convention, where each namespace lives locally and its
policy. **Replaces** `hf_sync.py` + every per-repo `hf_*.yaml` + every `fetch_*.sh`.

HF repos are git-backed (xet/LFS) → every push is a commit (history/revert on the HF side).
Local trees stay gitignored (their `/data` symlink targets never transplant).

Companion: `UFM-train/.docs_claude/plans/active/workspace-submodule-and-data-sync.md`.

## Namespaces → HF repos

| ns | local meaning | HF repo |
|---|---|---|
| `dataset` | render umbrellas (the shared `/data/user/jeffk/datasets` tree) | `jeffke613/refseg-datasets` |
| `asset` | usdz sim inputs (`graspable_objects`, `optflow_objects`) | `jeffke613/refseg-assets` |
| `checkpoint` | model weights (`sam/…`, `gim/…`, `verifier/…`) | `jeffke613/refseg-checkpoints` *(rename of today's `refseg-assets`)* |

## The `art` tool — `~/dotfiles/bin/art` (chmod +x, on PATH)

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.25", "pyyaml"]
# ///
"""art <verb> <ns> [name...] [-y] — generic artifact registry over HF.
ns: dataset | asset | checkpoint, defined per-repo in .artifacts.yaml (walked up from cwd):
  <ns>: {repo: owner/name, dir: <path rel to this file>, require_name: bool, ignore: [globs]}
verbs:
  ls   <ns>                list top-level artifact names in the namespace repo
  pull <ns> [name...]      download named subtree(s)/file(s) -> dir  (all if !require_name)
  push <ns> [name...]      upload named subtree(s) -> repo, or whole dir (incremental)"""
import sys, yaml
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, upload_folder, upload_large_folder

REPO_TYPE = "dataset"  # one HF repo per ns; dataset-type holds arbitrary files fine

def _cfg(ns):
    for base in (Path.cwd(), *Path.cwd().parents):
        f = base / ".artifacts.yaml"
        if f.is_file():
            spec = yaml.safe_load(f.read_text()) or {}
            if ns not in spec:
                sys.exit(f"namespace {ns!r} not in {f} (have: {', '.join(spec) or 'none'})")
            e = spec[ns]
            return (e["repo"], (f.parent / e["dir"]).resolve(),
                    e.get("require_name", False), e.get("ignore", []))
    sys.exit("no .artifacts.yaml found from cwd upward")

def main():
    raw = sys.argv[1:]
    yes = ("-y" in raw) or ("--yes" in raw)
    a = [x for x in raw if x not in ("-y", "--yes")]
    if len(a) < 2: sys.exit(__doc__)
    verb, ns, names = a[0], a[1], a[2:]
    repo, d, require_name, ignore = _cfg(ns)
    api = HfApi()
    if verb == "ls":
        for t in api.list_repo_tree(repo, repo_type=REPO_TYPE, recursive=False):
            print(t.path)
    elif verb == "pull":
        if require_name and not names:
            sys.exit(f"{ns}: name required (repo may be huge) — try `art ls {ns}`")
        allow = [p for n in names for p in (n, f"{n}/**")] or None
        snapshot_download(repo, repo_type=REPO_TYPE, local_dir=str(d), allow_patterns=allow)
        print(f"pulled {names or 'ALL'} <- {repo} -> {d}")
    elif verb == "push":
        if not yes and input(f"push {names or d} -> {repo}? [y/N] ").strip().lower() not in ("y","yes"):
            sys.exit("aborted")
        api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
        if names:
            for n in names:
                upload_folder(repo_id=repo, repo_type=REPO_TYPE, folder_path=str(d / n),
                              path_in_repo=n, ignore_patterns=ignore)
        else:
            upload_large_folder(repo_id=repo, repo_type=REPO_TYPE, folder_path=str(d),
                                ignore_patterns=ignore)
        print(f"pushed {names or 'ALL'} -> {repo}")
    else:
        sys.exit(__doc__)

if __name__ == "__main__":
    main()
```

`list_repo_tree` / `snapshot_download(allow_patterns=)` / `upload_folder(path_in_repo=)` /
`upload_large_folder(ignore_patterns=)` are all stock `huggingface_hub`. (`pull` matches a
file *or* a subtree via `[n, f"{n}/**"]`; `push <name>` operates on subdirs.)

## Aliases — `.functions.sh` (terse, like `pl`)

```bash
dsl(){ art ls dataset; }     dspull(){ art pull dataset "$@"; }     dspush(){ art push dataset "$@"; }
asl(){ art ls asset; }       aspull(){ art pull asset "$@"; }       aspush(){ art push asset "$@"; }
ckl(){ art ls checkpoint; }  ckpull(){ art pull checkpoint "$@"; }  ckpush(){ art push checkpoint "$@"; }
```

## Per-repo `.artifacts.yaml` (at each repo root; `dir` resolves relative to it → cwd-independent)

```yaml
# isaac_datagen/.artifacts.yaml
dataset: { repo: jeffke613/refseg-datasets, dir: src/isaac_datagen/datasets, require_name: true,  ignore: [debug/**] }
asset:   { repo: jeffke613/refseg-assets,   dir: assets,                     require_name: false, ignore: [ycb/**] }

# segmentation/.artifacts.yaml
dataset:    { repo: jeffke613/refseg-datasets,    dir: datasets,     require_name: true }
checkpoint: { repo: jeffke613/refseg-checkpoints, dir: checkpoints,  require_name: false }

# UFM-train/.artifacts.yaml
dataset: { repo: jeffke613/refseg-datasets, dir: datasets, require_name: true }
```

## isaac_datagen config relativization (launch cwd = `src/isaac_datagen/`)

`configs/randomized.yaml` (and `mixed.yaml`): `intrinsics_path: zed_K.npy` *(keep)*;
`objects_path → ../../assets/{optflow,graspable}_objects/...`; matcher configs →
`../../../reference_matching/src/reference_matching/configs/...`; `dataset_dir →
datasets/<umbrella>` (→ `/data` symlink). `mixed.yaml` `combined_dataset/` →
`../../assets/graspable_objects/<?>` ⚠ **OPEN-1**.

## HF migration (one-time)

1. **Rename** weights repo: `HfApi().move_repo("jeffke613/refseg-assets",
   "jeffke613/refseg-checkpoints", repo_type="dataset")`.
2. **segmentation:** rename local `assets/` → `checkpoints/`; `segmenter.yaml` /
   `gligen_training.yaml` `ckpt_path: ../assets/...` → `../checkpoints/...`;
   `verifier_ckpt: ../assets/verifier/...` → `../checkpoints/verifier/...`. ⚠ touches 2
   pinned config lines + the workspace symlink (`~/repo/checkpoints`).
3. `aspush` (creates `refseg-assets`, now usdz) from isaac_datagen; `dspush` (creates
   `refseg-datasets`) from the shared tree; **delete** `jeffke613/expanded-refseg`.
4. **Remove superseded:** `segmentation/scripts/hf_sync.py`, `segmentation/hf_dataset.yaml`,
   `refseg-workspace/{hf_assets.yaml, scripts/fetch_assets.sh}`. README points at `art`.

## Resulting workflow (any repo, anywhere)

```bash
cd <repo>
dspull shelf-optflow        # scoped (datasets are huge)
ckpull                      # all weights
aspull                      # all usdz (isaac_datagen only)
# ... generate / train ...
dspush                      # mirror new renders up (incremental; each push = an HF commit)
```

## Decision log / open items

- **Confirm:** rename segmentation local dir `assets/`→`checkpoints/` (cleaner; touches 2
  pinned config lines + workspace symlink) vs keep `assets/` as the checkpoint dir.
- **OPEN-1:** which `assets/graspable_objects/*` set(s) replace `combined_dataset` in `mixed.yaml`.
- ignore `debug/**` (11 G) from the dataset mirror; clean up the repo-root ad-hoc `datasets/`
  (only the `ycb_optflow` smoke render).

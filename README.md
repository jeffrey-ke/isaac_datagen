# isaac_datagen

Isaac Sim Replicator synthetic data generation for reference-prompted instance
segmentation and visual servoing.

## Quick start (local / glibc 2.35+)

```bash
uv sync --locked
uv run clean_datagen.py src/isaac_datagen/configs/mixed.yaml idx=0 num_frames=8
```

See `CLAUDE.md` for module index and data flow.

## PSC / Bridges-2 (Apptainer container)

Bridges-2 ships glibc 2.28; Isaac Sim 5.1 wheels need 2.35+. Use the container for
system libs only — Python packages come from the committed `uv.lock` via `uv sync --locked`.

**Prereqs**

- GPU compute node (`interact --gpu`); containers do not run on login nodes.
- `$HOME/.config/uv/uv.toml` pointing cache at ocean (not `$HOME/.cache/uv`):

  ```toml
  cache-dir = "/ocean/projects/cis260205p/jke2/.uv_cache"
  ```

**Build** (once, on a compute node):

```bash
cd /ocean/projects/cis260205p/jke2/refseg-workspace/isaac_datagen
singularity build --force containers/isaac_datagen.sif containers/isaac_datagen.def
# apptainer works too — same commands
```

The image (`containers/isaac_datagen.sif`, ~263 MB) holds Ubuntu 22.04 system deps only.
It does not contain Isaac or the `.venv`.

**Setup paths:**

```bash
export REFSEG_WS=/ocean/projects/cis260205p/jke2/refseg-workspace
export OCEAN_ROOT=/ocean/projects/cis260205p/jke2
export SIF="$REFSEG_WS/isaac_datagen/containers/isaac_datagen.sif"
```

**Install Python env** (first time, multi-GB download):

```bash
singularity exec --nv \
  --bind "$REFSEG_WS:/workspace" \
  --bind "$OCEAN_ROOT:$OCEAN_ROOT" \
  "$SIF" \
  bash -lc 'cd /workspace/isaac_datagen && uv sync --locked'
```

**Run datagen:**

```bash
singularity exec --nv \
  --bind "$REFSEG_WS:/workspace" \
  --bind "$OCEAN_ROOT:$OCEAN_ROOT" \
  "$SIF" \
  bash -lc 'cd /workspace/isaac_datagen && uv run clean_datagen.py src/isaac_datagen/configs/mixed.yaml idx=0 num_frames=1'
```

Any `uv run …` or `uv run python …` works the same way — always bind workspace + ocean,
use `--nv` for GPU/Isaac.

**Where things live on disk**

| What | Path |
|---|---|
| Container image | `containers/isaac_datagen.sif` |
| Python env | `isaac_datagen/.venv` |
| uv download cache | `/ocean/projects/cis260205p/jke2/.uv_cache` |
| Render outputs | `{dataset_dir}/render{idx:03d}/` (e.g. `datasets/mixed/render000/`) |

Configs set `dataset_dir` relative to the repo (e.g. `datasets/mixed`). Override on the
CLI if needed: `dataset_dir=/ocean/projects/cis260205p/jke2/my-renders`.

**Notes**

- Isaac EULA: image and `clean_datagen.py` set `OMNI_KIT_ACCEPT_EULA=YES` for batch runs.
- Do not run `uv lock` on native PSC host (glibc too old). Update lock elsewhere, commit, sync in container.
- Bind the full `refseg-workspace` (editable deps: `vision_core`, `reference_matching`, …).

More detail: [`containers/README.md`](containers/README.md).

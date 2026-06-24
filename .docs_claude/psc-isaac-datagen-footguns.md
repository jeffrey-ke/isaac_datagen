# PSC Isaac datagen — footguns & landmines

Hard-won lessons from bringing the Singularity Isaac-datagen pipeline up on PSC Bridges-2
(2026-06). Each entry: **symptom → root cause → fix**. Read before debugging a stuck render.

The golden rule that recurs: **one shared `.venv` and one shared asset tree are visible from both
the host (glibc 2.28) and the container (glibc 2.35) — but they belong to the container.** Most
landmines are some variant of "host touched a container-owned thing."

---

## 1. Container / GPU / Vulkan

- **`VkResult: ERROR_INCOMPATIBLE_DRIVER` / `Vulkan 1.1 is not supported` / `GPU Foundation is not
  initialized`.** `--nv` injects CUDA libs but **NOT the NVIDIA Vulkan/EGL ICD JSON** (documented
  apptainer gap: apptainer#2210, nvidia-container-toolkit#16/#1392). The loader finds no driver. →
  Provide the ICD. Test-bind: `--bind /usr/share/vulkan/icd.d/nvidia_icd.json --bind
  /usr/share/glvnd/egl_vendor.d/10_nvidia.json --env VK_ICD_FILENAMES=...`. Durable: bake
  `nvidia_icd.json`+`10_nvidia.json` (relative `library_path`) into the `.def` + set the env (what
  `render_amazon_l40s.sbatch` does at runtime).
- **`nvidia-smi` working proves CUDA, NOT Vulkan.** They fail independently. Verify Vulkan with
  `vulkaninfo --summary` under `--nv`.
- **`Skipping unsupported non-RTX GPU` / `No device could be created` (after Vulkan is fixed).** Isaac
  Sim 5.1 **requires hardware RT cores**. NVIDIA docs: *"GPUs without RT Cores (A100, H100) are not
  supported."* V100 too. → **On Bridges-2 the ONLY Isaac-capable GPU is the L40S** (`--gpus=l40s-48:1`).
  H100/A100/V100 are useless for the renderer even when idle/free.
- **Driver-version red herring.** Same error message also comes from a driver>535.255 misreport whose
  flag is `--/rtx/verifyDriverVersion/enabled=false` — that flag does **not** fix the missing-ICD or
  non-RTX cases (isaac-sim#357). The L40S driver (560.35.03) is older than the docs' 580.65.06 ask but
  works once the ICD is present.
- **Benign boot noise — do NOT chase these:** `GLFW initialization failed` / `failed to open default
  display` (headless, no X), `carb.audio … eDeviceLost` (no sound card), `NGX isn't enabled` (no
  DLSS), USD `OrthogonalizeBasis did not converge`, diffusers `safety_checker=None`, `accelerate was
  not found`, `Loading pipeline components` (DIFT model load — stderr mis-tagged `[Error]` by omni.kit).
- **Warm shader cache.** First boot compiles RTX shaders (~3 min); later boots on the same node reuse
  the cache (via mounted `$HOME`) and start in ~15 s. Expect the first render of a session to be slow.

## 2. uv & the container `.venv`

- **The `.venv` is container-built, on shared Ocean disk, and must be managed ONLY in-container.**
  Its wheels are `manylinux_2_35` (glibc 2.35) and its interpreter is the container's. The host can
  see it but its uv/Python are wrong-ABI.
- **Host project-mode `uv run`/`uv sync` corrupts it.** `uv run` is mutating: from a dir with
  `pyproject.toml` it validates+repairs the env for *this machine* first. Host and container both have
  `/usr/bin/python3.11` but **different versions** (host 3.11.9 / uv 0.11.21 vs container 3.11.0rc1 /
  uv 0.11.23), so host uv judged the venv stale, tried to recreate it, tore down `.venv/lib`, failed
  on permissions, and left it broken for both. Symptom later: in-container `uv run` dies on
  `failed to remove .venv/lib: Permission denied`. → For ad-hoc **host** Python use
  `uv run --no-project --with <pkg>`, a PEP-723 `uv run --script` (e.g. `art`), or `cd /tmp`. Never
  plain `uv run`/`uv sync` for the project on the host. Fix once corrupted: `rm -rf .venv` + rebuild
  in-container with `uv sync --locked` (a full reinstall — the Lustre `rm` alone took ~6 min).
- **`bash -c`, never `bash -lc`, inside `singularity exec`.** `-l` sources the bind-mounted host
  `~/.bash_profile`/`~/.bashrc` → `module`/`conda`/`fzf` errors **and** prepends `~/.local/bin`,
  shadowing the container's `/usr/local/bin/uv` with the host's. The image already puts uv on PATH.
- **`uv sync --locked` fails: "lockfile needs to be updated".** Sibling-repo metadata changed (e.g.
  `vision_core` rename) silently stales `uv.lock`. → Regenerate **in-container** (`uv lock`), commit.
  Do NOT `uv lock` on the native host (glibc 2.28 can't resolve the `manylinux_2_35` Isaac wheels).
- **Pre-warm to save GPU SU.** `uv sync` inside a GPU job bills GPU time for a multi-GB install. Run
  it first on a CPU node (container runs there too); the GPU job's `uv sync --locked` then no-ops.

## 3. Multi-repo workspace (editable siblings + HF assets)

- **Editable sibling repos must be at compatible commits.** `vision_core`/`reference_matching` import
  from source; a stale/detached HEAD → `ImportError` on a renamed symbol, or a *missing file* a config
  references (e.g. `random_proposal.yaml` added in an unpulled commit). The venv/lock can be fine.
  → `git switch master && git pull` each sibling off detached HEAD as a preflight.
- **Assets are HuggingFace-managed and gitignored** (`jeffke613/refseg-assets` via the `art`
  tool / `aspull`). Not in git, no LFS. A **partial `snapshot_download` silently drops files** (we
  lost 2/44 usdz to an xet hiccup). → `aspull` (= `art pull asset`) re-pulls missing files and skips
  cached. Verify integrity: per-object `meta` count must equal `usd_path` count.

## 4. Slurm / scheduling (Bridges-2)

- **Allocation `cis260205p` is GPU-only** (no RM/CPU). Even a trivial logging job must `--gpus=…:1`.
- **GPU request formula.** Whole-node `GPU`: `n` GPUs must be a multiple of 8. Fractional
  `GPU-shared`/`GPU-small`: `--gpus=<type>:1..4`, billed per GPU; cores come proportionally (L40S node
  = 192 cores / 8 GPUs → 24 cores per GPU, auto-assigned).
- **idle CPUs ≠ free GPUs.** A `mix` node can show idle cores in `sinfo %C` while every GPU is
  allocated (`scontrol show node` → `AllocTRES gres/gpu`). Don't assume capacity from `%C`.
- **L40S is scarce** (3 nodes, often down/full). ETA (`squeue --start`) can read multi-day, but it's a
  **pessimistic backfill estimate** — a **short walltime backfills ahead of the 4h/2-day jobs** (our
  30-60 min jobs landed in minutes despite a 2-day ETA). Keep render walltime tight.
- **No other RTX GPU to escape to** — L40S only; `GPU-shared` (1 GPU) beats whole-node `GPU` for it.
- **sbatch must propagate the inner exit code.** A failed `singularity exec`/`uv run` without
  `set -e` or `rc=$?; …; exit $rc` lets the job report **`COMPLETED 0:0` with 0 output**. Always end
  with `exit $rc`.
- **Run loop.** `jid=$(sbatch --parsable job.sbatch)` (id from the return, NOT the log) →
  `squeue -j $jid` (+`--start` for ETA) → `tail -F *_$jid.{out,err}` → `sacct -j $jid
  --format=…AllocTRES…,State,ExitCode` to audit what was charged.

## 5. The datagen config & pipeline

- **`amazon.yaml` is `mode=optflow`** (its `objects_path` is `optflow_objects/amazon/`). It omits
  `mode`, which is a mandatory no-default field → must pass `mode=optflow` (mixed.yaml/shelf.yaml omit
  it too).
- **Launch cwd must be `src/isaac_datagen/`** — config paths are relative to it (`zed_K.npy`,
  `../../assets/...`, `../../../reference_matching/...`). Entry point is the console script
  `isaac-datagen` / `isaac-datagen-pipeline`, NOT a repo-root `clean_datagen.py` (stale in old docs).
- **`dataset_dir` must pre-exist** — `RuntimeConfig.__post_init__` asserts `Path(dataset_dir).exists()`.
  `mkdir -p` it before the run.
- **`run_pipeline` is not mode-aware** — it always does render → cid/iid validate → proposals →
  inliers. For optflow that's fine: the optflow render dir nests an `ObsMask`, so it's consumable by
  phases 2 & 3. The interactive y/N gate auto-skips with no tty (batch).
- **obs count = `num_frames × num_targets`** (e.g. 300 × 3 = 900). Size walltime off the *obs* count,
  not `num_frames`. ~2 s/obs on L40S → 900 obs ≈ 30-35 min + boot.
- **Missing object geometry → render-time crash spam.** A placed object with no `usd_path` usdz makes
  the camera frame empty → `OptFlowWriter: write() called with no labeled instances — expected ≥1`,
  fired per-frame (floods logs). Non-fatal (Replicator continues) but those frames are lost. Caused
  entirely by the partial asset pull above; 44/44 assets → 0 such asserts.
- **The render is NOT resumable, and `finalize_metadata()` runs only at the very end.** A `TIMEOUT`
  mid-render leaves obs/masks on disk but **no `runtime.yaml`/catalog → unusable dataset**. Check
  `runtime.yaml` exists to confirm a complete dataset; re-run from scratch (clear the render dir) with
  adequate walltime.

## 6. Process / debugging discipline

- **Validate cheaply before spending GPU SU** — on a CPU node, in-container: (1) `uv sync --locked`
  (lock OK?), (2) `uv run --no-sync python -c "import yourpkg, vision_core, reference_matching"`
  (sibling imports?), (3) `load_config(cfg, [...])` (path asserts?). Each isolates one failure class
  for ~0 GPU cost. Most of our failures were caught here or in the first ~2 min of GPU boot.
- **Monitors flood on per-frame errors.** Filter to milestones + terminal states, exclude known spam
  (`grep -v 'no labeled instances'`), and report counts in a final summary line rather than streaming
  each occurrence.
- **Smoke first** (`num_frames=1`/small), then scale — but note a 1-frame render won't exercise
  `finalize` timing or the empty-frame path.

# One-command pipeline: render → proposals → inlier labels

## Context

Producing a verifier-ready render dir currently takes three manual invocations
(`isaac-datagen`, `isaac-datagen-proposals`, `isaac-datagen-inliers <render_dir>`).
User wants one command bundling all three (decision: all three phases; Python console
script, consistent with the existing entry-point pattern).

Key constraint: phases must stay **separate processes**. Isaac Sim's `SimulationApp`
only reliably releases the GPU when its process exits, and the proposer then needs
~14 GB itself. So the bundle is a thin subprocess orchestrator, not an in-process
import of the phase modules.

Resume semantics: phases 2 and 3 are individually resumable/idempotent
(skip-if-exists; phase 3 just relabels). Phase 1 is NOT — and re-rendering under an
existing `proposals/` would silently desync them (new scenes, stale skipped
proposals). Therefore the orchestrator **skips phase 1 whenever `obs/` is non-empty**
and tells the user to delete the render dir for a fresh render. Net effect: the whole
pipeline command is safely re-runnable after any interruption.

## Changes

### 1. NEW `src/isaac_datagen/run_pipeline.py`

Thin orchestrator: `load_config` (reused from `runtime_config.py`) only to compute
`render_dir` (needed for the phase-1 skip check and phase-3's dir argument); each
phase invoked via its console script from the same venv `bin/`, `check=True`
fail-fast, argv passed through verbatim to phases 1 and 2.

```python
"""Run the full reference-seg datagen pipeline: render → proposals → inlier labels.

Thin orchestrator: each phase runs as its OWN subprocess (Isaac Sim only releases
the GPU when its process exits; the proposer then needs it), chained fail-fast.
Phase 1 is skipped when the render dir already has obs/ frames — phases 2 and 3
are individually resumable, but re-rendering under an existing proposals/ would
silently desync them. Delete the render dir to force a fresh render.

Usage: isaac-datagen-pipeline <config.yaml> [key=value ...]
"""

import shutil
import subprocess
import sys
from pathlib import Path

from isaac_datagen.runtime_config import load_config


def _run(script: str, *args: str) -> None:
    exe = shutil.which(script, path=str(Path(sys.executable).parent)) or shutil.which(script)
    if exe is None:
        sys.exit(f"console script not found: {script} (uv sync?)")
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    try:
        subprocess.run([exe, *args], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"{script} failed with exit code {e.returncode} — fix and re-run "
                 f"(completed phases/frames are skipped on resume)")


def main():
    if len(sys.argv) < 2:
        print("usage: isaac-datagen-pipeline <config.yaml> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    runtime = load_config(sys.argv[1], sys.argv[2:])
    render_dir = Path(runtime.dataset_dir) / f"render{runtime.idx:03d}"

    obs = render_dir / "obs"
    n_obs = len(list(obs.glob("obs_*.png"))) if obs.is_dir() else 0
    if n_obs:
        print(f"phase 1: {obs} already has {n_obs} frames — skipping render "
              f"(delete {render_dir} to re-render)", flush=True)
    else:
        _run("isaac-datagen", *sys.argv[1:])
    _run("isaac-datagen-proposals", *sys.argv[1:])
    _run("isaac-datagen-inliers", str(render_dir))


if __name__ == "__main__":
    main()
```

### 2. `pyproject.toml` — `[project.scripts]` entry

`isaac-datagen-pipeline = "isaac_datagen.run_pipeline:main"`

## Verification

1. Import smoke: `uv run python -c "from isaac_datagen.run_pipeline import main"`.
2. End-to-end on the fully-done `cid-mask-verify/render900` (safe while the GPU is
   busy: phase 1 skips on existing obs/, phase 2 skips 4/4 and exits before loading
   the proposer, phase 3 is NN-free):
   `uv run isaac-datagen-pipeline configs/randomized.yaml idx=900 dataset_dir=cid-mask-verify`
   → expect the phase-1 skip line, `resume: 4/4 … skipping`, and a fresh phase-3
   stats line; all three `===` banners in order.
3. Fail-fast: run with a bogus override (e.g. nonexistent `proposer_config_path`) and
   confirm the chain stops at phase 2 with the exit-code message. (Optional.)

## Outcome (2026-06-05)

Shipped as planned: `src/isaac_datagen/run_pipeline.py` + `isaac-datagen-pipeline`
console script. Verified on the fully-done `cid-mask-verify/render900` while a real
phase-2 job ran on the GPU: phase-1 skip line, `resume: 4/4 … skipping` (proposer
never loaded), fresh phase-3 stats (78,265/152,981 — totals shifted from the
cid-mask-dual run because two frames were regenerated; DKM point sampling is
nondeterministic). Fail-fast path is plain `subprocess.run(check=True)` → message
with exit code; not separately exercised.

"""Shared `--help` epilog for the pipeline's front-door commands (isaac-datagen,
isaac-datagen-pipeline). Hand-maintained: mirrors pyproject.toml's [project.scripts]
comments and a scan of src/isaac_datagen/configs/*.yaml — update both when either drifts.
"""

_PHASES = """\
PHASES
  isaac-datagen <config.yaml> [key=value ...]
      phase 1 (Isaac): render a dataset (RGB-D + masks + reference features).
      e.g. isaac-datagen configs/mixed.yaml idx=0 num_frames=8

  isaac-datagen-proposals <render_dir> [key=value ...]
      phase 2 (no Isaac): add proposer point-prompts onto a rendered dataset.

  isaac-datagen-downsample-proposals <render_dir> [...]
      phase 2.5 (no Isaac): FPS-cap each class's proposals; run before inliers.

  isaac-datagen-inliers <render_dir> --eps <inlier_border_eps>
      phase 3 (no Isaac): label each proposal inlier/outlier for verifier training.

  isaac-datagen-pipeline <config.yaml> [key=value ...]
      all three phases as one resumable command — start here.
      e.g. isaac-datagen-pipeline configs/mixed.yaml idx=0 num_frames=8

  isaac-datagen-unseen <config.yaml> <source_render_dir> (--start S --end E | --split-manifest J) [...]
      channel-swap 'unseen 0-shot' eval dir from an existing phase-1 render dir.
"""

_OVERRIDES = """\
COMMON OVERRIDES (dotlist key=value, applied after the YAML)
  idx=N            render index -> dataset_dir/renderNNN/
  num_frames=N     frames to capture
  num_targets=N    grasp targets per frame
  mode=...         required (no default) for amazon.yaml / mixed.yaml / shelf.yaml —
                   pass mode=optflow or mode=reference_segmentation

  occasionally, for ablations:
  seed=N                                RNG stream (effective_seed = seed + idx)
  distant_intensity=N                   key-light intensity
  distant_light_offset=[x,y,z]          sun direction
  occluders_per_target=N                shadow-casting occluders per grasp target
  occluder_scale=N                      occluder cube size
  placement_args.max_column_height=N    stacker column height cap
"""

_CONFIGS = """\
CONFIGS (src/isaac_datagen/configs/)
  amazon.yaml                    phase-1 render, amazon-only catalog, LookAtPoser halo poses
                                  (mode not set — override on CLI)
  expanded-refseg.yaml           optflow re-render of legacy phase-1 render dirs (mode:optflow baked)
  expanded-refseg-v2.yaml        curated 10-class amazon-v2 + per-frame light/exposure jitter
  jagged-expanded-refseg-v2.yaml jagged column-height ablation of expanded-refseg-v2
  jagged2-expanded-refseg-v2.yaml  same as jagged-expanded-refseg-v2, seed=100 (distinct RNG stream)
  blues-expanded-refseg-v2.yaml  jagged-expanded-refseg-v2 restricted to the 4 blue-family classes
                                  (teal/aquamarine/blue/cyan RegexFilter on class)
  mixed.yaml                     heterogeneous multi-source catalog, GridFixedPoser (mode required)
  staggered.yaml                 mixed.yaml + column y-depth/x-gap stagger
  shelf.yaml                     multi-source + class-filter regex chain + occluders (mode required)
  random3_smoke.yaml             small smoke test, random 3-object subset
  tuna_only_smoke.yaml           single-class smoke test
"""

_PATHS = """\
KEY PATHS
  Launch cwd must be src/isaac_datagen/ — config-relative paths (assets, sibling
  reference_matching configs) resolve from there.
  dataset_dir must already exist (RuntimeConfig.__post_init__ asserts this).
  Configs:       src/isaac_datagen/configs/
  Footguns doc:  .docs_claude/psc-isaac-datagen-footguns.md
  Module index:  CLAUDE.md
"""

TLDR = "\n".join([_PHASES, _OVERRIDES, _CONFIGS, _PATHS])

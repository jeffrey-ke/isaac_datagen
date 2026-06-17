"""RuntimeConfig schema + YAML/dotlist loader.

Single source of truth for datagen configuration. Downstream code
(build_scene, make_replicator) depends only on RuntimeConfig.

Computed fields surface in YAML as ${call:<fn_name>[,arg,...]}. The
referenced function must exist at module level; typos raise AttributeError
at load time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
import os
import sys

from omegaconf import OmegaConf

from isaac_datagen.filters import FilterSpec


@dataclass
class RuntimeConfig:
    idx: int
    mode: str
    num_targets: int
    scene: str
    dataset_dir: str
    intrinsics_path: str

    descriptor_device: str
    proposer_device: str


    proposer_config_path: str
    descriptor_config_path: str

    # Object-placement policy: a class name from the placers.py registry, e.g.
    # "UntilExhaustedStacker". Its ctor kwargs come from placement_args (below).
    # Required (no default) so a config can't silently get the wrong policy.
    placement: str
    dome_light: bool

    dry_run: bool

    proposer_max_occlusion: float

    # Phase-3 inliers: a proposal counts as an inlier only if it lies ≥ this many px
    # inside its class union mask (border margin; see vision_core.mask_utils.coords_in_mask).
    inlier_border_eps: float

    num_frames: int | None = None
    grid_dims: tuple[int, int, int] | None = None


    seed: int = 1
    width: int = 1920
    height: int = 1080

    # Dry run (dotlist: dry_run=true): build the scene and plan poses exactly as the
    # real run, then export scene.usdz + baked debug cameras/grasp-axes and the planned
    # poses for offline (Blender) inspection — skipping the writer and RTX capture.

    # Phase-2 proposals: skip objects whose occlusion ratio is ≥ this. A class is
    # kept if its best-visible member passes; NaN (unknown) never passes.

    # Phase-2 frame window (contiguous sharding): process frames
    # [start_frame, end_frame); end_frame=None → through the last frame.
    start_frame: int = 0
    end_frame: int | None = None

    # run_pipeline only: >1 device → one phase-2 subprocess per device, splitting
    # the frame window contiguously. None/1 device → single proposer run.
    proposer_devices: tuple[str, ...] | None = None

    # RTX render cost / VRAM tuning (consumed by boot_sim). Defaults preserve the
    # prior hardcoded behavior; lower these to trade render quality/speed for VRAM.
    #
    #   field                    | default | effect
    #   -------------------------|---------|------------------------------------------------
    #   path_tracing_spp         | 256     | samples/pixel; lower = noisier, less mem, faster
    #   path_tracing_max_bounces | 12      | light bounces; lower = less memory/time
    #   enable_texture_streaming | False   | True = stream textures on demand (less VRAM)
    #   texture_streaming_budget | 0.6     | VRAM fraction cap for streaming (when enabled)
    #   debug_material_type      | -1      | 0 disables materials (flat output, big savings); -1 = normal
    path_tracing_spp: int = 256
    path_tracing_max_bounces: int = 12
    enable_texture_streaming: bool = False
    texture_streaming_budget: float = 0.6
    debug_material_type: int = -1
    xrange: tuple[float, float] = (0.550, 0.550)
    yrange: tuple[float, float] = (-0.22, 0.22)
    zrange: tuple[float, float] = (0.01, 0.01)
    target_to_baseline_ypr_desired: tuple[float, float, float] = (90, 0, 90)

    # Object-placement policy registry (placers.py): name a placer class in
    # `placement`, pass its ctor kwargs here verbatim (mirrors pose_generation_policy
    # / segmentation OptimConfig). e.g. {"column_height": 5} for UntilExhaustedStacker.
    placement_args: dict = field(default_factory=dict)
    # Legacy OccupancyGrid grid dims (i,j,k); no longer used by the live placement path
    # (placers.py retired OccupancyGrid), still read by debug_scripts. Optional.
    pallet_dims: tuple[int, int, int] | None = None

    # Pose-generation policy registry (posers.py): name a poser class, pass its
    # ctor kwargs verbatim. Mirrors segmentation OptimConfig (name + args).
    pose_generation_policy: str = "GridFixedPoser"
    pose_generation_policy_args: dict = field(default_factory=dict)

    occluders_per_target: int = 0
    occluder_pose_policy: str = "GridFixedPoser"
    occluder_pose_policy_args: dict = field(default_factory=dict)
    occluder_scale: float | None = None        # fixed cube scale; None → random uniform(0.04, 0.2)

    filter_specs: list[FilterSpec] = field(default_factory=list)

    texture_paths: tuple[str, ...] = ()
    background_textures: tuple[str, ...] = ()

    # ── Lighting: DistantLight key + DomeLight fill ──────────────────────────
    # DistantLight is the main source: parallel rays → uniform wall irradiance
    # (no inverse-square dark-wall falloff) and crisp directional shadows for the
    # occluders. It is AIMED at the grasp-target centroid via look_at+cv2opengl
    # (see build_scene), not hand-rotated. distant_light_offset is the "sun"
    # position relative to that centroid — DIRECTION ONLY matters (distant light),
    # default front-top into the -Y camera-facing faces. Dome is a low ambient
    # fill so shadowed faces don't crush. TUNE distant_intensity (and/or
    # exposure_time) on the first render to land fg_mean ~120-180; keep
    # dome_fill_intensity ~10-20% of the key.
    distant_light: bool = True
    distant_intensity: float = 3000.0
    distant_angle: float = 0.53                        # angular diameter (deg); raise → softer penumbra
    distant_light_offset: tuple[float, float, float] = (1.0, -3.0, 3.0)  # sun pos rel. to centroid; dir only
    dome_fill_intensity: float = 200.0

    # ── Lighting diagnostics (dark-box investigation) ────────────────────────
    # SUPERSEDED: the live recipe is the distant-key + dome-fill block above, not
    # dome-only. These fields remain for the seeded dome-intensity diagnostic path
    # only (jitter_dome=False by default → unused while the key light is static).
    # Historical note (render848): a FIXED, non-normalized dome at intensity 1000 under
    # exposure_time=1.0 → fg_mean ~178 was the validated dome-only config. dome_normalize=True
    # divides intensity by ~4π solid angle → starves the dome into the ACES toe (dark wall);
    # keep it False. dome_intensity_range only feeds the (off) jitter path. The dome is now a
    # low fill (dome_fill_intensity), not the main light.
    jitter_dome: bool = False
    dome_intensity_range: tuple[float, float] = (500.0, 1000.0)
    dome_normalize: bool = False
    log_lighting: bool = True                 # write <render_dir>/lighting_log.json

    # ── RTX exposure (dark-box fix) ──────────────────────────────────────────
    # The captured `rgb` AOV is post-tonemap LDR (ACES, /rtx/post/tonemap/op=6).
    # Auto-exposure is off, so exposure is fixed by the photographic triangle
    # below. The shipped carb default exposureTime=0.02s (a daylight shutter,
    # ~EV100 10) underexposed our dome-lit scene into the ACES toe → the box wall
    # quantized to exactly 0 until radiance cleared the toe, then snapped white
    # (the dark-box cliff). A longer shutter pulls the wall into the midtones;
    # lower f_number / higher film_iso also brighten. set_exposure=False leaves
    # the RTX defaults untouched.
    set_exposure: bool = True
    exposure_time: float = 1.0                # shutter seconds; 0.02 (old default) → 1.0 ≈ +5.6 stops
    f_number: float = 5.0                     # aperture f-stop; lower = brighter (∝ 1/f_number²)
    film_iso: float = 100.0                   # sensor ISO; higher = brighter (∝ iso/100)

    # Subframes accumulated per captured frame — the documented "materials not
    # loaded in time" / denoise knob (≥2 per Isaac 5.1 Replicator troubleshooting).
    # NOTE: this did NOT fix the intermittent all-black render; that is a
    # per-process renderer-init coin flip decided before the first frame, not a
    # per-frame settle issue. See .docs_claude/plans/active/render-darkness-investigation.md.
    rt_subframes: int = 20

    # app.update() frames to settle RTX before capture: lets MDL shaders compile, the dome
    # HDRI/textures stream in, and PT/denoiser state initialize, so frame 0 isn't a black
    # coin flip. This targets the per-process renderer-init flip rt_subframes did NOT fix
    # (see note above). Mirrors Isaac's camera-sensor warmup (isaacsim.sensors.camera tests:
    # N x app.update() before reading pixels). 0 disables.
    warmup_frames: int = 32

    # Inspection toggle: True → obs RGBA is fully opaque (alpha=255 everywhere) instead of
    # cropped to graspable-instance pixels (composite_rgba). Lets you view the whole frame —
    # shadows / background / occluders included. OFF by default; iid_mask still carries the true
    # foreground, so the dataset contract is intact.
    obs_full_alpha: bool = False

    # ── Optical-flow dataset (Plan 2) ────────────────────────────────────────
    # mode selects the orchestrator in clean_datagen.main():
    #   "reference_segmentation" (default, ObsMask dataset) or "optflow" (OptFlow dataset).
    # Object dataset dir(s) the active orchestrator places + captures. `mode` already
    # selects the collector that interprets these (collect_objects for
    # reference_segmentation, collect_preoptflow for optflow), so one field suffices
    # and is required for both modes.
    objects_path: list[str] = field(default_factory=list)

    def __post_init__(self):
        assert (self.num_frames is None) ^ (self.grid_dims is None), \
            "exactly one of num_frames / grid_dims must be set"
        assert self.start_frame >= 0 and (self.end_frame is None or self.end_frame > self.start_frame), \
            f"bad frame window [{self.start_frame}, {self.end_frame})"
        assert self.inlier_border_eps >= 0, f"inlier_border_eps must be ≥ 0: {self.inlier_border_eps}"
        lo, hi = self.dome_intensity_range
        assert lo <= hi, f"dome_intensity_range must have lo<=hi: {(lo, hi)}"
        assert self.exposure_time > 0, f"exposure_time must be > 0: {self.exposure_time}"
        assert self.f_number > 0, f"f_number must be > 0: {self.f_number}"
        assert self.film_iso > 0, f"film_iso must be > 0: {self.film_iso}"
        assert self.rt_subframes >= 1, f"rt_subframes must be >= 1: {self.rt_subframes}"
        assert self.mode in ("reference_segmentation", "optflow"), f"unknown mode: {self.mode!r}"
        assert self.objects_path, f"mode={self.mode} requires objects_path"
        assert Path(self.dataset_dir).exists(), f"dataset_dir missing: {self.dataset_dir}"
        assert Path(self.intrinsics_path).exists(), f"intrinsics_path missing: {self.intrinsics_path}"
        assert Path(self.proposer_config_path).exists(), f"proposer_config_path missing: {self.proposer_config_path}"
        assert Path(self.descriptor_config_path).exists(), f"descriptor_config_path missing: {self.descriptor_config_path}"

    @property
    def sampling(self) -> int | tuple[int, int, int]:
        if self.num_frames is not None:
            return self.num_frames
        assert self.grid_dims is not None
        return self.grid_dims

    @property
    def effective_seed(self) -> int:
        """Per-render seed: base seed offset by render idx, so each render dir is
        distinct yet reproducible (re-render same idx → identical scene)."""
        return self.seed + self.idx


def _glob_amazon_textures() -> list[str]:
    from isaac_datagen.scene import RESOURCE_PATH
    d = os.path.join(RESOURCE_PATH, "boxes", "textures")
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.startswith("amazon_texture_")
    )


def _call(name: str, *args):
    return getattr(sys.modules[__name__], name)(*args)


def register_resolvers():
    OmegaConf.register_new_resolver("call", _call, replace=True)


def load_config(yaml_path: str | Path, dotlist: list[str]) -> RuntimeConfig:
    from omegaconf import OmegaConf
    register_resolvers()
    schema = OmegaConf.structured(RuntimeConfig)
    yaml_cfg = OmegaConf.load(str(yaml_path))
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    merged = OmegaConf.merge(schema, yaml_cfg, cli_cfg)
    return cast(RuntimeConfig, OmegaConf.to_object(merged))

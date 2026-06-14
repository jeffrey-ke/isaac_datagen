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


@dataclass
class RuntimeConfig:
    idx: int
    num_targets: int
    scene: str
    dataset_dir: str
    intrinsics_path: str

    descriptor_device: str
    proposer_device: str

    graspable_objects_path: str

    proposer_config_path: str
    descriptor_config_path: str

    pallet_dims: tuple[int, int, int]
    dome_light: bool

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
    dry_run: bool = False

    # Object placement policy: "occupancy_grid" (static full-wall, uniform boxes,
    # requires exactly prod(pallet_dims) objects) or "until_exhausted_stacker"
    # (heterogeneous bboxes, columns of <= column_height, requires >= 1 object).
    placement: str = "occupancy_grid"
    # until_exhausted_stacker only: max objects per vertical column.
    column_height: int = 5

    # Phase-2 proposals: skip objects whose occlusion ratio is ≥ this. A class is
    # kept if its best-visible member passes; NaN (unknown) never passes.
    proposer_max_occlusion: float = 0.10

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

    # Pose-generation policy registry (posers.py): name a poser class, pass its
    # ctor kwargs verbatim. Mirrors segmentation OptimConfig (name + args).
    pose_generation_policy: str = "GridFixedPoser"
    pose_generation_policy_args: dict = field(default_factory=dict)

    # Shadow-occluder domain randomization (scene.add_shadow_occluders): per grasp
    # target, place this many invisible shapes (0 = off) that cast path-traced
    # shadows on the box but are hidden from the camera (primvars:hideForCamera).
    # Each is positioned once via its own poser (same registry as the camera),
    # sampled in the target frame and mapped to world via the target's target2world.
    occluders_per_target: int = 0
    occluder_pose_policy: str = "GridFixedPoser"
    occluder_pose_policy_args: dict = field(default_factory=dict)
    occluder_scale: float | None = None        # fixed cube scale; None → random uniform(0.04, 0.2)

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

    def __post_init__(self):
        assert (self.num_frames is None) ^ (self.grid_dims is None), \
            "exactly one of num_frames / grid_dims must be set"
        assert self.start_frame >= 0 and (self.end_frame is None or self.end_frame > self.start_frame), \
            f"bad frame window [{self.start_frame}, {self.end_frame})"
        assert self.inlier_border_eps >= 0, f"inlier_border_eps must be ≥ 0: {self.inlier_border_eps}"
        assert self.placement in ("occupancy_grid", "until_exhausted_stacker"), \
            f"unknown placement policy: {self.placement!r}"
        assert self.column_height >= 1, f"column_height must be >= 1: {self.column_height}"
        lo, hi = self.dome_intensity_range
        assert lo <= hi, f"dome_intensity_range must have lo<=hi: {(lo, hi)}"
        assert self.exposure_time > 0, f"exposure_time must be > 0: {self.exposure_time}"
        assert self.f_number > 0, f"f_number must be > 0: {self.f_number}"
        assert self.film_iso > 0, f"film_iso must be > 0: {self.film_iso}"
        assert self.rt_subframes >= 1, f"rt_subframes must be >= 1: {self.rt_subframes}"
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

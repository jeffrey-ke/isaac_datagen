
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
import os
import sys

from omegaconf import OmegaConf

from isaac_datagen.filters import FilterSpec


@dataclass
class LightJitterSpec:
    root: str
    pattern: str
    intensity_scale_range: tuple[float, float]


@dataclass
class RuntimeConfig:
    idx: int
    mode: str
    num_targets: int | None
    scene: str
    dataset_dir: str
    intrinsics_path: str

    descriptor_device: str
    proposer_device: str


    proposer_config_path: str
    descriptor_config_path: str

    placement: str
    dome_light: bool

    dry_run: bool

    inlier_border_eps: float

    num_frames: int | None = None
    grid_dims: tuple[int, int, int] | None = None


    seed: int = 1
    width: int = 1920
    height: int = 1080


    proposer_min_visible_ratio: float = 0.30
    proposer_tau_d: float = 0.001
    proposer_tau_r: float = 0.005
    proposer_min_visible_px: int = 60000
    proposer_max_occlusion: float = 1.0

    start_frame: int = 0
    end_frame: int | None = None

    proposer_devices: tuple[str, ...] | None = None

    path_tracing_spp: int = 256
    path_tracing_max_bounces: int = 12
    enable_texture_streaming: bool = False
    texture_streaming_budget: float = 0.6
    debug_material_type: int = -1
    xrange: tuple[float, float] = (0.550, 0.550)
    yrange: tuple[float, float] = (-0.22, 0.22)
    zrange: tuple[float, float] = (0.01, 0.01)
    target_to_baseline_ypr_desired: tuple[float, float, float] = (90, 0, 90)

    placement_args: dict = field(default_factory=dict)
    pallet_dims: tuple[int, int, int] | None = None

    pose_generation_policy: str = "GridFixedPoser"
    pose_generation_policy_args: dict = field(default_factory=dict)

    occluders_per_target: int = 0
    occluder_pose_policy: str = "GridFixedPoser"
    occluder_pose_policy_args: dict = field(default_factory=dict)
    occluder_scale: float | None = None

    filter_specs: list[FilterSpec] = field(default_factory=list)

    texture_paths: tuple[str, ...] = ()
    background_textures: tuple[str, ...] = ()

    distant_light: bool = True
    distant_intensity: float = 3000.0
    distant_angle: float = 0.53
    distant_light_offset: tuple[float, float, float] = (1.0, -3.0, 3.0)
    dome_fill_intensity: float = 200.0

    jitter_distant: bool = False
    distant_offset_jitter: float = 0.75
    distant_intensity_jitter: tuple[float, float] | None = None
    distant_temperature_jitter: tuple[float, float] | None = None

    jitter_dome: bool = False
    dome_intensity_range: tuple[float, float] = (500.0, 1000.0)
    dome_normalize: bool = False
    log_lighting: bool = True

    set_exposure: bool = True
    exposure_time: float = 1.0
    f_number: float = 5.0
    film_iso: float = 100.0

    rt_subframes: int = 20

    warmup_frames: int = 32

    obs_full_alpha: bool = False

    objects_path: list[str] = field(default_factory=list)

    scene_builder: str = "build_scene"
    scene_builder_args: dict = field(default_factory=dict)
    light_jitter_patterns: list[LightJitterSpec] = field(default_factory=list)

    def __post_init__(self):
        assert self.scene_builder, "scene_builder must name a scene_builders registry entry"
        for s in self.light_jitter_patterns:
            lo, hi = s.intensity_scale_range
            assert s.root and s.pattern and 0 < lo <= hi, f"bad LightJitterSpec: {s}"
        assert (self.num_frames is None) ^ (self.grid_dims is None), \
            "exactly one of num_frames / grid_dims must be set"
        assert self.start_frame >= 0 and (self.end_frame is None or self.end_frame > self.start_frame), \
            f"bad frame window [{self.start_frame}, {self.end_frame})"
        assert self.inlier_border_eps >= 0, f"inlier_border_eps must be ≥ 0: {self.inlier_border_eps}"
        lo, hi = self.dome_intensity_range
        assert lo <= hi, f"dome_intensity_range must have lo<=hi: {(lo, hi)}"
        assert self.distant_offset_jitter >= 0.0, \
            f"distant_offset_jitter must be >= 0: {self.distant_offset_jitter}"
        if self.distant_intensity_jitter is not None:
            lo, hi = self.distant_intensity_jitter
            assert 0 <= lo <= hi, f"distant_intensity_jitter must have 0<=lo<=hi: {(lo, hi)}"
        if self.distant_temperature_jitter is not None:
            lo, hi = self.distant_temperature_jitter
            assert 1000.0 <= lo <= hi <= 10000.0, \
                f"distant_temperature_jitter must be 1000<=lo<=hi<=10000 K: {(lo, hi)}"
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

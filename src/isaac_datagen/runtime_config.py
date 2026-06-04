"""RuntimeConfig schema + YAML/dotlist loader.

Single source of truth for datagen configuration. Downstream code
(build_scene, make_replicator) depends only on RuntimeConfig.

Computed fields surface in YAML as ${call:<fn_name>[,arg,...]}. The
referenced function must exist at module level; typos raise AttributeError
at load time.
"""

from __future__ import annotations
from dataclasses import dataclass
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

    num_frames: int | None = None
    grid_dims: tuple[int, int, int] | None = None


    seed: int = 1
    width: int = 1920
    height: int = 1080

    # Phase-2 proposals: skip objects whose occlusion ratio is ≥ this. A class is
    # kept if its best-visible member passes; NaN (unknown) never passes.
    proposer_max_occlusion: float = 0.10

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

    texture_paths: tuple[str, ...] = ()
    background_textures: tuple[str, ...] = ()

    def __post_init__(self):
        assert (self.num_frames is None) ^ (self.grid_dims is None), \
            "exactly one of num_frames / grid_dims must be set"
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

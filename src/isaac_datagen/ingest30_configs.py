import sys
from pathlib import Path

import yaml

from vision_core.script_args import ScriptArgs
from isaac_datagen.asset_catalogs import catalog_classes

_REFMATCH = "../../../reference_matching/src/reference_matching/configs"
PRODUCT_PATTERNS = ["model_snack*", "model_instant_beverages*", "model_flour*",
                    "model_detergent*", "model_sauces*", "model_cereal*", "model_drink101*"]

_COMMON = dict(
    mode="optflow", scene="empty", intrinsics_path="zed_K.npy",
    descriptor_device="cuda:0", proposer_device="cpu", dry_run=False,
    inlier_border_eps=0.0,
    proposer_config_path=f"{_REFMATCH}/grid_proposal.yaml",
    descriptor_config_path=f"{_REFMATCH}/descriptor.yaml",
    proposer_min_visible_ratio=0.30,
    set_exposure=True, exposure_time=1.0, f_number=5.0, film_iso=100.0,
    rt_subframes=10, warmup_frames=32,          # render speed (validated ~identical to spp256/rt20/bounces12)
    path_tracing_spp=96, path_tracing_max_bounces=6,
)

_JITTERED_LIGHTS = dict(
    dome_light=True, dome_fill_intensity=200.0,
    distant_light=True, distant_intensity=2000.0, distant_angle=0.53,
    distant_light_offset=[1.0, -3.0, 3.0],
    jitter_distant=True, distant_offset_jitter=2.0,
    distant_intensity_jitter=[500.0, 4000.0],
    distant_temperature_jitter=[4500.0, 8500.0],
    jitter_dome=True, dome_intensity_range=[100.0, 350.0],
    light_jitter_patterns=[],
)


def _halo(xr, yr, zr):
    return dict(pose_generation_policy="LookAtPoser",
                pose_generation_policy_args=dict(xrange=xr, yrange=yr, zrange=zr))


def _decentered_halo(xr, yr, zr, radii, cls):
    assert cls in radii, f"no pool_object_radius computed for class {cls!r} -- have: {sorted(radii)}"
    return dict(pose_generation_policy="DecenteredLookAtPoser",
                pose_generation_policy_args=dict(
                    xrange=xr, yrange=yr, zrange=zr,
                    intrinsics_path=_COMMON["intrinsics_path"], resolution=[1920, 1080],
                    object_radius=radii[cls], margin_deg=1.0, max_roll_deg=15.0))


POOL_POSERS = {"LookAtPoser": lambda xr, yr, zr, radii, cls: _halo(xr, yr, zr),
               "DecenteredLookAtPoser": _decentered_halo}


def _pool_poser(sa: ScriptArgs, cls: str) -> dict:
    assert sa.pool_poser in POOL_POSERS, \
        f"unknown pool_poser {sa.pool_poser!r} — valid: {sorted(POOL_POSERS)}"
    cfg = POOL_POSERS[sa.pool_poser]([0.3, 2.0], [-2.0, 2.0], [-0.7, 0.7],
                                      sa.pool_object_radius, cls)
    if sa.pool_offset_sampler:
        cfg["pose_generation_policy_args"]["offset_sampler"] = sa.pool_offset_sampler
    return cfg


def _disable_physics(classes):
    return [{"name": "DisablePhysics", "args": {"pattern": f"{c}*"}} for c in classes]


def base_config(sa: ScriptArgs, base_classes: list[str]) -> dict:
    return _COMMON | _JITTERED_LIGHTS | _halo([0.25, 0.85], [-0.42, 0.42], [-0.40, 0.40]) | dict(
        seed=sa.seeds.base, idx=0,
        num_targets=sa.base_num_targets, num_frames=sa.base_num_frames,
        dataset_dir=f"{sa.root}/datasets/base",
        scene_builder="build_scene",
        scene_builder_args={"grasp_frames": "catalog",
                            "orientation": {"name": "AlignGraspFronts",
                                            "args": {"azimuth_deg": -90}},  # fronts face -Y (bbox-grasp convention)
                            "mutations": _disable_physics(base_classes)},
        placement="UntilExhaustedStacker",
        placement_args={"max_column_height": 3, "min_y": 0.0, "max_y": 0.10,
                        "min_gap": 0.0, "max_gap": 0.50},
        objects_path=[str(sa.base_catalog)],
        filter_specs=[
            {"name": "ReplicateFilter",
             "args": {"key": "name", "value": "*", "count": sa.base_replicas}},
            {"name": "ShuffleFilter", "args": {"seed": "${idx}"}},
        ],
    )


def pool_config(sa: ScriptArgs, cls: str) -> dict:
    return _COMMON | _JITTERED_LIGHTS | _pool_poser(sa, cls) | dict(
        seed=sa.seeds.pools, idx=0,
        num_targets=None, num_frames=sa.pool_frames,
        dataset_dir=f"{sa.root}/datasets/pools/{cls}-1inst",
        scene_builder="build_scene",
        scene_builder_args={"grasp_frames": "catalog",
                            "mutations": _disable_physics([cls])},
        placement="UntilExhaustedStacker",
        placement_args={"max_column_height": 1},
        objects_path=[str(sa.ingest_catalog)],
        filter_specs=[{"name": "RegexFilter",
                       "args": {"key": "class", "value": f"^{cls}$"}}],
    )


def _store_config(sa: ScriptArgs, classes: list[str], dataset_dir: str, num_frames: int) -> dict:
    alternation = "^(" + "|".join(sorted(classes)) + ")$"
    return _COMMON | _halo([0.3, 0.9], [-0.3, 0.3], [-0.2, 0.3]) | dict(
        seed=sa.seeds.test, idx=0,
        num_targets=None, num_frames=num_frames,
        dataset_dir=dataset_dir,
        scene_builder="build_store_scene",
        scene_builder_args={
            "store_usd": "../../../usds/store001.usd",
            "product_patterns": PRODUCT_PATTERNS,
            "grasp_frame_policy": "FixedFaceGrasp",
            "grasp_frame_policy_args": {"face": "-Y"},
            "mutations": [{"name": "RemoveUntrackedProducts"}],
            "require_tracked_only": PRODUCT_PATTERNS,
        },
        dome_light=False, distant_light=False,
        light_jitter_patterns=[{"root": "/World/Store", "pattern": "*Light*",
                                "intensity_scale_range": [0.25, 8.0]}],
        placement="UntilExhaustedStacker", placement_args={},
        objects_path=[str(sa.base_catalog), str(sa.ingest_catalog)],
        filter_specs=[{"name": "RegexFilter",
                       "args": {"key": "class", "value": alternation}}],
    )


def test_store_config(sa: ScriptArgs, all_classes: list[str]) -> dict:
    return _store_config(sa, all_classes, f"{sa.root}/datasets/test/store",
                         sa.test_store_num_frames)
test_store_config.__test__ = False   # builder, not a pytest test, despite the name


def smoke_config(sa: ScriptArgs, cls: str) -> dict:
    return _store_config(sa, [cls], f"{sa.root}/smoke/{cls}", 3)


def test_composed_config(sa: ScriptArgs, all_classes: list[str]) -> dict:
    return base_config(sa, all_classes) | dict(
        seed=sa.seeds.test,
        num_targets=sa.test_composed_num_targets,
        num_frames=sa.test_composed_num_frames,
        dataset_dir=f"{sa.root}/datasets/test/composed",
        objects_path=[str(sa.base_catalog), str(sa.ingest_catalog)],
        filter_specs=[
            {"name": "ReplicateFilter",
             "args": {"key": "name", "value": "*", "count": sa.test_composed_replicas}},
            {"name": "ShuffleFilter", "args": {"seed": "${idx}"}},
        ],
    )
test_composed_config.__test__ = False   # builder, not a pytest test, despite the name


def write_all(sa: ScriptArgs) -> list[Path]:
    base_classes, ingest_classes = list(sa.base_classes), list(sa.ingest_classes)
    overlap = set(base_classes) & set(ingest_classes)
    assert not overlap, f"base/ingest share classes: {sorted(overlap)}"
    assert sorted(base_classes) == catalog_classes(sa.base_catalog), \
        "manifest base_classes drifted from assembled catalog"
    assert sorted(ingest_classes) == catalog_classes(sa.ingest_catalog), \
        "manifest ingest_classes drifted from assembled catalog"
    all_classes = sorted(base_classes + ingest_classes)

    out = Path(sa.root) / "configs" / "datagen"
    (out / "smoke").mkdir(parents=True, exist_ok=True)
    configs: dict[Path, dict] = {out / "base.yaml": base_config(sa, base_classes),
                                 out / "test-store.yaml": test_store_config(sa, all_classes),
                                 out / "test-composed.yaml": test_composed_config(sa, all_classes)}
    for cls in ingest_classes:
        configs[out / f"pool-{cls}.yaml"] = pool_config(sa, cls)
    for cls in all_classes:
        configs[out / "smoke" / f"{cls}.yaml"] = smoke_config(sa, cls)

    for path, cfg in configs.items():
        Path(cfg["dataset_dir"]).mkdir(parents=True, exist_ok=True)   # dataset_dir MUST pre-exist
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[ingest30-configs] N={len(base_classes)} M={len(all_classes)} "
          f"-> {len(configs)} configs under {out}")
    return sorted(configs)


def main():
    write_all(ScriptArgs.load(sys.argv[1]))

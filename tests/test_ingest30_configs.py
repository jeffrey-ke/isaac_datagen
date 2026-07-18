import yaml
import pytest

from vision_core.script_args import ScriptArgs
from isaac_datagen.ingest30_configs import (
    base_config, pool_config, test_store_config, test_composed_config,
    write_all, POOL_POSERS,
)

ARGS = dict(
    root="",                                    # filled per test
    base_assets=[], ingest_assets=[],           # writer never reads these; classes carry the info
    base_classes=["cereal001", "sauces001"],
    ingest_classes=["snack031", "flour001"],
    descriptor="CleanDiftFpn",
    descriptor_bake_config="refmatch/fpn.yaml",
    seeds=dict(base=3001, pools=3101, test=3201),
    base_num_dirs=2, base_num_targets=4, base_num_frames=10, base_replicas=5,
    pool_frames=100, test_store_num_frames=5,
    test_composed_num_dirs=1, test_composed_num_targets=4,
    test_composed_num_frames=10, test_composed_replicas=3,
    store_site_catalog="",              # filled per test by sa_for
    test_store_replicas=2, test_store_num_targets=20,
)


def fake_assembled(root, name, classes):
    meta = root / "catalogs" / name / "meta"
    meta.mkdir(parents=True)
    for i, c in enumerate(classes):
        (meta / f"meta_{i:04d}.yaml").write_text(yaml.safe_dump(
            # ycb-style: object name shares NO prefix with its class (the DisablePhysics trap)
            {"name": f"ycb_{i:03d}_{c}", "class": c, "store_prim": f"model_{c}/v_0"}))


def fake_site_catalog(root, n):
    import numpy as np
    meta = root / "sites" / "meta"
    gp = root / "sites" / "grasp_point"
    meta.mkdir(parents=True); gp.mkdir(parents=True)
    for i in range(n):
        (meta / f"meta_{i:04d}.yaml").write_text(yaml.safe_dump(
            {"name": f"s{i:03d}", "class": f"s{i:03d}", "store_prim": f"model_s{i:03d}/v_0"}))
        np.save(gp / f"grasp_point_{i:04d}.npy", np.eye(4, dtype=np.float32))
    return root / "sites"


def sa_for(tmp_path, n_sites=10):
    import copy
    d = copy.deepcopy(ARGS)
    root = tmp_path / "root"
    d["root"] = str(root)
    fake_assembled(root, "base", ["cereal001", "sauces001"])
    fake_assembled(root, "ingest", ["snack031", "flour001"])
    d["store_site_catalog"] = str(fake_site_catalog(root, n_sites))
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(d))
    return ScriptArgs.load(p)


def test_pool_config(tmp_path):
    sa = sa_for(tmp_path)
    cfg = pool_config(sa, "snack031")
    assert cfg["seed"] == 3101 and cfg["num_frames"] == 100
    assert cfg["dataset_dir"].endswith("datasets/pools/snack031-1inst")
    assert cfg["filter_specs"] == [
        {"name": "RegexFilter", "args": {"key": "class", "value": "^snack031$"}}]
    assert cfg["scene_builder_args"]["mutations"] == [
        {"name": "DisablePhysics", "args": {"pattern": "ycb_000_snack031*"}}]  # placed names only
    assert cfg["objects_path"] == [str(sa.ingest_catalog)]


def test_pool_config_default_poser_unchanged(tmp_path):
    sa = sa_for(tmp_path)                      # pool_poser defaults to "LookAtPoser"
    cfg = pool_config(sa, "snack031")
    assert cfg["pose_generation_policy"] == "LookAtPoser"
    assert cfg["pose_generation_policy_args"] == {
        "xrange": [0.3, 2.0], "yrange": [-2.0, 2.0], "zrange": [-0.7, 0.7]}


def test_pool_config_decentered_poser(tmp_path):
    import dataclasses

    sa = sa_for(tmp_path)
    sa = dataclasses.replace(sa, pool_poser="DecenteredLookAtPoser",
                             pool_object_radius={"snack031": 0.22, "flour001": 0.31})
    cfg = pool_config(sa, "snack031")
    assert cfg["pose_generation_policy"] == "DecenteredLookAtPoser"
    args = cfg["pose_generation_policy_args"]
    assert args["xrange"] == [0.3, 2.0] and args["yrange"] == [-2.0, 2.0] and args["zrange"] == [-0.7, 0.7]
    assert args["object_radius"] == 0.22       # snack031's radius, not flour001's
    assert args["intrinsics_path"] == "zed_K.npy"
    assert args["resolution"] == [1920, 1080]
    assert args["margin_deg"] == 1.0 and args["max_roll_deg"] == 15.0


def test_pool_poser_unknown_fails_loud(tmp_path):
    import dataclasses

    sa = sa_for(tmp_path)
    sa = dataclasses.replace(sa, pool_poser="TotallyMadeUp")
    with pytest.raises(AssertionError, match="TotallyMadeUp"):
        pool_config(sa, "snack031")


def test_decentered_poser_missing_radius_fails_loud(tmp_path):
    import dataclasses

    sa = sa_for(tmp_path)
    sa = dataclasses.replace(sa, pool_poser="DecenteredLookAtPoser",
                             pool_object_radius={"flour001": 0.31})   # snack031 missing
    with pytest.raises(AssertionError, match="snack031"):
        pool_config(sa, "snack031")


def test_pool_posers_registry_has_both():
    assert set(POOL_POSERS) == {"LookAtPoser", "DecenteredLookAtPoser"}


def test_pool_config_default_offset_sampler_has_no_override(tmp_path):
    sa = sa_for(tmp_path)                       # pool_offset_sampler defaults to {}
    cfg = pool_config(sa, "snack031")
    assert "offset_sampler" not in cfg["pose_generation_policy_args"]


def test_pool_config_log_offset_sampler_injected_with_default_poser(tmp_path):
    import dataclasses

    sa = sa_for(tmp_path)
    sa = dataclasses.replace(sa, pool_offset_sampler={
        "name": "log_uniform_offsets", "args": {"floor": 0.02}})
    cfg = pool_config(sa, "snack031")
    assert cfg["pose_generation_policy"] == "LookAtPoser"          # unaffected — independent axis
    assert cfg["pose_generation_policy_args"]["offset_sampler"] == {
        "name": "log_uniform_offsets", "args": {"floor": 0.02}}


def test_pool_config_log_offset_sampler_injected_with_decentered_poser(tmp_path):
    import dataclasses

    sa = sa_for(tmp_path)
    sa = dataclasses.replace(sa, pool_poser="DecenteredLookAtPoser",
                             pool_object_radius={"snack031": 0.22, "flour001": 0.31},
                             pool_offset_sampler={
                                 "name": "log_uniform_offsets", "args": {"floor": 0.02}})
    cfg = pool_config(sa, "snack031")
    assert cfg["pose_generation_policy"] == "DecenteredLookAtPoser"
    args = cfg["pose_generation_policy_args"]
    assert args["object_radius"] == 0.22                           # radii mechanism untouched
    assert args["offset_sampler"] == {
        "name": "log_uniform_offsets", "args": {"floor": 0.02}}


def test_base_config(tmp_path):
    sa = sa_for(tmp_path)
    cfg = base_config(sa)
    assert cfg["seed"] == 3001 and cfg["num_targets"] == 4 and cfg["num_frames"] == 10
    assert cfg["filter_specs"][0] == {
        "name": "ReplicateFilter", "args": {"key": "name", "value": "*", "count": 5}}
    muts = cfg["scene_builder_args"]["mutations"]
    assert [m["args"]["pattern"] for m in muts] == [
        "ycb_000_cereal001*", "ycb_001_sauces001*"]   # one freeze per placed base object


def test_store_config_repopulates(tmp_path):
    sa = sa_for(tmp_path)
    cfg = test_store_config(sa, ["cereal001", "flour001", "sauces001", "snack031"])
    assert cfg["seed"] == 3201
    assert cfg["scene_builder"] == "build_repopulated_store_scene"
    assert cfg["scene_builder_args"]["site_catalog"] == sa.store_site_catalog
    assert cfg["scene_builder_args"]["mutations"] == []          # repopulation, not RemoveUntrackedProducts
    assert cfg["num_targets"] == 20
    names = [f["name"] for f in cfg["filter_specs"]]
    assert names == ["ReplicateFilter", "ShuffleFilter"]         # balance + order, no RegexFilter
    assert cfg["filter_specs"][0]["args"]["count"] == sa.test_store_replicas
    assert cfg["objects_path"] == [str(sa.base_catalog), str(sa.ingest_catalog)]


def test_store_config_num_targets_null(tmp_path):
    sa = sa_for(tmp_path)
    sa.test_store_num_targets = -1                       # sentinel: all placed objects are targets
    cfg = test_store_config(sa, ["cereal001", "flour001", "sauces001", "snack031"])
    assert cfg["num_targets"] is None                    # <0 -> null
    sa.test_store_num_targets = 12
    assert test_store_config(sa, [])["num_targets"] == 12  # non-negative passes through unchanged


def test_write_all(tmp_path):
    sa = sa_for(tmp_path)
    written = write_all(sa)
    names = {p.name for p in written}
    assert names == {"base.yaml", "pool-snack031.yaml", "pool-flour001.yaml",
                      "test-store.yaml", "test-composed.yaml"}
    root = tmp_path / "root"
    assert (root / "datasets" / "pools" / "snack031-1inst").is_dir()
    for p in written:                                    # every file is valid yaml
        yaml.safe_load(p.read_text())


def test_write_all_no_smoke_configs(tmp_path):
    sa = sa_for(tmp_path)              # M=4 classes, replicate=2 (site cap lives in _init_manifest now)
    write_all(sa)
    out = tmp_path / "root" / "configs" / "datagen"
    assert (out / "test-store.yaml").exists()
    assert not (out / "smoke").exists()          # smoke configs no longer written


def test_composed_config_fields(tmp_path):
    sa = sa_for(tmp_path)
    cfg = test_composed_config(sa)
    assert cfg["seed"] == 3201                           # test seed, not base's 3001
    assert cfg["num_targets"] == 4 and cfg["num_frames"] == 10
    assert cfg["dataset_dir"].endswith("datasets/test/composed")
    assert cfg["objects_path"] == [str(sa.base_catalog), str(sa.ingest_catalog)]
    assert cfg["filter_specs"][0]["args"]["count"] == 3  # test_composed_replicas, not base_replicas
    muts = cfg["scene_builder_args"]["mutations"]
    assert {m["args"]["pattern"] for m in muts} == {     # every base+ingest object frozen by name
        "ycb_000_cereal001*", "ycb_001_sauces001*", "ycb_000_snack031*", "ycb_001_flour001*"}


_ORIENTATION = {"name": "AlignGraspFronts", "args": {"azimuth_deg": -90}}


def test_base_config_fronts_objects(tmp_path):
    sa = sa_for(tmp_path)
    cfg = base_config(sa)
    assert cfg["scene_builder_args"]["orientation"] == _ORIENTATION


def test_composed_config_fronts_objects(tmp_path):
    sa = sa_for(tmp_path)
    cfg = test_composed_config(sa)
    assert cfg["scene_builder_args"]["orientation"] == _ORIENTATION


def test_pool_and_store_configs_have_no_orientation(tmp_path):
    sa = sa_for(tmp_path)
    assert "orientation" not in pool_config(sa, "snack031")["scene_builder_args"]
    assert "orientation" not in test_store_config(sa, ["snack031"])["scene_builder_args"]

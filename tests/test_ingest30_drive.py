import argparse
from pathlib import Path

import pytest
import yaml

from vision_core.script_args import ScriptArgs
from isaac_datagen.ingest30_drive import (
    VERBS, _init_manifest, _unpack_arm_positionals, arm_commands, parse_args,
    curves_commands, score_commands, stage_bake, stage_flatten, stage_render,
)

MANIFEST = dict(
    root="", base_assets=[], ingest_assets=[],
    base_classes=["zebra"], ingest_classes=["apple", "kiwi"],
    descriptor="D", descriptor_bake_config="/refmatch/fpn.yaml",
    seeds=dict(base=1, pools=2, test=3),
    base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
    pool_frames=3, test_store_num_frames=1,
    test_composed_num_dirs=1, test_composed_num_targets=1,
    test_composed_num_frames=1, test_composed_replicas=1,
    store_site_catalog="/data/keep", test_store_replicas=9, test_store_num_targets=20,
)


def sa_(tmp_path):
    dg = tmp_path / "configs" / "datagen"
    dg.mkdir(parents=True)
    for name in ("base.yaml", "pool-apple.yaml", "pool-kiwi.yaml",
                 "test-store.yaml", "test-composed.yaml"):
        # _rendered() reads dataset_dir; none of these dirs exist -> all jobs emitted
        (dg / name).write_text(yaml.safe_dump(
            {"dataset_dir": str(tmp_path / "datasets" / name.removesuffix(".yaml"))}))
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(MANIFEST | {"root": str(tmp_path)}))
    return ScriptArgs.load(p)


def test_render_commands(tmp_path):
    cmds = stage_render(sa_(tmp_path))
    argvs = [" ".join(c.argv) for c in cmds]
    assert sum("base.yaml idx=0" in a for a in argvs) == 1
    assert sum("base.yaml idx=1" in a for a in argvs) == 1     # base_num_dirs=2
    assert sum("pool-apple.yaml idx=0" in a for a in argvs) == 1
    assert sum("test-composed.yaml idx=0" in a for a in argvs) == 1
    assert all(c.drop_pythonpath for c in cmds)                # Isaac needs clean env


def test_bake_commands(tmp_path):
    sa = sa_(tmp_path)
    for d in ("datasets/base", "datasets/pools/apple-1inst", "datasets/test/store"):
        (tmp_path / d).mkdir(parents=True)
    cmds = stage_bake(sa)
    joined = [" ".join(c.argv) for c in cmds]
    assert all("add-backbone" in a and "/refmatch/fpn.yaml" in a for a in joined)
    assert len(cmds) == 3


def test_arm_commands_two_phase():
    cmds = arm_commands("/r", "gligen", "b.yaml", "i.yaml",
                        label=None, all_data=False, allow_dirty=True)
    joined = [" ".join(c.argv) for c in cmds]
    assert len(cmds) == 2
    assert "ingest30-base-train /r gligen b.yaml --label gligen" in joined[0]
    assert "ingest30-loop /r gligen i.yaml --label gligen" in joined[1]
    assert all("--allow-dirty" in j for j in joined)
    assert all(c.drop_pythonpath for c in cmds)                # segmentation needs clean env


def test_arm_retrained_maps_to_closed():
    cmds = arm_commands("/r", "retrained", "r.yaml", None,
                        label=None, all_data=True, allow_dirty=False)
    (cmd,) = cmds                                              # base-train ONLY, no loop
    j = " ".join(cmd.argv)
    assert "ingest30-base-train /r closed r.yaml --label retrained --all-data" in j
    assert cmd.drop_pythonpath                                 # segmentation needs clean env
    with pytest.raises(AssertionError, match="retrained"):
        arm_commands("/r", "retrained", "r.yaml", "i.yaml",
                     label=None, all_data=True, allow_dirty=False)


def test_score_commands():
    (cmd,) = score_commands("/r", "gligen", "solo", steps=None)
    assert " ".join(cmd.argv).endswith("ingest30-score /r gligen --protocol solo")
    assert cmd.drop_pythonpath                                 # segmentation needs clean env


def test_curves_commands():
    (cmd,) = curves_commands("/r")
    assert " ".join(cmd.argv).endswith("ingest30-curves /r")
    assert cmd.drop_pythonpath                                 # segmentation needs clean env


def test_curves_cli_parsing():
    a = parse_args(["curves", "/r"])
    assert a.verb == "curves" and a.root == "/r"


def test_flatten_commands(tmp_path):
    (cmd,) = stage_flatten(sa_(tmp_path))
    assert " ".join(cmd.argv).endswith(f"ingest30-flatten {tmp_path}")
    assert cmd.drop_pythonpath                                 # segmentation needs clean env


def test_verb_registry_complete():
    assert list(VERBS) == ["init", "arm", "score", "curves"]


def test_arm_cli_parsing():
    # (a) contiguous positionals, flags trailing
    a = parse_args(["arm", "gligen", "b.yaml", "i.yaml", "/r",
                    "--label", "x", "--allow-dirty"])
    assert _unpack_arm_positionals(a.args_) == ("b.yaml", "i.yaml", "/r")
    assert a.label == "x" and a.allow_dirty and not a.all_data

    # (b) flags interspersed between the last two positionals
    a = parse_args(["arm", "gligen", "b.yaml", "i.yaml", "--label", "x", "/r"])
    assert _unpack_arm_positionals(a.args_) == ("b.yaml", "i.yaml", "/r")
    assert a.label == "x" and not a.allow_dirty and not a.all_data

    # (c) the spec transcript line: flag between the 2-item shape's two positionals
    a = parse_args(["arm", "retrained", "r.yaml", "--all-data", "/r"])
    assert _unpack_arm_positionals(a.args_) == ("r.yaml", None, "/r")
    assert a.all_data and a.label is None

    # (d) 2-item shape, flag trailing
    a = parse_args(["arm", "retrained", "r.yaml", "/r", "--all-data"])
    assert _unpack_arm_positionals(a.args_) == ("r.yaml", None, "/r")
    assert a.all_data and a.label is None

    # bad shape: neither 2 nor 3 positionals
    a = parse_args(["arm", "gligen", "only.yaml"])
    with pytest.raises(AssertionError, match="expects 'base_config root'"):
        _unpack_arm_positionals(a.args_)


def test_init_parser_accepts_store_site_flags():
    a = parse_args(["init", "b.txt", "i.txt", "/tmp/root",
                    "--descriptor", "CleanDiftFpn", "--descriptor-config", "d.yaml",
                    "--store-site-catalog", "/data/keep",
                    "--test-store-replicas", "7", "--test-store-num-targets", "15"])
    assert a.store_site_catalog == "/data/keep"
    assert a.test_store_replicas == 7
    assert a.test_store_num_targets == 15


def test_no_smoke_verb():
    with pytest.raises(SystemExit):                 # 'smoke' subparser is gone
        parse_args(["smoke", "/tmp/root"])


def test_init_refuses_over_cap_before_writing_manifest(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac
    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: {"base": ["a"], "ingest": ["b", "c"]}[Path(dest).name])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 2)   # only 2 curated sites

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=5, test_store_num_targets=20,
    )
    with pytest.raises(AssertionError, match="exceeds the 2 curated store sites"):
        _init_manifest(a)                              # 3 classes x 5 = 15 > 2 sites
    assert not (tmp_path / "manifest.yaml").exists()   # over-cap must write no manifest


def test_init_force_semantics(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac
    assembled = []

    def fake_assemble(paths, dest):
        dest = Path(dest)
        assert not dest.exists()                       # force must have cleared it
        dest.mkdir(parents=True)
        assembled.append(dest.name)
        return {"base": ["zebra"], "ingest": ["apple", "kiwi"]}[dest.name]

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "assemble_catalog", fake_assemble)
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)

    bake_cfg = tmp_path / "fpn.yaml"                    # absolute -> WS join keeps it as-is
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))

    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None, test_store_num_targets=20,
    )

    (tmp_path / "datasets").mkdir()                    # (a) renders present -> refuse
    with pytest.raises(AssertionError, match="datasets"):
        _init_manifest(a)
    (tmp_path / "datasets").rmdir()

    stale_dirs = [tmp_path / "catalogs" / "base", tmp_path / "catalogs" / "ingest",
                  tmp_path / "flat_test", tmp_path / "configs" / "datagen"]
    for d in stale_dirs:                               # (b) tool-owned regenerables -> cleared
        d.mkdir(parents=True)
        (d / "stale.txt").write_text("old")
    sa = _init_manifest(a)
    assert assembled == ["base", "ingest"]
    for d in stale_dirs:
        assert not (d / "stale.txt").exists()
    assert not (tmp_path / "flat_test").exists()       # flatten rebuilds it from new renders
    assert not (tmp_path / "configs" / "datagen").exists()  # write_all regenerates (in run_init)
    assert (tmp_path / "manifest.yaml").exists()
    assert sa.base_classes == ["zebra"] and sa.ingest_classes == ["apple", "kiwi"]


def test_init_descriptor_bake_config_mismatch(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    def fake_assemble(paths, dest):
        dest = Path(dest)
        dest.mkdir(parents=True)
        return {"base": ["zebra"], "ingest": ["apple"]}[dest.name]  # non-overlapping, m=2

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "assemble_catalog", fake_assemble)
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))

    def a_(descriptor):
        return argparse.Namespace(
            root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
            descriptor=descriptor, descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
            pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
            seed_base=1, seed_pools=2, seed_test=3,
            base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
            pool_frames=3, test_store_num_frames=1,
            test_composed_num_dirs=1, test_composed_num_targets=1,
            test_composed_num_frames=1, test_composed_replicas=1,
            store_site_catalog=str(tmp_path), test_store_replicas=None,
            test_store_num_targets=20,
        )

    with pytest.raises(AssertionError, match="descriptor 'E' != bake config .* name 'D'"):
        _init_manifest(a_("E"))                        # mismatch -> fail loud before any I/O

    sa = _init_manifest(a_("D"))                        # matching name -> proceeds
    assert sa.descriptor == "D"


def test_init_manifest_default_poser_skips_radius(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac
    import isaac_datagen.pool_object_radii as por

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    def fail_if_called(*a, **kw):
        raise AssertionError("compute_pool_object_radii must not run for LookAtPoser")
    monkeypatch.setattr(por, "compute_pool_object_radii", fail_if_called)

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
    )
    sa = _init_manifest(a)
    assert sa.pool_poser == "LookAtPoser"
    assert sa.pool_object_radius == {}


def test_init_manifest_decentered_poser_computes_radii(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac
    import isaac_datagen.pool_object_radii as por

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    seen = {}
    def fake_compute(ingest_catalog, nproc=None):
        seen["ingest_catalog"] = ingest_catalog
        return {"apple": 0.1, "kiwi": 0.2}
    monkeypatch.setattr(por, "compute_pool_object_radii", fake_compute)

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="DecenteredLookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
    )
    sa = _init_manifest(a)
    assert sa.pool_poser == "DecenteredLookAtPoser"
    assert sa.pool_object_radius == {"apple": 0.1, "kiwi": 0.2}
    assert seen["ingest_catalog"] == Path(tmp_path) / "catalogs" / "ingest"


def test_init_resume_pool_poser_mismatch_fails_loud(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))

    def a_(pool_poser, force):
        return argparse.Namespace(
            root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=force,
            descriptor="D", descriptor_config=str(bake_cfg), pool_poser=pool_poser,
            pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
            seed_base=1, seed_pools=2, seed_test=3,
            base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
            pool_frames=3, test_store_num_frames=1,
            test_composed_num_dirs=1, test_composed_num_targets=1,
            test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
        )

    _init_manifest(a_("LookAtPoser", force=True))          # first init: fresh root
    with pytest.raises(AssertionError, match="pool_poser"):
        _init_manifest(a_("DecenteredLookAtPoser", force=False))   # resume, different poser


def test_pool_poser_cli_flag_default_and_choices():
    a = parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml"])
    assert a.pool_poser == "LookAtPoser"

    a = parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml",
                    "--pool-poser", "DecenteredLookAtPoser"])
    assert a.pool_poser == "DecenteredLookAtPoser"

    with pytest.raises(SystemExit):
        parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml",
                    "--pool-poser", "NotAThing"])


def test_init_manifest_default_offset_sampler_is_empty(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
    )
    sa = _init_manifest(a)
    assert sa.pool_offset_sampler == {}


def test_init_manifest_log_offset_sampler_stored(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="log_uniform_offsets", pool_offset_sampler_floor=0.02,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
    )
    sa = _init_manifest(a)
    assert sa.pool_offset_sampler == {"name": "log_uniform_offsets", "args": {"floor": 0.02}}


def test_init_manifest_log_offset_sampler_needs_floor(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or [])

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="log_uniform_offsets", pool_offset_sampler_floor=None,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
    )
    with pytest.raises(AssertionError, match="pool-offset-sampler-floor"):
        _init_manifest(a)


def test_init_manifest_uniform_offset_sampler_rejects_floor(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or [])

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))
    a = argparse.Namespace(
        root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=True,
        descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
        pool_offset_sampler="uniform_offsets", pool_offset_sampler_floor=0.02,
        seed_base=1, seed_pools=2, seed_test=3,
        base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
        pool_frames=3, test_store_num_frames=1,
        test_composed_num_dirs=1, test_composed_num_targets=1,
        test_composed_num_frames=1, test_composed_replicas=1,
    )
    with pytest.raises(AssertionError, match="pool-offset-sampler-floor"):
        _init_manifest(a)


def test_init_resume_pool_offset_sampler_mismatch_fails_loud(tmp_path, monkeypatch):
    import isaac_datagen.asset_catalogs as ac

    monkeypatch.setattr(ac, "read_asset_list", lambda p: [f"{p}:asset"])
    monkeypatch.setattr(ac, "catalog_meta", lambda p: [{}] * 42)
    monkeypatch.setattr(ac, "assemble_catalog",
                        lambda paths, dest: Path(dest).mkdir(parents=True) or
                        (["zebra"] if Path(dest).name == "base" else ["apple", "kiwi"]))

    bake_cfg = tmp_path / "fpn.yaml"
    bake_cfg.write_text(yaml.safe_dump({"name": "D"}))

    def a_(sampler, floor, force):
        return argparse.Namespace(
            root=str(tmp_path), base_assets="b.txt", ingest_assets="i.txt", force=force,
            descriptor="D", descriptor_config=str(bake_cfg), pool_poser="LookAtPoser",
            pool_offset_sampler=sampler, pool_offset_sampler_floor=floor,
            seed_base=1, seed_pools=2, seed_test=3,
            base_num_dirs=2, base_num_targets=1, base_num_frames=1, base_replicas=1,
            pool_frames=3, test_store_num_frames=1,
            test_composed_num_dirs=1, test_composed_num_targets=1,
            test_composed_num_frames=1, test_composed_replicas=1,
        store_site_catalog=str(tmp_path), test_store_replicas=None,
        test_store_num_targets=1,
        )

    _init_manifest(a_("uniform_offsets", None, force=True))            # first init: fresh root
    with pytest.raises(AssertionError, match="pool_offset_sampler"):
        _init_manifest(a_("log_uniform_offsets", 0.02, force=False))   # resume, different sampler


def test_pool_offset_sampler_cli_flag_default_and_choices():
    a = parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml"])
    assert a.pool_offset_sampler == "uniform_offsets"
    assert a.pool_offset_sampler_floor is None

    a = parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml",
                    "--pool-offset-sampler", "log_uniform_offsets",
                    "--pool-offset-sampler-floor", "0.02"])
    assert a.pool_offset_sampler == "log_uniform_offsets"
    assert a.pool_offset_sampler_floor == 0.02

    with pytest.raises(SystemExit):
        parse_args(["init", "b.txt", "i.txt", "/r",
                    "--descriptor", "D", "--descriptor-config", "d.yaml",
                    "--pool-offset-sampler", "NotAThing"])

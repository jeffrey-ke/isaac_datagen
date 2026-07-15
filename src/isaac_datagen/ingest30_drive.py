import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from vision_core.script_args import ScriptArgs, SeedSeries

WS = Path(__file__).resolve().parents[3]          # workspace root


@dataclass
class Cmd:
    argv: list[str]
    cwd: Path
    drop_pythonpath: bool = False


def _datagen(cfg: Path, idx: int) -> Cmd:
    return Cmd(["uv", "run", "isaac-datagen", str(cfg), f"idx={idx}"],
               WS / "isaac_datagen" / "src" / "isaac_datagen", drop_pythonpath=True)


def _dgen(*argv: str) -> Cmd:
    return Cmd(["uv", "run", *argv], WS / "isaac_datagen")


def _seg(*argv: str) -> Cmd:
    return Cmd(["uv", "run", *argv], WS / "segmentation")


def _rendered(cfg: Path, idx: int) -> bool:
    dataset_dir = yaml.safe_load(cfg.read_text())["dataset_dir"]
    return (Path(dataset_dir) / f"render{idx:03d}" / "obs").is_dir()


def _dataset_roots(sa: ScriptArgs) -> list[Path]:
    root = Path(sa.root) / "datasets"
    return [root / "base", *sorted((root / "pools").glob("*")),
            *sorted((root / "test").glob("*"))]


def stage_render(sa: ScriptArgs) -> list[Cmd]:
    dg = Path(sa.root) / "configs" / "datagen"
    jobs = [(dg / "base.yaml", i) for i in range(sa.base_num_dirs)]
    jobs += [(p, 0) for p in sorted(dg.glob("pool-*.yaml"))]
    jobs += [(dg / "test-store.yaml", 0)]
    jobs += [(dg / "test-composed.yaml", i) for i in range(sa.test_composed_num_dirs)]
    return [_datagen(cfg, i) for cfg, i in jobs if not _rendered(cfg, i)]


def stage_bake(sa: ScriptArgs) -> list[Cmd]:
    return [_dgen("python", "-m", "isaac_datagen.migrate_descriptors_backbone",
                  "add-backbone", str(r), sa.descriptor_bake_config)
            for r in _dataset_roots(sa)]


def stage_squash(sa: ScriptArgs) -> list[Cmd]:
    return [_seg("m2f-squash-vis", str(r), "--in-place", "--min-visibility", "0.30")
            for r in _dataset_roots(sa)
            if not (r / "render000" / "squash_meta.yaml").exists()]


def stage_flatten(sa: ScriptArgs) -> list[Cmd]:
    return [_seg("ingest30-flatten", sa.root)]


INIT_STAGES = {"render": stage_render, "bake": stage_bake,
               "squash": stage_squash, "flatten": stage_flatten}
GPU_STAGES = {"render"}   # bake/squash/flatten read baked products; arm/score/smoke check inline


def arm_commands(root, ops, base_cfg, ingest_cfg, label, all_data, allow_dirty) -> list[Cmd]:
    dirty = ["--allow-dirty"] if allow_dirty else []
    if ops == "retrained":                       # spec §10: closed ops, base+pools, no loop
        assert ingest_cfg is None and all_data, \
            "retrained takes ONE config and implies --all-data (no ingest loop)"
        return [_seg("ingest30-base-train", root, "closed", base_cfg,
                     "--label", label or "retrained", "--all-data", *dirty)]
    assert ingest_cfg is not None, f"arm {ops!r} needs <base_config> <ingest_config>"
    assert not all_data, "--all-data belongs to the retrained arm"
    label = label or ops
    return [_seg("ingest30-base-train", root, ops, base_cfg, "--label", label, *dirty),
            _seg("ingest30-loop", root, ops, ingest_cfg, "--label", label, *dirty)]


def score_commands(root, label, protocol, steps) -> list[Cmd]:
    extra = ["--steps", steps] if steps else []
    return [_seg("ingest30-score", root, label, "--protocol", protocol, *extra)]


def run_cmd(cmd: Cmd, log) -> None:
    env = {k: v for k, v in os.environ.items()
           if not (cmd.drop_pythonpath and k == "PYTHONPATH")}
    print(f"[meta] {' '.join(cmd.argv)}  (cwd={cmd.cwd})", flush=True)
    subprocess.run(cmd.argv, cwd=cmd.cwd, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)


def run_stage(verb: str, stage: str, cmds: list[Cmd], root: str) -> None:
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if stage in GPU_STAGES:
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader"])
    with open(logs / f"{verb}-{stage}.log", "a") as log:
        for cmd in cmds:
            run_cmd(cmd, log)
    print(f"[meta] {verb}:{stage} done")


def _init_manifest(a) -> ScriptArgs:
    from isaac_datagen.asset_catalogs import assemble_catalog, read_asset_list

    root = Path(a.root)
    manifest = root / "manifest.yaml"
    base_assets = read_asset_list(a.base_assets)
    ingest_assets = read_asset_list(a.ingest_assets)
    if manifest.exists() and not a.force:
        sa = ScriptArgs.load(manifest)           # resume: root state is frozen
        assert sa.base_assets == base_assets and sa.ingest_assets == ingest_assets, \
            f"{manifest} exists with DIFFERENT asset lists — new lists need a new root (or --force)"
        print("[meta] init resuming from existing manifest")
        return sa
    base_classes = assemble_catalog(base_assets, root / "catalogs" / "base")
    ingest_classes = assemble_catalog(ingest_assets, root / "catalogs" / "ingest")
    overlap = set(base_classes) & set(ingest_classes)
    assert not overlap, f"base/ingest share classes: {sorted(overlap)}"
    sa = ScriptArgs(
        root=str(root), base_assets=base_assets, ingest_assets=ingest_assets,
        base_classes=base_classes, ingest_classes=ingest_classes,
        descriptor=a.descriptor, descriptor_bake_config=a.descriptor_config,
        seeds=SeedSeries(base=a.seed_base, pools=a.seed_pools, test=a.seed_test),
        base_num_dirs=a.base_num_dirs, base_num_targets=a.base_num_targets,
        base_num_frames=a.base_num_frames, base_replicas=a.base_replicas,
        pool_frames=a.pool_frames, test_store_num_frames=a.test_store_num_frames,
        test_composed_num_dirs=a.test_composed_num_dirs,
        test_composed_num_targets=a.test_composed_num_targets,
        test_composed_num_frames=a.test_composed_num_frames,
        test_composed_replicas=a.test_composed_replicas,
    )
    sa.save(manifest)
    return sa


def run_init(a) -> None:
    from isaac_datagen.ingest30_configs import write_all

    sa = _init_manifest(a)
    write_all(sa)                                # idempotent: same manifest -> same files
    for stage, fn in INIT_STAGES.items():
        run_stage("init", stage, fn(sa), sa.root)


def run_arm(a) -> None:
    assert (Path(a.root) / "manifest.yaml").exists(), f"{a.root}: not an inited root"
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    for i, cmd in enumerate(arm_commands(a.root, a.ops, a.base_config, a.ingest_config,
                                         a.label, a.all_data, a.allow_dirty)):
        run_stage("arm", f"{a.label or a.ops}-{i}", [cmd], a.root)


def run_score(a) -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    run_stage("score", a.label, score_commands(a.root, a.label, a.protocol, a.steps), a.root)


def run_smoke(a) -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    run_stage("smoke", "store", [_dgen("isaac-datagen-ingest30-smoke",
                                       str(Path(a.root) / "manifest.yaml"))], a.root)


VERBS = {"init": run_init, "arm": run_arm, "score": run_score, "smoke": run_smoke}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meta")
    sub = p.add_subparsers(dest="verb", required=True)

    ini = sub.add_parser("init")
    ini.add_argument("base_assets")
    ini.add_argument("ingest_assets")
    ini.add_argument("root")
    ini.add_argument("--descriptor", required=True)
    ini.add_argument("--descriptor-config", required=True)
    ini.add_argument("--force", action="store_true")
    ini.add_argument("--seed-base", type=int, default=3001)
    ini.add_argument("--seed-pools", type=int, default=3101)
    ini.add_argument("--seed-test", type=int, default=3201)
    # size defaults = the spec §4 example values; verify against
    # expanded-refseg-v2 at execution and adjust HERE only
    ini.add_argument("--base-num-dirs", type=int, default=3)
    ini.add_argument("--base-num-targets", type=int, default=10)
    ini.add_argument("--base-num-frames", type=int, default=100)
    ini.add_argument("--base-replicas", type=int, default=5)
    ini.add_argument("--pool-frames", type=int, default=100)
    ini.add_argument("--test-store-num-frames", type=int, default=5)
    ini.add_argument("--test-composed-num-dirs", type=int, default=2)
    ini.add_argument("--test-composed-num-targets", type=int, default=10)
    ini.add_argument("--test-composed-num-frames", type=int, default=50)
    ini.add_argument("--test-composed-replicas", type=int, default=3)

    arm = sub.add_parser("arm")
    arm.add_argument("ops")
    arm.add_argument("base_config")
    arm.add_argument("ingest_config", nargs="?", default=None)
    arm.add_argument("root")
    arm.add_argument("--label", default=None)
    arm.add_argument("--all-data", action="store_true")
    arm.add_argument("--allow-dirty", action="store_true")

    sco = sub.add_parser("score")
    sco.add_argument("label")
    sco.add_argument("root")
    sco.add_argument("--protocol", default="solo")
    sco.add_argument("--steps", default=None)

    smo = sub.add_parser("smoke")
    smo.add_argument("root")
    return p


def main():
    a = _parser().parse_args()
    VERBS[a.verb](a)

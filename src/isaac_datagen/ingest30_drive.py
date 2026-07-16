import argparse
import os
import shutil
import subprocess
import sys
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
    # segmentation never wants Isaac's env: if meta is launched from an
    # Isaac-sourced shell, an inherited PYTHONPATH could pull in Isaac's
    # torch 2.7 instead of segmentation's own venv's torch 2.11
    return Cmd(["uv", "run", *argv], WS / "segmentation", drop_pythonpath=True)


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
GPU_STAGES = {"render"}   # bake/squash/flatten read baked products; arm/score check inline


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
    from isaac_datagen.pool_object_radii import compute_pool_object_radii

    root = Path(a.root)
    manifest = root / "manifest.yaml"
    # descriptor_config resolves like stage_bake's add-backbone consumes it: relative
    # to WS/isaac_datagen. Mismatch here would otherwise surface as a KeyError at
    # flatten/train — checked on both fresh init and resume.
    bake_cfg_name = yaml.safe_load(
        (WS / "isaac_datagen" / a.descriptor_config).read_text())["name"]
    assert bake_cfg_name == a.descriptor, (
        f"--descriptor {a.descriptor!r} != bake config {a.descriptor_config!r} "
        f"name {bake_cfg_name!r}"
    )
    if a.pool_offset_sampler == "uniform_offsets":
        assert a.pool_offset_sampler_floor is None, (
            "--pool-offset-sampler-floor only applies with a non-default --pool-offset-sampler "
            f"(got --pool-offset-sampler-floor {a.pool_offset_sampler_floor})"
        )
        pool_offset_sampler = {}
    else:
        assert a.pool_offset_sampler_floor is not None, (
            f"--pool-offset-sampler {a.pool_offset_sampler!r} needs --pool-offset-sampler-floor"
        )
        pool_offset_sampler = {"name": a.pool_offset_sampler,
                               "args": {"floor": a.pool_offset_sampler_floor}}
    base_assets = read_asset_list(a.base_assets)
    ingest_assets = read_asset_list(a.ingest_assets)
    if manifest.exists() and not a.force:
        sa = ScriptArgs.load(manifest)           # resume: root state is frozen
        assert sa.base_assets == base_assets and sa.ingest_assets == ingest_assets, \
            f"{manifest} exists with DIFFERENT asset lists — new lists need a new root (or --force)"
        assert sa.pool_poser == a.pool_poser, (
            f"{manifest} exists with pool_poser={sa.pool_poser!r} — different --pool-poser "
            f"{a.pool_poser!r} needs a new root (or --force)"
        )
        assert sa.pool_offset_sampler == pool_offset_sampler, (
            f"{manifest} exists with pool_offset_sampler={sa.pool_offset_sampler!r} — different "
            f"--pool-offset-sampler needs a new root (or --force)"
        )
        print("[meta] init resuming from existing manifest")
        return sa
    assert not (root / "datasets").exists(), \
        f"{root}/datasets exists — --force rebuilds catalogs+manifest only; stale renders " \
        "must be deleted by hand (exact paths) if the asset lists changed"
    for stale in (root / "catalogs" / "base", root / "catalogs" / "ingest",
                  root / "flat_test", root / "configs" / "datagen"):
        if stale.exists():
            shutil.rmtree(stale)   # tool-owned regenerables; exact names, no globs
    base_classes = assemble_catalog(base_assets, root / "catalogs" / "base")
    ingest_classes = assemble_catalog(ingest_assets, root / "catalogs" / "ingest")
    overlap = set(base_classes) & set(ingest_classes)
    assert not overlap, f"base/ingest share classes: {sorted(overlap)}"
    pool_object_radius = {}
    if a.pool_poser == "DecenteredLookAtPoser":
        pool_object_radius = compute_pool_object_radii(root / "catalogs" / "ingest")
    from isaac_datagen.asset_catalogs import catalog_meta
    site_catalog = str(Path(a.store_site_catalog).resolve())
    n_sites = len(catalog_meta(Path(site_catalog)))
    m = len(base_classes) + len(ingest_classes)
    replicas = a.test_store_replicas if a.test_store_replicas is not None else n_sites // m
    assert replicas >= 1, (
        f"{m} classes vs {n_sites} store sites: no room to fill even one per class — "
        f"use fewer classes or a larger site catalog")
    assert m * replicas <= n_sites, (          # before sa.save: over-cap writes no manifest
        f"store repopulation: {m} classes x replicate {replicas} = {m * replicas} objects "
        f"exceeds the {n_sites} curated store sites. Reduce --test-store-replicas to "
        f"<= {n_sites // m}, or place overflow via the composed scene (no fixed-site limit).")
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
        store_site_catalog=site_catalog,
        test_store_replicas=replicas,
        test_store_num_targets=a.test_store_num_targets,
        pool_poser=a.pool_poser, pool_object_radius=pool_object_radius,
        pool_offset_sampler=pool_offset_sampler,
    )
    sa.save(manifest)
    return sa


def run_init(a) -> None:
    from isaac_datagen.ingest30_configs import write_all

    sa = _init_manifest(a)
    write_all(sa)                                # idempotent: same manifest -> same files
    for stage, fn in INIT_STAGES.items():
        run_stage("init", stage, fn(sa), sa.root)


def _unpack_arm_positionals(args_: list[str]) -> tuple[str, str | None, str]:
    if len(args_) == 2:
        base_config, root = args_
        return base_config, None, root
    if len(args_) == 3:
        base_config, ingest_config, root = args_
        return base_config, ingest_config, root
    assert False, (
        "arm expects 'base_config root' (2 positionals, e.g. retrained) or "
        f"'base_config ingest_config root' (3 positionals); got {len(args_)}: {args_!r}"
    )


def run_arm(a) -> None:
    base_config, ingest_config, root = _unpack_arm_positionals(a.args_)
    assert (Path(root) / "manifest.yaml").exists(), f"{root}: not an inited root"
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    for i, cmd in enumerate(arm_commands(root, a.ops, base_config, ingest_config,
                                         a.label, a.all_data, a.allow_dirty)):
        run_stage("arm", f"{a.label or a.ops}-{i}", [cmd], root)


def run_score(a) -> None:
    assert (Path(a.root) / "manifest.yaml").exists(), f"{a.root}: not an inited root"
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    run_stage("score", a.label, score_commands(a.root, a.label, a.protocol, a.steps), a.root)


VERBS = {"init": run_init, "arm": run_arm, "score": run_score}


def _parser() -> argparse.ArgumentParser:
    raw = dict(formatter_class=argparse.RawDescriptionHelpFormatter)
    p = argparse.ArgumentParser(
        prog="meta", **raw,
        description=(
            "ingest30 experiment driver: init -> arm -> score.\n\n"
            "Run from isaac_datagen/ as `uv run meta ...`. Output appends to\n"
            "<root>/logs/<verb>-<stage>.log. Every verb except init needs an inited\n"
            "root (<root>/manifest.yaml). GPU verbs print nvidia-smi memory used\n"
            "first -- check the GPU is free before launching."
        ),
        epilog=(
            "example transcript (root on /data):\n"
            "  meta init base.txt ingest.txt <root> --descriptor CleanDiftFpn \\\n"
            "      --descriptor-config ../reference_matching/src/reference_matching/"
            "configs/fpn_cleandift_123.yaml\n"
            "  meta arm gligen src/segmentation/configs/ingest30/base-train-gligen.yaml \\\n"
            "      src/segmentation/configs/ingest30/ingest-gligen.yaml <root> --allow-dirty\n"
            "  meta arm retrained src/segmentation/configs/ingest30/base-train-closed.yaml"
            " --all-data <root>\n"
            "  meta score gligen <root>"
        ),
    )
    sub = p.add_subparsers(dest="verb", required=True)

    ini = sub.add_parser(
        "init", **raw,
        help="assemble catalogs + manifest, write datagen configs, "
             "render -> bake -> squash -> flatten",
        description=(
            "Assemble asset catalogs, freeze <root>/manifest.yaml, write the datagen\n"
            "configs, then run render -> bake -> squash -> flatten.\n\n"
            "Resumable: re-running with the SAME asset lists resumes, skipping dirs\n"
            "already rendered/baked/squashed. Different lists against an existing\n"
            "manifest is a hard error -- use a new root, or --force."
        ),
        epilog=(
            "flags that change HOW MUCH / WHAT KIND of data gets generated\n"
            "(defaults + types are on each flag's own entry above):\n"
            "\n"
            "  poser / camera-offset distribution (WHAT KIND)\n"
            "    --pool-poser                  LookAtPoser (centered) | DecenteredLookAtPoser\n"
            "    --pool-offset-sampler         uniform_offsets | log_uniform_offsets (near-biased)\n"
            "    --pool-offset-sampler-floor   required with log_uniform_offsets only\n"
            "\n"
            "  base render volume (HOW MUCH)\n"
            "    --base-num-dirs               render dirs\n"
            "    --base-num-targets            capture targets per scene\n"
            "    --base-num-frames             frames per dir\n"
            "    --base-replicas               object copies placed per scene\n"
            "\n"
            "  pool render volume (HOW MUCH -- one dir per ingest class)\n"
            "    --pool-frames                 frames per pool dir\n"
            "\n"
            "  test render volume (HOW MUCH)\n"
            "    --test-store-num-frames       frames, store scene\n"
            "    --test-composed-num-dirs      composed render dirs\n"
            "    --test-composed-num-targets   capture targets per composed scene\n"
            "    --test-composed-num-frames    frames per composed dir\n"
            "    --test-composed-replicas      object copies placed per composed scene"
        ),
    )
    ini.add_argument("base_assets",
                     help="newline-separated reference-image paths; line count = N (base classes)")
    ini.add_argument("ingest_assets",
                     help="newline-separated reference-image paths; line count = M-N; "
                          "no class overlap with base (asserted)")
    ini.add_argument("root", help="experiment root; put it on /data")
    ini.add_argument("--descriptor", required=True,
                     help="must equal the `name:` inside --descriptor-config (asserted)")
    ini.add_argument("--descriptor-config", required=True,
                     help="descriptor bake config, path relative to isaac_datagen/")
    ini.add_argument("--store-site-catalog",
                     default=str(WS / "isaac_datagen" / "assets" / "optflow_objects"
                                 / "store001-optflow-objects-keep"),
                     help="curated store site catalog (default: store001-optflow-objects-keep)")
    ini.add_argument("--test-store-replicas", type=int, default=None,
                     help="ReplicateFilter count for the store leg; "
                          "default fills as many of the S sites as fit (S // M)")
    ini.add_argument("--test-store-num-targets", type=int, default=20,
                     help="camera vantage points per store dir; <0 -> null = every placed "
                          "object, one frame each (default: %(default)s)")
    ini.add_argument("--force", action="store_true",
                     help="rebuild tool-owned regenerables (catalogs/{base,ingest}, flat_test, "
                          "configs/datagen); refuses if datasets/ exists -- delete stale "
                          "renders by hand, exact names only")
    ini.add_argument("--pool-poser", default="LookAtPoser",
                     choices=["LookAtPoser", "DecenteredLookAtPoser"],
                     help="-1inst pool poser (default: %(default)s); DecenteredLookAtPoser "
                          "computes a per-class object_radius from each class's mesh bbox")
    ini.add_argument("--pool-offset-sampler", default="uniform_offsets",
                     choices=["uniform_offsets", "log_uniform_offsets"],
                     help="-1inst pool camera-offset distribution (default: %(default)s); "
                          "log_uniform_offsets biases toward the near/center bound, needs "
                          "--pool-offset-sampler-floor")
    ini.add_argument("--pool-offset-sampler-floor", type=float, default=None,
                     help="floor for --pool-offset-sampler log_uniform_offsets (required with "
                          "it, forbidden with the default uniform_offsets)")
    ini.add_argument("--seed-base", type=int, default=3001,
                     help="base-dataset seed (default: %(default)s)")
    ini.add_argument("--seed-pools", type=int, default=3101,
                     help="-1inst pool seed (default: %(default)s)")
    ini.add_argument("--seed-test", type=int, default=3201,
                     help="held-out test seed (default: %(default)s)")
    # size defaults = the spec §4 example values; verify against
    # expanded-refseg-v2 at execution and adjust HERE only
    ini.add_argument("--base-num-dirs", type=int, default=3,
                     help="base render dirs (default: %(default)s)")
    ini.add_argument("--base-num-targets", type=int, default=10,
                     help="capture targets per base scene (default: %(default)s)")
    ini.add_argument("--base-num-frames", type=int, default=100,
                     help="frames per base dir (default: %(default)s)")
    ini.add_argument("--base-replicas", type=int, default=5,
                     help="copies of each object placed in a base scene (default: %(default)s)")
    ini.add_argument("--pool-frames", type=int, default=100,
                     help="frames per -1inst pool dir, one dir per ingest class "
                          "(default: %(default)s)")
    ini.add_argument("--test-store-num-frames", type=int, default=5,
                     help="frames for the store test render (default: %(default)s)")
    ini.add_argument("--test-composed-num-dirs", type=int, default=2,
                     help="composed test render dirs (default: %(default)s)")
    ini.add_argument("--test-composed-num-targets", type=int, default=10,
                     help="capture targets per composed test scene (default: %(default)s)")
    ini.add_argument("--test-composed-num-frames", type=int, default=50,
                     help="frames per composed test dir (default: %(default)s)")
    ini.add_argument("--test-composed-replicas", type=int, default=3,
                     help="copies of each object placed in a composed test scene "
                          "(default: %(default)s)")

    arm = sub.add_parser(
        "arm", **raw,
        help="train one arm end to end (base train, then ingest loop)",
        description=(
            "Train one arm.\n\n"
            "  gligen|closed:  meta arm <ops> <base_config> <ingest_config> <root>\n"
            "                  runs ingest30-base-train, then ingest30-loop\n"
            "  retrained:      meta arm retrained <base_config> --all-data <root>\n"
            "                  ONE config, closed ops on base + all pools, no loop\n\n"
            "Config paths resolve from segmentation/ (the subprocess cwd), e.g.\n"
            "src/segmentation/configs/ingest30/base-train-gligen.yaml. Flags may be\n"
            "interspersed anywhere among the positionals."
        ),
    )
    arm.add_argument("ops", help="gligen | closed | retrained")
    arm.add_argument("args_", nargs="+", metavar="base_config [ingest_config] root",
                     help="3 positionals for gligen/closed, 2 for retrained")
    arm.add_argument("--label", default=None,
                     help="names runs/checkpacks/scores (default: the ops name)")
    arm.add_argument("--all-data", action="store_true",
                     help="retrained only: train on base + all pools")
    arm.add_argument("--allow-dirty", action="store_true",
                     help="launch despite uncommitted submodule changes")

    sco = sub.add_parser(
        "score", **raw,
        help="score a trained arm's checkpoint pack against flat_test",
        description=(
            "Score a trained arm's checkpoint pack against <root>/flat_test.\n\n"
            "A descriptor-stamp mismatch (pack vs flat_test) refuses to score by\n"
            "design -- re-flatten or re-bake, don't override the guard."
        ),
    )
    sco.add_argument("label", help="arm label used at train time")
    sco.add_argument("root")
    sco.add_argument("--protocol", default="solo",
                     help="scoring protocol registered in PROTOCOLS (default: %(default)s)")
    sco.add_argument("--steps", default=None,
                     help="comma-separated step tags to score, e.g. base,00,05 (default: all)")

    return p


ARM_VALUE_FLAGS = {"--label"}                       # arm flags that take one value
ARM_BOOL_FLAGS = {"--all-data", "--allow-dirty"}     # arm flags that take none


def _reorder_arm_argv(rest: list[str]) -> list[str]:
    # argparse can't split one nargs="+" positional run across a flag interspersed
    # between two of its values (stdlib limitation, cf. bpo-9338) — move arm's
    # flags after its positionals so parse_args always sees one contiguous run.
    positionals, flags = [], []
    it = iter(rest)
    for tok in it:
        name = tok.split("=", 1)[0]
        if name in ARM_BOOL_FLAGS or name in ARM_VALUE_FLAGS:
            flags.append(tok)
            if name in ARM_VALUE_FLAGS and "=" not in tok:
                flags.append(next(it))
        else:
            positionals.append(tok)
    return positionals + flags


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "arm":
        argv = argv[:1] + _reorder_arm_argv(argv[1:])
    return _parser().parse_args(argv)


def main():
    a = parse_args()
    VERBS[a.verb](a)

import csv
import subprocess
import sys
from pathlib import Path

from vision_core.script_args import ScriptArgs


def smoke_tally(smoke_root: Path) -> list[tuple[str, int]]:
    rows = []
    for d in sorted(p for p in Path(smoke_root).iterdir() if p.is_dir()):
        obs = d / "render000" / "obs"
        rows.append((d.name, len(list(obs.iterdir())) if obs.is_dir() else 0))
    return rows


def run_smokes(sa: ScriptArgs) -> None:
    cfgs = sorted((Path(sa.root) / "configs" / "datagen" / "smoke").glob("*.yaml"))
    assert cfgs, f"no smoke configs under {sa.root}/configs/datagen/smoke — run ingest30-configs first"
    for cfg in cfgs:
        print(f"[smoke] {cfg.stem}")
        subprocess.run(["uv", "run", "isaac-datagen", str(cfg.resolve()), "idx=0"],
                       check=True)


def write_report(sa: ScriptArgs) -> Path:
    smoke_root = Path(sa.root) / "smoke"
    rows = smoke_tally(smoke_root)
    report = smoke_root / "report.csv"
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "frames"])
        w.writerows(rows)
    zeros = [c for c, n in rows if n == 0]
    print(f"[smoke] report -> {report}; {len(zeros)} classes with ZERO frames: {zeros}")
    return report


def main():
    sa = ScriptArgs.load(sys.argv[1])
    if "--report-only" not in sys.argv:
        run_smokes(sa)
    write_report(sa)

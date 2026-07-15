# ingest30 Plan-A smoke: throwaway sizes, real mechanisms. NOT an experiment recipe.
# Manual walk of the `meta init` sequence (Plan C automates this).
from pathlib import Path

from vision_core.script_args import ScriptArgs, SeedSeries
from isaac_datagen.asset_catalogs import assemble_catalog, read_asset_list
from isaac_datagen.ingest30_configs import write_all

root = Path("/data/user/jeffk/ingest30/smokeA")
base_assets = read_asset_list(root / "base_assets.txt")
ingest_assets = read_asset_list(root / "ingest_assets.txt")
base_classes = assemble_catalog(base_assets, root / "catalogs" / "base")
ingest_classes = assemble_catalog(ingest_assets, root / "catalogs" / "ingest")
overlap = set(base_classes) & set(ingest_classes)
assert not overlap, f"base/ingest share classes: {sorted(overlap)}"

sa = ScriptArgs(
    root=str(root), base_assets=base_assets, ingest_assets=ingest_assets,
    base_classes=base_classes, ingest_classes=ingest_classes,
    descriptor="CleanDiftFpn",
    descriptor_bake_config="../reference_matching/src/reference_matching/configs/fpn_cleandift_123.yaml",
    seeds=SeedSeries(base=3001, pools=3101, test=3201),
    base_num_dirs=1, base_num_targets=2, base_num_frames=2, base_replicas=2,
    pool_frames=3, test_store_num_frames=2,
    test_composed_num_dirs=1, test_composed_num_targets=2,
    test_composed_num_frames=2, test_composed_replicas=2,
)
sa.save(root / "manifest.yaml")
write_all(sa)

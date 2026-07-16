import numpy as np
import pytest
import yaml

from isaac_datagen.store_mutations import Site, load_sites

KEEP = "assets/optflow_objects/store001-optflow-objects-keep"   # via the assets symlink; cwd = isaac_datagen


def test_load_sites_real_keep_catalog():
    sites = load_sites(KEEP)
    assert len(sites) == 283
    assert all(isinstance(s, Site) for s in sites)
    assert all(s.store_prim.endswith("/v_0") for s in sites)     # store_prim points at the v_0 geometry prim
    assert all(s.grasp.shape == (4, 4) for s in sites)
    assert len({s.cls for s in sites}) == 42


def test_load_sites_rejects_catalog_without_store_prim(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    (meta / "meta_0000.yaml").write_text(yaml.safe_dump({"name": "x001", "class": "x001"}))
    with pytest.raises(AssertionError, match="store_prim"):
        load_sites(str(tmp_path))

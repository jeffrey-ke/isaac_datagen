import pytest
import yaml

from isaac_datagen.asset_catalogs import (
    asset_classes, catalog_classes, catalog_meta, parse_asset, read_asset_list,
)


def fake_catalog(tmp_path, name, metas):
    cat = tmp_path / name
    (cat / "meta").mkdir(parents=True)
    (cat / "reference_image").mkdir()
    for i, m in enumerate(metas):
        (cat / "meta" / f"meta_{i:04d}.yaml").write_text(yaml.safe_dump(m))
        (cat / "reference_image" / f"reference_image_{i:04d}.png").write_bytes(b"x")
    return cat


METAS = [   # mirrors real store metas; duplicates share class
    {"name": "snack031", "class": "snack031", "store_prim": "model_snack031/v_0"},
    {"name": "snack031_1", "class": "snack031", "store_prim": "model_snack031_1/v_0"},
    {"name": "cereal001", "class": "cereal001", "store_prim": "model_cereal001/v_0"},
]


def ref(cat, i):
    return str(cat / "reference_image" / f"reference_image_{i:04d}.png")


def test_parse_asset(tmp_path):
    cat = fake_catalog(tmp_path, "keep", METAS)
    assert parse_asset(ref(cat, 2)) == (cat, 2)


def test_parse_asset_rejects_non_reference(tmp_path):
    cat = fake_catalog(tmp_path, "keep", METAS)
    with pytest.raises(AssertionError, match="reference_image"):
        parse_asset(str(cat / "meta" / "meta_0002.yaml"))
    with pytest.raises(AssertionError, match="exist"):
        parse_asset(ref(cat, 9))                       # index with no file


def test_asset_classes_order_and_duplicates(tmp_path):
    cat = fake_catalog(tmp_path, "keep", METAS)
    assert asset_classes([ref(cat, 2), ref(cat, 0)]) == ["cereal001", "snack031"]
    with pytest.raises(AssertionError, match="duplicate"):
        asset_classes([ref(cat, 0), ref(cat, 1)])      # snack031 twice (shelf dup)


def test_read_asset_list(tmp_path):
    cat = fake_catalog(tmp_path, "keep", METAS)
    lst = tmp_path / "assets.txt"
    lst.write_text(f"{ref(cat, 0)}\n\n{ref(cat, 2)}\n")
    assert read_asset_list(lst) == [ref(cat, 0), ref(cat, 2)]


def test_catalog_classes(tmp_path):
    cat = fake_catalog(tmp_path, "keep", METAS)
    assert catalog_classes(cat) == ["cereal001", "snack031"]


def test_not_a_catalog_fails(tmp_path):
    with pytest.raises(AssertionError, match="meta"):
        catalog_meta(tmp_path)

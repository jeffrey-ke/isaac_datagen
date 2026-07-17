import pytest


def _write_meta(catalog, classes):
    meta_dir = catalog / "meta"
    meta_dir.mkdir(parents=True)
    for i, cls in enumerate(classes):
        (meta_dir / f"meta_{i:04d}.yaml").write_text(
            f"class: {cls}\nname: {cls}\nstore_prim: model_{cls}/v_0\n")


def test_pool_usd_paths(tmp_path):
    from isaac_datagen.pool_object_radii import pool_usd_paths

    catalog = tmp_path / "ingest"
    _write_meta(catalog, ["snack031", "flour001"])

    paths = pool_usd_paths(catalog)
    assert paths == {
        "snack031": catalog / "usd_path" / "usd_path_0000.usdz",
        "flour001": catalog / "usd_path" / "usd_path_0001.usdz",
    }


def test_reassemble_happy_path():
    from isaac_datagen.pool_object_radii import _reassemble

    class_to_path = {"snack031": "path/a.usdz", "flour001": "path/b.usdz"}
    stdout = "path/a.usdz\t0.22\npath/b.usdz\t0.31\n"
    assert _reassemble(stdout, class_to_path) == {"snack031": 0.22, "flour001": 0.31}


def test_reassemble_out_of_order_is_fine():
    from isaac_datagen.pool_object_radii import _reassemble

    class_to_path = {"snack031": "path/a.usdz", "flour001": "path/b.usdz"}
    stdout = "path/b.usdz\t0.31\npath/a.usdz\t0.22\n"   # parallel workers finish out of order
    assert _reassemble(stdout, class_to_path) == {"snack031": 0.22, "flour001": 0.31}


def test_reassemble_missing_class_fails_loud():
    from isaac_datagen.pool_object_radii import _reassemble

    class_to_path = {"snack031": "path/a.usdz", "flour001": "path/b.usdz"}
    stdout = "path/a.usdz\t0.22\n"                      # flour001's worker never reported
    with pytest.raises(AssertionError, match="flour001"):
        _reassemble(stdout, class_to_path)


pytest.importorskip("pxr", reason="needs usd-core: uv run --with pytest --with usd-core")


def _write_usdz(usda_path, usdz_path, scale):
    from pxr import Gf, Usd, UsdGeom, UsdUtils

    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.Xform.Define(stage, "/W")
    cube = UsdGeom.Cube.Define(stage, "/W/c")
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    stage.GetRootLayer().Save()
    ok = UsdUtils.CreateNewUsdzPackage(str(usda_path), str(usdz_path))
    assert ok, f"failed to package {usdz_path}"


def test_compute_pool_object_radii_end_to_end(tmp_path):
    import numpy as np

    from isaac_datagen.pool_object_radii import compute_pool_object_radii

    catalog = tmp_path / "ingest"
    _write_meta(catalog, ["snack031", "cereal001"])
    usd_dir = catalog / "usd_path"
    usd_dir.mkdir(parents=True)
    scales = {"snack031": (0.05, 0.05, 0.1), "cereal001": (0.1, 0.09, 0.15)}
    for i, cls in enumerate(["snack031", "cereal001"]):
        _write_usdz(tmp_path / f"{cls}.usda", usd_dir / f"usd_path_{i:04d}.usdz", scales[cls])

    radii = compute_pool_object_radii(catalog, nproc=2)

    for cls, scale in scales.items():
        expected = float(np.linalg.norm(scale)) * 2
        assert radii[cls] == pytest.approx(expected, rel=1e-4)

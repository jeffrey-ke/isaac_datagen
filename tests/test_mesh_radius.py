import pytest

pytest.importorskip("pxr", reason="needs usd-core: uv run --with pytest --with usd-core")


def _write_cube(path, scale):
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/W")
    cube = UsdGeom.Cube.Define(stage, "/W/c")
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    stage.GetRootLayer().Save()


def test_mesh_radius_is_bbox_full_diagonal(tmp_path):
    import numpy as np

    from isaac_datagen.mesh_radius import mesh_radius

    usd_path = tmp_path / "obj.usda"
    _write_cube(usd_path, (0.1, 0.05, 0.2))

    expected = float(np.linalg.norm((0.1, 0.05, 0.2))) * 2
    assert mesh_radius(str(usd_path)) == pytest.approx(expected, rel=1e-5)


def test_mesh_radius_ignores_translation(tmp_path):
    """The radius is a size, not a position -- moving the mesh far from the
    file's own origin must not change it (we don't know exactly where the
    render-time origin sits relative to the mesh, so this must hold)."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    from isaac_datagen.mesh_radius import mesh_radius

    usd_path = tmp_path / "obj.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/W")
    cube = UsdGeom.Cube.Define(stage, "/W/c")
    cube.AddScaleOp().Set(Gf.Vec3f(0.1, 0.05, 0.2))
    cube.AddTranslateOp().Set(Gf.Vec3f(5.0, 5.0, 5.0))
    stage.GetRootLayer().Save()

    expected = float(np.linalg.norm((0.1, 0.05, 0.2))) * 2
    assert mesh_radius(str(usd_path)) == pytest.approx(expected, rel=1e-5)


def test_main_prints_bare_radius(tmp_path, monkeypatch, capsys):
    from isaac_datagen import mesh_radius as mr

    usd_path = tmp_path / "obj.usda"
    _write_cube(usd_path, (0.1, 0.05, 0.2))

    monkeypatch.setattr("sys.argv", ["mesh_radius", str(usd_path)])
    mr.main()
    out = capsys.readouterr().out.strip()
    # bare float only -- no path echoed; associating it back with the input path is the
    # xargs/shell wrapper's job (Task 3), not this tool's
    assert float(out) == pytest.approx(0.4582575763241539, rel=1e-5)


def test_mesh_radius_fails_loud_on_geometry_less_stage(tmp_path):
    """pxr's BBoxCache returns the empty-range sentinel (-3.4e38 per axis) for a
    stage with no mesh geometry at all -- NOT (0, 0, 0). Silently taking its norm
    would produce an enormous, nonsensical radius instead of a loud failure."""
    from pxr import Usd, UsdGeom

    from isaac_datagen.mesh_radius import mesh_radius

    usd_path = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/W")          # no mesh geometry at all
    stage.GetRootLayer().Save()

    with pytest.raises(AssertionError, match="empty/invalid bounding box"):
        mesh_radius(str(usd_path))

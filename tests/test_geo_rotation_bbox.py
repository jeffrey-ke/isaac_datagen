import pytest

pytest.importorskip("pxr", reason="needs usd-core: uv run --with pytest --with usd-core")


def test_wrapper_bbox_includes_geo_rotation():
    """Placer footprint measurement must see the orientation yaw applied on geo."""
    from pxr import Gf, Usd, UsdGeom

    from isaac_datagen.isaac_utils import local_bbox_range

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/W")
    UsdGeom.Xform.Define(stage, "/W/obj")
    geo = UsdGeom.Xform.Define(stage, "/W/obj/geo")
    cube = UsdGeom.Cube.Define(stage, "/W/obj/geo/c")
    cube.AddScaleOp().Set(Gf.Vec3f(0.2, 0.05, 0.05))

    wrapper = stage.GetPrimAtPath("/W/obj")
    r0 = local_bbox_range(wrapper).GetSize()
    geo.AddRotateZOp().Set(90.0)
    r1 = local_bbox_range(wrapper).GetSize()

    assert r0[0] == pytest.approx(r1[1], abs=1e-5)
    assert r0[1] == pytest.approx(r1[0], abs=1e-5)

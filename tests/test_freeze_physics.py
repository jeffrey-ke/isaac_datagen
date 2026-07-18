import types

import pytest

pytest.importorskip("pxr", reason="needs usd-core: uv run --with pytest --with usd-core")

from isaac_datagen.store_mutations import CaptureTarget, freeze_physics


def _stage_with(paths):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/W")
    return stage, {p: UsdGeom.Xform.Define(stage, p).GetPrim() for p in paths}


def test_freeze_disables_store_extracted_bodies():
    from pxr import UsdPhysics
    stage, prims = _stage_with(["/W/a"])
    UsdPhysics.RigidBodyAPI.Apply(prims["/W/a"])
    src = types.SimpleNamespace(meta={"name": "a", "store_prim": "p/v_0"})
    assert freeze_physics(stage, [CaptureTarget(src, "/W/a")]) == 1
    assert UsdPhysics.RigidBodyAPI(prims["/W/a"]).GetRigidBodyEnabledAttr().Get() is False


def test_freeze_tolerates_blender_assets_without_physics():
    stage, _ = _stage_with(["/W/b"])
    blender = types.SimpleNamespace(meta={"name": "b"})          # no store_prim, no physics API
    assert freeze_physics(stage, [CaptureTarget(blender, "/W/b")]) == 0


def test_freeze_fails_loud_when_store_source_has_no_body():
    stage, _ = _stage_with(["/W/c"])
    src = types.SimpleNamespace(meta={"name": "c", "store_prim": "q/v_0"})
    with pytest.raises(AssertionError, match="no rigid body"):
        freeze_physics(stage, [CaptureTarget(src, "/W/c")])

import sys
import types

import numpy as np
import pytest

pytest.importorskip("pxr", reason="needs usd-core: uv run --with pytest --with usd-core")


@pytest.fixture
def stage(monkeypatch):
    from pxr import Usd
    st = Usd.Stage.CreateInMemory()
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fake = types.ModuleType("isaacsim.core.utils.stage")
    fake.get_current_stage = lambda: st
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage", fake)
    return st


def _order_and_point(prim):
    from pxr import UsdGeom
    x = UsdGeom.Xformable(prim)
    ops = [op.GetOpName() for op in x.GetOrderedXformOps()]
    m = np.array(x.GetLocalTransformation())               # row-vector convention: p' = p @ M
    return ops, (np.array([1.0, 0.0, 0.0, 1.0]) @ m)[:3]


def test_pose_then_scale_is_trs(stage):
    from pxr import UsdGeom
    from isaac_datagen.isaac_utils import set_transform
    UsdGeom.Xform.Define(stage, "/W")
    w = UsdGeom.Xform.Define(stage, "/W/w").GetPrim()
    set_transform(w, translation=(1, 2, 3), rotation=(0, 0, 90))
    set_transform(w, scale=(0.5, 0.5, 0.5))
    ops, p = _order_and_point(w)
    assert ops == ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    assert np.allclose(p, [1.0, 2.5, 3.0], atol=1e-5)      # scale innermost: translation unscaled


def test_repose_keeps_scale(stage):
    from pxr import UsdGeom
    from isaac_datagen.isaac_utils import set_transform
    UsdGeom.Xform.Define(stage, "/W")
    w = UsdGeom.Xform.Define(stage, "/W/w").GetPrim()
    set_transform(w, translation=(9, 9, 9), rotation=(0, 0, 0))
    set_transform(w, scale=(0.5, 0.5, 0.5))
    set_transform(w, translation=(1, 2, 3), rotation=(0, 0, 90))   # re-pose reuses existing ops
    ops, p = _order_and_point(w)
    assert ops == ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    assert np.allclose(p, [1.0, 2.5, 3.0], atol=1e-5)

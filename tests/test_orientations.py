import numpy as np
import pytest

from isaac_datagen import orientations


def _frame(front):
    """Mesh-local grasp SE3 with +X = front, +Z = up (mesh_convert convention)."""
    x = np.asarray(front, float)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    g = np.eye(4)
    g[:3, :3] = np.column_stack([x, y, z])
    return g


@pytest.mark.parametrize("front,expected", [
    ([1, 0, 0], -90.0),    # canonical +X front -> quarter turn to -Y
    ([0, -1, 0], 0.0),     # already fronting -Y -> no-op
    ([-1, 0, 0], 90.0),    # -X front
    ([0, 1, 0], -180.0),   # +Y front -> half turn
])
def test_yaw_to_azimuth_face_frames(front, expected):
    yaw = orientations.yaw_to_azimuth(_frame(front), azimuth_deg=-90)
    assert np.isclose((yaw - expected) % 360.0, 0.0) or np.isclose((yaw - expected) % 360.0, 360.0)


def test_yaw_to_azimuth_rejects_vertical_front():
    g = np.eye(4)
    g[:3, :3] = np.column_stack([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # front = up
    with pytest.raises(AssertionError, match="not horizontal"):
        orientations.yaw_to_azimuth(g, azimuth_deg=-90, label="snack031: ")


def test_registry():
    assert orientations.get("AlignGraspFronts") is orientations.AlignGraspFronts
    with pytest.raises(KeyError):
        orientations.get("NoSuchPolicy")


def test_align_requires_azimuth():
    with pytest.raises(TypeError):
        orientations.AlignGraspFronts()  # azimuth_deg is required, no default


def test_plain_scene_spec_accepts_orientation():
    from isaac_datagen.scene import PlainSceneSpec
    spec = PlainSceneSpec(orientation={"name": "AlignGraspFronts",
                                       "args": {"azimuth_deg": -90}})
    assert spec.orientation["name"] == "AlignGraspFronts"


def test_plain_scene_spec_default_is_none():
    from isaac_datagen.scene import PlainSceneSpec
    assert PlainSceneSpec().orientation is None


def test_plain_scene_spec_rejects_bad_orientation():
    from isaac_datagen.scene import PlainSceneSpec
    with pytest.raises(KeyError):
        PlainSceneSpec(orientation={"name": "NoSuchPolicy"})
    with pytest.raises(TypeError):
        PlainSceneSpec(orientation={"name": "AlignGraspFronts"})  # missing azimuth_deg
    with pytest.raises(AssertionError):
        PlainSceneSpec(orientation={"policy": "AlignGraspFronts"})  # wrong shape

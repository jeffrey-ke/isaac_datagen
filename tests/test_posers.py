from pathlib import Path

import numpy as np

from isaac_datagen.posers import DecenteredLookAtPoser, LookAtPoser
from vision_core.pose_utils import cv2opengl, look_at

XR, YR, ZR = [0.3, 2.0], [-2.0, 2.0], [-0.7, 0.7]
ZED_K = Path(__file__).resolve().parents[1] / "src" / "isaac_datagen" / "zed_K.npy"
LOG_SPEC = {"name": "log_uniform_offsets", "args": {"floor": 0.02}}


def test_lookat_poser_default_unchanged():
    np.random.seed(7)
    got = LookAtPoser(XR, YR, ZR)(50)

    np.random.seed(7)
    offsets = np.random.uniform(*zip(XR, YR, ZR), size=(50, 3))    # today's exact inline call
    want = np.array([cv2opengl(look_at(np.zeros(3), off)) for off in offsets])

    assert np.array_equal(got, want)


def test_lookat_poser_log_sampler_reduces_mean_radius():
    np.random.seed(1)
    default_t = LookAtPoser(XR, YR, ZR)(5000)[:, :3, 3]
    np.random.seed(1)
    log_t = LookAtPoser(XR, YR, ZR, offset_sampler=LOG_SPEC)(5000)[:, :3, 3]

    assert np.linalg.norm(log_t, axis=1).mean() < np.linalg.norm(default_t, axis=1).mean()


def test_decentered_lookat_poser_position_parity_holds_with_custom_sampler():
    np.random.seed(3)
    look = LookAtPoser(XR, YR, ZR, offset_sampler=LOG_SPEC)(50)[:, :3, 3]

    np.random.seed(3)
    dec = DecenteredLookAtPoser(XR, YR, ZR, intrinsics_path=str(ZED_K), resolution=[1920, 1080],
                                object_radius=0.25, offset_sampler=LOG_SPEC)(50)[:, :3, 3]

    assert np.allclose(look, dec)

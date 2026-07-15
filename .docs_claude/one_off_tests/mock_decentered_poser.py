"""Track A -- pre-render verification mock for `DecenteredLookAtPoser`. Sim-free: `posers.py`
never imports Isaac, so plain `python3` (numpy + scipy) is enough -- no isaac_datagen .venv
needed. Re-creates the planning-session numeric mock from plan
`decentered-1inst-pools-and-shift-aug.md` (Verification > Track A -- pre-render), using the
real `zed_K.npy` intrinsics and the pool halo box (x [0.3,2.0], y +-2.0, z +-0.7) read off the
`-1inst` configs (`configs/emptyworld-optflow-snacks-kshot-*-1inst.yaml`).

Checks:
  1. seed-matched positions -- DecenteredLookAtPoser(200)[:, :3, 3] == LookAtPoser(200)[:, :3, 3]
  2. pointing exactness -- grasp origin projects onto the sampled pixel within 0.5px
  3. ground truth -- a dense 256-point object_radius sphere never leaves the frame, including
     frames pinned to the eroded-rect corners (worst case)
  4. over ~5000 frames -- origin-projection std, coverage range, close-up fallback fraction

Run: `python3 mock_decentered_poser.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ISAAC_DATAGEN_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ISAAC_DATAGEN_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "vision_core" / "src"))
sys.path.insert(0, str(ISAAC_DATAGEN_ROOT / "src"))

from vision_core.pose_utils import erode_frame_rect  # noqa: E402
from isaac_datagen.posers import LookAtPoser, DecenteredLookAtPoser  # noqa: E402

K_PATH = ISAAC_DATAGEN_ROOT / "src" / "isaac_datagen" / "zed_K.npy"
RESOLUTION = (1920, 1080)
XRANGE, YRANGE, ZRANGE = (0.3, 2.0), (-2.0, 2.0), (-0.7, 0.7)
OBJECT_RADIUS = 0.25
MARGIN_DEG = 1.0
MAX_ROLL_DEG = 15.0
SEED = 1001


def poser_kwargs() -> dict:
    return dict(xrange=XRANGE, yrange=YRANGE, zrange=ZRANGE,
                intrinsics_path=str(K_PATH), resolution=RESOLUTION,
                object_radius=OBJECT_RADIUS, margin_deg=MARGIN_DEG, max_roll_deg=MAX_ROLL_DEG)


def project_batch(pose_gl: np.ndarray, K: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """pose_gl: OpenGL-convention cam2world (as returned by the posers). Projects world points
    into CV pixel coords via K, undoing cv2opengl's column flip (self-inverse: pose_cv =
    pose_gl @ diag(1,-1,-1,1))."""
    D = np.diag([1.0, -1.0, -1.0, 1.0])
    pose_cv = pose_gl @ D
    world2cam = np.linalg.inv(pose_cv)
    hom = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=-1)
    p_cam = (world2cam @ hom.T).T[:, :3]
    uvw = (K @ p_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def sphere_points(radius: float, n: int) -> np.ndarray:
    """n points on a Fibonacci sphere of the given radius, centered at the target origin."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    return radius * np.stack([x, y, z], axis=-1)


def sphere_escapes(pose_gl: np.ndarray, K: np.ndarray, resolution) -> bool:
    w, h = resolution
    proj = project_batch(pose_gl, K, sphere_points(OBJECT_RADIUS, 256))
    in_frame = (proj[:, 0] >= 0) & (proj[:, 0] <= w) & (proj[:, 1] >= 0) & (proj[:, 1] <= h)
    return not bool(in_frame.all())


def instrumented_run(poser: DecenteredLookAtPoser, num_frames: int, seed: int):
    """Runs the REAL DecenteredLookAtPoser, recording every np.random.uniform draw so the
    per-frame sampled pixel (uv) and close-up-fallback flag can be recovered without
    duplicating _decentered's own logic. Only `erode_frame_rect` (already imported, pure, no
    RNG) is reused to predict -- from the recovered offsets -- which frames take the close-up
    branch (0 further draws) vs the decentering branch (3 further draws: u, v, roll), matching
    _decentered's control flow verbatim."""
    calls = []
    real_uniform = np.random.uniform

    def spy(*args, **kwargs):
        val = real_uniform(*args, **kwargs)
        calls.append(val)
        return val

    np.random.seed(seed)
    np.random.uniform = spy
    try:
        poses = poser(num_frames)
    finally:
        np.random.uniform = real_uniform

    offsets = calls[0]
    assert offsets.shape == (num_frames, 3)

    per_frame_uv: list[tuple[float, float] | None] = [None] * num_frames
    close_up = np.zeros(num_frames, dtype=bool)
    ci = 1
    for i, off in enumerate(offsets):
        ang_r = np.arcsin(min(OBJECT_RADIUS / np.linalg.norm(off), 1.0)) + np.radians(MARGIN_DEG)
        rect = erode_frame_rect(poser.K, poser.resolution, ang_r)
        if rect is None:
            close_up[i] = True
            continue
        per_frame_uv[i] = (float(calls[ci]), float(calls[ci + 1]))
        ci += 3
    assert ci == len(calls), f"RNG draw accounting mismatch: consumed {ci}, recorded {len(calls)}"
    return poses, offsets, per_frame_uv, close_up


def pose_at_forced_uv(poser: DecenteredLookAtPoser, off: np.ndarray, uv: tuple[float, float]) -> np.ndarray:
    """Exercises the REAL _decentered code path but forces the sampled pixel to `uv` (pinning
    to eroded-rect corners for the ground-truth worst-case check) by monkeypatching the u/v
    draws; the roll draw still runs for real."""
    seq = iter(uv)
    real_uniform = np.random.uniform

    def spy(lo, hi):
        try:
            return next(seq)
        except StopIteration:
            return real_uniform(lo, hi)

    np.random.uniform = spy
    try:
        return poser._decentered(off)
    finally:
        np.random.uniform = real_uniform


def check_seed_matched_positions(num_frames: int = 200) -> bool:
    np.random.seed(SEED)
    centered = LookAtPoser(XRANGE, YRANGE, ZRANGE)(num_frames)
    np.random.seed(SEED)
    decentered = DecenteredLookAtPoser(**poser_kwargs())(num_frames)
    ok = np.array_equal(centered[:, :3, 3], decentered[:, :3, 3])
    print(f"[1. seed-matched positions] n={num_frames} exact match: {ok}")
    return ok


def check_pointing_exactness(poses, per_frame_uv, K) -> bool:
    errs = []
    for pose, uv in zip(poses, per_frame_uv):
        if uv is None:
            continue
        proj = project_batch(pose, K, np.zeros((1, 3)))[0]
        errs.append(np.linalg.norm(proj - np.array(uv)))
    errs = np.array(errs)
    ok = bool(len(errs) and errs.max() < 0.5)
    print(f"[2. pointing exactness] n={len(errs)} max_err={errs.max():.4f}px mean_err={errs.mean():.4f}px "
          f"pass(<0.5px): {ok}")
    return ok


def check_sphere_ground_truth(poser, poses, offsets, close_up, K, resolution) -> bool:
    # close_up (eroded-rect-collapsed) frames stay centered -- baseline LookAtPoser behavior,
    # not a decentering guarantee; excluded here exactly as check_pointing_exactness and the
    # corner check below already do (uv is None / `if cu: continue`).
    decentered_poses = [pose for pose, cu in zip(poses, close_up) if not cu]
    n_bad_random = sum(sphere_escapes(pose, K, resolution) for pose in decentered_poses)
    print(f"[3a. sphere ground truth, sampled uv] n={len(decentered_poses)} (excludes {close_up.sum()} "
          f"close-up fallback frames) frames_with_escaping_points={n_bad_random}")

    n_tested = n_bad_corner = 0
    for off, cu in zip(offsets, close_up):
        if cu:
            continue
        ang_r = np.arcsin(min(OBJECT_RADIUS / np.linalg.norm(off), 1.0)) + np.radians(MARGIN_DEG)
        rect = erode_frame_rect(poser.K, poser.resolution, ang_r)
        corners = [(rect[0], rect[2]), (rect[1], rect[2]), (rect[0], rect[3]), (rect[1], rect[3])]
        for corner in corners:
            n_tested += 1
            pose = pose_at_forced_uv(poser, off, corner)
            if sphere_escapes(pose, K, resolution):
                n_bad_corner += 1
    print(f"[3b. sphere ground truth, eroded-rect corners (worst case)] n_tested={n_tested} "
          f"frames_with_escaping_points={n_bad_corner}")
    return n_bad_random == 0 and n_bad_corner == 0


def report_stats(poses, K, close_up, num_frames):
    # poses is (N, 4, 4); project_batch inverts one pose at a time (per-frame world2cam).
    proj = np.array([project_batch(p, K, np.zeros((1, 3)))[0] for p in poses])
    std = proj.std(axis=0)
    cov_x = (proj[:, 0].min(), proj[:, 0].max())
    cov_y = (proj[:, 1].min(), proj[:, 1].max())
    frac_centered = close_up.mean()
    print(f"[4. over {num_frames} frames] origin-projection std=({std[0]:.1f}, {std[1]:.1f})px "
          f"coverage x=[{cov_x[0]:.1f},{cov_x[1]:.1f}] y=[{cov_y[0]:.1f},{cov_y[1]:.1f}] "
          f"close-up fallback fraction={frac_centered:.3%}")
    print("    (plan expects: std ~ (432, 156)px, coverage ~ x[212,1700] y[208,871], close-up ~10%)")


def main():
    K = np.load(K_PATH)
    print(f"Using K from {K_PATH}:\n{K}")
    print(f"resolution={RESOLUTION} halo box: xrange={XRANGE} yrange={YRANGE} zrange={ZRANGE} "
          f"object_radius={OBJECT_RADIUS}\n")

    ok1 = check_seed_matched_positions(200)

    num_frames = 5000
    poser = DecenteredLookAtPoser(**poser_kwargs())
    poses, offsets, per_frame_uv, close_up = instrumented_run(poser, num_frames, seed=SEED)

    ok2 = check_pointing_exactness(poses, per_frame_uv, K)
    ok3 = check_sphere_ground_truth(poser, poses, offsets, close_up, K, RESOLUTION)
    report_stats(poses, K, close_up, num_frames)

    all_ok = ok1 and ok2 and ok3
    print(f"\nALL CHECKS PASSED: {all_ok}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

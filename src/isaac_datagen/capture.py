"""Replicator-driven stereo capture orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_datagen.pose_planning import plan_poses


def get_target2world(target_paths):
    """Compute local-to-world SE3 poses for a batch of prim paths.

    Args:
        target_paths: sequence of prim paths.

    Returns:
        (B, 4, 4) array of target-to-world SE3 matrices.
    """
    from pxr import UsdGeom, Usd
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    poses = []
    for target_path in target_paths:
        target_prim = stage.GetPrimAtPath(target_path)
        xformable = UsdGeom.Xformable(target_prim)
        gf_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        poses.append(np.array(gf_matrix).T)
    return np.stack(poses)


def broadcast(a: Sequence, b: Sequence):
    """Numpy-style elementwise pairing with length-1 broadcasting."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("sequences must be non-empty")
    if len(a) == len(b):
        return list(zip(a, b))
    if len(a) == 1:
        return [(a[0], y) for y in b]
    if len(b) == 1:
        return [(x, b[0]) for x in a]
    raise ValueError(f"sequences not broadcastable: lens {len(a)} and {len(b)}")


def _broadcast_pairs(writers: Sequence, cameras: Sequence):
    if len(writers) == 1 and len(cameras) > 1:
        # 1 writer drains many cameras: one attach call with union of rps,
        # because writer.attach clobbers _writer_id on a second call.
        union = [rp for cam in cameras for rp in cam.rps]
        return [(writers[0], union)]
    return [(w, list(cam.rps)) for w, cam in broadcast(writers, cameras)]


def attach_writers(pairs):
    for writer, rps in pairs:
        writer.attach(*rps)


@contextmanager
def capture_session(writers, cameras, n_frames, replicator, rt_subframes=4):
    """Open a Replicator capture scope around caller-defined per-frame ops.

    Enters rep.new_layer(), attaches writers to camera render products under
    broadcast rules, yields rep so the caller can open whatever triggers and
    modifiers they want, then drives n_frames steps and waits for completion.
    """
    rep = replicator.rep
    pairs = _broadcast_pairs(writers, cameras)
    with rep.new_layer():
        attach_writers(pairs)
        yield rep
    for _ in range(n_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)
    rep.orchestrator.wait_until_complete()


def move_prims(prims, pose_sequences, replicator):
    rep = replicator.rep
    for prim, poses in broadcast(prims, pose_sequences):
        positions = [p[:3, 3].tolist() for p in poses]
        rotations = [
            R.from_matrix(p[:3, :3]).as_euler('xyz', degrees=True).tolist()
            for p in poses
        ]
        with prim:
            rep.modify.pose(
                position=rep.distribution.sequence(positions),
                rotation=rep.distribution.sequence(rotations),
            )


def capture_with_poses(world_poses, writer, camera, replicator):
    """Move camera through world_poses and capture one frame per pose.

    Args:
        world_poses: sequence of (4,4) SE3 matrices in the world frame.
        writer: initialized Replicator Writer, not yet attached.
        camera: camera object with .prim_path and .rps attributes.
        replicator: scene replicator handle (has .rep and .apply_randomizers()).
    """
    rep = replicator.rep
    rig_node = rep.get.prim_at_path(camera.prim_path)
    with capture_session(
        writers=[writer],
        cameras=[camera],
        n_frames=len(world_poses),
        replicator=replicator,
    ) as rep:
        with rep.trigger.on_frame():
            move_prims([rig_node], [world_poses], replicator)
            replicator.apply_randomizers()


def make_index(target_to_baseline_ypr_desired, xrange, yrange, zrange,
               sampling, target_prim, zed, replicator, render_dir):
    from isaac_datagen.stereo_writer import StereoSampleWriter

    target_frame_poses = plan_poses(
        target_to_baseline_ypr_desired, xrange, yrange, zrange, sampling
    )
    target2world = get_target2world(target_prim)
    world_poses = target2world @ target_frame_poses
    offsets = [pose[:3, 3].tolist() for pose in target_frame_poses]

    stereo_writer = StereoSampleWriter(
        output_dir=str(render_dir),
        offsets=offsets,
        target2world=target2world,
    )

    capture_with_poses(world_poses, stereo_writer, zed, replicator)

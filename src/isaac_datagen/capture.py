"""Replicator-driven stereo capture orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_datagen import posers


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


def plan_capture(runtime, scene, rng):
    """Pick per-target grasp frames and the world camera poses.

    THE single computation both the real render (reference_segmentation) and the
    dry-run debug export depend on, so the two can never drift apart.

    Returns:
        (idx, grasp_points, world_poses): the chosen grasp-point indices, their
        prim paths, and the (B*N, 4, 4) world camera poses (targets × planned poses).
    """
    idx = rng.choice(len(scene.grasp_points), size=runtime.num_targets)
    grasp_points = [scene.grasp_points[i] for i in idx]
    target2worlds = get_target2world(grasp_points)                       # (B, 4, 4)
    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)                       # (N, 4, 4)
    # (B, N, 4, 4) -> (B*N, 4, 4): flatten target and pose dims into one batch.
    world_poses = np.einsum('bij,njk->bnik', target2worlds, target_frame_poses).reshape(-1, 4, 4)
    return idx, grasp_points, world_poses


def se3_to_pos_euler(pose):
    """SE3 -> (translation, euler_xyz_deg).

    THE single SE3 decomposition. move_prims feeds this to rep.modify.pose for the
    live capture; the dry-run baker feeds it to set_transform. Sharing it (rather
    than the application call, which differs) is what keeps the baked debug cameras
    landing exactly where the rendered cameras would.
    """
    return (pose[:3, 3].tolist(),
            R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist())


def set_prim_pose(prim_path, pose):
    """Author an SE3 world pose onto the prim at `prim_path`.

    Prim-path-string addressed; consistent with move_prims by construction —
    set_transform authors the same USD ops (xformOp:translate + xformOp:rotateXYZ)
    that rep.modify.pose authors, fed the identical euler triple from se3_to_pos_euler.
    """
    from isaacsim.core.utils.stage import get_current_stage
    from isaac_datagen.isaac_utils import set_transform
    translation, rotation = se3_to_pos_euler(pose)
    set_transform(get_current_stage().GetPrimAtPath(prim_path),
                  translation=translation, rotation=rotation)


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
def capture_session(writers, cameras, n_frames, replicator, rt_subframes=20):
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
        positions, rotations = zip(*(se3_to_pos_euler(p) for p in poses))
        with prim:
            rep.modify.pose(
                position=rep.distribution.sequence(list(positions)),
                rotation=rep.distribution.sequence(list(rotations)),
            )


def capture_with_poses(world_poses, writer, camera, replicator, rt_subframes=20):
    """Move camera through world_poses and capture one frame per pose.

    Args:
        world_poses: sequence of (4,4) SE3 matrices in the world frame.
        writer: initialized Replicator Writer, not yet attached.
        camera: camera object with .prim_path and .rps attributes.
        replicator: scene replicator handle (has .rep and .apply_randomizers()).
        rt_subframes: subframes per captured frame (PT material-load slack + denoise).
    """
    rep = replicator.rep
    rig_node = rep.get.prim_at_path(camera.prim_path)
    with capture_session(
        writers=[writer],
        cameras=[camera],
        n_frames=len(world_poses),
        replicator=replicator,
        rt_subframes=rt_subframes,
    ) as rep:
        with rep.trigger.on_frame():
            move_prims([rig_node], [world_poses], replicator)
            replicator.apply_randomizers()

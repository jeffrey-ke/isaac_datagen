
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaac_datagen import posers


def get_target2world(target_paths):
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


def plan_capture(runtime, scene):
    idx = (np.arange(len(scene.grasp_points)) if runtime.num_targets is None
           else np.random.choice(len(scene.grasp_points), size=runtime.num_targets))
    grasp_points = [scene.grasp_points[i] for i in idx]
    target2worlds = get_target2world(grasp_points)
    poser = posers.get(runtime.pose_generation_policy)(**runtime.pose_generation_policy_args)
    target_frame_poses = poser(runtime.num_frames)
    world_poses = np.einsum('bij,njk->bnik', target2worlds, target_frame_poses).reshape(-1, 4, 4)
    return idx, grasp_points, world_poses


def se3_to_pos_euler(pose):
    return (pose[:3, 3].tolist(),
            R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist())


def set_prim_pose(prim_path, pose):
    from isaacsim.core.utils.stage import get_current_stage
    from isaac_datagen.isaac_utils import set_transform
    translation, rotation = se3_to_pos_euler(pose)
    set_transform(get_current_stage().GetPrimAtPath(prim_path),
                  translation=translation, rotation=rotation)


def broadcast(a: Sequence, b: Sequence):
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
        union = [rp for cam in cameras for rp in cam.rps]
        return [(writers[0], union)]
    return [(w, list(cam.rps)) for w, cam in broadcast(writers, cameras)]


def attach_writers(pairs):
    for writer, rps in pairs:
        writer.attach(*rps)


@contextmanager
def capture_session(writers, cameras, n_frames, replicator, rt_subframes=20, per_frame=None):
    rep = replicator.rep
    pairs = _broadcast_pairs(writers, cameras)
    with rep.new_layer():
        attach_writers(pairs)
        yield rep
    for i in range(n_frames):
        if per_frame is not None:
            per_frame(i)
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
    rep = replicator.rep
    rig_node = rep.get.prim_at_path(camera.prim_path)
    with capture_session(
        writers=[writer],
        cameras=[camera],
        n_frames=len(world_poses),
        replicator=replicator,
        rt_subframes=rt_subframes,
        per_frame=replicator.per_frame,
    ) as rep:
        with rep.trigger.on_frame():
            move_prims([rig_node], [world_poses], replicator)
            replicator.apply_randomizers()

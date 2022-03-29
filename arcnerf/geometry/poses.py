# -*- coding: utf-8 -*-

import numpy as np
import torch

from .transformation import normalize


def invert_pose(poses):
    """ Change poses between c2w and w2c. Support numpy and torch

    Args:
        poses: (N, 4, 4)

    Returns:
        poses: (N, 4, 4)
    """
    if isinstance(poses, torch.FloatTensor):
        return torch.inverse(poses.clone())
    elif isinstance(poses, np.ndarray):
        return np.linalg.inv(poses.copy())


def center_poses(poses):
    """
    Centralize the pose, which changes the world coordinate center.
    The central of the poses is now the origin of world

    Args:
        poses: (N_images, 3, 4)

    Returns:
        poses_centered: (N_images, 3, 4) the centered poses
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg

    bottom = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, bottom], 1)  # (N_images, 4, 4)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3, :]  # (N_images, 3, 4)

    return poses_centered


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Args:
        poses: (N_images, 3, 4), should be c2w poses, but not w2c

    Returns:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)

    pose_avg = np.stack([x, y, z, center], 1)

    return pose_avg

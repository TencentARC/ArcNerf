# -*- coding: utf-8 -*-

import numpy as np
import torch

from .transformation import normalize


def invert_poses(poses):
    """ Change poses between c2w and w2c. Support numpy and torch

    Args:
        poses: (N, 4, 4)

    Returns:
        poses: (N, 4, 4)
    """
    if isinstance(poses, torch.Tensor):
        return torch.inverse(poses.clone())
    elif isinstance(poses, np.ndarray):
        return np.linalg.inv(poses.copy())


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the y axis: the normalized average y axis.
    3. Compute axis z': the average z axis.
    4. Compute x' = z' cross product y, then normalize it as the x axis.
    5. Compute the z axis: x cross product y.

    Note that at step 3, we cannot directly use z' as z axis since it's
    not necessarily orthogonal to y axis. We need to pass from x to z.

    Args:
        poses: (N_images, 4, 4), c2w poses

    Returns:
        pose_avg: (4, 4) the average pose c2w
    """
    poses_3x4 = poses[:, :3, :].copy()
    # 1. Compute the center
    center = poses_3x4[..., 3].mean(0)

    # 2. Compute the y axis
    y = normalize(poses_3x4[..., 1].mean(0))

    # 3. Compute axis z' (no need to normalize as it's not the final output)
    z_ = poses_3x4[..., 2].mean(0)

    # 4. Compute the x axis
    x = normalize(np.cross(y, z_))

    # 5. Compute the z axis (as x and y are normalized, y is already of norm 1)
    z = np.cross(x, y)

    pose_avg = np.stack([x, y, z, center], 1)
    homo = np.zeros(shape=(1, 4), dtype=pose_avg.dtype)
    homo[0, -1] = 1.0
    pose_avg = np.concatenate([pose_avg, homo], axis=0)

    return pose_avg

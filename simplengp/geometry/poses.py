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


def center_poses(poses, center_loc=None):
    """
    Centralize the pose, which changes the world coordinate center. Good for cams that are not centered at (0,0,0)
    If you now where camera's are looking at (like central of point_cloud), you can set center_loc and
        all camera will look to this new place, rotation not change.
    Else all camera will look at avg_pose instead (rotation applied)
    The avg_poses/center_loc is now the origin of world. This will change the orientation of cameras. Be careful to use.

    Args:
        poses: (N_images, 4, 4)
        center_loc: (3, ), if None, need to calculate the avg by cameras
                    else, every camera looks to it after adjustment

    Returns:
        poses_centered: (N_images, 4, 4) the centered poses
    """
    if center_loc is None:
        up = normalize(poses[:, :3, 1].mean(0))
        pose_avg = average_poses(poses)
        poses_centered = poses.copy()
        poses_centered[:, :3, 3] -= pose_avg[:3, 3]
        for idx in range(poses.shape[0]):
            poses_centered[idx, :3, :3] = look_at(poses[idx, :3, 3], pose_avg[:3, 3], up)[:3, :3]
    else:
        poses_centered = poses.copy()
        poses_centered[:, :3, 3] -= center_loc

    return poses_centered


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


def view_matrix(forward, cam_loc, up=np.array([0.0, 1.0, 0.0])):
    """Get view matrix(c2w matrix) given forward/up dir and cam_loc.
    Notice: The up direction is y+, but (0,0) ray is down-ward. It is consistent with c2w from dataset but strange

    Args:
        forward: direction of view. np(3, )
        up: up direction. np(3, ). By default up on y-axis
        cam_loc: view location. np(3, )

    Returns:
        view_mat: c2w matrix. np(4, 4)

    """
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    view_mat = np.stack((rot_x, rot_y, rot_z, cam_loc), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(view_mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [view_mat.shape[0], 1, 1])
    view_mat = np.concatenate((view_mat, hom_vec), axis=-2)

    return view_mat


def look_at(cam_loc, point, up=np.array([0.0, 1.0, 0.0])):
    """Get view matrix(c2w matrix) given cam_loc, look_at_point, and up dir
    origin is at cam_loc always.

    Args:
        cam_loc: view location. np(3, )
        point: look at destination. np(3, )
        up: up direction. np(3, ). By default up on y-axis

    Returns:
        view_mat: c2w matrix. np(4, 4)
    """
    forward = normalize(point - cam_loc)  # cam_loc -> point

    return view_matrix(forward, cam_loc, up)

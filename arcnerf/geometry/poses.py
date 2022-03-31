# -*- coding: utf-8 -*-

import numpy as np
import torch

from .sphere import uv_to_sphere_point, get_sphere_line, get_spiral_line, get_regular_sphere_line
from .transformation import normalize
from .triangle import circumcircle_from_triangle


def invert_poses(poses):
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


def center_poses(poses, center_loc=None):
    """
    Centralize the pose, which changes the world coordinate center. Good for cams that are not centered at (0,0,0)
    If you now where camera's are looking at (like central of point_cloud), you can set center_loc and
        all camera will look to this new place, rotation not change.
    Else all camera will look at avg_pose instead (rotatation applied
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
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Args:
        poses: (N_images, 4, 4), c2w poses

    Returns:
        pose_avg: (4, 4) the average pose c2w
    """
    poses_3x4 = poses[:, :3, :].copy()
    # 1. Compute the center
    center = poses_3x4[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses_3x4[..., 2].mean(0))

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses_3x4[..., 1].mean(0)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)

    pose_avg = np.stack([x, y, z, center], 1)
    homo = np.ones(shape=(1, 4), dtype=pose_avg.dtype)
    pose_avg = np.concatenate([pose_avg, homo], axis=0)

    return pose_avg


def view_matrix(forward, cam_loc, up=np.array([0.0, 1.0, 0.0])):
    """Get view matrix(c2w matrix) given forward/up dir and cam_loc.

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


def generate_cam_pose_on_sphere(
    mode,
    radius,
    n_cam,
    u_start=0,
    v_ratio=0,
    v_range=(1, 0),
    n_rot=3,
    upper=None,
    close=False,
    origin=(0, 0, 0),
    look_at_point=np.array([0, 0, 0])
):
    """Get custom camera poses on sphere, looking at origin

    Args:
        mode: Support three mode: 'random', 'regular', 'circle', 'spiral'
            - random: any cam on sphere surface
            - regular: regularly distribute on different levels, n_rot level
            - circle: cam on a sphere circle track, decided by u_start and v_ratio
            - spiral: cam on a spiral track, decided by u_start, v_range, n_rot
        radius: sphere radius
        n_cam: num of cam selected
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_ratio: vertical lift ratio, in (-1, 1). 0 is largest, pos is on above.
        v_range: a tuple of v (v_start, v_end), start and end v ratio of spiral line
                    vertical lift angle, in (-1, 0). 0 is largest circle level, pos is on above.
        n_rot: num of full rotation, by default 3
        upper: Control camera postion for get_regular_sphere_line
        close: if true, first one will be the same as last(for circle and regular)
        origin: origin of sphere, tuple of 3
        look_at_point: the point camera looked at, np(3,)

    Returns:
        c2w: np(n_cam, 4, 4) matrix of cam position, in order
    """
    assert mode in ['random', 'regular', 'circle', 'spiral'], 'Invalid mode, only random/circle/spiral'

    cam_poses = []
    xyz = None
    if mode == 'random':
        u = np.random.rand(n_cam) * np.pi * 2
        v = np.random.rand(n_cam) * np.pi
        xyz = uv_to_sphere_point(u, v, radius, origin)  # (n_cam, 3)
    if mode == 'regular':
        xyz = get_regular_sphere_line(radius, u_start, origin, n_rot, n_pts=n_cam, upper=upper, close=close)
    elif mode == 'circle':
        xyz = get_sphere_line(radius, u_start, v_ratio, origin, n_pts=n_cam, close=close)
    elif mode == 'spiral':
        xyz = get_spiral_line(radius, u_start, v_range, origin, n_rot, n_pts=n_cam)

    for idx in range(xyz.shape[0]):
        cam_loc = xyz[idx]
        c2w = look_at(cam_loc, look_at_point)[None, :]
        cam_poses.append(c2w)
    cam_poses = np.concatenate(cam_poses, axis=0)

    return cam_poses


def generate_can_pose_from_tri_circumcircle(verts, n_cam, up=np.array([0, 1, 0]), close=True):
    """Get cam pose on a circle that is the circumcircle of triangle

    Args:
        verts: np(3, 3), triangle verts, second dim is xyz
        close: if true, first one will be the same as last

    Returns:
        c2w: np(n_cam, 4, 4) matrix of cam position, in order
        origin: np(3,), origin of the circle in 3d space
        radius: radius of circle
    """
    origin, radius, normal, circle = circumcircle_from_triangle(verts, n_cam, close)

    cam_poses = []
    for idx in range(circle.shape[0]):
        cam_loc = circle[idx]
        c2w = look_at(cam_loc, origin, up=normal)[None, :]
        cam_poses.append(c2w)
    cam_poses = np.concatenate(cam_poses, axis=0)

    return cam_poses, origin, radius

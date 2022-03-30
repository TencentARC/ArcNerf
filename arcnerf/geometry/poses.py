# -*- coding: utf-8 -*-

import math

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


def get_sphere_surface(radius, origin=(0.0, 0.0, 0.0), n_pts=100):
    """Get sphere surface position. y is up-down axis

    Args:
        radius: radius fo sphere
        origin: a list of 3, xyz origin
        n_pts: num of point on each dim, by default 100.

    Returns:
        x, y, z: 2-dim location.
    """
    u = np.linspace(0, 2 * np.pi, 100)  # horizontal
    v = np.linspace(0, np.pi, 100)  # vertical
    x = radius * np.outer(np.cos(u), np.sin(v)) + origin[0]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + origin[2]
    y = radius * np.outer(np.sin(u), np.sin(v)) + origin[1]

    return x, y, z


def get_sphere_line(radius, u_start=0, v=0, origin=(0.0, 0.0, 0.0), n_pts=100):
    """Get sphere surface line different by angle. The circle is face up-down, rotate in counter-clockwise
     y is up-down axis

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v: vertical lift ratio, in (-1, 1). 0 is largest, pos is on above.
        origin: origin of sphere. np(3, )
        n_pts: num of point on line, by default 100.

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v <= 1, 'Invalid v ratio, (-1, 1) only'
    u = np.linspace(0, 1, n_pts) + 1
    u[u > 1.0] -= 1.0
    u *= 2 * np.pi
    v = (1 - v) * np.pi / 2.0

    x = radius * np.cos(u) * np.sin(v) + origin[0]
    y = radius * np.ones_like(u) * np.cos(v) + origin[1]
    z = radius * np.sin(u) * np.sin(v) + origin[2]

    line = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)

    return line


def get_spiral_line(radius, u_start=0, v_range=(1, -1), origin=(0.0, 0.0, 0.0), n_rot=3, n_pts=100):
    """Get spiral surface line, rotate in counter-clockwise

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_range: a tuple of v (v_start, v_end), start and end v ratio of spiral line
                    vertical lift angle, in (-1, 1). 0 is largest, pos is on above.
        origin: origin of sphere. np(3, )
        n_rot: num of full rotation, by default 3
        n_pts: num of point on line, by default 100.

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v_range[0] <= 1 and -1 <= v_range[0] <= 1,\
        'Invalid v range, start and end all in (-1, 1) only'
    n_pts_per_rot = math.ceil(float(n_pts) / float(n_rot))
    u = np.linspace(0, 1, n_pts_per_rot) + u_start
    u[u > 1.0] -= 1.0
    u *= 2 * np.pi
    u = np.concatenate([u] * n_rot)[:n_pts]
    v = np.linspace((1 - v_range[0]), (1 - v_range[1]), n_pts) * np.pi / 2.0

    print(u.shape)
    print(v.shape)

    x = radius * np.cos(u) * np.sin(v) + origin[0]
    y = radius * np.ones_like(u) * np.cos(v) + origin[1]
    z = radius * np.sin(u) * np.sin(v) + origin[2]

    line = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)

    return line


def generate_cam_pose_on_sphere(mode, radius, n_cam, v=0.5, origin=(0.0, 0.0, 0.0)):
    """Get custom camera poses on sphere, looking at origin

    Args:
        mode: Support three mode: 'random', 'circle', 'spiral'
        radius: sphere radius
        n_cam: num of cam selected
        v: vertical lift ratio, by default is 0.5(upper small sphere)
        origin: origin to look at, by default is (0,0,0)

    """
    assert mode in ['random', 'circle', 'spiral'], 'Invalid mode, only random/circle/spiral'


def get_u_start_from_pose():
    pass


def get_v_from_pose():
    pass

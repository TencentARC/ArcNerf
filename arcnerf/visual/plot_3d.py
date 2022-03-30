# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from .draw_camera import create_camera_model
from common.visual.draw_cv2 import get_colors


def transform_plt_space(pts, xyz_axis=0):
    """Transform any point in world space to plt space
    This will change y/z axis

    Args:
        pts: np.array in (n_pts, 3) or (3, n_pts)
        xyz_axis: 0 or 1, 0 means pts in (3, n_pts), else (n_pts, 3)

    """
    assert xyz_axis in [0, 1], 'invalid xyz_axis'
    assert len(pts.shape) == 2 and (pts.shape[0] == 3 or pts.shape[1] == 3), 'Shape should be (n, 3) or (3, n)'

    if xyz_axis == 1:
        pts = np.transpose(pts, [1, 0])  # (3, pts)

    # rot mat
    rot_mat = np.identity(3, dtype=pts.dtype)
    rot_mat[1, 1] = 0
    rot_mat[1, 2] = 1
    rot_mat[2, 1] = 1
    rot_mat[2, 2] = 0

    pts_rot = rot_mat @ pts
    if xyz_axis == 1:
        pts_rot = np.transpose(pts_rot, [1, 0])  # (pts, 3)

    return pts_rot


def draw_3d_components(
    c2w=None,
    cam_colors=None,
    points=None,
    point_size=20,
    point_colors=None,
    rays=None,
    ray_colors=None,
    title='',
    save_path=None
):
    """draw 3d component, including cameras, points, rays, etc
    For any pts in world space, you need to transform_plt_space to switch yz axis
    You can specified color for different cam/ray/point.

    Args:
        c2w: c2w pose stack in in shape(N_cam, 4, 4). None means not visual
        cam_colors: color in (N_cam, 3) or (3,), applied for each cam
        points: point in (N_p, 3) shape in world coord
        point_size: size of point, by default set up 20
        point_colors: color in (N_p, 3) or (3,), applied for each point
        rays: a tuple (rays_o, rays_d), each in (N_r, 3), in world coord
                rays_d is with actual len, if you want longer arrow, you need to extent rays_d
        ray_colors: color in (N_r, 3) or (3,), applied for each ray
        title: a string of figure title
        save_path: path to save the fig. None will only show fig
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')

    # set color. By default: point(green), cam(red), ray(blue)
    N_cam = c2w.shape[0]
    N_p = points.shape[0]
    N_r = rays[0].shape[0]
    if cam_colors is None:
        cam_colors = get_colors('red', to_int=False, to_np=True)
    if cam_colors.shape == (3, ):
        cam_colors = np.repeat(cam_colors[None, :], N_cam, axis=0)
    print(cam_colors.shape)
    assert cam_colors.shape == (N_cam, 3), 'Invalid cam colors shape...(N_cam, 3) or (3,)'

    if point_colors is None:
        point_colors = get_colors('green', to_int=False, to_np=True)
    if point_colors.shape == (3, ):
        point_colors = np.repeat(point_colors[None, :], N_p, axis=0)
    assert point_colors.shape == (N_p, 3), 'Invalid point colors shape...(N_p, 3) or (3,)'

    if ray_colors is None:
        ray_colors = get_colors('blue', to_int=False, to_np=True)
    if ray_colors.shape == (3, ):
        ray_colors = np.repeat(ray_colors[None, :], N_r, axis=0)
    assert ray_colors.shape == (N_r, 3), 'Invalid ray colors shape...(N_r, 3) or (3,)'

    # axis range
    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf
    axis_scale_factor = 0.25  # extent scale by such factor

    if c2w is not None:
        # set vis params, adjust by camera loc
        max_cam_pose_norm = np.linalg.norm(c2w[:, :3, 3], axis=-1).max()
        cam_width = 0.032 * max_cam_pose_norm
        cam_height = 0.024 * max_cam_pose_norm
        f_scale = 0.08 * max_cam_pose_norm

        # single camera_model in local coord. Each is a xxx
        camera_model = create_camera_model(cam_width, cam_height, f_scale)

        for idx in range(c2w.shape[0]):  # each camera
            mat = c2w[idx]
            for i in range(len(camera_model)):  # each polygon. (4, n_pts)
                X = np.zeros(camera_model[i].shape)
                # to world coord
                for j in range(X.shape[1]):  # each point in polygon, (4, )
                    X[:4, j] = mat @ camera_model[i][:4, j]
                X = transform_plt_space(X[:3, :])
                # draw in world coord. plot3D plots line betweem neighbour vertices
                ax.plot3D(X[0, :], X[1, :], X[2, :], color=cam_colors[idx])  # draw multi lines

                min_values = np.minimum(min_values, X.min(1))
                max_values = np.maximum(max_values, X.max(1))

    if points is not None:
        points_plt = transform_plt_space(points, xyz_axis=1)
        ax.scatter3D(points_plt[:, 0], points_plt[:, 1], points_plt[:, 2], color=point_colors, s=point_size)
        min_values = np.minimum(min_values, points_plt.min(0))
        max_values = np.maximum(max_values, points_plt.max(0))

    if rays is not None:
        rays_o = rays[0]
        rays_d = rays[1]
        rays_e = rays_o + rays_d
        rays_o_plt = transform_plt_space(rays_o, xyz_axis=1)
        rays_d_plt = transform_plt_space(rays_d, xyz_axis=1)
        rays_e_plt = transform_plt_space(rays_e, xyz_axis=1)
        ax.quiver(
            rays_o_plt[:, 0],
            rays_o_plt[:, 1],
            rays_o_plt[:, 2],
            rays_d_plt[:, 0],
            rays_d_plt[:, 1],
            rays_d_plt[:, 2],
            color=ray_colors
        )
        min_values = np.minimum(np.minimum(min_values, rays_o_plt.min(0)), rays_e_plt.min(0))
        max_values = np.maximum(np.maximum(max_values, rays_o_plt.max(0)), rays_e_plt.max(0))

    min_values -= (max_values - min_values) * axis_scale_factor
    max_values += (max_values - min_values) * axis_scale_factor

    X_min, Y_min, Z_min = min_values[0], min_values[1], min_values[2]
    X_max, Y_max, Z_max = max_values[0], max_values[1], max_values[2]
    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

    plt.close()

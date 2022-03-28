# -*- coding: utf-8 -*-
# refer: https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py

import matplotlib.pyplot as plt
import numpy as np


def inverse_homo_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -R.T.dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homo_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(width, height, f_scale):
    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2 * height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def draw_cameras(ax, cam_width, cam_height, f_scale, extrinsics):
    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(cam_width, cam_height, f_scale)  # TODO: Check this
    color = (1.0, 0.0, 0.0)  # red

    for idx in range(extrinsics.shape[0]):
        # R, _ = cv2.Rodrigues(extrinsics[idx, :3, :3])
        # cMo = np.eye(4, 4)
        # cMo[0:3, 0:3] = R
        cMo = extrinsics[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], True)
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=color)
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))

    return min_values, max_values


def draw_camera_extrinsic(extrinsics, save_path=None):
    """draw cameras on image
    :params: extrinsics: w2c pose stack in in shape(N, 4, 4)
    """
    # set vis params, adjust by camera loc
    max_cam_pose_norm = np.linalg.norm(extrinsics[:, :3, 3], axis=-1).max()
    cam_width = 0.032 * max_cam_pose_norm
    cam_height = 0.024 * max_cam_pose_norm
    f_scale = 0.04 * max_cam_pose_norm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')

    min_values, max_values = draw_cameras(ax, cam_width, cam_height, f_scale, extrinsics)

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
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

    plt.close()

# -*- coding: utf-8 -*-
# refer: https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py

import numpy as np


def create_camera_model(width, height, f_scale):
    """Return camera model in local coord. Cam is allow centered at (0,0,0)

    Args:
        width: local image plane width by half
        height: local image plane height by half
        f_scale: camera model frustum length

    Returns:
        a list of np(4, n_point). Each np(4, n_point) is n_point in local coord,
        they are transformed by c2w into world coord
    """
    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

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

    return [X_img_plane, X_center1, X_center2, X_center3, X_center4]

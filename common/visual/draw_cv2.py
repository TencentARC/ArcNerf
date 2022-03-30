# -*- coding: utf-8 -*-

import cv2
import numpy as np

from ..visual import get_colors


def check_pixel_in_img(p, img_shape, col_row_order=False):
    """Check whether a pixel is in a image give its shape
    :param: p is Tuple(). p[0] is col(x) / p[1] is row(y)
    :param: img_shape: list of 2. In height-width order
    :param: xy: the points is in col(x)-row(y) order or not
    """
    if col_row_order:
        return 0 <= p[0] < img_shape[0] and 0 <= p[1] < img_shape[1]
    else:
        return 0 <= p[0] < img_shape[1] and 0 <= p[1] < img_shape[0]


def draw_points_on_img(pixels, img_shape, color=None):
    """Draw the pixel colors"""
    if color is None:
        color = [(0, 0, 255)] * pixels.shape[0]  # in bgr order, 0-255 range
    elif isinstance(color, str):
        color = [get_colors(color)] * pixels.shape[0]

    img = np.ones(shape=(img_shape[0], img_shape[1], 3)) * 255
    img = img.astype(np.uint8)
    for i in range(pixels.shape[0]):
        p = (int(pixels[i, 0]), int(pixels[i, 1]))
        if check_pixel_in_img(p, img_shape):
            img[p[1], p[0], :] = color[i, :]

    return img


def draw_vert_on_img(img, pixels, show_index=False, color=None):
    """Draw vertices as points on image
    :param: pixel is in col(x)-row(y) order
    """
    if color is None:
        color = (0, 0, 255)  # in bgr order, 0-255 range
    elif isinstance(color, str):
        color = get_colors(color)

    pt_size = int(round(max(img.shape[:2]) / 1000) + 3)
    text_size = (max(img.shape[:2]) / 1000) * 6.0 / 5.0 + 0.33
    for i in range(pixels.shape[0]):
        p = (int(pixels[i, 0]), int(pixels[i, 1]))
        if check_pixel_in_img(p, img.shape[:2]):
            cv2.circle(img, p, pt_size, color, -1)
            if show_index:
                cv2.putText(img, str(i), p, fontFace=5, fontScale=text_size, color=(255, 0, 0), lineType=3)

    return img

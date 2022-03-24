# -*- coding: utf-8 -*-

import cv2
import numpy as np


def read_img(path, dtype=np.float32, norm_by_255=False, bgr2rgb=True):
    """Read img in RGB Order. Generally do no norm the image by 255"""
    try:
        img = cv2.imread(path)
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if dtype == np.float32 or dtype == np.float64:
            if img.dtype == np.uint8:
                img = img.astype(dtype)

        if norm_by_255:
            img /= 255.0

    except cv2.error:
        if img is None:
            raise RuntimeError('Image at {} read as None...'.format(path))
        else:
            raise RuntimeError(
                'Image at {} found error, read as image {}, dtype {}, shape {}'.format(path, img, img.dtype, img.shape)
            )

    return img


def img_to_uint8(img, transpose=None, std=None, mean=None, norm_by_255=True, rgb2bgr=True):
    """To revert a img in float type to uint8 for visualization
       Will do std -> mean -> norm_by_255 -> rgb2bgr in order
    Parameters
    ----------
    :param img: numpy array in (H, W, 3) shape
    :param transpose: Tuple
        permute the axes order if not None.
    :param rgb2bgr: bool
        whether to change color from rgb to bgr
    :param mean: list with same length of img channel
        the mean to apply
    :param std: list with same length of img channel
        the std to apply
    :param norm_by_255: bool
        whether to multiply the value by 255
    """
    if transpose:
        img = np.transpose(img, transpose)

    assert img.shape[-1] == 3, 'Only support image with rgb channel now'

    if std:
        assert img.shape[-1] == len(std)
        img *= std

    if mean:
        assert img.shape[-1] == len(mean)
        img += mean

    if norm_by_255:
        img *= 255.0

    if rgb2bgr:
        img = img[:, :, [2, 1, 0]]

    return img.astype(np.uint8)


def is_img_ext(file):
    """Check whether a filename is an image file by checking extension."""
    return file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))

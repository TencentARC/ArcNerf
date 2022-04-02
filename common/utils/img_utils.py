# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np


def read_img(path, dtype=np.float32, norm_by_255=False, bgr2rgb=True, gray=False):
    """Read img in RGB Order. Generally do no norm the image by 255"""
    try:
        img = cv2.imread(path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bgr2rgb = False

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


def img_scale(img, scale, interpolation=None):
    """Scale a image using cv2.resize(). scale > 1 is scale_up
    By default scale_up uses INTER_LINEAR, scale_down uses INTER_AREA. You can specify by youself.
    """
    if scale == 1:
        return img

    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    if scale > 1:
        interpolation = cv2.INTER_LINEAR if interpolation is None else interpolation
    elif scale < 1:
        interpolation = cv2.INTER_AREA if interpolation is None else interpolation

    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    return img


def img_to_uint8(img, transpose=None, std=None, mean=None, norm_by_255=True, rgb2bgr=True):
    """To revert a img in float type to uint8 for visualization
       Will do std -> mean -> norm_by_255 -> rgb2bgr in order
    Args:
        img: numpy array in (H, W, 3) shape
        transpose: Tuple. permute the axes order if not None.
        rgb2bgr: bool. whether to change color from rgb to bgr
        mean: list with same length of img channel. the mean to apply
        std: list with same length of img channel. the std to apply
        norm_by_255: bool. whether to multiply the value by 255
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


def get_n_img_in_dir(folder):
    """Get the num of image in directory"""
    return len([f for f in os.listdir(folder) if is_img_ext(f)])


def heic_to_png(heic_path):
    """Change a heic(ios format) to png file.
    Save .png image to the same directory with different extension.
    """
    from wand.image import Image
    img = Image(filename=heic_path)
    img.format = 'png'
    img.save(filename=heic_path.replace('.HEIC', '.png'))
    img.close()


def get_image_metadata(img_path):
    """Get image w,h,channel"""
    img = cv2.imread(img_path)
    if len(img.shape) == 2:
        h, w = img.shape[:2]
        channel = 0
    else:
        h, w, channel = img.shape

    return w, h, channel

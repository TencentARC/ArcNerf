# -*- coding: utf-8 -*-

import random

import numpy as np
import torchvision.transforms as transforms

from .color_adjust import ColorJitterCV2


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = 'The Augmentation Information: \n'

    transforms_list.append(ColorJitter(cfgs.jitter))
    aug_info += '  Add ColorJitter with level {}\n'.format(cfgs.jitter)

    return transforms.Compose(transforms_list), aug_info


class ColorJitter(object):
    """Color Jitter
    Require 'img' field in data. Must assume values are in (0, 255) range
    Still return an image in (0-255) range
    """

    def __init__(self, jitter):
        """ Jitter should be a value in [0, 1). Larger value indicates a stronger jitter"""
        self.jitter = jitter

    def __call__(self, data):
        jitter_prob = random.random()
        if jitter_prob > 0.5 or self.jitter is None:  # no always want to jitter
            return data

        assert 'img' in data.keys(), 'Please assert img in data...'
        img = data['img']

        if np.max(img) <= 1.0:
            raise RuntimeError('Please input an image with 0-255 range')

        jit = ColorJitterCV2(
            brightness=self.jitter, contrast=self.jitter, saturation=self.jitter, hue=self.jitter / 2.0
        )

        img = jit(img)
        data['img'] = img

        return data


class ImgNorm(object):
    """Normalize image by mean and std. Order is in RGB. Norm first than do mean-std adjustment
    Require 'img' field in data

    By default will not do mean-std adjustment, if you need, you need to input like
    img_mean = [0.485, 0.456, 0.406]; img_std = [0.229, 0.224, 0.225]
    which will be ImageNet mean and std
    """

    def __init__(self, img_mean=None, img_std=None, norm_by_255=True):
        self.mean = img_mean if img_mean else [0, 0, 0]
        self.std = img_std if img_std else [1, 1, 1]
        self.norm_by_255 = norm_by_255

    def __call__(self, data):
        assert 'img' in data.keys(), 'Please assert img in data...'

        img = data['img']
        if len(img) == 2:
            img = np.expand_dims(img, -1)
        assert img.shape[-1] == len(self.mean), 'Please input an image with channel == {}'.format(len(self.mean))

        if self.norm_by_255:
            img /= 255.0

        img -= self.mean
        img /= self.std
        data['img'] = img

        return data


class PermuteImg(object):
    """Permute image from (H, W, 3) to (3, H, W)"""

    def __init__(self):
        return

    def __call__(self, data):
        assert 'img' in data.keys(), 'Please assert img in data...'

        img = data['img']
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        img = np.transpose(img, (2, 0, 1))
        data['img'] = img

        return data

# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = ''

    if valid_key_in_cfgs(cfgs, 'augmentation'):
        if valid_key_in_cfgs(cfgs.augmentation, 'n_rays') and \
                get_value_from_cfgs_field(cfgs.augmentation, 'n_rays', 0) > 0:
            transforms_list.append(SampleRays(cfgs.augmentation.n_rays))
            aug_info += '  Add SampleRays with N_rays {}\n'.format(cfgs.augmentation.n_rays)

        if valid_key_in_cfgs(cfgs.augmentation, 'shuffle') and \
                get_value_from_cfgs_field(cfgs.augmentation, 'shuffle', False):
            transforms_list.append(ShuffleRays())
            aug_info += '  Add Rays shuffle\n'

    return transforms.Compose(transforms_list), aug_info


class SampleRays(object):
    """Sample rays from image and rays"""

    def __init__(self, n_rays=1024):
        self.n_rays = n_rays

    def __call__(self, inputs):
        select_idx = torch.randperm(inputs['img'].shape[0])

        inputs['img'] = inputs['img'][select_idx, :]
        inputs['rays_o'] = inputs['rays_o'][select_idx, :]
        inputs['rays_d'] = inputs['rays_d'][select_idx, :]

        if 'mask' in inputs:
            inputs['mask'] = inputs['mask'][select_idx, ...]

        return inputs


class ShuffleRays(object):
    """Shuffle rays and images"""

    def __init__(self):
        return

    def __call__(self, inputs):
        select_idx = torch.randperm(inputs['img'].shape[0])

        inputs['img'] = inputs['img'][select_idx, :]
        inputs['rays_o'] = inputs['rays_o'][select_idx, :]
        inputs['rays_d'] = inputs['rays_d'][select_idx, :]

        if 'mask' in inputs:
            inputs['mask'] = inputs['mask'][select_idx, ...]

        return inputs


def linear_to_srgb(x: torch.Tensor):
    """RGB from linear space to sRGB Space
    ref: https://entropymine.com/imageworsener/srgbformula/

    Args:
        x: linear rgb value in (B, 3), should be in (0~1)

    Returns:
        y: rgb value in sRGB space

    """
    assert torch.all(x <= 1.0) and torch.all(x >= 0.0), 'Input should be in (0~1)'

    return torch.where(x <= 0.00313066844250063, 12.92 * x, 1.055 * x**(1.0 / 2.4) - 0.055)


def srgb_to_linear(x):
    """RGB from sRGB space to linear Space
    ref: https://entropymine.com/imageworsener/srgbformula/

    Args:
        x: sRGB value in (B, 3), should be in (0~1)

    Returns:
        y: rgb value in linear space

    """
    assert torch.all(x <= 1.0) and torch.all(x >= 0.0), 'Input should be in (0~1)'

    return torch.where(x <= 0.0404482362771082, x / 12.92, ((x + 0.055) / 1.055)**2.4)

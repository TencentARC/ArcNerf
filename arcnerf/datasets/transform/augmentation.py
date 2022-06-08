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

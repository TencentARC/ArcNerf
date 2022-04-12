# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

from common.utils.cfgs_utils import valid_key_in_cfgs


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = ''

    if valid_key_in_cfgs(cfgs, 'augmentation'):
        if valid_key_in_cfgs(cfgs.augmentation, 'n_rays'):
            transforms_list.append(SampleRays(cfgs.augmentation.n_rays))
            aug_info += '  Add SampleRays with N_rays {}\n'.format(cfgs.augmentation.n_rays)

    return transforms.Compose(transforms_list), aug_info


class SampleRays(object):
    """Sample rays from image and rays"""

    def __init__(self, n_rays=1024):
        self.n_rays = n_rays

    def __call__(self, inputs):
        n_total = inputs['img'].shape[0]
        device = inputs['img'].device
        select_idx = torch.randint(0, n_total, size=[self.n_rays]).to(device)

        inputs['img'] = inputs['img'][select_idx, :]
        inputs['mask'] = inputs['mask'][select_idx] if inputs['mask'] is not None else None
        inputs['rays_o'] = inputs['rays_o'][select_idx, :]
        inputs['rays_d'] = inputs['rays_d'][select_idx, :]

        return inputs

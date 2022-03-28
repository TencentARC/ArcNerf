# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = ''

    if hasattr(cfgs, 'augmentation') and cfgs.augmentation is not None:
        if hasattr(cfgs.augmentation, 'N_rays') and cfgs.augmentation.N_rays is not None:
            transforms_list.append(SampleRays(cfgs.augmentation.N_rays))
            aug_info += '  Add SampleRays with N_rays {}\n'.format(cfgs.augmentation.N_rays)

    return transforms.Compose(transforms_list), aug_info


class SampleRays(object):
    """Sample rays from image and rays"""

    def __init__(self, N_rays=1024):
        self.N_rays = N_rays

    def __call__(self, inputs):
        N_total = inputs['img'].shape[0]
        device = inputs['img'].device
        select_idx = torch.randint(0, N_total, size=[self.N_rays]).to(device)

        inputs['img'] = inputs['img'][select_idx, :]
        inputs['mask'] = inputs['mask'][select_idx] if inputs['mask'] is not None else None
        inputs['rays_o'] = inputs['rays_o'][select_idx, :]
        inputs['rays_d'] = inputs['rays_d'][select_idx, :]

        return inputs

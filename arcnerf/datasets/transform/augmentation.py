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

        if valid_key_in_cfgs(cfgs.augmentation, 'transfer_rgb'):
            type = get_value_from_cfgs_field(cfgs.augmentation.transfer_rgb, 'type', 'linear_to_srgb')
            transforms_list.append(TransferRGBSpace(type))
            aug_info += '  Add RGB space transfer - {}\n'.format(type)

        if valid_key_in_cfgs(cfgs.augmentation, 'blend_bkg_color'):
            bkg_color = get_value_from_cfgs_field(cfgs.augmentation.blend_bkg_color, 'bkg_color', [1.0, 1.0, 1.0])
            transforms_list.append(BlendBkgColor(bkg_color))
            aug_info += '  Add Blend bkg color - {}\n'.format(bkg_color)

    return transforms.Compose(transforms_list), aug_info


class SampleRays(object):
    """Sample rays from image and rays"""

    def __init__(self, n_rays=1024):
        self.n_rays = n_rays

    def __call__(self, inputs):
        # get sample idx
        select_idx = torch.randperm(inputs['img'].shape[0])[:self.n_rays]

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
        # get full selection idx
        select_idx = torch.randperm(inputs['img'].shape[0])

        inputs['img'] = inputs['img'][select_idx, :]
        inputs['rays_o'] = inputs['rays_o'][select_idx, :]
        inputs['rays_d'] = inputs['rays_d'][select_idx, :]

        if 'mask' in inputs:
            inputs['mask'] = inputs['mask'][select_idx, ...]

        return inputs


class BlendBkgColor(object):
    """Blend bkg color"""

    def __init__(self, bkg_color):
        self.bkg_color = bkg_color

    def __call__(self, inputs):
        dtype = inputs['img'].dtype
        device = inputs['img'].device

        bkg_color = torch.tensor(self.bkg_color, dtype=dtype, device=device)[None]  # (1, 3)
        mask = inputs['mask'][:, None]  # (B, 1)

        inputs['img'] = inputs['img'] * mask + (1.0 - mask) * bkg_color

        return inputs


class TransferRGBSpace(object):
    """Transfer the rgb space of image rays. Must be rgb ordered image."""

    def __init__(self, t_type):
        self.t_type = t_type
        assert self.t_type in ['linear_to_srgb', 'srgb_to_linear'], 'Not support {}'.format(t_type)

    def __call__(self, inputs):
        mask = inputs['mask'] if 'mask' in inputs else None

        if self.t_type == 'linear_to_srgb':
            inputs['img'] = linear_to_srgb(inputs['img'], mask)
        elif self.t_type == 'srgb_to_linear':
            inputs['img'] = srgb_to_linear(inputs['img'])
        else:
            raise NotImplementedError('Transfer {} not supported...'.format(self.t_type))

        return inputs


def linear_to_srgb(x: torch.Tensor, mask=None):
    """RGB from linear space to sRGB Space, alpha must be applied
    ref: https://entropymine.com/imageworsener/srgbformula/

    Args:
        x: linear rgb value in (B, 3), should be in (0~1)
        mask: mask in (B, ), alpha channel, could be None

    Returns:
        y: rgb value in sRGB space

    """
    # apply alpha
    if mask is not None:
        _mask = torch.repeat_interleave(mask[:, None], x.shape[1], dim=1)
        x = torch.where(_mask != 0, x / _mask, x)

    # transfer func
    x = torch.where(x <= 0.00313066844250063, 12.92 * x, 1.055 * x**(1.0 / 2.4) - 0.055)

    return x


def srgb_to_linear(x):
    """RGB from sRGB space to linear Space
    ref: https://entropymine.com/imageworsener/srgbformula/

    Args:
        x: sRGB value in (B, 3), should be in (0~1)

    Returns:
        y: rgb value in linear space

    """
    return torch.where(x <= 0.0404482362771082, x / 12.92, ((x + 0.055) / 1.055)**2.4)

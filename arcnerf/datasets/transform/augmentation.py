# -*- coding: utf-8 -*-

import torchvision.transforms as transforms

from common.datasets.transform.augmentation import (ColorJitter, ImgNorm, PermuteImg)


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = ''

    if hasattr(cfgs, 'augmentation') and cfgs.augmentation is not None:
        if hasattr(cfgs.augmentation, 'jitter') and cfgs.augmentation.jitter is not None:
            transforms_list.append(ColorJitter(cfgs.augmentation.jitter))
            aug_info += '  Add ColorJitter with level {}\n'.format(cfgs.augmentation.jitter)

    transforms_list.append(ImgNorm(norm_by_255=True))
    aug_info += '  Add ImgNorm'
    transforms_list.append(PermuteImg())
    aug_info += '  Add Permute'

    return transforms.Compose(transforms_list), aug_info

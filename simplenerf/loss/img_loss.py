# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import LOSS_REGISTRY
from common.utils.torch_utils import mean_tensor_by_mask


@LOSS_REGISTRY.register()
class ImgLoss(nn.Module):
    """loss for image. """

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                keys: key used for loss sum. By default 'rgb'.
                      'rgb_coarse'/'rgb_fine' for two stage network
                use_mask: use mask for average calculation. By default False.
                do_mean: calculate the mean of loss. By default True.
        """
        super(ImgLoss, self).__init__()
        self.keys = get_value_from_cfgs_field(cfgs, 'keys', ['rgb'])
        self.use_mask = get_value_from_cfgs_field(cfgs, 'use_mask', False)
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, data, output):
        """
        Args:
            output['rgb'/'rgb_fine'/'rgb_coarse']: (B, N_rays, 3). output rgb depends on keys
            data['img']: (B, N_rays, 3)
            data['mask']: (B, N_rays), only if used mask

        Returns:
            loss: (1, ) mean loss. RGB value in (0~1)
                   if not do_mean, return (B, N_rays, 3) loss
        """
        device = output[self.keys[0]].device
        gt = data['img'].to(device)
        if self.use_mask:
            mask = data['mask'].to(device)

        loss = 0.0
        for k in self.keys:
            loss += self.loss(output[k], gt)  # (B, N_rays, 3)

        if self.use_mask:
            loss = mean_tensor_by_mask(loss.mean(-1), mask)
        else:
            loss = loss.mean()

        return loss

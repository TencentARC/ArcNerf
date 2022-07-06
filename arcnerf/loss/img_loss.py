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
                loss_type: select loss type such as 'MSE'/'L1'. By default MSE
                internal_weights: If set, will multiply factors to each weights. By default None.
                use_mask: use mask for average calculation. By default False.
                do_mean: calculate the mean of loss. By default True.
        """
        super(ImgLoss, self).__init__()
        self.keys = get_value_from_cfgs_field(cfgs, 'keys', ['rgb'])
        self.loss = self.parse_loss(cfgs)
        self.internal_weights = get_value_from_cfgs_field(cfgs, 'internal_weights', None)
        self.use_mask = get_value_from_cfgs_field(cfgs, 'use_mask', False)
        self.do_mean = get_value_from_cfgs_field(cfgs, 'do_mean', True)

    @staticmethod
    def parse_loss(cfgs):
        loss_type = get_value_from_cfgs_field(cfgs, 'loss_type', 'MSE')
        if loss_type == 'MSE':
            loss = nn.MSELoss(reduction='none')
        elif loss_type == 'L1':
            loss = nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError('Loss type {} not support in img loss...'.format(loss_type))

        return loss

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
        for idx, k in enumerate(self.keys):
            if self.internal_weights is not None:
                loss += self.internal_weights[idx] * self.loss(output[k], gt)  # (B, N_rays, 3)
            else:
                loss += self.loss(output[k], gt)  # (B, N_rays, 3)

        if self.do_mean:  # (1,)
            if self.use_mask:
                loss = mean_tensor_by_mask(loss.mean(-1), mask)
            else:
                loss = loss.mean()

        return loss

# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskLoss(nn.Module):
    """loss for mask"""

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                keys: key used for loss sum. By default 'mask'.
                      'mask_coarse'/'mask_fine' for two stage network
                loss_type: select loss type such as 'MSE'/'L1'/'BCE'. By default MSE
        """
        super(MaskLoss, self).__init__()
        self.keys = get_value_from_cfgs_field(cfgs, 'keys', ['mask'])
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, data, output):
        """
        Args:
            output['mask'/'mask_fine'/'mask_coarse']: (B, N_rays). output mask depends on keys
            data['mask']: (B, N_rays)

        Returns:
            loss: (1, ) mean loss. error value in (0~1)
                    if not do_mean, return (B, N_rays) loss
        """
        device = output[self.keys[0]].device
        gt = data['mask'].to(device)

        loss = 0.0
        for k in self.keys:
            loss += self.loss(output[k], gt)  # (B, N_rays)

        loss = loss.mean()

        return loss

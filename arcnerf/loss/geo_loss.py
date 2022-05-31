# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import LOSS_REGISTRY
from common.utils.torch_utils import mean_tensor_by_mask


@LOSS_REGISTRY.register()
class EikonalLoss(nn.Module):
    """Eikonal loss, regularize the normal has norm 1"""

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                key: key used for loss sum. By default 'normal'.
                      'normal_pts' for pts
                loss_type: select loss type such as 'MSE'/'L1'. By default MSE
                use_mask: use mask for average calculation. By default False.
                do_mean: calculate the mean of loss. By default True.
        """
        super(EikonalLoss, self).__init__()
        self.key = get_value_from_cfgs_field(cfgs, 'key', 'normal')
        self.loss = self.parse_loss(cfgs)
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
            raise NotImplementedError('Loss type {} not support in geo loss...'.format(loss_type))

        return loss

    def forward(self, data, output):
        """
        Args:
            output['normal'/'normal_pts']: (B, N_rays, (N_pts), 3). normal output depends on keys
            data['mask']: (B, N_rays), only if used mask

        Returns:
            Eikonal: (1, ) mean Eikonal loss.
                     if not do_mean, return (B, N_rays, (N_pts)) loss
        """
        dtype = output[self.key].dtype
        device = output[self.key].device
        out = output[self.key]
        if self.use_mask:
            mask = data['mask'].to(device)  # (B, N_rays)

        norm = torch.norm(out, dim=-1)  # (B, N_rays, (N_pts))
        norm_ones = torch.ones_like(norm, dtype=dtype).to(device)

        loss = self.loss(norm, norm_ones)

        if self.do_mean:
            if self.use_mask:
                if len(loss.shape) == 3:  # expand for pts-dim
                    mask = torch.repeat_interleave(mask.unsqueeze(-1), loss.shape[-1], -1)  # (B, N_rays, N_pts)
                loss = mean_tensor_by_mask(loss, mask)
            else:
                loss = loss.mean()

        return loss

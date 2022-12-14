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
        norm_ones = torch.ones_like(norm, dtype=dtype, device=device)

        loss = self.loss(norm, norm_ones)

        if self.do_mean:
            if self.use_mask:
                if len(loss.shape) == 3:  # expand for pts-dim
                    mask = torch.repeat_interleave(mask.unsqueeze(-1), loss.shape[-1], -1)  # (B, N_rays, N_pts)
                loss = mean_tensor_by_mask(loss, mask)
            else:
                loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class RegMaskLoss(nn.Module):
    """Regularize the mask to make opacity 0/1"""

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                keys: key used for loss sum. By default 'mask'.
                do_mean: calculate the mean of loss. By default True.
        """
        super(RegMaskLoss, self).__init__()
        self.keys = get_value_from_cfgs_field(cfgs, 'keys', ['mask'])
        self.do_mean = get_value_from_cfgs_field(cfgs, 'do_mean', True)

    def forward(self, data, output):
        """
        Args:
            output['mask'/'mask_fine'/'mask_coarse']: (B, N_rays). output mask depends on keys

        Returns:
            reg mask loss: (1, ) mean loss.
                     if not do_mean, return (B, N_rays) loss
        """
        loss = 0.0
        for k in self.keys:
            loss += cal_nll_loss(output[k])  # (B, N_rays)

        if self.do_mean:
            loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class RegWeightsLoss(nn.Module):
    """Regularize the weights to make opacity 0/1"""

    def __init__(self, cfgs=None):
        """
        Args:
            cfgs: a obj with following attributes:
                key: key used for loss sum. By default 'weights'. All key add 'progress_'
                do_mean: calculate the mean of loss. By default True.
        """
        super(RegWeightsLoss, self).__init__()
        self.keys = get_value_from_cfgs_field(cfgs, 'keys', ['weights'])
        self.keys = ['progress_' + k for k in self.keys]
        self.do_mean = get_value_from_cfgs_field(cfgs, 'do_mean', True)

    def forward(self, data, output):
        """
        Args:
            output['progress_weights'/'progress_weights_fine'/'progress_weights_coarse']: (B, N_rays, N_pts).
                output weights depends on keys. pts on each weights

        Returns:
            reg weights loss: (1, ) mean loss.
                     if not do_mean, return (B, N_rays, N_pts) loss
        """
        loss = 0.0
        for k in self.keys:
            assert k in output.keys(), 'You must turn debug.get_progress=True for this loss...'
            loss += cal_nll_loss(output[k])  # (B, N_rays)

        if self.do_mean:
            loss = loss.mean()

        return loss


def cal_nll_loss(t, eps=1e-5):
    """neg-loglikehood loss = -o*log(o)"""
    loss = torch.zeros_like(t, dtype=t.dtype, device=t.device)
    zeros_mask = t < eps  # any neg and 0 value
    loss[~zeros_mask] = -t[~zeros_mask] * torch.log(t[~zeros_mask])

    return loss

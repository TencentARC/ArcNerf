# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskCFLoss(nn.Module):
    """MSE loss for mask and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(MaskCFLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, data, output):
        """
        Args:
            output['mask_coarse']: (B, N_rays). Coarse mask output
            output['mask_fine']: (B, N_rays), optional. Fine mask output
            data['mask']: (B, N_rays)

        Returns:
            loss: (1, ) mean loss. error value in (0~1)
        """
        device = output['mask_coarse'].device
        gt = data['mask'].to(device)

        loss = self.loss(output['mask_coarse'], gt)  # (B, N_rays)
        if 'mask_fine' in output:
            loss += self.loss(output['mask_fine'], gt)
        loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class MaskCFL1Loss(MaskCFLoss):
    """L1 loss for mask and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(MaskCFL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')


@LOSS_REGISTRY.register()
class MaskCFBCELoss(MaskCFLoss):
    """BCE loss for mask and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(MaskCFBCELoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')


@LOSS_REGISTRY.register()
class MaskLoss(nn.Module):
    """Simple MSE loss for Mask"""

    def __init__(self, cfgs=None):
        super(MaskLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, data, output):
        """
        Args:
            output['mask']: (B, N_rays). mask output
            data['mask']: (B, N_rays)

        Returns:
            loss: (1, ) mean loss. error value in (0~1)
        """
        device = output['mask'].device
        gt = data['mask'].to(device)

        loss = self.loss(output['mask'], gt)
        loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class MaskL1Loss(MaskLoss):
    """L1 loss for mask and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(MaskLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')


@LOSS_REGISTRY.register()
class MaskBCELoss(MaskLoss):
    """BCE loss for mask and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(MaskLoss, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

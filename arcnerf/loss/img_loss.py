# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY
from common.utils.torch_utils import mean_tensor_by_mask


@LOSS_REGISTRY.register()
class ImgCFLoss(nn.Module):
    """MSE loss for image and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(ImgCFLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.use_mask = False

    def forward(self, data, output):
        """
        Args:
            output['rgb_coarse']: (B, N_rays, 3). Coarse output
            output['rgb_fine']: (B, N_rays, 3), optional. Fine output
            data['img']: (B, N_rays, 3)
            data['mask']: (B, N_rays), only if used mask

        Returns:
            loss: (1, ) mean loss. RGB value in (0~1)
        """
        device = output['rgb_coarse'].device
        gt = data['img'].to(device)
        if self.use_mask:
            mask = data['mask'].to(device)

        loss = self.loss(output['rgb_coarse'], gt)  # (B, N_rays, 3)
        if 'rgb_fine' in output:
            loss += self.loss(output['rgb_fine'], gt)

        if self.use_mask:
            loss = mean_tensor_by_mask(loss.mean(-1), mask)
        else:
            loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class ImgCFL1Loss(ImgCFLoss):
    """L1 loss for image and coarse/fine output with mask. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(ImgCFL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')


@LOSS_REGISTRY.register()
class ImgCFMaskLoss(ImgCFLoss):
    """MSE loss for image and coarse/fine output with mask. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(ImgCFMaskLoss, self).__init__(cfgs)
        self.loss = nn.MSELoss(reduction='none')
        self.use_mask = True


@LOSS_REGISTRY.register()
class ImgCFMaskL1Loss(ImgCFLoss):
    """L1 loss for image and coarse/fine output with mask. Use for two stage network"""

    def __init__(self, cfgs=None):
        super(ImgCFMaskL1Loss, self).__init__(cfgs)
        self.loss = nn.L1Loss(reduction='none')
        self.use_mask = True


@LOSS_REGISTRY.register()
class ImgLoss(nn.Module):
    """Simple MSE loss for rgb"""

    def __init__(self, cfgs=None):
        super(ImgLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.use_mask = False

    def forward(self, data, output):
        """
        Args:
            output['rgb']: (B, N_rays, 3). img output
            data['img']: (B, N_rays, 3)
            data['mask']: (B, N_rays), only if used mask

        Returns:
            loss: (1, ) mean loss. RGB value in (0~1)
        """
        device = output['rgb'].device
        gt = data['img'].to(device)
        if self.use_mask:
            mask = data['mask'].to(device)

        loss = self.loss(output['rgb'], gt)
        if self.use_mask:
            loss = mean_tensor_by_mask(loss.mean(-1), mask)
        else:
            loss = loss.mean()

        return loss


@LOSS_REGISTRY.register()
class ImgL1Loss(ImgLoss):
    """Simple L1 loss for rgb"""

    def __init__(self, cfgs=None):
        super(ImgL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')


@LOSS_REGISTRY.register()
class ImgMaskLoss(ImgLoss):
    """Simple MSE loss for rgb with mask"""

    def __init__(self, cfgs=None):
        super(ImgMaskLoss, self).__init__(cfgs)
        self.loss = nn.MSELoss(reduction='none')
        self.use_mask = True


@LOSS_REGISTRY.register()
class ImgMaskL1Loss(ImgLoss):
    """Simple L1 loss for rgb with mask"""

    def __init__(self, cfgs=None):
        super(ImgMaskL1Loss, self).__init__(cfgs)
        self.loss = nn.L1Loss(reduction='none')
        self.use_mask = True

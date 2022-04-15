# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ImgCFLoss(nn.Module):
    """MSE loss for image and coarse/fine output. Use for two stage network"""

    def __init__(self, cfgs):
        super(ImgCFLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, data, output):
        """
        Args:
            output['rgb_coarse']: (B, N_rays, 3). Coarse output
            output['rgb_fine']: (B, N_rays, 3), optional. Fine output
            data['img']: (B, N_rays, 3)

        Returns:
            loss: (1, ) mean loss. RGB value in (0~1)
        """
        device = output['rgb_coarse'].device
        gt = data['img'].to(device)

        loss = self.loss(output['rgb_coarse'], gt)
        if 'rgb_fine' in output:
            loss += self.loss(output['rgb_fine'], gt)

        return loss


@LOSS_REGISTRY.register()
class ImgLoss(nn.Module):
    """Simple MSE loss for rgb"""

    def __init__(self, cfgs):
        super(ImgLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, data, output):
        """
        Args:
            output['rgb']: (B, N_rays, 3). Coarse output
            output['rgb']: (B, N_rays, 3), optional. Fine output
            data['img']: (B, N_rays, 3)

        Returns:
            loss: (1, ) mean loss. RGB value in (0~1)
        """
        device = output['rgb'].device
        gt = data['img'].to(device)

        loss = self.loss(output['rgb'], gt)

        return loss

# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class L2Loss(nn.Module):
    """Simple Img loss for comparing input and gt"""

    def __init__(self, cfgs):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, data, output):
        device = output['img'].device
        gt = data['gt'].to(device)

        return self.loss(output['img'], gt)

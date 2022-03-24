# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RegLoss(nn.Module):
    """Simple Reg loss for input"""

    def __init__(self, cfgs):
        super(RegLoss, self).__init__()

    def forward(self, data, output):
        out = output['img']
        reduce_dim = tuple(range(1, len(out.shape)))
        loss = torch.sum(out, dim=reduce_dim)
        loss = torch.sum(loss) / out.shape[0]

        return loss

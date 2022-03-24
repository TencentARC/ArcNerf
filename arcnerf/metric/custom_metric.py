# -*- coding: utf-8 -*-

import torch.nn as nn

from common.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class M1(nn.Module):
    """Simple M1 Metric for comparing input and gt"""

    def __init__(self, cfgs):
        super(M1, self).__init__()
        self.M1 = nn.MSELoss()

    def forward(self, data, output):
        device = output['img'].device
        gt = data['gt'].to(device)

        return self.M1(output['img'], gt)


@METRIC_REGISTRY.register()
class M2(nn.Module):
    """Simple M2 Metric for comparing input and gt"""

    def __init__(self, cfgs):
        super(M2, self).__init__()
        self.M2 = nn.MSELoss()

    def forward(self, data, output):
        device = output['img'].device
        gt = data['gt'].to(device)

        return self.M2(output['img'], gt)

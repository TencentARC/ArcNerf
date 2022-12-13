# -*- coding: utf-8 -*-

import torch.nn as nn


class BaseModel(nn.Module):
    """Base model class"""

    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

    def forward(self, x):
        raise NotImplementedError('Please implement the forward func...')

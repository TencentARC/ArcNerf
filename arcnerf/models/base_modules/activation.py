# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from arcnerf.ops.trunc_exp import TruncExp
from common.utils.cfgs_utils import get_value_from_cfgs_field, dict_to_obj


class Sine(nn.Module):

    def __init__(self, w0=30.0):
        """Sine activation function
        Ref: "Implicit Neural Representations with Periodic Activation Functions"
              https://github.com/vsitzmann/siren
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w0 * x)


def get_activation(cfg, default=dict_to_obj({'type': 'relu'})):
    """Get activation from cfg. Type is specified by cfg.type"""
    if cfg is None:
        cfg = default

    if cfg.type.lower() == 'relu':
        act = nn.ReLU(inplace=True)
    elif cfg.type.lower() == 'softplus':
        beta = get_value_from_cfgs_field(cfg, 'beta', 100)
        act = nn.Softplus(beta=beta)
    elif cfg.type.lower() == 'leakyrelu':
        slope = get_value_from_cfgs_field(cfg, 'slope', 0.01)
        act = nn.LeakyReLU(negative_slope=slope, inplace=True)
    elif cfg.type.lower() == 'sine':
        w = get_value_from_cfgs_field(cfg, 'w', 30)
        act = Sine(w0=w)
    elif cfg.type.lower() == 'sigmoid':
        act = nn.Sigmoid()
    elif cfg.type.lower() == 'truncexp':
        clip = get_value_from_cfgs_field(cfg, 'clip', 15.0)
        act = TruncExp(clip)
    else:
        raise NotImplementedError('No activation class {}'.format(cfg.type))

    return act

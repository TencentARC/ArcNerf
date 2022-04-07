# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Sine(nn.Module):

    def __init__(self, w0=30.0):
        """Sine activation function
        Ref: "Implicit Neural Representations with Periodic Activation Functions"
              https://github.com/vsitzmann/siren
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.FloatTensor):
        return torch.sin(self.w0 * x)


def get_activation(cfg):
    """Get activation from cfg. Type is specified by cfg.type"""
    if cfg is None:
        return nn.ReLU(inplace=True)

    if cfg.type.lower() == 'relu':
        act = nn.ReLU(inplace=True)
    elif cfg.type.lower() == 'softplus':
        beta = cfg.beta if hasattr(cfg, 'beta') else 100
        act = nn.Softplus(beta=beta)
    elif cfg.type.lower() == 'leakyrelu':
        slope = cfg.slope if hasattr(cfg, 'slope') else 0.01
        act = nn.LeakyReLU(negative_slope=slope, inplace=True)
    elif cfg.type.lower() == 'sine':
        w = cfg.w if hasattr(cfg, 'w') else 30
        act = Sine(w0=w)
    elif cfg.type.lower() == 'sigmoid':
        act = nn.Sigmoid()
    else:
        raise NotImplementedError('No activation class {}'.format(cfg.type))

    return act

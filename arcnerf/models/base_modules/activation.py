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

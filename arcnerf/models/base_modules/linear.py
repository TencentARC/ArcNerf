# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from .activation import Sine


class DenseLayer(nn.Linear):
    """Dense Layer(Linear) with activation"""

    def __init__(self, input_dim, out_dim, activation=nn.ReLU(inplace=True), bias=True):
        """
        Args:
            input_dim: input dim
            out_dim: output dim
            activation: activation function.
                    By default use ReLU. Others can be (LeakyReLU, Sigmoid, Tanh, Softplus) etc
            bias: Whether to use bias. By default True
        Returns:
            out: (B, out_dim) tensor
        """
        super().__init__(input_dim, out_dim, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        out = self.activation(out)

        return out


class SirenLayer(nn.Linear):
    """Dense Layer with sine activation
       Ref: "Implicit Neural Representations with Periodic Activation Functions"
             https://github.com/vsitzmann/siren
             https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(self, input_dim, out_dim, is_first=False, bias=True):
        """
        Args:
            input_dim: input dim
            out_dim: output dim
            is_first: bool. If first layer, use 1/input_dim for weight init,
                            else use sqrt(c/input_dim) / w0 for init
            bias: Whether to use bias. By default True
        Returns:
            out: (B, out_dim) tensor
        """
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, bias=bias)
        self.activation = Sine(self.w0)

    def reset_parameters(self):
        super().reset_parameters()  # init bias
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        out = self.activation(out)

        return out

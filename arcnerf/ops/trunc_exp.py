# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class TruncExpOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, x, clip):
        """
        Args:
            x: input tensor
            clip: clip for backward

        Returns:
             output: output tensor with the same size
        """
        ctx.save_for_backward(x)
        ctx.clip = clip

        return torch.exp(x)

    @staticmethod
    def backward(ctx, grad):
        """
        Args:
            grad: grad on output tensor

        Returns:
             grad_x: grad on input tensor
        """
        grad = grad.contiguous()  # make it contiguous
        grad_x = grad * torch.exp(ctx.saved_tensors[0].clamp(-ctx.clip, ctx.clip))

        return grad_x, None


class TruncExp(nn.Module):
    """A torch.nn class that use the TruncExp function"""

    def __init__(self, clip=15.0):
        """
        Args:
            clip: clip value for backward. By default 15.0
        """
        super(TruncExp, self).__init__()
        self.clip = clip

    def forward(self, x):
        """
        Args:
            x: any tensor

        Returns:
             output: output tensor with the same shape
        """

        return TruncExpOps.apply(x, self.clip)

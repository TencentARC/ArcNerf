# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import _sh_encode


class SHEncodeOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, xyz, degree):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz direction, normalized

        Returns:
             output: torch tensor with (B, degree**2) shape
        """
        xyz = xyz.contiguous()  # make it contiguous
        output = _sh_encode.sh_encode_forward(xyz, degree)
        ctx.save_for_backward(xyz)
        ctx.degree = degree

        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Args:
            grad: torch tensor with (B, degree**2) shape, the grad on output

        Returns:
             grad_xyz: tensor of shape (B, 3), grad on the input xyz
        """
        grad = grad.contiguous()  # make it contiguous
        grad_xyz = _sh_encode.sh_encode_backward(grad, *ctx.saved_tensors, ctx.degree)

        return grad_xyz, None


class SHEncode(nn.Module):
    """A torch.nn class that use the SHEncode function"""

    def __init__(self, degree):
        """
        Args:
            degree: num of degree to expand, generally between 1~5
        """
        super(SHEncode, self).__init__()
        self.degree = degree

    def forward(self, xyz):
        """
        Args:
            xyz: tensor of shape (B, 3), xyz direction, normalized

        Returns:
             output: torch tensor with (B, degree**2) shape
        """
        assert xyz.shape[-1] == 3, 'Must be (B, 3)'

        return SHEncodeOps.apply(xyz, self.degree)

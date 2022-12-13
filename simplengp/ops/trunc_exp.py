# -*- coding: utf-8 -*-

import torch


class TruncExpOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, x, clip):
        ctx.save_for_backward(x)
        ctx.clip = clip

        return torch.exp(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()  # make it contiguous
        grad_x = grad * torch.exp(ctx.saved_tensors[0].clamp(-ctx.clip, ctx.clip))

        return grad_x, None


def trunc_exp(x, clip=15.0):
    return TruncExpOps.apply(x, clip)

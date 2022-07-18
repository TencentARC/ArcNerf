# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import _hashgrid_encode


class HashGridEncodeOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, xyz, embeddings, n_levels, n_feat_per_entry, offsets, resolutions, min_xyz, max_xyz):
        """
        Args:
            xyz: tensor of shape (B, D), xyz position
            embeddings: embeddings vec in shape (n_total_embed, F)
            n_levels: num of levels of embeddings(L), by default 16
            n_feat_per_entry: num of feat for each entry in hashmap(F), by default 2
            offsets: a list of offset of each level, len is L+1
            resolutions: a list of resolution at each level, len is L
            min_xyz: a list of D, the min_xyz position of the grid
            max_xyz: a list of D, the max_xyz position of the grid

        Returns:
            output: embed xyz with (B, L*F) shape
        """
        dtype = xyz.dtype
        device = xyz.device

        # keep dimension
        ctx.B = xyz.shape[0]
        ctx.L = n_levels
        ctx.F = n_feat_per_entry

        xyz = xyz.contiguous()  # make it contiguous
        embeddings = embeddings.contiguous()  # make it contiguous
        # change list to tensor
        _offsets = torch.tensor(offsets, dtype=torch.int).to(device)  # (L+1, )
        _resolutions = torch.tensor(resolutions, dtype=torch.int).to(device)  # (L, )
        _min_xyz = torch.tensor(min_xyz, dtype=dtype).to(device)  # (D, )
        _max_xyz = torch.tensor(max_xyz, dtype=dtype).to(device)  # (D, )

        # forward
        output, weights, hash_idx, valid = _hashgrid_encode.hashgrid_encode_forward(
            xyz, embeddings, n_levels, n_feat_per_entry, _offsets, _resolutions, _min_xyz, _max_xyz
        )

        ctx.save_for_backward(xyz, embeddings, weights, hash_idx, valid)

        # reshape output from (B, L, F) -> (B, L*F)
        output = output.view(xyz.shape[0], -1)  # (B, L*F)

        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Args:
            grad: tensor in (B, L*F) shape, the grad on final output

        Returns:
            grad_xyz: tensor of shape (B, D), grad on xyz position
            grad_embeddings: tensor in (n_total_embed, F), grad on embeddings vec
        """
        grad = grad.contiguous().view(ctx.B, ctx.L, ctx.F)  # make it contiguous, change to (B, L, F)
        grad_xyz, grad_embeddings = _hashgrid_encode.hashgrid_encode_backward(grad, *ctx.saved_tensors)

        return grad_xyz, grad_embeddings, None, None, None, None, None, None


class HashGridEncode(nn.Module):
    """A torch.nn class that use the HashGridEncode function"""

    def __init__(self, n_levels, n_feat_per_entry, offsets, resolutions):
        """
        Args:
            n_levels: num of levels of embeddings(L), by default 16
            n_feat_per_entry: num of feat for each entry in hashmap(F), by default 2
            offsets: a list of offset of each level, len is L+1
            resolutions: a list of resolution at each level, len is L
        """
        super(HashGridEncode, self).__init__()
        self.n_levels = n_levels
        self.n_feat_per_entry = n_feat_per_entry
        self.offsets = offsets
        self.resolutions = resolutions

    def forward(self, xyz, embeddings, min_xyz, max_xyz):
        """
        Args:
            xyz: tensor of shape (B, D), xyz position
            embeddings: embeddings vec in shape (n_total_embed, F)
            min_xyz: a list of D, the min_xyz position of the grid
            max_xyz: a list of D, the max_xyz position of the grid

        Returns:
             output: torch tensor with (B, L*F) shape
        """
        assert len(min_xyz) == xyz.shape[1] and len(max_xyz) == xyz.shape[1], 'Incorrect boundary size'
        assert embeddings.shape == (self.offsets[-1], self.n_feat_per_entry), 'embeddings must be (n_total_embed, F)'

        return HashGridEncodeOps.apply(
            xyz, embeddings, self.n_levels, self.n_feat_per_entry, self.offsets, self.resolutions, min_xyz, max_xyz
        )

# -*- coding: utf-8 -*-

import warnings

import torch

try:
    import _volume_func
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False
    warnings.warn('Volume_func Ops not build...')


class CheckOccOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, xyz, bitfield, range, n_grid):
        output = _volume_func.check_pts_in_occ_voxel(xyz, bitfield, range, n_grid)

        return output


@torch.no_grad()
def check_pts_in_occ_voxel_cuda(xyz, bitfield, xyz_range, n_grid):
    """Check whether voxel_idx are the same as occ voxel_idx

    Args:
        xyz: (B, 3), pts position
        bitfield: (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
        xyz_range: (3, 2) min/max xyz position
        n_grid: resolution of the volume

    Return:
        pts_in_occ_voxel: (B, ) bool tensor whether each pts is in occupied voxels
    """
    return CheckOccOps.apply(xyz, bitfield, xyz_range, n_grid)

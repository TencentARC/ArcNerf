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


class AABBOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, rays_o, rays_d, aabb_range, eps):
        near, far, pts, mask = _volume_func.aabb_intersection(rays_o, rays_d, aabb_range, eps)

        return near, far, pts, mask


@torch.no_grad()
def ray_aabb_intersection_cuda(rays_o, rays_d, aabb_range, eps=1e-7):
    """Ray aabb intersection with volume range

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        aabb_range: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
        eps: error threshold for parallel comparison, by default 1e-7

    Return:
        near: near intersection zvals. (N_rays, N_v)
        far:  far intersection zvals. (N_rays, N_v)
        pts: (N_rays, N_v, 2, 3), each ray has near/far two points with each volume.
        mask: (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
    """
    return AABBOps.apply(rays_o, rays_d, aabb_range, eps)

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
    def forward(ctx, xyz, bitfield, aabb_range, n_grid):
        n_pts = xyz.shape[0]
        aabb_range = torch.permute(aabb_range, (1, 0)).contiguous()  # (2, 3)

        output = torch.zeros((n_pts, ), dtype=torch.bool, device=xyz.device)

        _volume_func.check_pts_in_occ_voxel(xyz, bitfield, aabb_range, n_grid, output)

        return output


@torch.no_grad()
def check_pts_in_occ_voxel_cuda(xyz, bitfield, aabb_range, n_grid):
    """Check whether voxel_idx are the same as occ voxel_idx

    Args:
        xyz: (B, 3), pts position
        bitfield: (N_grid, N_grid, N_grid), bool tensor indicating each voxel's occupancy
        aabb_range: (3, 2) min/max xyz position
        n_grid: resolution of the volume

    Return:
        pts_in_occ_voxel: (B, ) bool tensor whether each pts is in occupied voxels
    """
    return CheckOccOps.apply(xyz, bitfield, aabb_range, n_grid)


# ------------------------------------------------------------------------------------------------ #


class AABBOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, rays_o, rays_d, aabb_range):
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
        aabb_range = torch.permute(aabb_range, (0, 2, 1)).contiguous()  # (N_v, 2, 3)

        n_rays = rays_o.shape[0]
        n_v = aabb_range.shape[0]
        near = torch.zeros((n_rays, n_v), dtype=rays_o.dtype, device=rays_o.device)
        far = torch.zeros((n_rays, n_v), dtype=rays_o.dtype, device=rays_o.device)
        pts = torch.zeros((n_rays, n_v, 2, 3), dtype=rays_o.dtype, device=rays_o.device)
        mask = torch.zeros((n_rays, n_v), dtype=torch.bool, device=rays_o.device)

        _volume_func.aabb_intersection(rays_o, rays_d, aabb_range, near, far, pts, mask)
        return near, far, pts, mask


@torch.no_grad()
def ray_aabb_intersection_cuda(rays_o, rays_d, aabb_range):
    """Ray aabb intersection with volume range

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        aabb_range: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume

    Return:
        near: near intersection zvals. (N_rays, N_v)
        far:  far intersection zvals. (N_rays, N_v)
        pts: (N_rays, N_v, 2, 3), each ray has near/far two points with each volume.
        mask: (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
    """
    return AABBOps.apply(rays_o, rays_d, aabb_range)


# ------------------------------------------------------------------------------------------------ #


class SparseVolumeSampleOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance):
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
        n_rays = rays_o.shape[0]
        aabb_range = torch.permute(aabb_range, (1, 0)).contiguous()  # (2, 3)

        zvals = torch.zeros((n_rays, n_pts), dtype=rays_o.dtype, device=rays_o.device)
        mask = torch.zeros((n_rays, n_pts), dtype=torch.bool, device=rays_o.device)

        _volume_func.sparse_volume_sampling(
            rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance, zvals, mask
        )

        return zvals, mask


@torch.no_grad()
def sparse_volume_sampling(rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance=0.0):
    """Sample pts in sparse volume. The output is a compact tensor

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        near: (N_rays, 1) near distance for each ray
        far: (N_rays, 1) far distance for each ray
        n_pts: max num of sampling pts on each ray,
        dt: const dt for searching
        aabb_range: (3, 2) bounding box range
        n_grid: resolution
        bitfield: bitfield in (n_grid, n_grid, n_grid) bool tensor
        near_distance: near distance for sampling. By default 0.0.


    Return:
        zvals: (N_rays, N_pts), sampled points zvals on each rays. At mose n_pts for each ray,
                but generally it only samples <100, pts in 128*3 sparse volume.
                Remaining zvals will be the same as last zval and masked as False.
        mask: (N_rays, N_pts), show whether each pts is valid in the rays
    """
    return SparseVolumeSampleOps.apply(
        rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance
    )


# ------------------------------------------------------------------------------------------------ #


class ReduceMaxOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, full_tensor, idx, n_group):
        full_tensor = full_tensor.contiguous()  # make it contiguous
        idx = idx.contiguous()  # make it contiguous

        uni_tensor = torch.zeros((n_group, ), dtype=full_tensor.dtype, device=full_tensor.device)

        _volume_func.tensor_reduce_max(full_tensor, idx, n_group, uni_tensor)

        return uni_tensor


@torch.no_grad()
def tensor_reduce_max(full_tensor, idx, n_group):
    """Get the max by index group

    Args:
        full_tensor: full value tensor, (N, )
        idx: index of each row (N, )
        n_group: num of group (N_uni)

    Return:
        uni_tensor: (N_uni. ) maximum of each unique group
    """
    return ReduceMaxOps.apply(full_tensor, idx, n_group)

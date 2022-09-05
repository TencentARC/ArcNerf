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
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
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


class SparseVolumeSampleOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance, perturb):
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
        zvals, mask = _volume_func.sparse_volume_sampling(
            rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance, perturb
        )

        return zvals, mask


@torch.no_grad()
def sparse_volume_sampling(
    rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance=0.0, perturb=False
):
    """Sample pts in sparse volume. The output is a compact tensor

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        near: (N_rays, 1) near distance for each ray
        near: (N_rays, 1) far distance for each ray
        n_pts: max num of sampling pts on each ray,
        dt: const dt for searching
        near_distance: near distance for sampling. By default 0.0.
        perturb: whether to perturb the first zval, use in training only. by default False

    Return:
        zvals: (N_rays, N_pts), sampled points zvals on each rays. At mose n_pts for each ray,
                but generally it only samples <100, pts in 128*3 sparse volume.
                Remaining zvals will be the same as last zval and masked as False.
        mask: (N_rays, N_pts), show whether each pts is valid in the rays
    """
    return SparseVolumeSampleOps.apply(
        rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance, perturb
    )


class ReduceMaxOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, full_tensor, idx, n_group):
        full_tensor = full_tensor.contiguous()  # make it contiguous
        idx = idx.contiguous()  # make it contiguous
        uni_tensor = _volume_func.tensor_reduce_max(full_tensor, idx, n_group)

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

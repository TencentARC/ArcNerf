# -*- coding: utf-8 -*-

import warnings

import torch

try:
    import _bitfield_func
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False
    warnings.warn('bitfield_func Ops not build...')

# ------------------------------------------------------------------------------------------------ #


class SparseVolumeSampleBitOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance):
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
        n_rays = rays_o.shape[0]
        aabb_range = torch.permute(aabb_range, (1, 0)).contiguous()  # (2, 3)

        zvals = torch.zeros((n_rays, n_pts), dtype=rays_o.dtype, device=rays_o.device)
        mask = torch.zeros((n_rays, n_pts), dtype=torch.bool, device=rays_o.device)

        _bitfield_func.sparse_volume_sampling_bit(
            rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance, zvals, mask
        )

        return zvals, mask


@torch.no_grad()
def sparse_volume_sampling_bit(rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance=0.0):
    """Sample pts in sparse volume with bitfield representation. The output is a compact tensor

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        near: (N_rays, 1) near distance for each ray
        far: (N_rays, 1) far distance for each ray
        n_pts: max num of sampling pts on each ray,
        dt: const dt for searching
        aabb_range: (3, 2) bounding box range
        n_grid: resolution
        bitfield: bitfield in (n_grid**3) uint8 tensor
        near_distance: near distance for sampling. By default 0.0.

    Return:
        zvals: (N_rays, N_pts), sampled points zvals on each rays. At mose n_pts for each ray,
                but generally it only samples <100, pts in 128*3 sparse volume.
                Remaining zvals will be the same as last zval and masked as False.
        mask: (N_rays, N_pts), show whether each pts is valid in the rays
    """
    return SparseVolumeSampleBitOps.apply(
        rays_o, rays_d, near, far, n_pts, dt, aabb_range, n_grid, bitfield, near_distance
    )


# -------------------------------------------------- ------------------------------------ #


class GenerateGridSamples(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid, n_elements, density_grid_ema_step, n_grid, thresh):
        device = density_grid.device
        positions_uniform = torch.empty((n_elements, 3), dtype=density_grid.dtype, device=device)
        indices_uniform = torch.empty((n_elements, ), dtype=torch.int32, device=device)

        _bitfield_func.generate_grid_samples(
            density_grid, density_grid_ema_step, n_elements, n_grid, thresh, positions_uniform, indices_uniform
        )

        return positions_uniform, indices_uniform


@torch.no_grad()
def generate_grid_samples(density_grid, n_elements, density_grid_ema_step, thresh, n_grid):
    """Generate grid samples in each voxel. The function generates inputs in [0, 1) range"""
    return GenerateGridSamples.apply(density_grid, n_elements, density_grid_ema_step, thresh, n_grid)


# -------------------------------------------------- ------------------------------------ #


class SplatGridSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density, density_grid_indices, n_samples, density_grid_tmp):

        _bitfield_func.splat_grid_samples(density, density_grid_indices, n_samples, density_grid_tmp)

        return density_grid_tmp


@torch.no_grad()
def splat_grid_samples(density, density_grid_indices, n_samples, density_grid_tmp):
    """Update the max density value for each voxel grid"""
    return SplatGridSample.apply(density, density_grid_indices, n_samples, density_grid_tmp)


# -------------------------------------------------- ------------------------------------ #


class EmaGridSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid_tmp, density_grid, n_elements, decay):
        _bitfield_func.ema_grid_samples_nerf(density_grid_tmp, n_elements, decay, density_grid)

        return density_grid


@torch.no_grad()
def ema_grid_samples_nerf(density_grid_tmp, density_grid, n_elements, decay):
    """Update density_grid by sample density in ema way"""
    return EmaGridSample.apply(density_grid_tmp, density_grid, n_elements, decay)


# -------------------------------------------------- ------------------------------------ #


class UpdateBitfield(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid):
        _bitfield_func.update_bitfield(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid)

        return density_grid_bitfield


@torch.no_grad()
def update_bitfield(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid):
    """Update density bitfield by density value"""
    return UpdateBitfield.apply(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid)


# -------------------------------------------------- ------------------------------------ #


class CountBitfield(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid_bitfield, n_grid):
        counter = torch.zeros((1, ), dtype=torch.float32, device=density_grid_bitfield.device)
        _bitfield_func.count_bitfield(density_grid_bitfield, counter, n_grid)

        return float(counter[0].item())


@torch.no_grad()
def count_bitfield(density_grid_bitfield, n_grid):
    """Count the num of occ bit in volume"""
    return CountBitfield.apply(density_grid_bitfield, n_grid)

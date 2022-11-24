# -*- coding: utf-8 -*-

import warnings

import torch

try:
    import _multivol_func
    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False
    warnings.warn('Multivol Ops not build...')

# ------------------------------------------------------------------------------------------------ #


class SparseVolumeSampleOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(
        ctx, rays_o, rays_d, near, far, n_pts, cone_angle, min_step, max_step, min_aabb_range, aabb_range, n_grid,
        n_cascade, bitfield, near_distance, inclusive
    ):
        rays_o = rays_o.contiguous()  # make it contiguous
        rays_d = rays_d.contiguous()  # make it contiguous
        n_rays = rays_o.shape[0]
        min_aabb_range = torch.permute(min_aabb_range, (1, 0)).contiguous()  # (2, 3)
        aabb_range = torch.permute(aabb_range, (1, 0)).contiguous()  # (2, 3)

        zvals = torch.zeros((n_rays, n_pts), dtype=rays_o.dtype, device=rays_o.device)
        mask = torch.zeros((n_rays, n_pts), dtype=torch.bool, device=rays_o.device)

        _multivol_func.sparse_sampling_in_multivol_bitfield(
            rays_o, rays_d, near, far, n_pts, cone_angle, min_step, max_step, min_aabb_range, aabb_range, n_grid,
            n_cascade, bitfield, near_distance, inclusive, zvals, mask
        )

        return zvals, mask


@torch.no_grad()
def sparse_sampling_in_multivol_bitfield(
    rays_o,
    rays_d,
    near,
    far,
    n_pts,
    cone_angle,
    min_step,
    max_step,
    min_aabb_range,
    aabb_range,
    n_grid,
    n_cascade,
    bitfield,
    near_distance=0.0,
    inclusive=False
):
    """Sample pts in sparse volume. The output is a compact tensor

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        near: (N_rays, 1) near distance for each ray
        far: (N_rays, 1) far distance for each ray
        n_pts: max num of sampling pts on each ray,
        cone_angle: for mip stepping sampling. 0 means const dt
        min_step: min stepping distance
        max_step: max stepping distance
        min_aabb_range: (3, 2) bbox range for inner volume
        aabb_range: (3, 2) bounding box range
        n_grid: resolution
        n_cascade: cascade level
        bitfield: bitfield in (n_grid, n_grid, n_grid) bool tensor
        near_distance: near distance for sampling. By default 0.0.
        inclusive: If True, include the inner volume. Else, only sample pts after passing it.
                  True serves for full-scene sampling, False for bkg only. By default False.

    Return:
        zvals: (N_rays, N_pts), sampled points zvals on each rays. At mose n_pts for each ray,
                but generally it only samples <100, pts in 128*3 sparse volume.
                Remaining zvals will be the same as last zval and masked as False.
        mask: (N_rays, N_pts), show whether each pts is valid in the rays
    """
    return SparseVolumeSampleOps.apply(
        rays_o, rays_d, near, far, n_pts, cone_angle, min_step, max_step, min_aabb_range, aabb_range, n_grid, n_cascade,
        bitfield, near_distance, inclusive
    )


# -------------------------------------------------- ------------------------------------ #


class GenerateGridSamples(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid, n_elements, aabb_range, density_grid_ema_step, n_cascade, n_grid, thresh, inclusive):
        device = density_grid.device
        positions_uniform = torch.empty((n_elements, 3), dtype=density_grid.dtype, device=device)
        indices_uniform = torch.empty((n_elements, ), dtype=torch.int32, device=device)
        aabb_range = torch.permute(aabb_range, (1, 0)).contiguous()  # (2, 3)

        _multivol_func.generate_grid_samples_multivol(
            density_grid, density_grid_ema_step, n_elements, aabb_range, n_cascade, n_grid, thresh, inclusive,
            positions_uniform, indices_uniform
        )

        return positions_uniform, indices_uniform


@torch.no_grad()
def generate_grid_samples_multivol(
    density_grid, n_elements, aabb_range, density_grid_ema_step, n_cascade, n_grid, thresh, inclusive
):
    """Generate grid samples in each voxel for multivol excluding the inner one"""
    return GenerateGridSamples.apply(
        density_grid, n_elements, aabb_range, density_grid_ema_step, n_cascade, n_grid, thresh, inclusive
    )


# -------------------------------------------------- ------------------------------------ #


class UpdateBitfield(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascade, inclusive):
        _multivol_func.update_bitfield_multivol(
            density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascade, inclusive
        )

        return density_grid_bitfield


@torch.no_grad()
def update_bitfield_multivol(
    density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascade, inclusive
):
    """Update density bitfield by density value"""
    return UpdateBitfield.apply(
        density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascade, inclusive
    )

# -*- coding: utf-8 -*-

import torch
try:
    import _dense_grid
except ImportError:
    raise NotImplementedError("You have not build the customized ops...run `sh scripts/install_ops.sh`...")


class EmaGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density_grid_tmp, density_grid, n_elements, decay):
        _dense_grid.ema_grid_samples_nerf(density_grid_tmp, n_elements, decay, density_grid)

        return density_grid


@torch.no_grad()
def ema_grid_samples_nerf(density_grid_tmp, density_grid, n_elements, decay):
    """Update density_grid by sample density in ema way"""
    return EmaGridSample.apply(density_grid_tmp, density_grid, n_elements, decay)

# -------------------------------------------------- ------------------------------------ #


class UpdateBitfield(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascades):
        _dense_grid.update_bitfield(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascades)

        return density_grid_bitfield


@torch.no_grad()
def update_bitfield(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascades):
    """Update density bitfield by density value"""
    return UpdateBitfield.apply(density_grid, density_grid_mean, density_grid_bitfield, thres, n_grid, n_cascades)

# -------------------------------------------------- ------------------------------------ #


class SplatGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, density_grid_indices, n_samples, dt, density_grid_tmp):

        _dense_grid.splat_grid_samples(
            density, density_grid_indices, n_samples, dt, density_grid_tmp
        )

        return density_grid_tmp


@torch.no_grad()
def splat_grid_samples(density, density_grid_indices, n_samples, dt, density_grid_tmp):
    """Update the max density value for each voxel grid"""
    return SplatGridSample.apply(density, density_grid_indices, n_samples, dt, density_grid_tmp)

# -------------------------------------------------- ------------------------------------ #


class GenerateGridSamples(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density_grid, n_elements, density_grid_ema_step, max_cascade, n_grid, thresh):
        device = density_grid.device
        positions_uniform = torch.empty((n_elements, 3), dtype=density_grid.dtype, device=device)
        indices_uniform = torch.empty((n_elements, ), dtype=torch.int32, device=device)

        _dense_grid.generate_grid_samples(
            density_grid, density_grid_ema_step, n_elements,
            max_cascade, n_grid, thresh, positions_uniform, indices_uniform
        )

        return positions_uniform, indices_uniform


@torch.no_grad()
def generate_grid_samples(density_grid, n_elements, density_grid_ema_step, max_cascade, thresh, n_grid):
    """Generate grid samples in each voxel"""
    return GenerateGridSamples.apply(
        density_grid, n_elements, density_grid_ema_step, max_cascade, thresh, n_grid
    )


# -------------------------------------------------- ------------------------------------ #


class CountBitfield(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density_grid_bitfield, n_grid, level):
        counter = torch.zeros((1,), dtype=torch.float32, device=density_grid_bitfield.device)
        _dense_grid.count_bitfield(density_grid_bitfield, counter, n_grid, level)

        return float(counter[0].item())


@torch.no_grad()
def count_bitfield(density_grid_bitfield, n_grid, level):
    """Count the num of occ bit in volume"""
    return CountBitfield.apply(density_grid_bitfield, n_grid, level)


# -------------------------------------------------- ------------------------------------ #


class GetOccPC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density_grid_bitfield, n_grid):
        counter = torch.zeros((1,), dtype=torch.float32, device=density_grid_bitfield.device)
        pc = torch.zeros((n_grid**3, 3), dtype=torch.float32, device=density_grid_bitfield.device)

        _dense_grid.get_occ_pc(density_grid_bitfield, pc, counter, n_grid)

        max_pc = int(counter[0].item())

        return pc[:max_pc]


@torch.no_grad()
def get_occ_pc(density_grid_bitfield, n_grid):
    """It only gets the most inner level"""
    return GetOccPC.apply(density_grid_bitfield, n_grid)

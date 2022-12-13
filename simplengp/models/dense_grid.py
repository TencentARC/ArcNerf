# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.torch_utils import chunk_processing
from simplengp.ops import (
    count_bitfield,
    generate_grid_samples,
    get_occ_pc,
    splat_grid_samples,
    ema_grid_samples_nerf,
    update_bitfield
)


class DenseGrid(nn.Module):
    """A dense grid with bitfield. """

    def __init__(self, cfgs, chunk_pts, n_sample):
        super(DenseGrid, self).__init__()

        self.dtype = torch.float32

        # cfgs
        self.n_grid = get_value_from_cfgs_field(cfgs, 'n_grid', 128)
        self.n_cascades = get_value_from_cfgs_field(cfgs, 'n_cascades', 8)
        self.epoch_optim = get_value_from_cfgs_field(cfgs, 'epoch_optim', 16)
        self.epoch_optim_warmup = get_value_from_cfgs_field(cfgs, 'epoch_optim_warmup', 256)
        self.opa_thres = get_value_from_cfgs_field(cfgs, 'opa_thres', 0.01)
        self.decay = get_value_from_cfgs_field(cfgs, 'decay', 0.95)

        # range related cfgs
        self.aabb_scale = get_value_from_cfgs_field(cfgs, 'aabb_scale', None)
        if self.aabb_scale is None:
            self.aabb_scale = int(2 ** (self.n_cascades - 1))

        self.max_cascade = 0
        while (1 << self.max_cascade) < self.aabb_scale:
            self.max_cascade += 1

        # aabb_range related to aabb_scale
        self.aabb_range = get_value_from_cfgs_field(cfgs, 'aabb_range', None)
        if self.aabb_range is None:
            self.aabb_range = [-self.aabb_scale / 2.0 + 0.5, self.aabb_scale / 2.0 + 0.5]

        self.chunk_pts = chunk_pts
        self.n_sample = n_sample

        # counters
        self.n_elements_per_level = self.n_grid ** 3  # num of element in each level
        self.n_elements = self.n_cascades * self.n_elements_per_level  # total num of grid

        density_bitfield = (torch.ones((self.n_elements // 8,), dtype=torch.uint8) * 255).type(torch.uint8)  # all 1
        self.register_buffer('density_bitfield', density_bitfield)

        density_grid = torch.zeros((self.n_elements,), dtype=self.dtype)
        self.register_buffer('density_grid', density_grid)

        density_grid_tmp = torch.zeros((self.n_elements,), dtype=self.dtype)
        self.register_buffer('density_grid_tmp', density_grid_tmp)

        self.ema_step = 0

    def get_bitfield(self):
        """Get the bitfield"""
        return self.density_bitfield

    def get_n_grid(self):
        """Get resolution"""
        return self.n_grid

    def get_n_cascades(self):
        """Get n cascades level"""
        return self.n_cascades

    def get_n_elements(self):
        """Get the total num of elements"""
        return self.n_elements

    def get_density_grid_mean(self):
        """Get the mean density on the inner level"""
        density_grid_mean = float(self.density_grid[:self.n_elements_per_level].clamp_min(0.0).mean().item())
        density_grid_mean = torch.ones((1,), dtype=self.dtype, device=self.density_grid.device) * density_grid_mean

        return density_grid_mean

    def get_aabb_range(self):
        """Get the bbox range"""
        return self.aabb_range

    def min_step_size(self):
        """This is the step size for (0, 1) cube"""
        return math.sqrt(3.0) / self.n_sample

    def max_step_size(self):
        """This is the step size for each diag of voxel in largest volume"""
        return math.sqrt(3.0) * self.aabb_scale / self.n_grid

    def step_size(self):
        """This the step size for aabb_range cube"""
        return math.sqrt(3.0) * (self.aabb_range[1] - self.aabb_range[0]) / self.n_sample

    def update_density_grid(self, epoch, dense_func):
        """Update density grid and bitfield periodically"""
        if epoch > 0:
            if epoch < self.epoch_optim_warmup:
                self._update_density_grid(self.n_elements, 0, dense_func)
            else:
                self._update_density_grid(self.n_elements // 4, self.n_elements // 4, dense_func)

    def _update_density_grid(self, n_uniform, n_nonuniform, dense_func):
        """Core update func"""
        n_total_sample = n_uniform + n_nonuniform

        # get sample
        pos_uniform, idx_uniform = generate_grid_samples(
            self.density_grid, n_uniform, self.ema_step, self.max_cascade, self.n_grid, -0.01
        )
        pos_nonuniform, idx_nonuniform = generate_grid_samples(
            self.density_grid, n_nonuniform, self.ema_step, self.max_cascade, self.n_grid, self.opa_thres
        )

        # merge
        pos_sample = torch.cat([pos_uniform, pos_nonuniform], dim=0)  # (n_total, 3)
        idx_sample = torch.cat([idx_uniform, idx_nonuniform], dim=0)  # (n_total, )

        # get sample point density
        density = chunk_processing(dense_func, self.chunk_pts, False, pos_sample)  # (n_total,)

        # update tmp density grid, always use min_step_size in inner volume
        self.density_grid_tmp.zero_()  # reset
        self.density_grid_tmp = splat_grid_samples(
            density, idx_sample, n_total_sample, self.min_step_size(), self.density_grid_tmp
        )

        # ema update density grid
        self.density_grid = ema_grid_samples_nerf(
            self.density_grid_tmp, self.density_grid, self.n_elements, self.decay
        )

        # update density mean and bitfield
        density_grid_mean = self.get_density_grid_mean()
        self.density_bitfield = update_bitfield(
            self.density_grid, density_grid_mean, self.density_bitfield, self.opa_thres, self.n_grid, self.n_cascades
        )

        self.ema_step = self.ema_step + 1

    def get_bitfield_count(self, level=0):
        """Get the num of 1 in bitfield and occupied ratio. Level is for each cascade"""
        bitcount = count_bitfield(self.density_bitfield, self.n_grid, level)
        occ_ratio = float(bitcount) / float(self.n_elements / self.n_cascades)  # for this level

        return bitcount, occ_ratio

    def get_occ_pc(self):
        """Get the occupied grid centroid as point cloud"""
        pc = get_occ_pc(self.density_bitfield, self.n_grid)

        return pc

    @staticmethod
    def div_round_up(val, divisor):
        return (val + divisor - 1) // divisor

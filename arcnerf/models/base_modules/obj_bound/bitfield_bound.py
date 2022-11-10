# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.volume import Volume
from arcnerf.ops.bitfield_func import (
    count_bitfield, generate_grid_samples, sparse_volume_sampling_bit, splat_grid_samples, ema_grid_samples_nerf,
    update_bitfield, CUDA_BACKEND_AVAILABLE
)
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from . import BOUND_REGISTRY
from .basic_bound import BasicBound


@BOUND_REGISTRY.register()
class BitfieldBound(BasicBound):
    """A volume structure bounding the object in bitfield.
    The one used in instant-ngp but only one level
    """

    def __init__(self, cfgs):
        super(BitfieldBound, self).__init__(cfgs)
        assert valid_key_in_cfgs(cfgs, 'bitfield'), 'You must have bitfield in the cfgs'
        assert CUDA_BACKEND_AVAILABLE, 'bitfield require CUDA functionality'

        self.cfgs = cfgs
        self.read_optim_cfgs()

        # set up the volume
        bitfield_cfgs = cfgs.bitfield
        if get_value_from_cfgs_field(bitfield_cfgs, 'n_grid') is None:
            bitfield_cfgs.n_grid = 128

        # volume params, use for aabb calculation
        self.volume = Volume(**bitfield_cfgs.__dict__)
        self.n_grid = self.volume.get_n_grid()
        self.n_elements = self.n_grid**3

        # set bitfield for pruning
        if self.get_optim_cfgs('epoch_optim') is not None:  # setup bitfield for pruning
            # density bitfield
            density_bitfield = (torch.ones((self.n_elements // 8, ), dtype=torch.uint8) *
                                255).type(torch.uint8)  # all 1
            self.register_buffer('density_bitfield', density_bitfield)
            # density
            density_grid = torch.zeros((self.n_elements, ), dtype=torch.float32)
            self.register_buffer('density_grid', density_grid)
            # tmp density
            density_grid_tmp = torch.zeros((self.n_elements, ), dtype=torch.float32)
            self.register_buffer('density_grid_tmp', density_grid_tmp)
            # for random sample
            self.ema_step = 0

    def get_obj_bound(self):
        """Get the real obj bounding structure"""
        return self.density_bitfield if self.get_optim_cfgs('epoch_optim') is not None else None

    def get_n_grid(self):
        """Get the num of grid """
        return self.n_grid

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        params = super().read_optim_cfgs()
        params['near_distance'] = get_value_from_cfgs_field(self.cfgs, 'near_distance', 0.0)

        return params

    def get_near_far_from_rays(self, inputs, **kwargs):
        """Get the near/far zvals from rays using volume bounding. Coarsely sample is allow

        Returns:
            near, far: torch.tensor (B, 1) each
            mask_rays: torch.tensor (B,), each rays validity
        """
        # in the coarse volume to save aabb cal time
        near, far, _, mask_rays = self.volume.ray_volume_intersection(inputs['rays_o'], inputs['rays_d'])

        return near, far, mask_rays[:, 0]

    def get_zvals_from_near_far(
        self,
        near,
        far,
        n_pts,
        inference_only=False,
        inverse_linear=False,
        perturb=False,
        rays_o=None,
        rays_d=None,
        **kwargs
    ):
        """Sample in the density bitfield.

        Returns:
            near, far: torch.tensor (B, 1) each
            mask_pts: torch.tensor (B, n_pts), each pts validity on each ray.
                    This helps the following network reduces duplicated computation.

        """
        const_dt = self.volume.get_diag_len() / n_pts
        zvals, mask_pts = sparse_volume_sampling_bit(
            rays_o,
            rays_d,
            near,
            far,
            n_pts,
            const_dt,
            self.volume.get_range(),
            self.volume.get_n_grid(),
            self.density_bitfield,
            near_distance=self.get_optim_cfgs('near_distance')
        )

        return zvals, mask_pts

    def get_density_grid_mean(self):
        """Get the mean density"""
        density_grid_mean = float(self.density_grid.clamp_min(0.0).mean().item())
        density_grid_mean = torch.ones((1, ), dtype=self.density_grid.dtype,
                                       device=self.density_grid.device) * density_grid_mean

        return density_grid_mean

    def get_bitfield_count(self, level=0):
        """Get the num of 1 in bitfield and occupied ratio. Level is for each cascade"""
        bitcount = count_bitfield(self.density_bitfield, self.n_grid)
        occ_ratio = float(bitcount) / float(self.n_elements)  # for this level

        return bitcount, occ_ratio

    @torch.no_grad()
    def optimize(self, cur_epoch=0, n_pts=128, get_est_opacity=None):
        """Overwrite the optimization function for volume pruning with bitfield

        In warmup stage, sample all cells and update
        Else in pose-warmup stage, uniform sampled 1/4 cells from all and 1/4 cells from occupied cells.
        """
        epoch_optim = self.get_optim_cfgs('epoch_optim')
        epoch_optim_warmup = self.get_optim_cfgs('epoch_optim_warmup')

        if cur_epoch <= 0 or epoch_optim is None or cur_epoch % epoch_optim != 0:
            return

        # select pts randomly in some voxels
        if epoch_optim_warmup is not None and cur_epoch < epoch_optim_warmup:
            self._update_density_grid(self.n_elements, 0, get_est_opacity, n_pts)
        else:
            self._update_density_grid(self.n_elements // 4, self.n_elements // 4, get_est_opacity, n_pts)

    def _update_density_grid(self, n_uniform, n_nonuniform, get_est_opacity, n_pts):
        """Core update func"""
        n_total_sample = n_uniform + n_nonuniform

        # get sample
        pos_uniform, idx_uniform = generate_grid_samples(
            self.density_grid, n_uniform, self.ema_step, self.n_grid, -0.01
        )
        pos_nonuniform, idx_nonuniform = generate_grid_samples(
            self.density_grid, n_nonuniform, self.ema_step, self.n_grid, self.get_optim_cfgs('opa_thres')
        )

        # merge
        pos_sample = torch.cat([pos_uniform, pos_nonuniform], dim=0)  # (n_total, 3)
        idx_sample = torch.cat([idx_uniform, idx_nonuniform], dim=0)  # (n_total, )

        # adjust position
        scale = self.volume.get_range()[:, 1] - self.volume.get_range()[:, 0]  # (3,)
        pos_sample = pos_sample * scale[None] + self.volume.get_range()[:, 0][None]

        # get the opacity by call the funcition
        dt = self.volume.get_diag_len() / float(n_pts)  # only consider n coarse sample pts
        opacity = get_est_opacity(dt, pos_sample)  # (N,)

        # update tmp density grid, always use min_step_size in inner volume
        self.density_grid_tmp.zero_()  # reset
        self.density_grid_tmp = splat_grid_samples(opacity, idx_sample, n_total_sample, self.density_grid_tmp)

        # ema update density grid
        self.density_grid = ema_grid_samples_nerf(
            self.density_grid_tmp, self.density_grid, self.n_elements, self.get_optim_cfgs('ema_optim_decay')
        )

        # update density mean and bitfield
        density_grid_mean = self.get_density_grid_mean()
        self.density_bitfield = update_bitfield(
            self.density_grid, density_grid_mean, self.density_bitfield, self.get_optim_cfgs('opa_thres'), self.n_grid
        )

        self.ema_step = self.ema_step + 1

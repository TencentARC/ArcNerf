# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.ray import aabb_ray_intersection, get_ray_points_by_zvals
from arcnerf.geometry.volume import Volume
from arcnerf.ops.bitfield_func import splat_grid_samples, ema_grid_samples_nerf
from arcnerf.ops.multivol_func import (
    generate_grid_samples_multivol, sparse_sampling_in_multivol_bitfield, update_bitfield_multivol,
    CUDA_BACKEND_AVAILABLE
)
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .bkg_model import BkgModel
from .base_modules import build_geo_model, build_radiance_model


@MODEL_REGISTRY.register()
class MultiVol(BkgModel):
    """ Multi-Volume model with several resolution. It is the one used in instant-ngp,
        but inner volume is removed
    """

    def __init__(self, cfgs):
        super(MultiVol, self).__init__(cfgs)
        assert CUDA_BACKEND_AVAILABLE, 'bitfield require CUDA functionality'

        self.cfgs = cfgs
        self.optim_cfgs = self.read_optim_cfgs()

        # network
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)

        # volume setting
        vol_cfgs = self.cfgs.model.basic_volume
        if get_value_from_cfgs_field(vol_cfgs, 'n_grid') is None:
            vol_cfgs.n_grid = 128
        self.n_cascade = vol_cfgs.n_cascade
        assert self.n_cascade > 1, 'You should have at least 2 cascades...'
        self.n_grid = vol_cfgs.n_grid
        self.basic_volume = Volume(**vol_cfgs.__dict__)

        # min vol & max vol range
        origin = self.basic_volume.get_origin()  # (3, )
        max_len = [x * 2**(self.n_cascade - 1) for x in self.basic_volume.get_len()]
        self.max_volume = Volume(origin=origin, xyz_len=max_len)

        # those params for sampling
        self.cone_angle = get_value_from_cfgs_field(self.cfgs.model.rays, 'cone_angle', 0.0)
        self.min_step = self.basic_volume.get_diag_len() / self.get_ray_cfgs('n_sample')
        self.max_step = self.max_volume.get_diag_len() / self.n_grid

        # set bitfield for pruning
        self.n_elements = self.n_grid**3
        self.total_n_elements = self.n_elements * (self.n_cascade - 1)
        # density bitfield
        density_bitfield = (torch.ones((self.total_n_elements // 8, ), dtype=torch.uint8) * 255).type(torch.uint8)
        self.register_buffer('density_bitfield', density_bitfield)
        # density
        density_grid = torch.zeros((self.total_n_elements, ), dtype=torch.float32)
        self.register_buffer('density_grid', density_grid)
        # tmp density
        density_grid_tmp = torch.zeros((self.total_n_elements, ), dtype=torch.float32)
        self.register_buffer('density_grid_tmp', density_grid_tmp)
        # for random sample
        self.ema_step = 0

    def get_near_far_from_rays(self, rays_o, rays_d):
        """Get the near/far zvals from rays using outer volume. The cameras are in the max volume.

        Returns:
            near, far: torch.tensor (B, 1) each
        """
        aabb_range = self.max_volume.get_range()[None].to(rays_o.device)  # (1, 3, 2)
        # do not use the CUDA version since the far calculation is not accurate
        near, far, _, _ = aabb_ray_intersection(rays_o, rays_d, aabb_range, force_torch=True)  # (B, 1) * 2

        return near, far

    def get_zvals_from_near_far(self, near, far, n_pts, rays_o, rays_d, **kwargs):
        """Sample in the muitl-res density bitfield.

        Returns:
            zvals: torch.tensor (B, 1) of zvals
            mask_pts: torch.tensor (B, n_pts), each pts validity on each ray.
                    This helps the following network reduces duplicated computation.
        """
        zvals, mask = sparse_sampling_in_multivol_bitfield(
            rays_o,
            rays_d,
            near,
            far,
            n_pts,
            self.cone_angle,
            self.min_step,
            self.max_step,
            self.basic_volume.get_range(),
            self.max_volume.get_range(),
            self.n_grid,
            self.n_cascade,
            self.density_bitfield,
            near_distance=self.get_optim_cfgs('near_distance')
        )

        return zvals, mask

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        n_rays = rays_o.shape[0]

        near, far = self.get_near_far_from_rays(rays_o, rays_d)
        zvals, mask_pts = self.get_zvals_from_near_far(near, far, self.get_ray_cfgs('n_sample'), rays_o, rays_d)

        # keep the largest sample. Actually some duplicate pts, but calc all.
        max_num_pts = max(1, int(mask_pts.sum(dim=1).max()))
        zvals = zvals[:, :max_num_pts]  # (n_rays, max_pts)

        # get points, expand rays_d to all pts
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N, 3)
        rays_d_repeat = torch.repeat_interleave(rays_d.unsqueeze(1), pts.shape[1], dim=1)  # (B, N, 3)

        # flatten
        pts = pts.view(-1, 3)  # (BN, 3)
        rays_d_repeat = rays_d_repeat.view(-1, 3)  # (BN, 3)

        # get sigma and rgb, . shape in (N_valid_pts, ...)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts, rays_d_repeat
        )

        # reshape to (B, N_sample, ...)
        sigma = sigma.view(n_rays, -1)  # (B, N_sample)
        radiance = radiance.view(n_rays, -1, 3)  # (B, N_sample, 3)

        output = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

    def get_density_grid_mean(self):
        """Get the mean density of each level"""
        density_grid_mean = float(self.density_grid.clamp_min(0.0).mean().item())
        density_grid_mean = torch.ones((1, ), dtype=self.density_grid.dtype,
                                       device=self.density_grid.device) * density_grid_mean

        return density_grid_mean

    @torch.no_grad()
    def optimize(self, cur_epoch=0):
        """Overwrite the optimization function for volume pruning with multivol

        In warmup stage, sample all cells and update
        Else in pose-warmup stage, uniform sampled 1/4 cells from all and 1/4 cells from occupied cells.
        """
        epoch_optim = self.get_optim_cfgs('epoch_optim')
        epoch_optim_warmup = self.get_optim_cfgs('epoch_optim_warmup')

        if cur_epoch <= 0 or epoch_optim is None or cur_epoch % epoch_optim != 0:
            return

        # select pts randomly in some voxels
        n_pts = self.get_ray_cfgs('n_sample')
        if epoch_optim_warmup is not None and cur_epoch < epoch_optim_warmup:
            self._update_density_grid(self.total_n_elements, 0, n_pts)
        else:
            self._update_density_grid(self.total_n_elements // 4, self.total_n_elements // 4, n_pts)

    def _update_density_grid(self, n_uniform, n_nonuniform, n_pts):
        """Core update func"""
        n_total_sample = n_uniform + n_nonuniform

        # get sample. The inner level will be ignored
        pos_uniform, idx_uniform = generate_grid_samples_multivol(
            self.density_grid, n_uniform, self.basic_volume.get_range(), self.ema_step, self.n_cascade, self.n_grid,
            -0.01
        )
        pos_nonuniform, idx_nonuniform = generate_grid_samples_multivol(
            self.density_grid, n_nonuniform, self.basic_volume.get_range(), self.ema_step, self.n_cascade, self.n_grid,
            self.get_optim_cfgs('opa_thres')
        )

        # merge
        pos_sample = torch.cat([pos_uniform, pos_nonuniform], dim=0)  # (n_total, 3)
        idx_sample = torch.cat([idx_uniform, idx_nonuniform], dim=0)  # (n_total, )

        # get the opacity by call the function
        dt = self.basic_volume.get_diag_len() / float(n_pts)
        opacity = self.get_est_opacity(dt, pos_sample)  # (N,)

        # update tmp density grid, always use min_step_size in inner volume
        self.density_grid_tmp.zero_()  # reset
        self.density_grid_tmp = splat_grid_samples(opacity, idx_sample, n_total_sample, self.density_grid_tmp)

        # ema update density grid
        self.density_grid = ema_grid_samples_nerf(
            self.density_grid_tmp, self.density_grid, self.total_n_elements, self.get_optim_cfgs('ema_optim_decay')
        )

        # update density mean and bitfield
        density_grid_mean = self.get_density_grid_mean()
        self.density_bitfield = update_bitfield_multivol(
            self.density_grid, density_grid_mean, self.density_bitfield, self.get_optim_cfgs('opa_thres'), self.n_grid,
            self.n_cascade
        )

        self.ema_step = self.ema_step + 1

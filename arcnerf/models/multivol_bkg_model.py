# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.ray import aabb_ray_intersection
from arcnerf.geometry.volume import Volume
from arcnerf.ops.multivol_func import sparse_sampling_in_multivol_bitfield, CUDA_BACKEND_AVAILABLE
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from .bkg_model import BkgModel


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
        # output = {}

        near, far = self.get_near_far_from_rays(rays_o, rays_d)
        zvals, mask_pts = self.get_zvals_from_near_far(near, far, self.get_ray_cfgs('n_sample'), rays_o, rays_d)

        # keep the largest sample
        max_num_pts = max(1, int(mask_pts.sum(dim=1).max()))
        zvals = zvals[:, :max_num_pts]  # (n_rays, max_pts)
        mask_pts = mask_pts[:, :max_num_pts]  # (n_rays, max_pts)

    def optimize(self, cur_epoch=0):
        """Optimize the bitfield"""

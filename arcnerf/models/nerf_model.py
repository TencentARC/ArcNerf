# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_near_far_from_rays, get_zvals_from_near_far, ray_marching
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class NeRF(Base3dModel):
    """Single forward Nerf model. 8 layers in GeoNet and 1 layer in RadianceNet
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        self.geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # ray_cfgs
        self.rays_cfgs = self.read_ray_cfgs()
        self.chunk_size = self.cfgs.model.chunk_size

    def forward(self, inputs):
        """
        TODO: How to due with background, how mask can be applied
        Args:
            inputs['img']: torch.tensor (B, N, 3), rgb value in 0-1
            inputs['rays_o']: torch.tensor (B, N, 3), cam_loc/ray_start position
            inputs['rays_d']: torch.tensor (B, N, 3), view dir(normed)
            inputs['mask']: torch.tensor (B, N), mask value in {0, 1}. optional
            inputs['bound']: torch.tensor (B, 2)
        Returns:
            output is a dict with following keys:
                rgb: torch.tensor (B, N, 3)
                depth: torch.tensor (B, N)
                mask: torch.tensor (B, N)
        """
        rays_o = inputs['rays_o'].view(-1, 3)  # (BN, 3)
        rays_d = inputs['rays_d'].view(-1, 3)  # (BN, 3)
        batch_size, n_rays_per_batch = inputs['rays_o'].shape[:2]

        # get bounds for object, (BN, 1) * 2
        bounds = None
        if 'bounds' in inputs:
            bounds = torch.repeat_interleave(inputs['bounds'], n_rays_per_batch, dim=0)
        near, far = get_near_far_from_rays(
            rays_o, rays_d, bounds, self.rays_cfgs['near'], self.rays_cfgs['far'], self.rays_cfgs['bounding_radius']
        )

        # get zvals
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.rays_cfgs['n_sample'],
            inverse_linear=self.rays_cfgs['inverse_linear'],
            perturb=self.rays_cfgs['perturb']
        )  # (BN, N_sample)

        # get points
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (BN, N_sample, 3)
        pts = pts.view(-1, 3)  # (BN*N_sample, 3)

        # get sigma and rgb,  expand rays_d to all pts. shape in (BN*N_sample, dim)
        sigma, feature = chunk_processing(self.geo_net, self.chunk_size, pts)
        rays_d = torch.repeat_interleave(rays_d, self.rays_cfgs['n_sample'], dim=0)
        radiance = chunk_processing(self.radiance_net, self.chunk_size, pts, rays_d, None, feature)

        # reshape, ray marching and get color/weights
        sigma = sigma.view(-1, self.rays_cfgs['n_sample'], 1)[..., 0]  # (BN, N_sample)
        radiance = radiance.view(-1, self.rays_cfgs['n_sample'], 3)  # (BN, N_sample, 3)
        rgb, depth, mask, _ = ray_marching(
            sigma, radiance, zvals, self.rays_cfgs['add_inf_z'], self.rays_cfgs['noise_std']
        )

        output = {
            'rgb': rgb.view(batch_size, n_rays_per_batch, 3),  # (B, N, 3)
            'depth': depth.view(batch_size, n_rays_per_batch),  # (B, N)
            'mask': mask.view(batch_size, n_rays_per_batch),  # (B, N)
        }

        return output

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor):
        """This function forward pts and view dir directly, only for inference the geometry/color

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord
            view_dir: torch.tensor (N_pts, 3) view dir associate with each point
        Returns:
            output is a dict with following keys:
                sigma: torch.tensor (N_pts), density value for each point
                rgb: torch.tensor (N_pts, 3), color for each point
        """
        sigma, feature = chunk_processing(self.geo_net, self.chunk_size, pts)
        rays_d = normalize(view_dir)
        rgb = chunk_processing(self.radiance_net, self.chunk_size, pts, rays_d, None, feature)

        return sigma[..., 0], rgb

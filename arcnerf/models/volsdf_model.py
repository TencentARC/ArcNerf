# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class VolSDF(Base3dModel):
    """ VolSDF model. 8 layers in GeoNet and 4 layer in RadianceNet
        Model SDF and convert it to density.
        ref: https://github.com/lioryariv/volsdf
             https://github.com/ventusff/neurecon#volume-rendering--3d-implicit-surface
    """

    def __init__(self, cfgs):
        super(VolSDF, self).__init__(cfgs)
        self.geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # custom rays cfgs for upsampling
        self.rays_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        self.rays_cfgs['n_iter'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_iter', 4)
        # radius init for object
        self.radius_init = get_value_from_cfgs_field(self.cfgs.model.geometry, 'radius_init', 1.0)
        self.ln_beta, self.beta_min, self.speed_factor = self.get_params()

    @staticmethod
    def sigma_reverse():
        """It use SDF(inside object is smaller)"""
        return True

    def get_params(self):
        """Get scale param"""
        dtype = next(self.geo_net.parameters()).dtype
        init_beta = get_value_from_cfgs_field(self.cfgs.model.params, 'init_beta', 0.1)
        beta_min = get_value_from_cfgs_field(self.cfgs.model.params, 'beta_min', 0.0001)
        speed_factor = get_value_from_cfgs_field(self.cfgs.model.params, 'speed_factor', 10)
        ln_beta = nn.Parameter(data=torch.tensor([np.log(init_beta) / speed_factor], dtype=dtype), requires_grad=True)

        return ln_beta, beta_min, speed_factor

    def forward_beta(self):
        """Return scale = exp(ln_beta * speed)"""
        return torch.exp(self.ln_beta * self.speed_factor)

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        n_rays = rays_o.shape[0]

        # get bounds for object
        near, far = self._get_near_far_from_rays(inputs)  # (B, 1) * 2

        # get coarse zvals
        zvals = self.get_zvals_from_near_far(near, far, inference_only)  # (B, N_sample)

        # up-sample zvals, (B, N_total(N_sample+N_importance)) for zvals,  (B, 1) for zvals_surface
        zvals, zvals_surface = self._upsample_zvals(rays_o, rays_d, zvals, inference_only)

        # get points
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_total, 3)
        pts = pts.view(-1, 3)  # (B*N_total, 3)

        # get sdf, rgb and normal, expand rays_d to all pts. shape in (B*N_total, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, int(pts.shape[0] / n_rays), dim=0)
        sdf, radiance, normal_pts = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts, rays_d_repeat
        )

        # reshape
        sdf = sdf.view(n_rays, -1)  # (B, N_total)
        radiance = radiance.view(n_rays, -1, 3)  # (B, N_total, 3)
        normal_pts = normal_pts.view(n_rays, -1, 3)  # (B, N_total, 3)

        # convert sdf to sigma
        sigma = sdf_to_sigma(sdf, self.forward_beta(), self.beta_min)  # (B, N_total)

        # ray marching, sdf will not be used for calculation but for recording
        output = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)

        # add normal. For normal map, needs to normalize each pts
        output['normal'] = torch.sum(output['weights'].unsqueeze(-1) * normalize(normal_pts), -2)  # (B, 3)
        if not inference_only:
            output['params'] = {'beta': float(self.forward_beta().clone())}
            output['normal_pts'] = self.get_eikonal_pts(rays_o, rays_d, zvals_surface)  # (B, 2, 3), do not normalize it

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Rewrite for normal handling"""
        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sdf, rgb, _ = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts, rays_d
        )

        return sdf, rgb

    @staticmethod
    def _forward_pts_dir(
        geo_net,
        radiance_net,
        pts: torch.Tensor,
        rays_d: torch.Tensor = None,
    ):
        """Rewrite to use normal processing """
        sdf, feature, normal = geo_net.forward_with_grad(pts)
        radiance = radiance_net(pts, rays_d, normal, feature)

        return sdf[..., 0], radiance, normal

    def upsample_zvals(self, rays_o: torch.Tensor, rays_d: torch.Tensor, zvals: torch.Tensor, inference_only, s=32):
        """Upsample zvals if N_importance > 0

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals: tensor (B, N_sample), coarse zvals for all rays
            inference_only: affect the sample_pdf deterministic. By default False(For train)
            s: factor for up-sample. By default 32

        Returns:
            zvals: tensor (B, N_sample + N_importance), up-sample zvals near the surface
            zvals_surface: (B, 1) zvals on surface
        """
        if self.rays_cfgs['n_importance'] <= 0:
            return zvals

        # dtype = zvals.dtype
        # device = zvals.device
        # n_sample_per_iter = self.rays_cfgs['n_importance'] // self.rays_cfgs['n_iter']

        zvals_surface = None
        return zvals, zvals_surface

    def get_eikonal_pts(self, rays_o: torch.Tensor, rays_d: torch.Tensor, zvals_surface: torch.Tensor):
        """Get random eikonal pts from bounding sphere and near surface
        sec 3.5 in paper.

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals_surface: tensor (B, 1), zvals on surface

        Returns:
            pts: torch.tensor (B, 2, 3), in sphere pts + surface pts
        """
        dtype = rays_o.dtype
        device = rays_o.device
        bounding_radius = self.get_ray_cfgs('bounding_radius')

        # random pts in sphere, (B, 1, 3)
        pts_rand = torch.empty(size=(rays_o.shape[0], 1, 3), dtype=dtype)\
            .uniform_(-bounding_radius, bounding_radius).to(device)
        pts_rand = pts_rand / torch.norm(pts_rand, -1, keepdim=True) * bounding_radius  # make sure in sphere

        # pts on surface, (B, 1, 3)
        pts_surface = get_ray_points_by_zvals(rays_o, rays_d, zvals_surface)

        pts_all = torch.cat([pts_rand, pts_surface], dim=1)  # (B, 2, 3)

        return pts_all


def sdf_to_sigma(sdf: torch.Tensor, beta, beta_min=0.0001):
    """Turn sdf to sigma. When beta goes to 0, weights focus more on surface
    ----------------------------------------
    eq.2 and eq.3 in paper
    sigma = alpha(0.5*exp(sdf/beta)) if -sdf <= 0(sdf >=0)
    sigma = alpha(1 - 0.5*exp(-sdf/beta)) if -sdf > 0 (sdf < 0)
    ----------------------------------------

    Args:
        sdf: tensor (B, N_pts) of pts
        beta: beta for cdf adjustment
        beta_min: add to beta in case beta too small

    Returns:
        sigma: tensor (B, N_pts) of pts
    """
    beta = beta + beta_min
    alpha = 1 / beta

    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    sigma = alpha * torch.where(sdf >= 0, exp, 1 - exp)

    return sigma

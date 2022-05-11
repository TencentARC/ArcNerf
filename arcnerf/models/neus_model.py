# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_zvals_from_near_far, ray_marching, sample_pdf, alpha_to_weights
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class Neus(Base3dModel):
    """ Neus model. 8 layers in GeoNet and 4 layer in RadianceNet
        ref: https://lingjie0206.github.io/papers/NeuS
             https://github.com/ventusff/neurecon#volume-rendering--3d-implicit-surface
    """

    def __init__(self, cfgs):
        super(Neus, self).__init__(cfgs)
        self.geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # custom rays cfgs for upsampling
        self.rays_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        self.rays_cfgs['n_iter'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_iter', 4)
        # radius init for object
        self.radius_init = get_value_from_cfgs_field(self.cfgs.model.geometry, 'radius_init', 1.0)
        self.inv_s, self.speed_factor = self.get_params()
        self.anneal_end = get_value_from_cfgs_field(self.cfgs.model.params, 'anneal_end', 0)

    def get_params(self):
        """Get scale param"""
        dtype = next(self.geo_net.parameters()).dtype
        init_var = get_value_from_cfgs_field(self.cfgs.model.params, 'init_var', 0.05)
        speed_factor = get_value_from_cfgs_field(self.cfgs.model.params, 'speed_factor', 10)
        inv_s = nn.Parameter(data=torch.tensor([-np.log(init_var) / speed_factor], dtype=dtype), requires_grad=True)

        return inv_s, speed_factor

    def _forward_scale(self):
        """Return scale = exp(inv_s * speed)"""
        return torch.exp(self.inv_s * self.speed_factor)

    def get_cos_anneal(self, cur_epoch):
        """Get the cos_anneal_ratio"""
        if self.anneal_end == 0:
            return 1.0

        return min(1.0, cur_epoch / self.anneal_end)

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        n_rays = rays_o.shape[0]

        # get bounds for object, (B, 1) * 2
        near, far = self._get_near_far_from_rays(inputs)

        # get coarse zvals
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.rays_cfgs['n_sample'],
            inverse_linear=self.rays_cfgs['inverse_linear'],
            perturb=self.rays_cfgs['perturb'] if not inference_only else False
        )  # (B, N_sample)

        # up-sample zvals
        zvals = self._upsample_zvals(rays_o, rays_d, zvals, inference_only)  # (B, N_total(N_sample+N_importance))

        # use mid pts in section for sdf
        mid_zvals = 0.5 * (zvals[..., 1:] + zvals[..., :-1])
        mid_pts = get_ray_points_by_zvals(rays_o, rays_d, mid_zvals)  # (B, N_total-1, 3)
        mid_pts = mid_pts.view(-1, 3)  # (B*N_total-1, 3)

        # get sdf, feature and normal
        rays_d_repeat = torch.repeat_interleave(rays_d, int(mid_pts.shape[0] / n_rays), dim=0)
        sdf, radiance, normal_pts = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, mid_pts, rays_d_repeat
        )

        # reshape, change to real weight
        sdf = sdf.view(n_rays, -1)  # (B, N_total-1)
        rays_d_repeat = rays_d_repeat.view(n_rays, -1, 3)  # (B, N_total-1, 3)
        radiance = radiance.view(n_rays, -1, 3)  # (B, N_total-1, 3)
        normal_pts = normal_pts.view(n_rays, -1, 3)  # (B, N_total-1, 3)

        # estimate sdf for section pts using mid pts sdf
        cos_anneal_ratio = 1.0 if inference_only else self.get_cos_anneal(cur_epoch)

        # rays and normal are opposite, slope is neg (dot prod of rays and normal)
        slope = torch.sum(rays_d_repeat * normal_pts, dim=-1, keepdim=True)[..., 0]  # (B, N_total-1)
        iter_slope = -(F.relu(-slope * 0.5 + 0.5) * (1 - cos_anneal_ratio) + F.relu(-slope) * cos_anneal_ratio)  # neg
        alpha = sdf_to_alpha(sdf, zvals, iter_slope, self._forward_scale())  # (B, N_total-1)

        # ray marching
        output = ray_marching(
            sdf,  # will not use, just for record
            radiance,
            mid_zvals,
            self.rays_cfgs['add_inf_z'],  # rgb mode should be False, sigma mode should True
            self.rays_cfgs['noise_std'] if not inference_only else 0.0,
            weights_only=False,
            white_bkg=self.rays_cfgs['white_bkg'],
            alpha=alpha
        )

        # add normal. For normal map, needs to normalize each pts
        output['normal'] = torch.sum(output['weights'].unsqueeze(-1) * normalize(normal_pts), -2)  # (B, 3)
        if not inference_only:
            output['params'] = {'scale': float(self._forward_scale().clone())}
            output['normal_pts'] = normal_pts  # (B, N_total-1, 3), do not normalize it

        # merge rgb with background, only support this mode since sdf/sigma can not be blended toegther
        output = self._merge_bkg_rgb(inputs, output, inference_only)

        if get_progress:  # this save the sigma with out blending bkg, only in foreground
            for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights', 'radiance']:
                n_fg = self._get_n_fg(sdf)
                output['progress_{}'.format(key)] = output[key][:, :n_fg].detach()  # (B, N_sample(-1))
            output['sigma_reverse3d'] = True  # for rays 3d visual of sdf
        else:  # pop to reduce memory
            for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights', 'radiance']:
                output.pop(key)

        return output

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Rewrite for normal handling"""
        gpu_on_func = True if (self.is_cuda() and not pts.is_cuda) else False

        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sdf, rgb, _ = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, gpu_on_func, self.geo_net, self.radiance_net, pts, rays_d
        )

        return sdf, rgb

    @staticmethod
    def _forward_pts_dir(
        geo_net,
        radiance_net,
        pts: torch.Tensor,
        rays_d: torch.Tensor = None,
    ):
        """Core forward function to forward. Use chunk progress to call it will save memory for feature.
        Rewrite to use normal.

        Args:
            pts: (B, 3) xyz points
            rays_d: (B, 3) view dir(normalize)

        Return:
            sdf: (B, ) sigma value
            radiance: (B, 3) rgb value in float
        """
        sdf, feature, normal = geo_net.forward_with_grad(pts)
        radiance = radiance_net(pts, rays_d, normal, feature)

        return sdf[..., 0], radiance, normal

    def _upsample_zvals(self, rays_o: torch.Tensor, rays_d: torch.Tensor, zvals: torch.Tensor, inference_only, s=32):
        """Upsample zvals if N_importance > 0

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals: tensor (B, N_sample), coarse zvals for all rays
            inference_only: affect the sample_pdf deterministic. By default False(For train)
            s: factor for up-sample. By default 32

        Returns:
            zvals: tensor (B, N_sample + N_importance), up-sample zvals near the surface
        """
        if self.rays_cfgs['n_importance'] <= 0:
            return zvals

        dtype = zvals.dtype
        device = zvals.device
        n_sample_per_iter = self.rays_cfgs['n_importance'] // self.rays_cfgs['n_iter']
        for i in range(self.rays_cfgs['n_iter']):
            # cal sdf from network
            n_rays, n_pts = zvals.shape[:2]
            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)
            pts = pts.view(-1, 3)
            sdf = chunk_processing(self.geo_net.forward_geo_value, self.chunk_pts, False, pts)  # (B*N_pts)
            sdf = sdf.view(n_rays, n_pts)  # (B, N_pts)

            # find min slope
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            prev_zvals, next_zvals = zvals[:, :-1], zvals[:, 1:]
            slope = (next_sdf - prev_sdf) / (next_zvals - prev_zvals + 1e-5)  # (B, N_pts-1)

            zeros_pad = torch.zeros([n_rays, 1], dtype=dtype).to(device)
            prev_slope = torch.cat([zeros_pad, slope[:, :-1]], dim=-1)  # (B, N_pts-1)
            slope = torch.cat([prev_slope, slope], dim=-1)  # (B, 2*N_pts-2)
            slope, _ = torch.min(slope, dim=-1, keepdim=True)
            slope = slope.clamp(-10.0, 0.0)  # (B, )

            # upsample by alpha from cdf. eq.13 of paper.
            alpha = sdf_to_alpha(mid_sdf, zvals, slope, s * (2**(i + 1)))  # (s * 2^i for i=1,2,3)
            _, weights = alpha_to_weights(alpha)  # (B, N_pts-1)
            upsample_zvals = sample_pdf(
                zvals, weights, n_sample_per_iter, not self.rays_cfgs['perturb'] if not inference_only else True
            ).detach()  # (B, N_up)

            zvals = torch.cat([zvals, upsample_zvals], dim=-1)
            zvals, _ = torch.sort(zvals, dim=-1)

        return zvals


def sdf_to_cdf(sdf: torch.Tensor, s):
    """Turn sdf to cdf function using sigmoid function

    Args:
        sdf: tensor (B, N_pts)
        s: scale factor
    """
    return torch.sigmoid(sdf * s)


def sdf_to_alpha(mid_sdf: torch.Tensor, zvals: torch.Tensor, slope: torch.Tensor, s):
    """Turn sdf to alpha

    Args:
        mid_sdf: tensor (B, N_pts-1) of mid pts
        zvals: tensor (B, N_pts)
        slope: tensor (B, N_pts-1)/(B, ) of mid pts
        s: scale factor

    Returns:
        alpha: tensor (B, N_pts-1) of mid pts
    """
    dist = zvals[:, 1:] - zvals[:, :-1]
    prev_esti_sdf = mid_sdf - slope * dist * 0.5  # > mid_sdf, (B, N_pts-1)
    next_esti_sdf = mid_sdf + slope * dist * 0.5  # < mid_sdf, (B, N_pts-1)
    prev_cdf = sdf_to_cdf(prev_esti_sdf, s)
    next_cdf = sdf_to_cdf(next_esti_sdf, s)

    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)  # (B, N_pts-1)
    alpha = alpha.clip(0.0, 1.0)

    return alpha

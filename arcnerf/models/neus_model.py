# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import sample_pdf, alpha_to_weights
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .sdf_model import SdfModel
from .base_modules import build_geo_model, build_radiance_model


@MODEL_REGISTRY.register()
class Neus(SdfModel):
    """ Neus model.
        Model SDF and convert it to alpha.
        The neus model uses sphere of interest that is larger than the object but smaller than camera sphere.
        ref: https://lingjie0206.github.io/papers/NeuS
             https://github.com/ventusff/neurecon#volume-rendering--3d-implicit-surface
    """

    def __init__(self, cfgs):
        super(Neus, self).__init__(cfgs)
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # custom rays cfgs for upsampling
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        self.ray_cfgs['n_iter'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_iter', 4)
        # radius init for object
        self.radius_init = get_value_from_cfgs_field(self.cfgs.model.geometry, 'radius_init', 1.0)
        self.inv_s, self.speed_factor = self.get_params()
        self.anneal_end = get_value_from_cfgs_field(self.cfgs.model.params, 'anneal_end', 0)
        # Use bounding sphere in NeuS
        self.radius_bound = get_value_from_cfgs_field(self.cfgs.model.rays, 'radius_bound', 1.5)

    def get_params(self):
        """Get scale param"""
        dtype = next(self.parameters()).dtype
        init_var = get_value_from_cfgs_field(self.cfgs.model.params, 'init_var', 0.05)
        speed_factor = get_value_from_cfgs_field(self.cfgs.model.params, 'speed_factor', 10)
        inv_s = nn.Parameter(data=torch.tensor([-np.log(init_var) / speed_factor], dtype=dtype), requires_grad=True)

        return inv_s, speed_factor

    def forward_scale(self):
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
        zvals = inputs['zvals']  # (B, 1)
        mask_pts = inputs['mask_pts']  # (B, n_pts)
        bkg_color = inputs['bkg_color']  # (B, 3)

        # up-sample zvals with mask_pts
        zvals, mask_pts = self.upsample_zvals(rays_o, rays_d, zvals, mask_pts, inference_only)  # (B, N_total)

        # use mid pts in section for sdf
        mid_zvals, zvals, mask_mid_pts = self.handle_mid_pts(zvals, mask_pts)  # (B, N), (B, N+1), (B, N)

        # get pts sigma/rgb/normal  (B, N_sample, ...)
        sdf, radiance, normal_pts = self.get_sdf_radiance_normal_by_mask_pts(
            self.geo_net, self.radiance_net, rays_o, rays_d, mid_zvals, mask_mid_pts, inference_only
        )
        rays_d_repeat = torch.repeat_interleave(rays_d.unsqueeze(1), mid_zvals.shape[1], dim=1)  # (B, N_total, 3)

        # estimate sdf for section pts using mid pts sdf
        cos_anneal_ratio = 1.0 if inference_only else self.get_cos_anneal(cur_epoch)

        # rays and normal are opposite, slope is neg (dot prod of rays and normal). convert sdf to alpha
        slope = torch.sum(rays_d_repeat * normal_pts, dim=-1, keepdim=True)[..., 0]  # (B, N_total)
        iter_slope = -(F.relu(-slope * 0.5 + 0.5) * (1 - cos_anneal_ratio) + F.relu(-slope) * cos_anneal_ratio)  # neg
        alpha = sdf_to_alpha(sdf, zvals, iter_slope, self.forward_scale())  # (B, N_total)

        # ray marching, sdf will not be used for calculation but for recording
        output = self.ray_marching(
            sdf, radiance, mid_zvals, alpha=alpha, inference_only=inference_only, bkg_color=bkg_color
        )

        # add normal. For normal map, needs to normalize each pts
        output['normal'] = torch.sum(output['weights'].unsqueeze(-1) * normalize(normal_pts), -2)  # (B, 3)
        if not inference_only:
            output['params'] = {'scale': float(self.forward_scale().clone())}
            output['normal_pts'] = normal_pts  # (B, N_total, 3), do not normalize it

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

    def upsample_zvals(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        zvals: torch.Tensor,
        mask_pts=None,
        inference_only=False,
        s=32
    ):
        """Upsample zvals if N_importance > 0

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals: tensor (B, N_sample), coarse zvals for all rays
            mask_pts: tensor (B, N_sample) whether each pts is valid. None means all valid.
            inference_only: affect the sample_pdf deterministic. By default False(For train)
            s: factor for up-sample. By default 32

        Returns:
            zvals: tensor (B, N_sample + N_importance), up-sample zvals near the surface
            mask_pts: new mask in (B, N_sample + N_importance)
        """
        if self.get_ray_cfgs('n_importance') <= 0:
            return zvals, mask_pts

        dtype = zvals.dtype
        device = zvals.device
        n_sample_per_iter = self.get_ray_cfgs('n_importance') // self.get_ray_cfgs('n_iter')
        for i in range(self.get_ray_cfgs('n_iter')):
            # cal sdf from network
            n_rays, n_pts = zvals.shape[:2]
            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals).view(-1, 3)  # (B*N_pts, 3)
            sdf = self.forward_pts(pts).view(n_rays, n_pts)  # (B, N_pts)

            # find min slope
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            prev_zvals, next_zvals = zvals[:, :-1], zvals[:, 1:]
            slope = (next_sdf - prev_sdf) / (next_zvals - prev_zvals + 1e-5)  # (B, N_pts-1)

            zeros_pad = torch.zeros([n_rays, 1], dtype=dtype, device=device)
            prev_slope = torch.cat([zeros_pad, slope[:, :-1]], dim=-1)  # (B, N_pts-1)
            slope = torch.stack([prev_slope, slope], dim=-1)  # (B, N_pts-1, 2)
            slope, _ = torch.min(slope, dim=-1, keepdim=False)  # (B, N_pts-1)
            slope = slope.clamp(-10.0, 0.0)  # (B, N_pts-1)

            # clip the slope only in the sphere of interest
            pts = pts.view(n_rays, n_pts, 3)  # (B, N_pts, 3)
            radius = torch.norm(pts, dim=-1)  # (B, N_pts)
            inside_sphere = (radius[:, :-1] < self.radius_bound) | (radius[:, 1:] < self.radius_bound)
            slope = slope * inside_sphere

            # upsample by alpha from cdf. eq.13 of paper.
            alpha = sdf_to_alpha(mid_sdf, zvals, slope, s * (2**(i + 1)), clip=False)  # (s * 2^i for i=1,2,3)
            _, weights = alpha_to_weights(alpha)  # (B, N_pts-1)
            zvals_on_surface = sample_pdf(
                zvals, weights, n_sample_per_iter, not self.get_ray_cfgs('perturb') if not inference_only else True
            ).detach()  # (B, N_up)

            zvals = torch.cat([zvals, zvals_on_surface], dim=-1)
            zvals, _ = torch.sort(zvals, dim=-1)

            mask_pts = self.merge_full_mask(mask_pts, zvals_on_surface)

        return zvals, mask_pts

    def handle_mid_pts(self, zvals, mask_pts):
        """Handling the mid_pts from NeuS given mask_pts"""
        dtype = zvals.dtype
        device = zvals.device

        if mask_pts is None:
            sample_dist = (zvals[:, -1] - zvals[:, 0]) / self.get_ray_cfgs('n_sample') * 0.5  # (B,)
            mid_zvals = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N-1)
            # append an extra zval to the end (follow the original implementation)
            final_mid_zvals = mid_zvals[:, -1] + sample_dist  # (B,)
            final_zvals = zvals[:, -1] + sample_dist  # (B,)
            mid_zvals = torch.cat([mid_zvals, final_mid_zvals.unsqueeze(-1)], dim=-1)  # (B, N)
            zvals = torch.cat([zvals, final_zvals.unsqueeze(-1)], dim=-1)  # (B, N+1)
        else:  # Need to adjust the last position
            sample_dist = (zvals[:, -1] - zvals[:, 0]) / self.get_ray_cfgs('n_sample') * 0.5  # (B,)
            zeros_mask = torch.zeros((mask_pts.shape[0], 1), dtype=torch.bool, device=device)
            ones_mask = torch.ones((mask_pts.shape[0], 1), dtype=torch.bool, device=device)
            final_zvals = zvals[:, -1] + sample_dist * 2.0  # (B,)
            _zvals = torch.ones((zvals.shape[0], zvals.shape[1] + 1), dtype=dtype,
                                device=device) * final_zvals.unsqueeze(1)  # (B, N+1)
            _mask_pts = torch.cat([mask_pts, zeros_mask], dim=1)  # (B, N+1)
            _zvals[_mask_pts] = zvals[mask_pts]  # (B, N+1)

            # mid zvals, but the last one will be the same zvals
            mid_zvals = 0.5 * (_zvals[..., 1:] + _zvals[..., :-1])  # (B, N)

            zvals = _zvals  # (B, N+1)
            mask_pts = torch.cat([ones_mask, mask_pts[:, :-1]], dim=1)  # (B, N+1) extent one valid pts

        return mid_zvals, zvals, mask_pts

    def get_est_opacity(self, dt, pts):
        """NeuS model convert sdf with slope to alpha(opacity)"""
        n_pts = pts.shape[0]
        # Make fake sdf on ray to get pts opacity
        rays_d = -normalize(pts)  # assume points to (0,0,0), (B, 3)
        sdf, _, normal = chunk_processing(self.geo_net.forward_with_grad, self.chunk_pts, False, pts)  # (B, 1), (B, 3)

        # rays and normal are opposite, slope is neg (dot prod of rays and normal). convert sdf to alpha
        slope = torch.sum(rays_d * normal, dim=-1, keepdim=True)  # (B, 1)
        zvals = torch.zeros((n_pts, 2), dtype=pts.dtype, device=pts.device)  # (B, 2)
        zvals[:, 1] += (dt * 1.0 / math.sqrt(3))  # mask the dist to diag dt to zvals
        iter_slope = -F.relu(-slope)  # neg
        opacity = sdf_to_alpha(sdf, zvals, iter_slope, self.forward_scale())  # (B, 1)

        return opacity[:, 0]


def sdf_to_cdf(sdf: torch.Tensor, s):
    """Turn sdf to cdf function using sigmoid function

    Args:
        sdf: tensor (B, N_pts)
        s: scale factor
    """
    return torch.sigmoid(sdf * s)


def sdf_to_pdf(sdf: torch.Tensor, s):
    """Turn sdf to pdf function using sigmoid function

    Args:
        sdf: tensor (B, N_pts)
        s: scale factor
    """
    esx = torch.exp(-sdf * s)
    return s * esx / ((1 + esx)**2)


def sdf_to_alpha(mid_sdf: torch.Tensor, zvals: torch.Tensor, mid_slope: torch.Tensor, s, clip=True):
    """Turn sdf to alpha. When s goes to inf, weights focus more on surface

    Args:
        mid_sdf: tensor (B, N_pts-1) of mid pts
        zvals: tensor (B, N_pts)
        mid_slope: tensor (B, N_pts-1) of mid pts
        s: scale factor
        clip: whether to clip alpha to 0-1

    Returns:
        alpha: tensor (B, N_pts-1) of mid pts
    """
    dist = zvals[:, 1:] - zvals[:, :-1]
    prev_esti_sdf = mid_sdf - mid_slope * dist * 0.5  # > mid_sdf, (B, N_pts-1)
    next_esti_sdf = mid_sdf + mid_slope * dist * 0.5  # < mid_sdf, (B, N_pts-1)
    prev_cdf = sdf_to_cdf(prev_esti_sdf, s)
    next_cdf = sdf_to_cdf(next_esti_sdf, s)

    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)  # (B, N_pts-1)
    if clip:
        alpha = alpha.clip(0.0, 1.0)

    return alpha

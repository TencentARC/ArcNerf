# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base_3d_model import Base3dModel
from .base_modules import build_geo_model, build_radiance_model
from arcnerf.render.ray_helper import get_zvals_from_near_far
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class MipNeRF(Base3dModel):
    """ MipNerf model.
        The mip-NeRF model that handles multi-res image using gaussian representation and encoding
        ref: https://github.com/google/mipnerf
    """

    def __init__(self, cfgs):
        super(MipNeRF, self).__init__(cfgs)
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # get gaussian fn
        self.gaussian_fn = get_value_from_cfgs_field(self.cfgs.model.rays, 'gaussian_fn', 'cone')
        # handle embed
        self.ipe_embed_freq = get_value_from_cfgs_field(self.cfgs.model.rays, 'ipe_embed_freq', 12)
        assert self.cfgs.model.geometry.input_ch == 6 * self.ipe_embed_freq, \
            'Incorrect input ch, should be {}'.format(6 * self.ipe_embed_freq)
        assert self.cfgs.model.geometry.embed_freq == 0, 'Should not have extra embedding in geometry model'

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        rays_r = inputs['rays_r']  # (B, 1)

        # get bounds for object
        near, far = self.get_near_far_from_rays(inputs)  # (B, 1) * 2

        # get zvals for each intervals
        zvals = self.get_zvals_from_near_far(near, far, inference_only)  # (B, N_sample+1)

        # get conical frustum
        means, covs = self.get_conical_frustum(rays_o, rays_d, rays_r, zvals)  # (B, N_sample, 3) * 2
        means = means.view(-1, 3)  # (B*N_sample, 3)
        covs = covs.view(-1, 3)  # (B*N_sample, 3)

        # integrated_pos_enc
        pts_embed, _ = self.integrated_pos_enc(means, covs, self.ipe_embed_freq)  # (B*N_sample, 6F)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts_embed, rays_d_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

        # ray marching, use mid pts as interval representation
        zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
        output = self.ray_marching(sigma, radiance, zvals_mid, inference_only=inference_only)

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

    def get_zvals_from_near_far(self, near: torch.Tensor, far: torch.Tensor, inference_only=False):
        """Get the zvals from near/far. Get n+1 zvals for each interval representation.

        It will use ray_cfgs['n_sample'] to select coarse samples.
        Other sample keys are not allowed.

        Args:
            near: torch.tensor (B, 1) near z distance
            far: torch.tensor (B, 1) far z distance
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.

        Returns:
            zvals: torch.tensor (B, N_sample+1)
        """
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.get_ray_cfgs('n_sample') + 1,  # get extra zvals
            inverse_linear=self.get_ray_cfgs('inverse_linear'),
            perturb=self.get_ray_cfgs('perturb') if not inference_only else False
        )  # (B, N_sample+1)

        return zvals

    def get_conical_frustum(self, rays_o, rays_d, rays_r, zvals):
        """Get the mean/cov representation of the conical frustum

        Args:
            rays_o: torch.tensor (B, 3) rays origin
            rays_d: torch.tensor (B, 3) rays direction
            rays_r: torch.tensor (B, 1) radius
            zvals: torch.tensor (B, N+1) sample zvals for each intervals.

        Returns:
            means: means of the ray (B, N, 3)
            covs: covariances of the ray (B, N, 3)
        """
        t_start = zvals[:, :-1]
        t_end = zvals[:, 1:]
        if self.gaussian_fn == 'cone':
            gaussian_fn = conical_frustum_to_gaussian
        elif self.gaussian_fn == 'cylinder':
            gaussian_fn = cylinder_to_gaussian
        else:
            raise NotImplementedError('Invalid gaussian function {}'.format(self.gaussian_fn))
        means, covs = gaussian_fn(rays_d, t_start, t_end, rays_r)  # (B, N, 3) * 2
        means = means + rays_o.unsqueeze(1)  # (B, N, 3)

        return means, covs

    def integrated_pos_enc(self, means, covs, ipe_embed_freq):
        """Get positional encoding from means/cov representation

        Args:
            means: means of the ray (B, 3)
            covs: covariances of the ray (B, 3)
            ipe_embed_freq: maximum embed freq, F

        Returns:
            embed_mean_out: embedded of mean xyz, (B, 3F)
            embed_cov_out: embedded of cov (B, 3F)
        """
        scales = [2**i for i in range(0, ipe_embed_freq)]
        embed_mean = []
        embed_cov = []
        for scale in scales:
            embed_mean.append(means * scale)
            embed_cov.append(covs * scale**2)
        embed_mean = torch.cat(embed_mean, dim=-1)  # (N, 3F)
        embed_cov = torch.cat(embed_cov, dim=-1)  # (N, 3F)
        embed_mean = torch.cat([embed_mean, embed_mean + 0.5 * np.pi], dim=-1)  # (N, 6F)
        embed_cov = torch.cat([embed_cov, embed_cov], dim=-1)  # (N, 6F)

        def safe_trig(x, fn, t=100 * np.pi):
            return fn(torch.where(torch.abs(x) < t, x, x % t))

        embed_mean_out = torch.exp(-0.5 * embed_cov) * safe_trig(embed_mean, torch.sin)
        embed_cov_out = torch.clamp_min(
            0.5 * (1 - torch.exp(-2.0 * embed_cov) * safe_trig(2.0 * embed_mean, torch.cos)) - embed_mean_out**2, 0.0
        )

        return embed_mean_out, embed_cov_out

    def surface_render(
        self,
        inputs,
        method='secant_root_finding',
        n_step=128,
        n_iter=20,
        threshold=0.01,
        level=50.0,
        grad_dir='descent'
    ):
        """For density model, the surface is not exactly accurate. Not suggest ot use this func"""
        raise NotImplementedError('Do not support surface render for mipnerf')


def conical_frustum_to_gaussian(rays_d, t_start, t_end, rays_r):
    """Turn conical frustum into gaussian representation
    Sec 3.1 in paper

    Args:
        rays_d: torch.tensor (B, 3) rays direction
        t_start: (B, N) start zvals for each interv
        t_end: (B, N) end zvals for each interval
        rays_r: torch.tensor (B, 1) basic radius

    Returns:
        means: means of the ray (B, N, 3)
        covs: covariances of the ray (B, N, 3)
    """
    mu = (t_start + t_end) / 2.0  # (B, N)
    hw = (t_end - t_start) / 2.0  # (B, N)
    common_term = 3.0 * mu**2 + hw**2  # (B, N)
    t_mean = mu + (2.0 * mu * hw**2) / common_term  # (B, N)
    t_var = (hw**2) / 3.0 - (4.0 / 15.0) * ((hw**4 * (12.0 * mu**2 - hw**2)) / common_term**2)  # (B, N)
    r_var = rays_r**2 * ((mu**2) / 4.0 + (5.0 / 12.0) * hw**2 - (4.0 / 15.0) * (hw**4) / common_term)  # (B, N)
    mean, covs = lift_gaussian(rays_d, t_mean, t_var, r_var)

    return mean, covs


def cylinder_to_gaussian(rays_d, t_start, t_end, rays_r):
    """Turn cylinder frustum into gaussian representation

    Args:
        rays_d: torch.tensor (B, 3) rays direction
        t_start: (B, N) start zvals for each interv
        t_end: (B, N) end zvals for each interval
        rays_r: torch.tensor (B, 1) radius

    Returns:
        means: means of the ray (B, N, 3)
        covs: covariances of the ray (B, N, 3)
    """
    t_mean = (t_start + t_end) / 2.0  # (B, N)
    t_var = (t_end - t_start)**2 / 12.0  # (B, N)
    r_var = rays_r**2 / 4.0  # (B, N)
    mean, covs = lift_gaussian(rays_d, t_mean, t_var, r_var)

    return mean, covs


def lift_gaussian(rays_d, t_mean, t_var, r_var):
    """Lift mu/t to rays gaussian mean/var

    Args:
        rays_d: direction (B, 3)
        t_mean: mean (B, N) of each interval along ray
        t_var: variance (B, N) of each interval along ray
        r_var: variance (B, N) of each interval perpendicular to ray

    Returns:
        means: means of the ray (B, N, 3)
        covs: covariances of the ray (B, N, 3)
    """
    mean = rays_d.unsqueeze(1) * t_mean.unsqueeze(-1)  # (B, N, 3)
    d_mag_sq = torch.clamp_min(torch.sum(rays_d**2, dim=-1, keepdim=True), 1e-10)  # (B, 3)
    d_outer_diag = rays_d**2  # (B, 3)
    null_outer_diag = 1 - d_outer_diag / d_mag_sq  # (B, 3)
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]  # (B, N, 3)
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]  # (B, N, 3)
    cov_diag = t_cov_diag + xy_cov_diag  # (B, N, 3)

    return mean, cov_diag

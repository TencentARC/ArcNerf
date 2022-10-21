# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .sdf_model import SdfModel
from .base_modules import build_geo_model, build_radiance_model


@MODEL_REGISTRY.register()
class VolSDF(SdfModel):
    """ VolSDF model.
        Model SDF and convert it to density.
        ref: https://github.com/lioryariv/volsdf
             https://github.com/ventusff/neurecon#volume-rendering--3d-implicit-surface
    """

    def __init__(self, cfgs):
        super(VolSDF, self).__init__(cfgs)
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # custom rays cfgs for upsampling
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        self.ray_cfgs['n_eval'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_eval', 128)
        self.ray_cfgs['n_iter'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_iter', 5)
        assert self.get_ray_cfgs('n_iter'), 'You must have at least one iter for sampling'
        self.ray_cfgs['beta_iter'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'beta_iter', 10)
        self.ray_cfgs['eps'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'eps', 0.1)
        # radius init for object
        self.radius_init = get_value_from_cfgs_field(self.cfgs.model.geometry, 'radius_init', 1.0)
        self.ln_beta, self.beta_min, self.speed_factor = self.get_params()
        # Use bounding radius sampling in volsdf
        self.radius_bound = get_value_from_cfgs_field(self.cfgs.model.rays, 'radius_bound', 1.5)

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

    def get_n_coarse_sample(self):
        """use N_eval instead of using N_sample """
        return self.get_ray_cfgs('n_eval')

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        zvals = inputs['zvals']  # (B, 1)
        mask_pts = inputs['mask_pts']  # (B, n_pts)
        bkg_color = inputs['bkg_color']  # (B, 3)
        n_rays = rays_o.shape[0]

        # sample zvals near surface, (B, N_total(N_sample+N_importance)) for zvals,  (B, 1) for zvals_surface
        zvals, zvals_surface, mask_pts = self.upsample_zvals(
            rays_o, rays_d, zvals, self.forward_pts, mask_pts, inference_only
        )

        # get pts sigma/rgb/normal  (B, N_sample, ...)
        sdf, radiance, normal_pts = self.get_sdf_radiance_normal_by_mask_pts(
            self.geo_net, self.radiance_net, rays_o, rays_d, zvals, mask_pts, inference_only
        )

        # convert sdf to sigma
        sigma = sdf_to_sigma(sdf, self.forward_beta(), self.beta_min)  # (B, N_total)

        # ray marching, sdf will not be used for calculation but for recording
        output = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only, bkg_color=bkg_color)
        normal_pts = normal_pts[:, :output['weights'].shape[1]]  # in case add_inf_z

        # add normal. For normal map, needs to normalize each pts
        output['normal'] = torch.sum(output['weights'].unsqueeze(-1) * normalize(normal_pts), -2)  # (B, 3)
        if not inference_only:
            output['params'] = {'beta': float(self.forward_beta().clone())}
            # only sample some pts in sphere and on surface
            eikonal_pts = self.get_eikonal_pts(rays_o, rays_d, zvals_surface).view(-1, 3)  # (B*2, 3)
            rays_d_repeat = torch.repeat_interleave(rays_d, int(eikonal_pts.shape[0] / n_rays), dim=0)
            _, _, normal_eikonal_pts = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, eikonal_pts,
                rays_d_repeat
            )
            output['normal_pts'] = normal_eikonal_pts.view(n_rays, -1, 3)  # (B, 2, 3), do not normalize it

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

    def get_est_opacity(self, dt, pts):
        """VolSDF model convert sdf to sigma directly"""
        sdf = self.forward_pts(pts)  # (B,)
        density = sdf_to_sigma(sdf, self.forward_beta(), self.beta_min)
        opacity = 1.0 - torch.exp(-torch.relu(density) * dt)  # (B,)

        return opacity

    def upsample_zvals(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        zvals: torch.Tensor,
        sdf_func,
        mask_pts=None,
        inference_only=False
    ):
        """Sample zvals near surface

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals: tensor (B, N_eval), coarse zvals for all rays, uniformly distributed in (near, far)
            sdf_func: generally the model.forward_pts() func to get sdf from pts.
                       You can call it outside for debug.
            mask_pts: tensor (B, N_sample) whether each pts is valid. None means all valid.
            inference_only: affect the sample_pdf deterministic. By default False(For train)

        Returns:
            zvals: tensor (B, N_sample + N_importance), sample zvals near the surface
            zvals_surface: (B, 1) zvals on surface
            mask_pts: new mask in (B, N_sample + N_importance). This not work. Always return None.
        """
        dtype = zvals.dtype
        device = zvals.device
        n_rays = zvals.shape[0]

        samples, samples_idx = zvals, None
        sdf = None

        beta0 = self.forward_beta().detach()
        eps = self.get_ray_cfgs('eps')
        cur_iter, not_converge = 0, True

        # Get maximum beta from the upper bound (Lemma 2)
        dists = zvals[:, 1:] - zvals[:, :-1]  # (B, N_eval-1)
        log_eps_one = torch.log(torch.tensor(eps + 1.0, dtype=dtype, device=device))
        bound = (1.0 / (4.0 * log_eps_one)) * (dists**2.).sum(-1)
        beta = torch.sqrt(bound)

        # Algorithm 1
        while not_converge and cur_iter < self.get_ray_cfgs('n_iter'):
            pts = get_ray_points_by_zvals(rays_o, rays_d, samples).view(-1, 3)  # (B*N_eval, 3)
            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                sample_sdf = sdf_func(pts).unsqueeze(-1)  # (B*N_eval, 1)

            if samples_idx is not None:
                sdf_cat = torch.cat([sdf, sample_sdf.reshape(n_rays, -1)], dim=-1)  # (B, (iter+1)*N_eval)
                sdf = torch.gather(sdf_cat, 1, samples_idx)  # (B, (iter+1)*N_eval) sorted concat sdf
            else:
                sdf = sample_sdf.view(n_rays, -1)  # (B, N_eval), init uniform pts

            # Calculating the bound d* (Theorem 1)
            dists = zvals[:, 1:] - zvals[:, :-1]
            d_star = self.get_d_star(zvals, sdf)  # (B, iter*N_eval-1)

            # Updating beta using line search
            cur_error = self.get_error_bound(beta0, sdf, zvals, d_star)  # (B, )
            beta[cur_error <= eps] = beta0
            beta_min, beta_max = beta0.repeat(zvals.shape[0]), beta  # (B, ) * 2
            for _ in range(self.get_ray_cfgs('beta_iter')):
                beta_mid = (beta_min + beta_max) / 2.  # (B, )
                cur_error = self.get_error_bound(beta_mid.unsqueeze(-1), sdf, zvals, d_star)
                beta_max[cur_error <= eps] = beta_mid[cur_error <= eps]
                beta_min[cur_error > eps] = beta_mid[cur_error > eps]
            beta = beta_max  # (B, )

            # get new weights
            sigma = sdf_to_sigma(sdf, beta.unsqueeze(-1), self.beta_min)
            output = self.ray_marching(sigma, None, zvals, True, None, inference_only, weights_only=False)
            trans_shift, weights = output['trans_shift'], output['weights']  # (B, iter*N_eval) * 2

            # check converge
            cur_iter += 1
            not_converge = beta.max() >= beta0

            det = not self.get_ray_cfgs('perturb') if not inference_only else True
            if not_converge and cur_iter < self.get_ray_cfgs('n_iter'):  # not converge
                n_pts = self.get_ray_cfgs('n_eval')
                bound_opacity = self.get_integral_bound(trans_shift, beta.unsqueeze(-1), d_star, dists)
                pdf = bound_opacity  # (B, N_eval-1)
                det = True  # force det resample
            else:  # converge, final sample N_sample
                n_pts = self.get_ray_cfgs('n_sample')
                pdf = weights[..., :-1]  # (B, N_sample-1)

            # sample in the iter
            samples = sample_pdf(zvals, pdf, n_pts, det).detach()  # (B, N_eval/sample)

            if not_converge and cur_iter < self.get_ray_cfgs('n_iter'):  # not converge, add to zvals
                zvals, samples_idx = torch.sort(torch.cat([zvals, samples], -1), -1)  # (B, (iter+1)*N_eval)

        zvals_sample = samples  # (B, N_sample)

        # add more pts on the whole ray. Actually take more pts not on surface
        if self.get_ray_cfgs('n_importance') > 0:
            n_importance = self.get_ray_cfgs('n_importance')
            if inference_only:
                sample_idx = torch.linspace(0, zvals.shape[1] - 1, n_importance, dtype=torch.long, device=device)
            else:
                sample_idx = torch.randperm(zvals.shape[1], device=device)[:n_importance]
            zvals_extra = zvals[:, sample_idx]  # (B, N_importance)
            zvals_sample, _ = torch.sort(torch.cat([zvals_sample, zvals_extra], -1), -1)  # (B, N_sample + N_importance)

        # follow volsdf original repo, sampled on the whole ray
        idx = torch.randint(zvals_sample.shape[-1], (zvals_sample.shape[0], ), device=device)
        zvals_surface = torch.gather(zvals_sample, 1, idx.unsqueeze(-1))

        return zvals_sample, zvals_surface, None

    def get_error_bound(self, beta, sdf: torch.Tensor, zvals: torch.Tensor, d_star: torch.Tensor, max_per_ray=True):
        """Calculate the error bound from approximate integration.
        Theorem 1, eq.12 in paper.

        Args:
            beta: beta value. single value or tensor (B, N_pts/1)
            sdf: torch.tensor (B, N_pts) sdf on ray
            zvals: torch.tensor (B, N_pts) zvals of the points
            d_star: torch.tensor (B, N_pts-1) the bounding d in each interval (zi, zi+1)
            max_per_ray: If True, return the max error on each ray. By default True

        Returns:
            bound_opacity: torch.tensor (B,) the max bounding error of each ray from integral
                           If not max_per_ray, return (B, N_pts-1)
        """
        dtype = zvals.dtype
        device = zvals.device

        dists = zvals[:, 1:] - zvals[:, :-1]
        sigma = sdf_to_sigma(sdf, beta, self.beta_min)  # (B, N_pts)
        zeros = torch.zeros(dists.shape[0], 1, dtype=dtype, device=device)  # (B, 1)
        shifted_free_energy = torch.cat([zeros, dists * sigma[:, :-1]], dim=-1)  # (B, N_pts)
        integral_esti = torch.cumsum(shifted_free_energy, dim=-1)
        bound_opacity = self.get_integral_bound(integral_esti, beta, d_star, dists)  # (B, N_pts-1)

        if max_per_ray:
            bound_opacity = bound_opacity.max(-1)[0]  # (B,)

        return bound_opacity

    @staticmethod
    def get_d_star(zvals: torch.Tensor, sdf: torch.Tensor):
        """Calculate the d_star value between interval

        Args:
            zvals: torch.tensor (B, N_pts) zvals of the points
            sdf: torch.tensor (B, N_pts) sdf on ray

        Returns:
            d_star: torch.tensor (B, N_pts-1) the bounding d in each interval (zi, zi+1)
        """
        dtype = zvals.dtype
        device = zvals.device

        dists = zvals[:, 1:] - zvals[:, :-1]
        a, b, c = dists, sdf[:, :-1].abs(), sdf[:, 1:].abs()  # (B, N_pts-1)
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(zvals.shape[0], zvals.shape[1] - 1, dtype=dtype, device=device)  # (B, N_pts-1)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0  # (B, N_eval-1)
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (sdf[:, 1:].sign() * sdf[:, :-1].sign() == 1) * d_star  # (B, N_pts-1)

        return d_star

    @staticmethod
    def get_integral_bound(integral_esti, beta, d_star, dists):
        """Sub-func under error bound calculation

        Args:
            integral_esti: estimated integral value
            beta: beta value. single value or tensor (B, N_pts/1)
            d_star: torch.tensor (B, N_pts-1) the bounding d in each interval (zi, zi+1)
            dists: torch.tensor (B, N_pts-1) dists of intervals.

        Returns:
            bound_opacity: torch.tensor (B, N_pts-1) the bounding error of each ray from integral
        """
        error_per_section = torch.exp(-d_star / beta) * (dists**2.) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)  # (B, N_pts)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_esti[:, :-1])

        return bound_opacity

    def get_eikonal_pts(self, rays_o: torch.Tensor, rays_d: torch.Tensor, zvals_surface: torch.Tensor):
        """Get random eikonal pts from bounding sphere and near surface
        sec 3.5 in paper.

        Args:
            rays_o: torch.tensor (B, 3), cam_loc/ray_start position
            rays_d: torch.tensor (B, 3), view dir(assume normed)
            zvals_surface: tensor (B, 1), zvals on surface. If None, do not sample on surface

        Returns:
            pts_all: torch.tensor (B, 2, 3), in sphere pts + surface pts. (B, 1, 3) if zvals_surface is None.
        """
        dtype = rays_o.dtype
        device = rays_o.device

        # random pts in sphere, (B, 1, 3)
        pts_rand = torch.empty(size=(rays_o.shape[0], 1, 3), dtype=dtype, device=device)\
            .uniform_(-self.radius_bound, self.radius_bound)
        pts_rand = pts_rand / torch.norm(pts_rand, dim=-1, keepdim=True).max() * self.radius_bound  # make sure in bound

        # pts on surface, (B, 1, 3)
        pts_surface = None
        if zvals_surface is not None:
            pts_surface = get_ray_points_by_zvals(rays_o, rays_d, zvals_surface)

        # merge pts
        if pts_surface is None:
            pts_all = pts_rand  # (B, 1, 3)
        else:
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
        beta: beta for cdf adjustment. single value or tensor (B, N_pts/1)
        beta_min: add to beta in case beta too small

    Returns:
        sigma: tensor (B, N_pts) of pts
    """
    beta = beta + beta_min
    alpha = 1 / beta

    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    sigma = alpha * torch.where(sdf >= 0, exp, 1 - exp)

    return sigma

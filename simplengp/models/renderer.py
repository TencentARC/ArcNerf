# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import get_value_from_cfgs_field
from simplengp.ops import calc_rgb_bp, calc_rgb_nobp, fill_ray_marching_inputs


class Renderer(nn.Module):
    """A ray marching renderer """

    def __init__(self, cfgs):
        super(Renderer, self).__init__()

        # cfgs
        self.early_stop = get_value_from_cfgs_field(cfgs, 'early_stop', 0.0)
        self.bkg_color = get_value_from_cfgs_field(cfgs, 'bkg_color', [1.0, 1.0, 1.0])  # default white
        self.torch_render = get_value_from_cfgs_field(cfgs, 'torch_render', False)

    def render(self, sigma, radiance, numsteps_in, dt, bkg_color, inference_only):
        """Render rgb"""
        if self.torch_render:
            # get input
            _sigma, _radiance, _dt = prepare_ray_marching_data(sigma, radiance, dt, numsteps_in)
            return ray_marching_render(_sigma, _radiance, _dt, bkg_color)

        if inference_only or bkg_color is None:  # val and test
            return self.render_test(sigma, radiance, numsteps_in, dt)
        else:
            return self.render_train(sigma, radiance, numsteps_in, dt, bkg_color)

    def render_train(self, sigma, radiance, numsteps_in, dt, bkg_color):
        """Render train rays with bkg_color, need backward"""
        return calc_rgb_bp(sigma, radiance, numsteps_in, dt, bkg_color, self.early_stop)

    @torch.no_grad()
    def render_test(self, sigma, radiance, numsteps_in, dt):
        """Render test rays with pre-setting bkg_color"""
        bkg_color = torch.tensor(self.bkg_color, dtype=sigma.dtype, device=sigma.device)

        return calc_rgb_nobp(sigma, radiance, numsteps_in, dt, bkg_color, self.early_stop)


def ray_marching_render(sigma, radiance, dt, bkg_color):
    """Ray marching in torch, which is the same as NeRF"""
    alpha = 1 - torch.exp(-torch.relu(sigma) * dt)
    trans_shift, weights = alpha_to_weights(alpha)  # (N_rays, N_p) * 2
    mask = torch.sum(weights, -1)  # (N_rays)

    rgb = torch.sum(weights.unsqueeze(-1) * radiance, -2)  # (N_rays, 3)
    if bkg_color is not None:
        assert bkg_color.shape[0] == rgb.shape[0] or bkg_color.shape[0] == 1, 'Only bkg with N_rays/1 allowed..'
        rgb = rgb + trans_shift[:, -1:] * bkg_color

    return rgb, mask


def prepare_ray_marching_data(sigma, radiance, dt, numsteps_in):
    """Prepare tensor in (n_rays, n_pts)"""
    n_rays = numsteps_in.shape[0]
    max_n_pts = int(torch.max(numsteps_in[:, 0]))

    _sigma = torch.zeros((n_rays, max_n_pts), dtype=sigma.dtype, device=sigma.device)  # (n_rays, n_pts)
    _radiance = torch.zeros((n_rays, max_n_pts, 3), dtype=radiance.dtype, device=radiance.device)  # (n_rays, n_pts, 3)
    _dt = torch.zeros((n_rays, max_n_pts), dtype=radiance.dtype, device=radiance.device)  # (n_rays, n_pts)

    _sigma, _radiance, _dt = fill_ray_marching_inputs(sigma, radiance, dt, numsteps_in, _sigma, _radiance, _dt)

    return _sigma, _radiance, _dt


def alpha_to_weights(alpha: torch.Tensor):
    """Alpha to transmittance and weights"""
    dtype = alpha.dtype
    device = alpha.device
    alpha_one = torch.ones_like(alpha[:, :1], dtype=dtype, device=device)
    trans_shift = torch.cat([alpha_one, 1 - alpha + 1e-10], -1)  # (N_rays, N_p+1)
    trans_shift = torch.cumprod(trans_shift, -1)[:, :-1]  # (N_rays, N_p)
    weights = alpha * trans_shift  # (N_rays, N_p)

    return trans_shift, weights

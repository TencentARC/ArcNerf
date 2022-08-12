# -*- coding: utf-8 -*-

import torch

from .fg_model import FgModel
from .base_modules.encoding.gaussian_encoder import Gaussian
from .base_modules import build_geo_model, build_radiance_model
from arcnerf.render.ray_helper import sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class MipNeRF(FgModel):
    """ MipNerf model.
        The mip-NeRF model that handles multi-res image using gaussian representation and encoding
        ref: https://github.com/google/mipnerf
    """

    def __init__(self, cfgs):
        super(MipNeRF, self).__init__(cfgs)
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # importance sampling
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        # set gaussian
        gaussian_fn = get_value_from_cfgs_field(self.cfgs.model.rays.gaussian, 'gaussian_fn', 'cone')
        self.gaussian = Gaussian(gaussian_fn)
        # blur coarse weights
        self.blur_coarse_weights = get_value_from_cfgs_field(self.cfgs.model.rays, 'blur_coarse_weights', False)

    def get_n_coarse_sample(self):
        """make one more sample for interval modeling"""
        return self.get_ray_cfgs('n_sample') + 1

    def _forward(self, inputs, zvals, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """zvals is in shape (n_sample+1)"""
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        rays_r = inputs['rays_r']  # (B, 1)
        n_rays = rays_o.shape[0]
        output = {}

        # get mean/cov representation of intervals
        intervals = self.gaussian(zvals, rays_o, rays_d, rays_r)  # (B, N_sample, 6)
        intervals = intervals.view(-1, intervals.shape[-1])  # (B*N_sample, 6)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, int(intervals.shape[0] / n_rays), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, intervals, rays_d_repeat
        )

        # reshape
        sigma = sigma.view(n_rays, -1)  # (B, N_sample)
        radiance = radiance.view(n_rays, -1, 3)  # (B, N_sample, 3)

        # ray marching for coarse network, keep the coarse weights for next stage, use mid pts for interval
        zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
        output_coarse = self.ray_marching(sigma, radiance, zvals_mid, inference_only=inference_only)
        coarse_weights = output_coarse['weights']

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals, do not concat with original ones
            zvals = self.upsample_zvals(zvals_mid, coarse_weights, inference_only)  # (B, N_importance+1)

            # get mean/cov representation of intervals
            intervals = self.gaussian(zvals, rays_o, rays_d, rays_r)  # (B, N_importance, 6)
            intervals = intervals.view(-1, intervals.shape[-1])  # (B*N_importance, 6)

            # get sigma and rgb, expand rays_d to all pts. shape in (B*N_importance, ...)
            rays_d_repeat = torch.repeat_interleave(rays_d, int(intervals.shape[0] / n_rays), dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, intervals, rays_d_repeat
            )

            # reshape
            sigma = sigma.view(n_rays, -1)  # (B, N_importance)
            radiance = radiance.view(n_rays, -1, 3)  # (B, N_importance, 3)

            # ray marching for fine network, keep the coarse weights for next stage, use mid pts for interval
            zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
            output_fine = self.ray_marching(sigma, radiance, zvals_mid, inference_only=inference_only)

            # handle progress
            output['fine'] = self.output_get_progress(output_fine, get_progress)

        # adjust two stage output
        output = self.adjust_coarse_fine_output(output, inference_only)

        return output

    def upsample_zvals(self, zvals: torch.Tensor, weights: torch.Tensor, inference_only=True):
        """Upsample zvals if N_importance > 0. Similar to nerf. But it returns resampled zvals only

        Args:
            zvals: tensor (B, N_sample), coarse zvals for all rays
            weights: tensor (B, N_sample) (B, N_sample(-1))
            inference_only: affect the sample_pdf deterministic. By default False(For train)

        Returns:
            zvals: tensor (B, N_importance+1), up-sample zvals near the surface
        """
        if self.blur_coarse_weights:
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)  # (B, N_sample+2)
            weights_max = torch.max(weights_pad[..., :-1], weights_pad[..., 1:])  # (B, N_sample+1)
            weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])  # (B, N_sample)

        weights_coarse = weights[:, 1:self.get_n_coarse_sample() - 2]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse,
            self.get_ray_cfgs('n_importance') + 1, not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()

        return _zvals

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

# -*- coding: utf-8 -*-

import torch

from .fg_model import FgModel
from .base_modules import build_geo_model, build_radiance_model
from arcnerf.render.ray_helper import sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class NeRF(FgModel):
    """ Nerf model.
        The two-stage nerf use coarse/fine models for different stage, instead of using just one.
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        self.coarse_geo_net = build_geo_model(self.cfgs.model.geometry)
        self.coarse_radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # custom rays cfgs
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        self.ray_cfgs['shared_network'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'shared_network', False)
        # set fine model if n_importance > 0
        if self.get_ray_cfgs('n_importance') > 0:
            if self.get_ray_cfgs('shared_network'):  # use the same network
                self.fine_geo_net = self.coarse_geo_net
                self.fine_radiance_net = self.coarse_radiance_net
            else:  # separate network
                self.fine_geo_net = build_geo_model(self.cfgs.model.geometry)
                self.fine_radiance_net = build_radiance_model(self.cfgs.model.radiance)

    def get_net(self):
        """Get the actual net for usage"""
        if self.get_ray_cfgs('n_importance') > 0:
            geo_net = self.fine_geo_net
            radiance_net = self.fine_radiance_net
        else:
            geo_net = self.coarse_geo_net
            radiance_net = self.coarse_radiance_net

        return geo_net, radiance_net

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model"""
        self.coarse_geo_net.pretrain_siren()
        if self.get_ray_cfgs('n_importance') > 0:
            self.fine_geo_net.pretrain_siren()

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        zvals = inputs['zvals']  # (B, n_pts)
        mask_pts = inputs['mask_pts']  # (B, n_pts)
        bkg_color = inputs['bkg_color']  # (B, 3)
        output = {}

        # get coarse pts sigma/rgb  (B, N_sample, ...)
        sigma, radiance = self.get_sigma_radiance_by_mask_pts(
            self.coarse_geo_net, self.coarse_radiance_net, rays_o, rays_d, zvals, mask_pts
        )
        # ray marching for coarse network, keep the coarse weights for next stage
        output_coarse = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only, bkg_color=bkg_color)
        coarse_weights = output_coarse['weights']

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals
            zvals = self.upsample_zvals(zvals, coarse_weights, inference_only)  # TODO: Upsample new mask

            # get upsampled pts sigma/rgb  (B, N_total, ...)
            sigma, radiance = self.get_sigma_radiance_by_mask_pts(
                self.fine_geo_net,
                self.fine_radiance_net,
                rays_o,
                rays_d,
                zvals,
                None  # TODO: Full mask
            )

            # ray marching for fine network
            output_fine = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only, bkg_color=bkg_color)

            # handle progress
            output['fine'] = self.output_get_progress(output_fine, get_progress)

        # adjust two stage output
        output = self.adjust_coarse_fine_output(output, inference_only)

        return output

    def upsample_zvals(self, zvals: torch.Tensor, weights: torch.Tensor, inference_only=True):
        """Upsample zvals if N_importance > 0

        Args:
            zvals: tensor (B, N_sample), coarse zvals for all rays
            weights: tensor (B, N_sample) (B, N_sample(-1))
            inference_only: affect the sample_pdf deterministic. By default False(For train)

        Returns:
            zvals: tensor (B, N_sample + N_importance), up-sample zvals near the surface
        """
        weights_coarse = weights[:, 1:self.get_ray_cfgs('n_sample') - 1]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse, self.get_ray_cfgs('n_importance'),
            not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()
        zvals, _ = torch.sort(torch.cat([zvals, _zvals], -1), -1)  # (B, N_sample+N_importance=N_total)

        return zvals

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
        assert grad_dir == 'descent', 'Invalid for density model in nerf...'
        assert method != 'sphere_tracing', 'Do not support for density model in nerf...'

        # call parent class
        output = super().surface_render(inputs, method, n_step, n_iter, threshold, level, grad_dir)

        return output

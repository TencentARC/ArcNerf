# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules.encoding.gaussian_encoder import Gaussian
from .base_modules import build_geo_model, build_radiance_model
from arcnerf.render.ray_helper import get_zvals_from_near_far, sample_pdf
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
        # importance sampling
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        # set gaussian
        gaussian_fn = get_value_from_cfgs_field(self.cfgs.model.rays.gaussian, 'gaussian_fn', 'cone')
        self.gaussian = Gaussian(gaussian_fn)

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        rays_r = inputs['rays_r']  # (B, 1)
        output = {}

        # get bounds for object
        near, far = self.get_near_far_from_rays(inputs)  # (B, 1) * 2

        # get zvals for each intervals
        zvals = self.get_zvals_from_near_far(near, far, inference_only)  # (B, N_sample+1)

        # get mean/cov representation of intervals
        intervals = self.gaussian(zvals, rays_o, rays_d, rays_r)  # (B, N_sample, 6)
        intervals = intervals.view(-1, intervals.shape[-1])  # (B*N_sample, 6)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, intervals, rays_d_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

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
            intervals = intervals.view(-1, intervals.shape[-1])  # (B*N_sample, 6)

            # get sigma and rgb, expand rays_d to all pts. shape in (B*N_importance, ...)
            rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_importance'), dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, intervals, rays_d_repeat
            )

            # reshape
            sigma = sigma.view(-1, self.get_ray_cfgs('n_importance'))  # (B, N_importance)
            radiance = radiance.view(-1, self.get_ray_cfgs('n_importance'), 3)  # (B, N_importance, 3)

            # ray marching for fine network, keep the coarse weights for next stage, use mid pts for interval
            zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
            output_fine = self.ray_marching(sigma, radiance, zvals_mid, inference_only=inference_only)

            # handle progress
            output['fine'] = self.output_get_progress(output_fine, get_progress)

        # adjust two stage output
        output = self.adjust_coarse_fine_output(output, inference_only)

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

    def upsample_zvals(self, zvals: torch.Tensor, weights: torch.Tensor, inference_only=True):
        """Upsample zvals if N_importance > 0. Similar to nerf. But it returns resampled zvals only

        Args:
            zvals: tensor (B, N_sample), coarse zvals for all rays
            weights: tensor (B, N_sample) (B, N_sample(-1))
            inference_only: affect the sample_pdf deterministic. By default False(For train)

        Returns:
            zvals: tensor (B, N_importance+1), up-sample zvals near the surface
        """
        weights_coarse = weights[:, 1:self.get_ray_cfgs('n_sample') - 1]  # (B, N_sample-2)
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

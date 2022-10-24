# -*- coding: utf-8 -*-

import torch

from arcnerf.render.ray_helper import sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .fg_model import FgModel
from .base_modules.encoding.gaussian_encoder import Gaussian
from .base_modules import build_geo_model, build_radiance_model


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
        # blur weights
        self.blur_coarse_weights = get_value_from_cfgs_field(self.cfgs.model.rays, 'blur_coarse_weights', False)

    def get_n_coarse_sample(self):
        """make one more sample for interval modeling"""
        return self.get_ray_cfgs('n_sample') + 1

    def get_sigma_radiance_by_mask_pts(
        self, geo_net, radiance_net, rays_o, rays_d, intervals, mask_pts=None, inference_only=False
    ):
        """Process the pts/dir by mask_pts. Only process valid zvals to save computation

        Args:
            geo_net: geometry net
            radiance_net: radiance net
            rays_o: (B, 3) rays origin
            rays_d: (B, 3) rays direction(normalized)
            intervals: (B, N_pts, 6) intervals on each ray
            mask_pts: (B, N_pts) whether each pts is valid. If None, process all the pts
            inference_only: Whether its in the inference mode

        Returns:
            sigma: (B, N_pts) sigma on all pts. Duplicated pts share the same value
            radiance: (B, N_pts, 3) rgb on all pts. Duplicated pts share the same value
        """
        n_rays = intervals.shape[0]
        n_pts = intervals.shape[1]
        dtype = intervals.dtype
        device = intervals.device

        # get points, expand rays_d to all pts
        rays_d_repeat = torch.repeat_interleave(rays_d.unsqueeze(1), n_pts, dim=1)  # (B, N_pts, 3)

        if mask_pts is None:
            intervals = intervals.view(-1, 6)  # (B*N_pts, 6)
            rays_d_repeat = rays_d_repeat.view(-1, 3)  # (B*N_pts, 3)
        else:
            intervals = intervals[mask_pts].view(-1, 6)  # (N_valid_pts, 6)
            rays_d_repeat = rays_d_repeat[mask_pts].view(-1, 3)  # (N_valid_pts, 3)
            # adjust dynamic batchsize factor when mask_pts is not None
            if not inference_only:
                self.adjust_dynamicbs_factor(mask_pts)

        # get sigma and rgb, . shape in (N_valid_pts, ...)
        _sigma, _radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, geo_net, radiance_net, intervals, rays_d_repeat
        )

        # reshape to (B, N_sample, ...) by fill duplicating pts
        if mask_pts is None:
            sigma = _sigma.view(n_rays, -1)  # (B, N_sample)
            radiance = _radiance.view(n_rays, -1, 3)  # (B, N_sample, 3)
        else:
            last_pts_idx = torch.cumsum(mask_pts.sum(dim=1), dim=0) - 1  # index on flatten sigma/radiance
            last_sigma, last_radiance = _sigma[last_pts_idx], _radiance[last_pts_idx]  # (B,) (B, 3)
            sigma = torch.ones((n_rays, n_pts), dtype=dtype, device=device) * last_sigma.unsqueeze(1)
            radiance = torch.ones((n_rays, n_pts, 3), dtype=dtype, device=device) * last_radiance.unsqueeze(1)
            sigma[mask_pts] = _sigma
            radiance[mask_pts] = _radiance

        return sigma, radiance

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        """zvals is in shape (n_sample+1)"""
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        rays_r = inputs['rays_r']  # (B, 1)
        zvals = inputs['zvals']  # (B, 1)
        mask_pts = inputs['mask_pts']  # (B, n_pts)
        bkg_color = inputs['bkg_color']  # (B, 3)
        output = {}

        # get mean/cov representation of intervals
        intervals = self.gaussian(zvals, rays_o, rays_d, rays_r)  # (B, N_sample, 6)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        sigma, radiance = self.get_sigma_radiance_by_mask_pts(
            self.geo_net, self.radiance_net, rays_o, rays_d, intervals, mask_pts, inference_only
        )

        # ray marching for coarse network, keep the coarse weights for next stage, use mid pts for interval
        zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
        output_coarse = self.ray_marching(
            sigma, radiance, zvals_mid, inference_only=inference_only, bkg_color=bkg_color
        )
        coarse_weights = output_coarse['weights']

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals, do not concat with original ones
            zvals = self.upsample_zvals(zvals_mid, coarse_weights, inference_only)  # (B, N_importance+1)

            # get mean/cov representation of intervals
            intervals = self.gaussian(zvals, rays_o, rays_d, rays_r)  # (B, N_importance, 6)

            # get upsampled pts sigma/rgb  (B, N_importance, ...), new Mask must be None
            sigma, radiance = self.get_sigma_radiance_by_mask_pts(
                self.geo_net, self.radiance_net, rays_o, rays_d, intervals, None, inference_only
            )

            # ray marching for fine network, keep the coarse weights for next stage, use mid pts for interval
            zvals_mid = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
            output_fine = self.ray_marching(
                sigma, radiance, zvals_mid, inference_only=inference_only, bkg_color=bkg_color
            )

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
            weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:]) + 0.01  # (B, N_sample)

        weights_coarse = weights[:, 1:self.get_n_coarse_sample() - 2]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse,
            self.get_ray_cfgs('n_importance') + 1, not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()

        return _zvals

    def get_est_opacity(self, dt, pts):
        """For mip-nerf model, seem volume pruning is hard to work for any single pts"""
        raise NotImplementedError('Do not support opa calculation for mipnerf')

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

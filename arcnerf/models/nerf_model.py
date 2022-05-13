# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class NeRF(Base3dModel):
    """ Nerf model. 8 layers in GeoNet and 1 layer in RadianceNet
        The two-stage nerf use coarse/fine models for different stage, instead of using just one.
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        self.coarse_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.coarse_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # custom rays cfgs
        self.ray_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        # set fine model if n_importance > 0
        if self.get_ray_cfgs('n_importance') > 0:
            self.fine_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
            self.fine_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model"""
        self.coarse_geo_net.pretrain_siren()
        if self.get_ray_cfgs('n_importance') > 0:
            self.fine_geo_net.pretrain_siren()

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        output = {}

        # get bounds for object
        near, far = self.get_near_far_from_rays(inputs)  # (B, 1) * 2

        # coarse model
        # get zvals
        zvals = self.get_zvals_from_near_far(near, far, inference_only)  # (B, N_sample)  # (B, N_sample)

        # get points
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = pts.view(-1, 3)  # (B*N_sample, 3)

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.coarse_geo_net, self.coarse_radiance_net, pts,
            rays_d_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

        # ray marching for coarse network, keep the coarse weights for next stage
        output_coarse = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)
        coarse_weights = output_coarse['weights']

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals
            zvals = self.upsample_zvals(zvals, coarse_weights, inference_only)
            n_total = self.get_ray_cfgs('n_sample') + self.get_ray_cfgs('n_importance')

            # get upsampled pts
            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_total, 3)
            pts = pts.view(-1, 3)  # (B*N_total, 3)

            # get sigma and rgb, expand rays_d to all pts. shape in (B*N_total, ...)
            rays_d_repeat = torch.repeat_interleave(rays_d, n_total, dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.fine_geo_net, self.fine_radiance_net, pts,
                rays_d_repeat
            )

            # reshape
            sigma = sigma.view(-1, n_total)  # (B, n_total)
            radiance = radiance.view(-1, n_total, 3)  # (B, n_total, 3)

            # ray marching for fine network
            output_fine = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)

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
        weights_coarse = weights[:, :self.get_ray_cfgs('n_sample') - 2]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse, self.get_ray_cfgs('n_importance'),
            not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()
        zvals, _ = torch.sort(torch.cat([zvals, _zvals], -1), -1)  # (B, N_sample+N_importance=N_total)

        return zvals

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Rewrite for two stage implementation. """
        if self.get_ray_cfgs('n_importance') > 0:
            geo_net = self.fine_geo_net
            radiance_net = self.fine_radiance_net
        else:
            geo_net = self.coarse_geo_net
            radiance_net = self.coarse_radiance_net

        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sigma, rgb = chunk_processing(self._forward_pts_dir, self.chunk_pts, False, geo_net, radiance_net, pts, rays_d)

        return sigma, rgb

    def forward_pts(self, pts: torch.Tensor):
        """Rewrite for two stage implementation. """
        if self.get_ray_cfgs('n_importance') > 0:
            geo_net = self.fine_geo_net
        else:
            geo_net = self.coarse_geo_net

        sigma, _ = chunk_processing(geo_net, self.chunk_pts, False, pts)

        return sigma[..., 0]

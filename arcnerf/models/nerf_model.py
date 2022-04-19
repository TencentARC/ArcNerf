# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_near_far_from_rays, get_zvals_from_near_far, ray_marching, sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


@MODEL_REGISTRY.register()
class NeRF(Base3dModel):
    """ Nerf model. 8 layers in GeoNet and 1 layer in RadianceNet
        ref: https://www.matthewtancik.com/nerf
    """

    def __init__(self, cfgs):
        super(NeRF, self).__init__(cfgs)
        self.coarse_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
        self.coarse_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)
        # custom rays cfgs
        self.rays_cfgs['n_importance'] = get_value_from_cfgs_field(self.cfgs.model.rays, 'n_importance', 0)
        # set fine model if n_importance > 0
        if self.rays_cfgs['n_importance'] > 0:
            self.fine_geo_net = GeoNet(**self.cfgs.model.geometry.__dict__)
            self.fine_radiance_net = RadianceNet(**self.cfgs.model.radiance.__dict__)

    def _forward(self, inputs, inference_only=False, get_progress=False):
        """
        All the tensor are in chunk. B is total num of rays by grouping different samples in batch
        Args:
            inputs['img']: torch.tensor (B, 3), rgb value in 0-1
            inputs['rays_o']: torch.tensor (B, 3), cam_loc/ray_start position
            inputs['rays_d']: torch.tensor (B, 3), view dir(assume normed)
            inputs['mask']: torch.tensor (B,), mask value in {0, 1}. optional
            inputs['bound']: torch.tensor (B, 2)
            inference_only: If True, will not output coarse results. By default False
            get_progress: If True, output some progress for recording, can not used in inference only mode.
                          By default False

        Returns:
            output is a dict with following keys:
                coarse_rgb: torch.tensor (B, 3), only if inference_only=False
                coarse_depth: torch.tensor (B,), only if inference_only=False
                coarse_mask: torch.tensor (B,), only if inference_only=False
                fine_rgb: torch.tensor (B, 3)
                fine_depth: torch.tensor (B,)
                fine_mask: torch.tensor (B,)
                If get_progress is True:
                    TODO: visual progress
        """
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        output = {}

        # get bounds for object, (B, 1) * 2
        bounds = None
        if 'bounds' in inputs:
            bounds = inputs['bounds'] if 'bounds' in inputs else None
        near, far = get_near_far_from_rays(
            rays_o, rays_d, bounds, self.rays_cfgs['near'], self.rays_cfgs['far'], self.rays_cfgs['bounding_radius']
        )

        # coarse model
        # get zvals
        zvals = get_zvals_from_near_far(
            near,
            far,
            self.rays_cfgs['n_sample'],
            inverse_linear=self.rays_cfgs['inverse_linear'],
            perturb=self.rays_cfgs['perturb'] if not inference_only else False
        )  # (B, N_sample)

        # get points
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = pts.view(-1, 3)  # (B*N_sample, 3)

        # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_sample, dim)
        sigma, feature = chunk_processing(self.coarse_geo_net, self.chunk_pts, pts)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.rays_cfgs['n_sample'], dim=0)
        radiance = chunk_processing(self.coarse_radiance_net, self.chunk_pts, pts, rays_d_repeat, None, feature)

        # reshape, ray marching and get color/weights
        sigma = sigma.view(-1, self.rays_cfgs['n_sample'], 1)[..., 0]  # (B, N_sample)
        radiance = radiance.view(-1, self.rays_cfgs['n_sample'], 3)  # (B, N_sample, 3)

        # ray marching. If two stage and inference only, get weights from single stage.
        weights_only = inference_only and self.rays_cfgs['n_importance'] > 0
        output_coarse = ray_marching(
            sigma,
            radiance,
            zvals,
            self.rays_cfgs['add_inf_z'],
            self.rays_cfgs['noise_std'] if not inference_only else 0.0,
            weights_only=weights_only
        )
        if not weights_only:
            output['rgb_coarse'] = output_coarse['rgb']  # (B, 3)
            output['depth_coarse'] = output_coarse['depth']  # (B,)
            output['mask_coarse'] = output_coarse['mask']  # (B,)

        # fine model
        if self.rays_cfgs['n_importance'] > 0:
            weights_coarse = output_coarse['weights'][:, :self.rays_cfgs['n_sample'] - 2]  # (B, N_sample-2)
            zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
            _zvals = sample_pdf(
                zvals_mid, weights_coarse, self.rays_cfgs['n_importance'],
                not self.rays_cfgs['perturb'] if not inference_only else True
            ).detach()
            zvals, _ = torch.sort(torch.cat([zvals, _zvals], -1), -1)  # (B, N_sample+N_importance=N_total)
            n_total = self.rays_cfgs['n_sample'] + self.rays_cfgs['n_importance']

            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_total, 3)
            pts = pts.view(-1, 3)  # (B*N_total, 3)

            # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_total, dim)
            sigma, feature = chunk_processing(self.fine_geo_net, self.chunk_pts, pts)
            rays_d_repeat = torch.repeat_interleave(rays_d, n_total, dim=0)
            radiance = chunk_processing(self.fine_radiance_net, self.chunk_pts, pts, rays_d_repeat, None, feature)

            # reshape, ray marching and get color/weights
            sigma = sigma.view(-1, n_total, 1)[..., 0]  # (B, n_total)
            radiance = radiance.view(-1, n_total, 3)  # (B, n_total, 3)

            output_fine = ray_marching(
                sigma, radiance, zvals, self.rays_cfgs['add_inf_z'],
                self.rays_cfgs['noise_std'] if not inference_only else 0.0
            )

            output['rgb_fine'] = output_fine['rgb']  # (B, 3)
            output['depth_fine'] = output_fine['depth']  # (B,)
            output['mask_fine'] = output_fine['mask']  # (B,)

        return output

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor):
        """This function forward pts and view dir directly, only for inference the geometry/color

        Args:
            pts: torch.tensor (N_pts, 3), pts in world coord
            view_dir: torch.tensor (N_pts, 3) view dir associate with each point
        Returns:
            output is a dict with following keys:
                sigma: torch.tensor (N_pts), density value for each point
                rgb: torch.tensor (N_pts, 3), color for each point
        """
        if self.rays_cfgs['n_importance'] > 0:
            geo_net = self.fine_geo_net
            radiance_net = self.fine_radiance_net
        else:
            geo_net = self.coarse_geo_net
            radiance_net = self.coarse_radiance_net

        sigma, feature = chunk_processing(geo_net, self.chunk_pts, pts)
        rays_d = normalize(view_dir)  # norm view dir
        rgb = chunk_processing(radiance_net, self.chunk_pts, pts, rays_d, None, feature)

        return sigma[..., 0], rgb

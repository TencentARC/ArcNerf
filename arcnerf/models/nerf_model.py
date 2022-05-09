# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules import GeoNet, RadianceNet
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.geometry.transformation import normalize
from arcnerf.render.ray_helper import get_zvals_from_near_far, ray_marching, sample_pdf
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

    def pretrain_siren(self):
        """Pretrain siren layer of implicit model"""
        self.coarse_geo_net.pretrain_siren()
        if self.rays_cfgs['n_importance'] > 0:
            self.fine_geo_net.pretrain_siren()
        if self.bkg_model is not None:
            self.bkg_model.pretrain_siren()

    def _forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        output = {}

        # get bounds for object, (B, 1) * 2
        near, far = self._get_near_far_from_rays(inputs)

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

        # get sigma and rgb, expand rays_d to all pts. shape in (B*N_sample, dim)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.rays_cfgs['n_sample'], dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.fine_geo_net, self.fine_radiance_net, pts, rays_d_repeat
        )

        # reshape, ray marching and get color/weights
        sigma = sigma.view(-1, self.rays_cfgs['n_sample'])  # (B, N_sample)
        radiance = radiance.view(-1, self.rays_cfgs['n_sample'], 3)  # (B, N_sample, 3)

        # merge sigma from background and get result together
        sigma_all, radiance_all, zvals_all = self._merge_bkg_sigma(inputs, sigma, radiance, zvals, inference_only)

        # ray marching. If two stage and inference only, get weights from single stage.
        weights_only = inference_only and self.rays_cfgs['n_importance'] > 0
        output_coarse = ray_marching(
            sigma_all,
            radiance_all,
            zvals_all,
            self.rays_cfgs['add_inf_z'],  # rgb mode should be False, sigma mode should True
            self.rays_cfgs['noise_std'] if not inference_only else 0.0,
            weights_only=weights_only,
            white_bkg=self.rays_cfgs['white_bkg']
        )
        if not weights_only:
            # merge rgb with background
            output_coarse = self._merge_bkg_rgb(inputs, output_coarse, inference_only)

            output['rgb_coarse'] = output_coarse['rgb']  # (B, 3)
            output['depth_coarse'] = output_coarse['depth']  # (B,)
            output['mask_coarse'] = output_coarse['mask']  # (B,)

        if get_progress:  # this save the sigma with out blending bkg, only in foreground
            for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights']:
                n_fg = self._get_n_fg(sigma)
                output['progress_{}'.format(key)] = output_coarse[key][:, :n_fg].detach()  # (B, N_sample(-1))

        # fine model. resample is only performed in foreground zvals
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
            rays_d_repeat = torch.repeat_interleave(rays_d, n_total, dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.fine_geo_net, self.fine_radiance_net, pts,
                rays_d_repeat
            )

            # reshape, ray marching and get color/weights
            sigma = sigma.view(-1, n_total)  # (B, n_total)
            radiance = radiance.view(-1, n_total, 3)  # (B, n_total, 3)

            # merge sigma from background and get result together
            sigma_all, radiance_all, zvals_all = self._merge_bkg_sigma(inputs, sigma, radiance, zvals, inference_only)

            output_fine = ray_marching(
                sigma_all,
                radiance_all,
                zvals_all,
                self.rays_cfgs['add_inf_z'],  # rgb mode should be False, sigma mode should True
                self.rays_cfgs['noise_std'] if not inference_only else 0.0,
                white_bkg=self.rays_cfgs['white_bkg']
            )

            # merge rgb with background
            output_fine = self._merge_bkg_rgb(inputs, output_fine, inference_only)

            output['rgb_fine'] = output_fine['rgb']  # (B, 3)
            output['depth_fine'] = output_fine['depth']  # (B,)
            output['mask_fine'] = output_fine['mask']  # (B,)

            if get_progress:  # replace with fine, in foreground only
                for key in ['sigma', 'zvals', 'alpha', 'trans_shift', 'weights']:
                    n_fg = self._get_n_fg(sigma)
                    output['progress_{}'.format(key)] = output_fine[key][:, :n_fg].detach()  # (B, N_sample(-1))

        return output

    @torch.no_grad()
    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Rewrite for two stage implementation. """
        gpu_on_func = True if (self.is_cuda() and not pts.is_cuda) else False
        if self.rays_cfgs['n_importance'] > 0:
            geo_net = self.fine_geo_net
            radiance_net = self.fine_radiance_net
        else:
            geo_net = self.coarse_geo_net
            radiance_net = self.coarse_radiance_net

        # the feature takes large memory, should not keep all to process
        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype).to(pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sigma, rgb = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, gpu_on_func, geo_net, radiance_net, pts, rays_d
        )

        return sigma, rgb

    @torch.no_grad()
    def forward_pts(self, pts: torch.Tensor):
        """Rewrite for two stage implementation. """
        gpu_on_func = True if (self.is_cuda() and not pts.is_cuda) else False
        if self.rays_cfgs['n_importance'] > 0:
            geo_net = self.fine_geo_net
        else:
            geo_net = self.coarse_geo_net

        sigma, _ = chunk_processing(geo_net, self.chunk_pts, gpu_on_func, pts)

        return sigma[..., 0]

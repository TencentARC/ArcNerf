# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.render.ray_helper import get_zvals_outside_sphere, sample_pdf
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing
from .base_3d_model import Base3dModel
from .base_modules import build_geo_model, build_radiance_model


class BkgModel(Base3dModel):
    """Class for bkg model. Child class of Base3dModel.
    It can also be used as foreground model if you want.

    Do not use two-stage model when it's used as a bkg_model.
    """

    def __init__(self, cfgs):
        super(BkgModel, self).__init__(cfgs)

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        raise NotImplementedError('Please implement the forward func...')

    def get_zvals_outside_sphere(self, rays_o: torch.Tensor, rays_d: torch.Tensor, inference_only=False):
        """Get the zvals from ray-sphere intersection.

        It will use ray_cfgs['n_sample'] to select samples.
                    ray_cfgs['bounding_radius'] as the inner sphere radius.
        Other sample keys are not allowed.

        Args:
            rays_o: torch.tensor (B, 1) near z distance
            rays_d: torch.tensor (B, 1) far z distance
            inference_only: If True, will not pertube the zvals. used in eval/infer model. Default False.

        Returns:
            zvals: torch.tensor (B, N_sample) zvlas of ray-sphere intersection
            radius: torch.tensor (B, N_sample) radius of each sphere
        """
        zvals, radius = get_zvals_outside_sphere(
            rays_o,
            rays_d,
            self.get_ray_cfgs('n_sample'),
            self.get_ray_cfgs('bounding_radius'),
            perturb=self.get_ray_cfgs('perturb') if not inference_only else False
        )  # (B, N_sample), (N_sample, )
        radius = torch.repeat_interleave(radius.unsqueeze(0).unsqueeze(-1), rays_o.shape[0], 0)  # (B, N_sample)

        return zvals, radius

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        optim_cfgs = self.cfgs.model.optim
        params = {
            'near_distance': get_value_from_cfgs_field(optim_cfgs, 'near_distance', 0.0),
            'epoch_optim': get_value_from_cfgs_field(optim_cfgs, 'epoch_optim', 16),  # You must optimize the volume
            'epoch_optim_warmup': get_value_from_cfgs_field(optim_cfgs, 'epoch_optim_warmup', 256),
            'ema_optim_decay': get_value_from_cfgs_field(optim_cfgs, 'ema_optim_decay', 0.95),
            'opa_thres': get_value_from_cfgs_field(optim_cfgs, 'opa_thres', 0.01)
        }

        return params

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key"""
        if key is None:
            return self.optim_cfgs

        return self.optim_cfgs[key]

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by key"""
        self.optim_cfgs[key] = value

    def optimize(self, cur_epoch=0):
        """Optimize the bkg structure. Support ['multivol'] now."""
        return


@MODEL_REGISTRY.register()
class NeRFPP(BkgModel):
    """ Nerf++ model.
        Process bkg points only. Do not support geometric extraction.
        ref: https://arxiv.org/abs/2010.07492
    """

    def __init__(self, cfgs):
        super(NeRFPP, self).__init__(cfgs)
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

        # check bounding_radius
        assert self.get_ray_cfgs('bounding_radius') is not None, 'Please specify the bounding radius for nerf++ model'

    def get_net(self):
        """Get the actual net for usage"""
        if self.get_ray_cfgs('n_importance') > 0:
            geo_net = self.fine_geo_net
            radiance_net = self.fine_radiance_net
        else:
            geo_net = self.coarse_geo_net
            radiance_net = self.coarse_radiance_net

        return geo_net, radiance_net

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        output = {}

        # get zvals for background, intersection from Multi-Sphere(MSI) (B, N_sample)
        zvals, radius = self.get_zvals_outside_sphere(rays_o, rays_d, inference_only)

        # get points and change to (x/r, y/r, z/r, 1/r). Only when rays_o is (0,0,0) all points xyz norm as same.
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = torch.cat([pts / radius, 1 / radius], dim=-1)
        pts = pts.view(-1, 4)  # (B*N_sample, 4)

        # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.coarse_geo_net, self.coarse_radiance_net, pts,
            rays_d_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

        # ray marching
        output_coarse = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)
        coarse_weights = output_coarse['weights']

        # handle progress
        output['coarse'] = self.output_get_progress(output_coarse, get_progress)

        # fine model
        if self.get_ray_cfgs('n_importance') > 0:
            # get upsampled zvals
            zvals = self.upsample_zvals(zvals, coarse_weights, inference_only)
            # get points and change to (x/r, y/r, z/r, 1/r). Only when rays_o is (0,0,0) all points xyz norm as same.
            pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_total, 3)
            # radius is the norm of pts
            radius = torch.norm(pts, dim=-1)[..., None]  # (B, N_total, 1)
            pts = torch.cat([pts / radius, 1 / radius], dim=-1)
            pts = pts.view(-1, 4)  # (B*N_sample, 4)

            # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_total, ...)
            n_total = self.get_ray_cfgs('n_importance') + self.get_ray_cfgs('n_sample')
            rays_d_repeat = torch.repeat_interleave(rays_d, n_total, dim=0)
            sigma, radiance = chunk_processing(
                self._forward_pts_dir, self.chunk_pts, False, self.fine_geo_net, self.fine_radiance_net, pts,
                rays_d_repeat
            )

            # reshape
            sigma = sigma.view(-1, n_total)  # (B, N_total)
            radiance = radiance.view(-1, n_total, 3)  # (B, N_total, 3)

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
            mask_pts: new mask in (B, N_sample + N_importance)
        """
        weights_coarse = weights[:, 1:self.get_ray_cfgs('n_sample') - 1]  # (B, N_sample-2)
        zvals_mid = 0.5 * (zvals[..., 1:] + zvals[..., :-1])  # (B, N_sample-1)
        _zvals = sample_pdf(
            zvals_mid, weights_coarse, self.get_ray_cfgs('n_importance'),
            not self.get_ray_cfgs('perturb') if not inference_only else True
        ).detach()
        zvals, _ = torch.sort(torch.cat([zvals, _zvals], -1), -1)  # (B, N_sample+N_importance=N_total)

        return zvals

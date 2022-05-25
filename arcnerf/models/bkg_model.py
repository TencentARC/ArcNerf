# -*- coding: utf-8 -*-

import torch

from .base_3d_model import Base3dModel
from .base_modules import build_geo_model, build_radiance_model
from arcnerf.geometry.ray import get_ray_points_by_zvals
from arcnerf.render.ray_helper import get_zvals_outside_sphere
from common.utils.registry import MODEL_REGISTRY
from common.utils.torch_utils import chunk_processing


class BkgModel(Base3dModel):
    """Class for bkg model. Child class of Based3dModel.
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


@MODEL_REGISTRY.register()
class NeRFPP(BkgModel):
    """ Nerf++ model.
        Process bkg points only. Do not support geometric extractration.
        ref: https://arxiv.org/abs/2010.07492
    """

    def __init__(self, cfgs):
        super(NeRFPP, self).__init__(cfgs)
        self.geo_net = build_geo_model(self.cfgs.model.geometry)
        self.radiance_net = build_radiance_model(self.cfgs.model.radiance)
        # check bounding_radius
        assert self.get_ray_cfgs('bounding_radius') is not None, 'Please specify the bounding radius for nerf++ model'

    def forward(self, inputs, inference_only=False, get_progress=False, cur_epoch=0, total_epoch=300000):
        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)

        # get zvals for background, intersection from Multi-Sphere(MSI) (B, N_sample)
        zvals, radius = self.get_zvals_outside_sphere(rays_o, rays_d, inference_only)

        # get points and change to (x/r, y/r, z/r, 1/r). Only when rays_o is (0,0,0) all points xyz norm as same.
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)  # (B, N_sample, 3)
        pts = torch.cat([pts / radius, 1 / radius], dim=-1)
        pts = pts.view(-1, 4)  # (B*N_sample, 4)

        # get sigma and rgb,  expand rays_d to all pts. shape in (B*N_sample, ...)
        rays_d_repeat = torch.repeat_interleave(rays_d, self.get_ray_cfgs('n_sample'), dim=0)
        sigma, radiance = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, self.geo_net, self.radiance_net, pts, rays_d_repeat
        )

        # reshape
        sigma = sigma.view(-1, self.get_ray_cfgs('n_sample'))  # (B, N_sample)
        radiance = radiance.view(-1, self.get_ray_cfgs('n_sample'), 3)  # (B, N_sample, 3)

        # ray marching
        output = self.ray_marching(sigma, radiance, zvals, inference_only=inference_only)

        # handle progress
        output = self.output_get_progress(output, get_progress)

        return output

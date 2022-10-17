# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.ray import surface_ray_intersection
from arcnerf.geometry.transformation import normalize
from common.utils.torch_utils import chunk_processing
from .fg_model import FgModel


class SdfModel(FgModel):
    """ SDF model. Modelling geo in sdf and convert it to alpha.
        Methods include Neus/volSDF, etc.
    """

    def __init__(self, cfgs):
        super(SdfModel, self).__init__(cfgs)

    @staticmethod
    def sigma_reverse():
        """It use SDF(inside object is smaller)"""
        return True

    def get_est_opacity(self, dt, pts):
        """Get the estimated opacity at certain pts. You have to rewrite this function in sdf-like models"""
        raise NotImplementedError('You must implement the function in sdf-like models')

    def forward_pts_dir(self, pts: torch.Tensor, view_dir: torch.Tensor = None):
        """Rewrite to use normal processing """
        geo_net, radiance_net = self.get_net()
        if view_dir is None:
            rays_d = torch.zeros_like(pts, dtype=pts.dtype, device=pts.device)
        else:
            rays_d = normalize(view_dir)  # norm view dir

        sigma, rgb, _ = chunk_processing(
            self._forward_pts_dir, self.chunk_pts, False, geo_net, radiance_net, pts, rays_d
        )

        return sigma, rgb

    @staticmethod
    def _forward_pts_dir(
        geo_net,
        radiance_net,
        pts: torch.Tensor,
        rays_d: torch.Tensor = None,
    ):
        """Rewrite to use normal processing """
        sdf, feature, normal = geo_net.forward_with_grad(pts)
        radiance = radiance_net(pts, rays_d, normal, feature)

        return sdf[..., 0], radiance, normal

    def surface_render(
        self, inputs, method='sphere_tracing', n_step=128, n_iter=20, threshold=0.01, level=0.0, grad_dir='ascent'
    ):
        assert level == 0.0, 'Invalid level for sdf model...'
        assert grad_dir == 'ascent', 'Invalid grad_dir for sdf model...'

        rays_o = inputs['rays_o']  # (B, 3)
        rays_d = inputs['rays_d']  # (B, 3)
        dtype = rays_o.dtype
        device = rays_o.device
        n_rays = rays_o.shape[0]

        # get bounds for object
        near, far, valid_rays = self.get_near_far_from_rays(inputs)  # (B, 1) * 2

        # get the network
        geo_net, radiance_net = self.get_net()

        # get surface pts for valid_rays
        if valid_rays is None or torch.all(valid_rays):
            zvals, pts, mask = surface_ray_intersection(
                rays_o, rays_d, geo_net.forward_geo_value, method, near, far, n_step, n_iter, threshold, level, grad_dir
            )

        else:
            zvals_valid, pts_valid, mask_valid = surface_ray_intersection(
                rays_o[valid_rays], rays_d[valid_rays], geo_net.forward_geo_value, method, near[valid_rays],
                far[valid_rays], n_step, n_iter, threshold, level, grad_dir
            )

            # full output update by valid rays
            zvals = torch.ones((n_rays, 1), dtype=dtype, device=device) * zvals_valid.max()
            pts = torch.ones((n_rays, 3), dtype=dtype, device=device)
            mask = torch.zeros((n_rays, ), dtype=torch.bool, device=device)
            zvals[valid_rays], pts[valid_rays], mask[valid_rays] = zvals_valid, pts_valid, mask_valid

        rgb = torch.ones((n_rays, 3), dtype=dtype, device=device)  # white bkg
        normal = torch.zeros((n_rays, 3), dtype=dtype, device=device)
        depth = zvals  # at max zvals after far
        mask_float = mask.type(dtype)

        # in case all rays do not hit the surface
        if torch.any(mask):
            _, rgb_mask, normal_mask = self._forward_pts_dir(geo_net, radiance_net, pts[mask], rays_d[mask])
            # forward mask pts/dir for color and normal
            rgb[mask] = rgb_mask
            normal[mask] = normal_mask

        output = {
            'rgb': rgb,  # (B, 3)
            'depth': depth,  # (B,)
            'mask': mask_float,  # (B,)
            'normal': normal  # (B, 3)
        }

        return output

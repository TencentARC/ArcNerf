# -*- coding: utf-8 -*-

import torch

from arcnerf.geometry.ray import sphere_ray_intersection


def get_near_far_from_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    bounds: torch.Tensor = None,
    near_hardcode=None,
    far_hardcode=None,
    bounding_radius=None
):
    """Get near, far zvals from rays. Hard-reset by near/far_hardcode

    Args:
        rays_o: tensor(N_rays, 3), ray origin
        rays_d: tensor(N_rays, 3), ray direction
        bounds: tensor(N_rays, 2), input bounds, generally obtained from data with point_cloud
        near_hardcode: If not None, will force all near to be this value
        far_hardcode: If not None, will force all far to be this value
        bounding_radius: If not None, will use this to calculate the ray-sphere intersection as near/far

    Returns:
        near: tensor(N_rays, 1), near zvals
        far:  tensor(N_rays, 1), far zvals
    """

    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]

    if near_hardcode is None or far_hardcode is None:
        if bounds is None and bounding_radius is None:
            raise NotImplementedError('You must specify near/far in some place...')

        if bounds is None:
            near, far, _, mask = sphere_ray_intersection(rays_o, rays_d, radius=bounding_radius)  # (BN, 1)
        else:
            # TODO: When use bounds from dataset, it may only cover range with object(far is not enough),
            # TODO: no background applied. You may need to extent far or use extra background layer for such dataset
            near, far = bounds[:, 0:1], bounds[:, 1:2]

        # hard set for near/far
        if near_hardcode is not None:
            near = near * 0 + near_hardcode
        if far_hardcode is not None:
            far = far * 0 + far_hardcode
    else:
        near = torch.ones(size=(n_rays, 1), dtype=dtype).to(device) * near_hardcode  # (BN, 1)
        far = torch.ones(size=(n_rays, 1), dtype=dtype).to(device) * far_hardcode  # (BN, 1)

    return near, far

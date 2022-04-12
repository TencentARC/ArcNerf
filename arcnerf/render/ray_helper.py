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


def get_zvals_from_near_far(near, far, n_pts, inclusive=True, inverse_linear=False, perturb=False):
    """Get zvals from near/far distance

    Args:
        near: tensor(N_rays, 1), near zvals
        far: tensor(N_rays, 1), far zvals
        n_pts: num of points sampled in (near, far)
        inclusive: If True, zvals include near,far. If False, only in range not inclusive. By default True.
        inverse_linear: If False, uniform sampling in (near, far). If True, use inverse sampling and closer to near.
                        By default False.
        perturb: If True, disturb sampling in all segment. By default False.

    Returns:
        zvals: (N_rays, n_pts), each ray samples n_pts points
    """
    device = near.device
    dtype = near.dtype

    if inclusive:
        t_vals = torch.linspace(0.0, 1.0, n_pts, dtype=dtype).to(device)  # (N_pts,)
    else:
        t_vals = torch.linspace(0.0, 1.0, n_pts + 2, dtype=dtype)[1:-1].to(device)  # (N_pts,)

    if inverse_linear:
        zvals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)  # (N_rays, N_pts)
    else:
        zvals = near * (1 - t_vals) + far * t_vals

    if perturb:
        _mids = .5 * (zvals[..., 1:] + zvals[..., :-1])
        _upper = torch.cat([_mids, zvals[..., -1:]], -1)
        _lower = torch.cat([zvals[..., :1], _mids], -1)
        _z_rand = torch.rand(size=_upper.shape, dtype=dtype).to(device)
        zvals = _lower + (_upper - _lower) * _z_rand  # (N_rays, N_pts)

    return zvals

# -*- coding: utf-8 -*-

import numpy as np
import torch

from arcnerf.geometry.projection import pixel_to_world
from arcnerf.geometry.ray import sphere_ray_intersection
from arcnerf.geometry.transformation import normalize
from common.utils.torch_utils import torch_to_np


def get_rays(W, H, intrinsic: torch.Tensor, c2w: torch.Tensor, index: np.ndarray = None, n_rays=-1, to_np=False):
    """Get rays in world coord from camera.
    No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        c2w: torch.tensor(4, 4) cam pose. cam_to_world transform
        index: sample ray by (i, j) index from (W, H), np.array/torch.tensor(N_ind, 2) for (i, j) index
                first index is X and second is Y, any index should be in range (0, W-1) and (0, H-1)
        n_rays: random sample ray by such num if it > 0
        to_np: if to np, return np array instead of torch.tensor

    Returns:
        a ray_bundle with rays_o and rays_d. Each is in dim (N_ray, 3).
             If no sampler is used, return (WH, 3) num of rays
        ind_unroll: sample index in list of (N_ind, ) for index in (WH, ) range
    """
    assert (index is None) or n_rays <= 0, 'You are not allowed to sampled both by index and N_ray'
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype), torch.linspace(0, H - 1, H, dtype=dtype)
    )  # i, j: (W, H)
    pixels = torch.stack([i, j], dim=-1).view(-1, 2).unsqueeze(0).to(device)  # (1, WH, 2)

    # index unroll
    if index is not None:
        assert len(index.shape) == 2 and index.shape[-1] == 2, 'invalid shape, should be (N_rays, 2)'
        if isinstance(index, np.ndarray):
            index = torch.tensor(index, dtype=torch.long).to(device)
        else:
            index = index.type(torch.long).to(device)
        index = index[:, 0] * H + index[:, 1]  # (N_rays, ) unroll from (i, j)
    # sample by N_rays
    if n_rays > 0:
        index = np.random.choice(range(0, W * H), n_rays, replace=False)  # (N_rays, )
        index = torch.tensor(index, dtype=torch.long).to(device)
    # sampled by index
    if index is not None:
        pixels = pixels[:, index, :]
        index = torch_to_np(index).tolist()

    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype).to(device)  # (1, WH/N_rays)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH/N_rays, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH/N_rays, 3)
    # normalized dir
    rays_d = normalize(rays_d)[0]  # (WH/N_rays, 3)
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH/N_rays, 3)

    if to_np:
        rays_o = torch_to_np(rays_o)
        rays_d = torch_to_np(rays_d)

    return rays_o, rays_d, index


def equal_sample(n_rays_w, n_rays_h, W, H):
    """Equal sample i,j index on img with (W, H)

    Args:
        n_rays_w: num of samples on each row (x direction)
        n_rays_h: num of samples on each col (y direction)
        W: image width
        H: image height

    Returns:
        index: np.array(n_rays_w*n_rays_h, 2) equally sampled grid
    """

    i, j = np.meshgrid(np.linspace(0, W - 1, n_rays_w), np.linspace(0, H - 1, n_rays_h))
    index = np.stack([i, j], axis=-1).reshape(-1, 2)

    return index


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


def get_zvals_from_near_far(
    near: torch.Tensor, far: torch.Tensor, n_pts, inclusive=True, inverse_linear=False, perturb=False
):
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


def ray_marching(sigma: torch.Tensor, radiance: torch.Tensor, zvals: torch.Tensor, add_inf_z=False, noise_std=0.0):
    """Ray marching and get color for each ray, get weight for each ray
        For p_i, the delta_i is p_i -> p_i+1 on the right side
        The full function is:
        - alpha_i = (1 - exp(- relu(sigma) * delta_i): prob not pass this point on ray
                - when sigma = 0, alpha_i = 0, light must pass this point
                - when sigma is large, alpha_i closer to 1, prob not pass this point is large, light will be block
                - when delta_i is large(last distance inf), light will not pass it (prob = 1)
        - Ti = mul_1_i-1(1 - alpha_i): accumulated transmittance, prob light pass all previous points
                - T0 = 1, pass previous prob is high at beginning
                - Tn closer 0, prob light pass previous point gets lower
        - Ti * alpha_i: prob pass all previous point but stop at this
                - at max when alpha_i get high
        - C_ray = sum_i_1_N(Ti * alpha_i * C_i): accumulated color at each point

    Args:
        sigma: (N_rays, N_pts), density value
        radiance: (N_rays, N_pts, 3), radiance value for each point
        zvals: (N_rays, N_pts), zvals for ray in unit-length
        add_inf_z: If True, add inf zvals(1e10) for calculation. If False, ignore last point for rgb/depth calculation
        noise_std: noise level add to density if noise > 0, used for training. By default 0.0.

    Returns:
        rgb: (N_rays, 3), rgb for each ray
        depth: (N_rays), weighted zvals for each ray that approximate for the surface
        mask: (N_ray), accumulated weights for the ray. Like a mask for background
                    If =1, the ray is stop some whether.
                    If =0, the ray does not stop, it pass through air.
                If you add rgb + (1-mask), background will become all 1.
        weights: (N_rays, N_pts) if add_inf_z else (N_rays, N_pts-1). Use for normalizing other values like normals.
    """
    dtype = sigma.dtype
    device = sigma.device
    n_rays, n_pts = sigma.shape[:2]

    deltas = zvals[:, 1:] - zvals[:, :-1]  # (N_rays, N_pts-1)
    if add_inf_z:  # (N_rays, N_pts)
        deltas = torch.cat([deltas, torch.ones(size=(n_rays, 1), dtype=dtype).to(device)], dim=-1)
    else:
        sigma = sigma[:, :-1]  # (N_rays, N_pts-1)
        radiance = radiance[:, :-1, ]  # (N_rays, N_pts-1, 3)
        zvals = zvals[:, :-1]  # (N_rays, N_pts-1)

    noise = 0.0
    if noise_std > 0.0:
        noise = torch.randn(sigma.shape, dtype=dtype).to(device) * noise_std

    # alpha_i = (1 - exp(- relu(sigma) * delta_i)
    alpha = 1 - torch.exp(deltas * torch.relu(sigma * noise))  # (N_rays, N_p)
    # Ti = mul_1_i-1(1 - alpha_i)
    alpha_one = torch.ones_like(alpha[:, :1], dtype=dtype).to(device)
    trans_shift = torch.cat([alpha_one, 1 - alpha + 1e-10], -1)  # (N_rays, N_p+1)
    # weight_i = Ti * alpha_i
    weights = alpha * torch.cumprod(trans_shift, -1)[:, :-1]  # (N_rays, N_p)
    # rgb = sum(weight_i * radiance_i)
    rgb = torch.sum(weights.unsqueeze(-1) * radiance, -2)  # (N_rays, 3)
    # depth = sum(weight_i * zvals_i)
    depth = torch.sum(weights * zvals, -1)  # (N_rays)
    # accumulated weight(mask)
    mask = torch.sum(weights, -1)  # (N_rays)

    return rgb, depth, mask, weights

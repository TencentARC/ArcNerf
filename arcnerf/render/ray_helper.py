# -*- coding: utf-8 -*-

import numpy as np
import torch

from arcnerf.geometry.projection import pixel_to_world
from arcnerf.geometry.ray import sphere_ray_intersection
from arcnerf.geometry.transformation import normalize
from common.utils.torch_utils import torch_to_np


def get_rays(
    W,
    H,
    intrinsic: torch.Tensor,
    c2w: torch.Tensor,
    wh_order=True,
    index=None,
    n_rays=-1,
    to_np=False,
    ndc=False,
    ndc_near=1.0,
    center_pixel=False,
    normalize_rays_d=True
):
    """Get rays in world coord from camera.
    No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        c2w: torch.tensor(4, 4) cam pose. cam_to_world transform
        wh_order: If True, the rays are flatten in column-major. If False, in row-major. By default True
        index: sample ray by (i, j) index from (W, H), np.array/torch.tensor(N_ind, 2) for (i, j) index
                first index is X and second is Y, any index should be in range (0, W-1) and (0, H-1)
        n_rays: random sample ray by such num if it > 0
        to_np: if to np, return np array instead of torch.tensor
        ndc: If True, change rays to ndc space, you can then change near far to 0,1. By default False
        ndc_near: near zvals. By default 1.0.
        center_pixel: If True, use the center pixel from (0.5, 0.5) instead of corner(0, 0)
        normalize_rays_d: normalize the rays_d. By default True

    Returns:
        rays_o: origin (N_ray, 3) tensor. If no sampler is used, return (WH, 3) num of rays
        rays_d: direction (N_ray, 3) tensor. If no sampler is used, return (WH, 3) num of rays
        index: sample index in list of (N_ind, ) for index in (WH, ) range
        rays_r: rays radius from mip-nerf, (N_ray, 1) tensor.

    """
    assert (index is None) or n_rays <= 0, 'You are not allowed to sampled both by index and N_ray'
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype, device=device),
        torch.linspace(0, H - 1, H, dtype=dtype, device=device)
    )  # i, j: (W, H)
    pixels = torch.stack([i, j], dim=-1).view(-1, 2).unsqueeze(0)  # (1, WH, 2)

    if center_pixel:
        pixels += 0.5

    # index unroll
    if index is not None:
        assert len(index.shape) == 2 and index.shape[-1] == 2, 'invalid shape, should be (N_rays, 2)'
        if isinstance(index, np.ndarray):
            index = torch.tensor(index, dtype=torch.long, device=device)
        else:
            index = index.type(torch.long).to(device)
        index = index[:, 0] * H + index[:, 1]  # (N_rays, ) unroll from (i, j)
    # sample by N_rays
    if n_rays > 0:
        index = np.random.choice(range(0, W * H), n_rays, replace=False)  # (N_rays, )
        index = torch.tensor(index, dtype=torch.long, device=device)
    # sampled by index
    if index is not None:
        pixels = pixels[:, index, :]
        index = torch_to_np(index).tolist()

    # reorder if full rays
    if not wh_order and index is None and n_rays <= 0:  # (HW, 2)
        pixels = pixels.squeeze(0).contiguous().view(W, H, 2).permute(1, 0, 2).contiguous().view(-1, 2).unsqueeze(0)

    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype, device=device)  # (1, WH/N_rays)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH/N_rays, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH/N_rays, 3)
    rays_d = rays_d[0]  # (WH/N_rays, 3)
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH/N_rays, 3)

    # chang to ndc
    if ndc:
        rays_o, rays_d = get_ndc_rays(rays_o, rays_d, W, H, intrinsic, ndc_near)
    else:
        # normalize rays for non_ndc case
        if normalize_rays_d:
            rays_d = normalize(rays_d)  # (WH/N_rays, 3)

    if to_np:
        rays_o = torch_to_np(rays_o)
        rays_d = torch_to_np(rays_d)

    # rays_r is the radius introduced in mip-nerf
    rays_r = None
    if index is None and n_rays <= 0:  # only in full image mode
        if wh_order:
            dirs = rays_d.clone().view(W, H, 3)  # (W, H, 3)
            dx = torch.sqrt(torch.sum((dirs[:-1, ...] - dirs[1:, ...])**2, -1))  # (W-1, H, 1)
            dx = torch.cat([dx, dx[-2:-1, ...]], dim=0)  # (W, H, 1)
        else:
            dirs = rays_d.clone().view(H, W, 3)  # (H, W, 3)
            dx = torch.sqrt(torch.sum((dirs[:, :-1, ...] - dirs[:, 1:, ...])**2, -1))  # (H, W-1, 1)
            dx = torch.cat([dx, dx[:, -2:-1, ...]], dim=1)  # (H, W, 1)
        rays_r = dx.unsqueeze(-1) * 2 / torch.sqrt(torch.tensor([12], dtype=dx.dtype, device=dx.device))
        rays_r = rays_r.view(-1, 1)

    return rays_o, rays_d, index, rays_r


def get_ndc_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, W, H, intrinsic: torch.Tensor, near=1.0):
    """Change rays in original space to ndc space

    Args:
        rays_o: rays origin in original world space
        rays_d: rays direction in original world space
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        near: near zvals. By default 1.0.

    Returns:
        a ray_bundle with rays_o and rays_d. Each is in dim (N_ray, 3).
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    f_x, f_y = intrinsic[0, 0], intrinsic[1, 1]
    # Projection
    o0 = -1. / (W / (2. * f_x)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * f_y)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * f_x)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * f_y)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


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
    bounding_radius=None,
):
    """Get near, far zvals from rays. Hard-reset by near/far_hardcode

    Args:
        rays_o: tensor(N_rays, 3), ray origin
        rays_d: tensor(N_rays, 3), ray direction
        bounds: tensor(N_rays, 2), input bounds, generally obtained from data with point_cloud.
                 Use ray-sphere far bound whenever bounding_radius is not None to restrict the pts in sphere.
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
            radius = torch.tensor([bounding_radius], dtype=dtype, device=device)
            near, far, _, _ = sphere_ray_intersection(rays_o, rays_d, radius=radius)  # (N_rays, 1)
        else:
            near, far = bounds[:, 0:1], bounds[:, 1:2]
            if bounding_radius is not None:  # restrict the far end bound if radius is set
                radius = torch.tensor([bounding_radius], dtype=dtype, device=device)
                _, far_bound, _, _ = sphere_ray_intersection(rays_o, rays_d, radius=radius)  # (N_rays, 1)
                far[far > far_bound] = far_bound[far > far_bound]

        # hard set for near/far
        if near_hardcode is not None:
            near = near * 0 + near_hardcode
        if far_hardcode is not None:
            far = far * 0 + far_hardcode
    else:
        near = torch.ones(size=(n_rays, 1), dtype=dtype, device=device) * near_hardcode  # (N_rays, 1)
        far = torch.ones(size=(n_rays, 1), dtype=dtype, device=device) * far_hardcode  # (N_rays, 1)

    # in case near >= far, cast far as near + 1e-5
    far[far <= near] = near[far <= near] + 1e-5

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
        t_vals = torch.linspace(0.0, 1.0, n_pts, dtype=dtype, device=device)  # (N_pts,)
    else:
        t_vals = torch.linspace(0.0, 1.0, n_pts + 2, dtype=dtype, device=device)[1:-1]  # (N_pts,)

    if inverse_linear:  # +1e-8 in case nan
        zvals = 1.0 / (1.0 / (near + 1e-8) * (1.0 - t_vals) + 1.0 / (far + 1e-8) * t_vals)  # (N_rays, N_pts)
    else:
        zvals = near + (far - near) * t_vals

    if perturb:
        zvals = perturb_interval(zvals)  # (N_rays, N_pts)

    return zvals


def get_zvals_from_near_far_fix_step(
    near: torch.Tensor, far: torch.Tensor, fix_t, n_pts, inclusive=True, perturb=False
):
    """Get zvals from near/far distance with fix step t

    Args:
        near: tensor(N_rays, 1), near zvals
        far: tensor(N_rays, 1), far zvals
        fix_t: fix step t
        n_pts: num of points sampled in (near, far)
        inclusive: If True, zvals include near,far. If False, only in range not inclusive. By default True.
        perturb: If True, disturb sampling in all segment. By default False.

    Returns:
        zvals: (N_rays, n_pts), each ray samples n_pts points.
                If zvals >= far, will keep the far in following pts.
        mask_pts: (N_rays, n_pts), bool tensor indicating each ray samples points validity.
                If zvals >= far, will mask duplicated points as False
    """
    assert fix_t > 0, 'Only allow positive step...'
    n_rays = near.shape[0]
    dtype = near.dtype
    device = near.device

    zvals = torch.ones((n_rays, n_pts), dtype=dtype, device=device)  # (N_rays, N_pts)
    mask_pts = torch.ones((n_rays, n_pts), dtype=torch.bool, device=device)  # (N_rays, N_pts)

    # init as near
    if inclusive:
        zvals = zvals * near
    else:
        zvals = zvals * (near + fix_t)

    # step forward and clamp in (near, far)
    step = torch.arange(0, n_pts, 1, device=device)[None]  # (1, N_pts)
    zvals = zvals + step * fix_t  # (N_rays, N_pts)
    zvals = zvals.clamp(near, far)

    # invalid for the pts that is same as last column
    zvals_col_diff = (zvals[:, 1:] - zvals[:, :-1] == 0.0)  # (N_rays, N_pts-1)
    zvals_col_diff = torch.cat([torch.zeros((n_rays, 1), dtype=torch.bool, device=device), zvals_col_diff],
                               dim=1)  # (N_rays, N_pts)
    mask_pts[zvals_col_diff] = False

    # perturb valid pts only
    if perturb or True:
        zvals = perturb_interval_with_mask(zvals, mask_pts)

    return zvals, mask_pts


def get_zvals_outside_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, n_pts, radius, perturb=False):
    """Get zvals outside a bounding radius

    Args:
        rays_o: tensor(N_rays, 3), ray origin
        rays_d: tensor(N_rays, 3), ray direction
        n_pts: num of point to sample
        radius: float value of the bounding sphere radius
        perturb: If True, disturb sampling in all segment. By default False.

    Returns:
        zvals: (N_rays, N_pts), each ray samples n_pts points
        sphere_radius: (N_pts, ) radius of the ball
    """
    device = rays_o.device
    dtype = rays_o.dtype

    t_vals = torch.linspace(0.0, 1.0, n_pts + 2, dtype=dtype, device=device)[1:-1]  # (N_pts,)
    sphere_radius = radius / torch.flip(t_vals, dims=[-1])  # (N_pts,), extend from radius -> inf
    if perturb:
        sphere_radius = perturb_interval(sphere_radius[None])[0]  # (N_pts, )
    zvals = get_zvals_from_sphere_radius(rays_o, rays_d, sphere_radius)  # (N_rays, N_pts)

    return zvals, sphere_radius


def get_zvals_from_sphere_radius(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_radius: torch.Tensor):
    """Get zvals from sphere radius

    Args:
        rays_o: tensor(N_rays, 3), ray origin
        rays_d: tensor(N_rays, 3), ray direction
        sphere_radius: tensor(N, ), multiple sphere layers radius

    Returns:
        zvals: (N_rays, N), each ray samples n_pts points.
              If points do not intersect, will use 0 for it.
    """
    _, zvals, _, _ = sphere_ray_intersection(rays_o, rays_d, sphere_radius)

    return zvals


def perturb_interval(vals: torch.Tensor):
    """Perturb sampling in the intervals

    Args:
        vals: tensor (B, N), sampling in N pts.

    Return:
        vals: perturb sampling (B, N)
    """
    dtype = vals.dtype
    device = vals.device
    _mids = .5 * (vals[..., 1:] + vals[..., :-1])  # (N_rays, N_pts-1)
    _upper = torch.cat([_mids, vals[..., -1:]], -1)  # (N_rays, N_pts)
    _lower = torch.cat([vals[..., :1], _mids], -1)  # (N_rays, N_pts)
    _z_rand = torch.rand(size=_upper.shape, dtype=dtype, device=device)
    vals = _lower + (_upper - _lower) * _z_rand  # (N_rays, N_pts)

    return vals


def perturb_interval_with_mask(vals: torch.Tensor, mask=None):
    """Perturb sampling in the intervals with mask.
    Since the unmasked values in each row is the same as the last masked val, change them as well.

    Args:
        vals: tensor (B, N), sampling in N pts. vals in each row will ending with same values if mask=False.
        mask: tensor (B, N), each pts

    Return:
        vals: perturb sampling (B, N) on valid vals
    """
    device = vals.device
    vals_perturb = perturb_interval(vals)
    if mask is None:
        return vals_perturb

    if mask is not None:
        vals[mask] = vals_perturb[mask]
        # last valid pts
        last_idx = torch.sum(mask, dim=1) - 1  # (B, )
        row_idx = torch.arange(vals.shape[0], dtype=last_idx.dtype, device=device)  # (B, )
        last_idx = torch.cat([row_idx[:, None], last_idx[:, None]], dim=1)  # (B, 2)
        last_value = vals[last_idx[:, 0], last_idx[:, 1]][:, None]  # (B, 1)
        # clamp the other vals
        vals = vals.clamp(vals[:, 0:1], last_value)

        return vals


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_sample, det=False, eps=1e-5):
    """Weighted sampling in bins by pdf weights

    Args:
        bins: (B, n_pts), each bin contain ordered n_pts, sample in such (n_pts-1) intervals
        weights: (B, n_pts-1), the weight for each interval
        n_sample: resample num
        det: If det, uniform sample in (0,1), else random sample in (0,1), by default False
        eps: small value, 1e-5

    Returns:
        samples: (B, n_sample) sampled based on pdf from bins. sorted for each sample
    """
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, n_pts-1)
    cdf = torch.cumsum(pdf, -1)  # (B, n_pts-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (B, n_pts)
    samples = sample_cdf(bins, cdf, n_sample, det)

    return samples


def sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, n_sample, det=False, eps=1e-5):
    """Weighted sampling in bins by cdf weights

    Args:
        bins: (B, n_pts), each bin contain ordered n_pts, sample in such (n_pts-1) intervals
        cdf: (B, n_pts), the accumulated weight for each pts, increase from 0~1
        n_sample: resample num
        det: If det, uniform sample in (0,1), else random sample in (0,1), by default False
        eps: small value, 1e-5

    Returns:
        samples: (B, n_sample) sampled based on cdf from bins. sorted for each sample
    """
    # Take uniform samples
    device = bins.device
    n_pts = bins.shape[-1]

    if det:
        u = torch.linspace(0.0, 1.0, steps=n_sample, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_sample])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_sample], device=device)
    u = u.contiguous()  # (B, n_sample)

    # inverse cdf, get index
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # (B, n_sample)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, n_pts - 1)
    inds_g = torch.stack([below, above], -1).view(-1, 2 * n_sample)  # (B, n_sample*2)

    cdf_g = torch.gather(cdf, 1, inds_g).view(-1, n_sample, 2)
    bins_g = torch.gather(bins, 1, inds_g).view(-1, n_sample, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # (B, n_sample)

    # sort
    samples, _ = torch.sort(samples, -1)  # (B, n_sample)

    return samples


def ray_marching(
    sigma: torch.Tensor,
    radiance: torch.Tensor,
    zvals: torch.Tensor,
    add_inf_z=False,
    noise_std=0.0,
    weights_only=False,
    white_bkg=False,
    alpha: torch.Tensor = None,
    bkg_color: torch.Tensor = None
):
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
        sigma: (N_rays, N_pts), density value, can use alpha directly. optional if alpha is input
        radiance: (N_rays, N_pts, 3), radiance value for each point. If none, will not cal rgb from weighted radiance
        zvals: (N_rays, N_pts), zvals for ray in unit-length
        add_inf_z: If True, add inf zvals(1e10) for calculation. If False, ignore last point for rgb/depth calculation
        noise_std: noise level add to density if noise > 0, used for training and sigma mode. By default 0.0.
        weights_only: If True, return weights only, used in inference time for hierarchical sampling
        white_bkg: If True, make the accum weight=0 rays as 1.
        alpha: (N_rays, N_pts), if provide, do not use sigma to calculate alpha, optional
        bkg_color: If not None, a (N_rays, 3) or (1, 3) tensor attaching the background. will not use white_bkg anyway.

    Returns:
        output a dict with following keys:
            rgb: (N_rays, 3), rgb for each ray, None if radiance not input
            depth: (N_rays), weighted zvals for each ray that approximate for the surface
            mask: (N_ray), accumulated weights for the ray. Like a mask for background
                        If =1, the ray is stop some whether.
                        If =0, the ray does not stop, it pass through air.
                    If you add rgb + (1-mask), background will become all 1.
            sigma: (N_rays, N_pts/N_pts-1), sigma after add_inf_z
            zvals:  (N_rays, N_pts/N_pts-1), zvals after add_inf_z
            alpha: (N_rays, N_pts/N_pts-1). prob not pass the point
            trans_shift: (N_rays, N_pts/N_pts-1). prob pass all previous light
            weights: (N_rays, N_pts/N_pts-1). Use for weighting other values like normals.
    """
    dtype = zvals.dtype
    device = zvals.device
    n_rays = zvals.shape[0]

    assert sigma is not None or alpha is not None, 'Can not be None for both alpha and sigma..'

    deltas = zvals[:, 1:] - zvals[:, :-1]  # (N_rays, N_pts-1)
    deltas[torch.abs(deltas) < 1e-5] = 0.0  # small value handling
    assert torch.all(deltas >= 0), 'zvals is not all increase....'

    _sigma = sigma
    _radiance = radiance
    _zvals = zvals

    # add an inf distance as last to keep original dimension
    if add_inf_z:  # (N_rays, N_pts)
        deltas = torch.cat([deltas, 1e10 * torch.ones(size=(n_rays, 1), dtype=dtype, device=device)], dim=-1)
    else:
        if alpha is None:  # only do that if alpha is None
            _sigma = sigma[:, :-1] if sigma is not None else None  # (N_rays, N_pts-1)
            _radiance = radiance[:, :-1, :] if radiance is not None else None  # (N_rays, N_pts-1, 3)
            _zvals = zvals[:, :-1]  # (N_rays, N_pts-1)

    # use sigma to get alpha
    if alpha is None:
        noise = 0.0
        if noise_std > 0.0:
            noise = torch.randn(_sigma.shape, dtype=dtype, device=device) * noise_std

        # alpha_i = (1 - exp(- relu(sigma + noise) * delta_i)
        alpha = 1 - torch.exp(-torch.relu(_sigma + noise) * deltas)  # (N_rays, N_p)

    trans_shift, weights = alpha_to_weights(alpha)  # (N_rays, N_p) * 2

    # depth = sum(weight_i * zvals_i)
    depth = torch.sum(weights * _zvals, -1)  # (N_rays)
    # accumulated weight(mask)
    mask = torch.sum(weights, -1)  # (N_rays)

    # rgb = sum(weight_i * radiance_i)
    if _radiance is not None:
        rgb = torch.sum(weights.unsqueeze(-1) * _radiance, -2)  # (N_rays, 3)
        if bkg_color is not None:
            assert bkg_color.shape[0] == rgb.shape[0] or bkg_color.shape[0] == 1, 'Only bkg with N_rays/1 allowed..'
            rgb = rgb + trans_shift[:, -1:] * bkg_color
        else:
            if white_bkg:  # where mask = 0, rgb = 1
                rgb = rgb + (1.0 - mask[:, None])  # or you can use tran_shift[-1]
    else:
        rgb = None

    if weights_only:
        output = {'weights': weights}
        return output

    output = {
        'rgb': rgb,  # (N_rays, 3)
        'depth': depth,  # (N_rays)
        'mask': mask,  # (N_rays)
        'sigma': _sigma,  # (N_rays, N_pts/N_pts-1)
        'radiance': _radiance,  # (N_rays, N_pts/N_pts-1)
        'zvals': _zvals,  # (N_rays, N_pts/N_pts-1)
        'alpha': alpha,  # (N_rays, N_pts/N_pts-1)
        'trans_shift': trans_shift,  # (N_rays, N_pts/N_pts-1)
        'weights': weights  # (N_rays, N_pts/N_pts-1)
    }

    return output


def alpha_to_weights(alpha: torch.Tensor):
    """Alpha to transmittance and weights
    - trans_shift(Ti = mul_1_i-1(1 - alpha_i)): accumulated transmittance, prob light pass all previous points
            - T0 = 1, pass previous prob is high at beginning
            - Tn closer 0, prob light pass previous point gets lower
    - weights(Ti * alpha_i): prob pass all previous point but stop at this
            - at max when alpha_i get high

    Args:
        alpha: tensor (N_rays, N_p), prob not pass this point on ray

    Returns:
        trans_shift: tensor (N_rays, N_p), accumulated transmittance, prob light pass all previous points
        weights: tensor (N_rays, N_p), prob pass all previous point but stop at this
    """
    dtype = alpha.dtype
    device = alpha.device
    # Ti = mul_1_i-1(1 - alpha_i)
    alpha_one = torch.ones_like(alpha[:, :1], dtype=dtype, device=device)
    trans_shift = torch.cat([alpha_one, 1 - alpha + 1e-10], -1)  # (N_rays, N_p+1)
    trans_shift = torch.cumprod(trans_shift, -1)[:, :-1]  # (N_rays, N_p)
    # weight_i = Ti * alpha_i
    weights = alpha * trans_shift  # (N_rays, N_p)

    return trans_shift, weights


def sample_ray_marching_output_by_index(output, index=None, n_rays=1, sigma_scale=2.0):
    """Sample output from ray marching by index, which is directly used for 2d visualization

    Args:
        output: output from ray_marching progress samples, with 'sigma', 'zvals', etc
                each is torch.tensor or np array
        index: a list of index to select. If None, use n_rays to sample.
        n_rays: num of sampled rays, by default, 1
        sigma_scale: used to scale pos sigma value up by this value for visual consistency. By default 2.0

    Returns:
        out_list: a list contain dicts for each ray sample
                Each dict has points, lines, legends which are lists of [x, y] and str for 2d visual
        sample_index: the index sampled
    """
    total_rays = output['zvals'].shape[0]
    n_pts_per_ray = output['zvals'].shape[1]

    if index is None:
        sample_index = np.random.choice(range(total_rays), n_rays, replace=False).tolist()
    else:
        sample_index = index

    out_list = []
    for idx in sample_index:
        res = {'points': [], 'lines': [], 'legends': []}
        # zvals as x
        x = torch_to_np(output['zvals'][idx]).tolist()
        res['points'].append([x, [-1] * n_pts_per_ray])
        # sigma, pos will be norm to (0-scale), neg will be norm to (-1, 0)
        sigma = torch_to_np(output['sigma'][idx]).copy()
        sigma_max = float(sigma.max())
        sigma_min = float(sigma.min())
        if sigma_max > 0:
            sigma[sigma > 0] = sigma[sigma > 0] / sigma_max * sigma_scale
        if sigma_min < 0:
            sigma[sigma < 0] = sigma[sigma < 0] / (np.abs(sigma_min) * 1.2)
        sigma = sigma.tolist()
        res['lines'].append([x, sigma])
        res['legends'].append('sigma(max={:.1f})'.format(sigma_max))
        # alpha
        alpha = torch_to_np(output['alpha'][idx]).tolist()
        res['lines'].append([x, alpha])
        res['legends'].append('alpha')
        # trans_shift
        trans_shift = torch_to_np(output['trans_shift'][idx]).tolist()
        res['lines'].append([x, trans_shift])
        res['legends'].append('trans_shift')
        # weights
        weights = torch_to_np(output['weights'][idx]).tolist()
        res['lines'].append([x, weights])
        res['legends'].append('weights')

        out_list.append(res)

    return out_list, sample_index


def make_sample_rays(near=2.0, far=4.0, n_pts=32, v_max=2.0, v_min=-1.0, sdf=True):
    """Make a synthetic sdf ray from + -> 0 -> - -> 0 -> +
    (near)           (far)
     + (max)         + (max)
      +             +
       +           +
        +         +
         0       0
          -     -
           -   -
            - -(min)

    Args:
        near: near zval. By default 2.0
        far: far zval. By default 4.0
        n_pts: num of total points. By default 32
        v_max: max value. By default 2.0
        v_min: min value. By default -1.0
        sdf: If True, value goes from v_max->0->v_min->0->v_max.
             Else from -v_max->0->-v_min->->0-v_max.
             By default True.

    Returns:
        output: a dict with following:
            zvals: np (1, N_pts), zvals from near to far
            zvals_list: list (N_pts)
            vals: np (1, N_pts), sdf or sigma values
            vals_list: list (N_pts)
            mid_zvals: (1, N_pts-1), middle of zvals
            mid_zvals_list: list (N_pts-1)
            mid_vals: (1, N_pts-1), middle of values
            mid_vals_list: list (N_pts-1)
            mid_slope: (1, N_pts-1), slope of middle values
            mid_slope_lost: list (N_pts-1)
    """
    assert v_max > 0 > v_min, 'Assert v_max > 0 > v_min in sample generation'
    assert n_pts % 2 == 0, 'Put input even num of pts'

    half_pts = int(n_pts / 2)
    zvals = np.linspace(near, far, n_pts)[None]
    zvals_list = zvals[0].tolist()

    mid_zvals = 0.5 * (zvals[:, 1:] + zvals[:, :-1])
    mid_zvals_list = mid_zvals[0].tolist()

    vals = np.concatenate([np.linspace(v_max, v_min, half_pts), np.linspace(v_min, v_max, half_pts)])[None]
    if not sdf:  # -v_max -> -v_min -> -v_max
        vals = -1 * vals
    vals_list = vals[0].tolist()

    mid_vals = 0.5 * (vals[:, 1:] + vals[:, :-1])
    mid_vals_list = mid_vals[0].tolist()

    mid_slope = (vals[:, 1:] - vals[:, :-1]) / (zvals[:, 1:] - zvals[:, :-1] + 1e-5)  # (1, N_pts-1)
    mid_slope_list = mid_slope[0].tolist()

    output = {
        'zvals': zvals,  # (1, N_pts)
        'zvals_list': zvals_list,  # (N_pts)
        'vals': vals,  # (1, N_pts)
        'vals_list': vals_list,  # (1, N_pts)
        'mid_zvals': mid_zvals,  # (1, N_pts-1)
        'mid_zvals_list': mid_zvals_list,  # (N_pts-1)
        'mid_vals': mid_vals,  # (1, N_pts-1)
        'mid_vals_list': mid_vals_list,  # (N_pts-1)
        'mid_slope': mid_slope,  # (1, N_pts-1)
        'mid_slope_list': mid_slope_list  # (N_pts-1)
    }

    return output


def handle_valid_mask_zvals(zvals: torch.Tensor, mask: torch.Tensor):
    """This function helps to move the valid pts at the beginning of each ray.

    ----------------------------------––----------
    e.g. (For a single rays)
    zvals = [0.0,  0.2,   0.4,   0.6,  0.8,  1.0]
    mask =  [True, False, False, True, True, False]
    ->
    zvals = [0.0,  0.6,  0.8,  0.8,   0.8,   0.8]
    mask =  [True, True, True, False, False, False]
    ----------------------------------––----------

    It could helps the network to process valid pts in geoNet/radianceNet with duplication,
    but do upsample/raymarching without each ray having a different number of pts.

    Args:
        zvals: (N_rays, N_pts) 2d tensor of zvals for pts on each rays
                zvals on each rays should be increase, or all is 0.
        mask: (N_rays, N_pts) indicating validity of each pts on each ray. Bool tensor.
               The T/F is located

    Returns:
        zvals: (N_rays, N_pts) with processed zvals
        mask: (N_rays, N_pts) with process mask
    """
    dtype = zvals.dtype
    device = zvals.device
    assert len(zvals.shape) == 2 and zvals.shape == mask.shape, 'Both tensor should be in (B, N)'

    # pts on the whole ray is invalid, make all zvals to be 0
    invalid_idx = torch.all(~mask, dim=1)  # (N_rays)
    zvals[invalid_idx] = 0.0

    # pts on the same rays has same zvals, and all pts are valid
    zvals_diff = zvals[:, 1:] - zvals[:, :-1]
    keep_one_idx = torch.logical_and(torch.all(torch.abs(zvals_diff) < 1e-7, dim=1), torch.all(mask, dim=1))  # (N_rays)
    mask[keep_one_idx, 1:] = False  # just keep one valid pts

    # other rays
    valid_idx = torch.logical_and(~invalid_idx, ~keep_one_idx)  # (N_rays)
    zvals_valid = zvals[valid_idx]  # (N_valid, N_pts)
    mask_valid = mask[valid_idx]  # (N_valid, N_pts)

    # mask True value/zvals at beginning for each rays
    mask_sort = torch.sort(
        mask_valid.type(torch.uint8), descending=True, dim=1
    )[0].type(torch.bool)  # sort on n_valid rays only, int for cuda
    zvals_valid[mask_sort] = zvals_valid[mask_valid]

    # make the final values
    last_idx = torch.sum(mask_sort, dim=1) - 1  # (N_valid, )
    row_idx = torch.arange(zvals_valid.shape[0], dtype=last_idx.dtype, device=device)  # (N_valid, )
    last_idx = torch.cat([row_idx[:, None], last_idx[:, None]], dim=1)  # (N_valid, 2)
    last_value = zvals_valid[last_idx[:, 0], last_idx[:, 1]][:, None]  # (N_valid, 1)
    zvals_last_value = torch.ones_like(zvals_valid, dtype=dtype, device=device) * last_value  # (N_valid, N_pts)
    zvals_last_value[mask_sort] = zvals_valid[mask_sort]

    # write back
    zvals[valid_idx] = zvals_last_value
    mask[valid_idx] = mask_sort

    return zvals, mask

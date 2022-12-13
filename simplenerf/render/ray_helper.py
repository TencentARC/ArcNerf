# -*- coding: utf-8 -*-

import torch

from simplenerf.geometry.projection import pixel_to_world


def get_rays(W, H, intrinsic: torch.Tensor, c2w: torch.Tensor, wh_order=True, ndc=False, ndc_near=1.0):
    """Get rays in world coord from camera.
    No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        c2w: torch.tensor(4, 4) cam pose. cam_to_world transform
        wh_order: If True, the rays are flatten in column-major. If False, in row-major. By default True
        ndc: If True, change rays to ndc space, you can then change near far to 0,1. By default False
        ndc_near: near zvals. By default 1.0.

    Returns:
        a ray_bundle with rays_o and rays_d. Each is in dim (N_ray, 3).
             If no sampler is used, return (WH, 3) num of rays
    """
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype, device=device), torch.linspace(0, H - 1, H, dtype=dtype, device=device)
    )  # i, j: (W, H)
    pixels = torch.stack([i, j], dim=-1).view(-1, 2).unsqueeze(0)  # (1, WH, 2)

    # reorder if full rays
    if not wh_order:  # (HW, 2)
        pixels = pixels.squeeze(0).contiguous().view(W, H, 2).permute(1, 0, 2).contiguous().view(-1, 2).unsqueeze(0)

    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype, device=device)  # (1, WH/N_rays)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH/N_rays, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH/N_rays, 3)
    rays_d = rays_d[0]
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH/N_rays, 3)

    # view dirs is in non-ndc space
    view_dirs = rays_d.clone()

    # chang to ndc
    if ndc:
        rays_o, rays_d = get_ndc_rays(rays_o, rays_d, W, H, intrinsic, ndc_near)

    return rays_o, rays_d, view_dirs


def get_ndc_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, W, H, intrinsic: torch.Tensor, near=1.0):
    """Change rays in original space to ndc space

    Args:
        rays_o: rays origin in original world space (N_ray, 3)
        rays_d: rays direction in original world space  (N_ray, 3)
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


def get_near_far_from_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    bounds: torch.Tensor = None,
    near_hardcode=None,
    far_hardcode=None,
):
    """Get near, far zvals from rays. Hard-reset by near/far_hardcode

    Args:
        rays_o: tensor(N_rays, 3), ray origin
        rays_d: tensor(N_rays, 3), ray direction
        bounds: tensor(N_rays, 2), input bounds, generally obtained from data with point_cloud.
                 Use ray-sphere far bound whenever bounding_radius is not None to restrict the pts in sphere.
        near_hardcode: If not None, will force all near to be this value
        far_hardcode: If not None, will force all far to be this value

    Returns:
        near: tensor(N_rays, 1), near zvals
        far:  tensor(N_rays, 1), far zvals
    """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]

    if bounds is not None:
        near, far = bounds[:, 0:1], bounds[:, 1:2]
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
    near: torch.Tensor, far: torch.Tensor, n_pts, inverse_linear=False, perturb=False
):
    """Get zvals from near/far distance

    Args:
        near: tensor(N_rays, 1), near zvals
        far: tensor(N_rays, 1), far zvals
        n_pts: num of points sampled in (near, far)
        inverse_linear: If False, uniform sampling in (near, far). If True, use inverse sampling and closer to near.
                        By default False.
        perturb: If True, disturb sampling in all segment. By default False.

    Returns:
        zvals: (N_rays, n_pts), each ray samples n_pts points
    """
    device = near.device
    dtype = near.dtype

    t_vals = torch.linspace(0.0, 1.0, n_pts, dtype=dtype, device=device)  # (N_pts,)

    if inverse_linear:  # +1e-8 in case nan
        zvals = 1.0 / (1.0 / (near + 1e-8) * (1.0 - t_vals) + 1.0 / (far + 1e-8) * t_vals)  # (N_rays, N_pts)
    else:
        zvals = near + (far - near) * t_vals

    if perturb:
        zvals = perturb_interval(zvals)  # (N_rays, N_pts)

    return zvals


def perturb_interval(vals: torch.Tensor):
    """Perturb sampling in the intervals

    Args:
        vals: tensor (B, N), sampling in N pts.

    Return:
        vals: pertube sampling (B, N)
    """
    dtype = vals.dtype
    device = vals.device
    _mids = .5 * (vals[..., 1:] + vals[..., :-1])  # (N_rays, N_pts-1)
    _upper = torch.cat([_mids, vals[..., -1:]], -1)  # (N_rays, N_pts)
    _lower = torch.cat([vals[..., :1], _mids], -1)  # (N_rays, N_pts)
    _z_rand = torch.rand(size=_upper.shape, dtype=dtype, device=device)
    vals = _lower + (_upper - _lower) * _z_rand  # (N_rays, N_pts)

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
    below = torch.clamp(inds - 1, 0, n_pts - 1)
    above = torch.clamp(inds, 0, n_pts - 1)
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
    rays_d: torch.Tensor,
    noise_std=0.0,
    weights_only=False,
    white_bkg=False,
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
        rays_d: (N_rays, 3), the real rays direction of each ray, used for adjust dists length by direction
        noise_std: noise level add to density if noise > 0, used for training and sigma mode. By default 0.0.
        weights_only: If True, return weights only, used in inference time for hierarchical sampling
        white_bkg: If True, make the accum weight=0 rays as 1.

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
            trans_shift: (N_rays, N_pts/N_pts-1). prob pass all previous light
            weights: (N_rays, N_pts/N_pts-1). Use for weighting other values like normals.
    """
    dtype = zvals.dtype
    device = zvals.device
    n_rays = zvals.shape[0]

    deltas = zvals[:, 1:] - zvals[:, :-1]  # (N_rays, N_pts-1)
    assert torch.all(deltas >= 0), 'zvals is not all increase....'

    # add an inf distance as last to keep original dimension
    deltas = torch.cat([deltas, 1e10 * torch.ones(size=(n_rays, 1), dtype=dtype, device=device)], dim=-1)
    # adjust distance by direction
    deltas = deltas * torch.norm(rays_d, dim=-1).unsqueeze(-1)  # (N_rays, N_pts)

    # use sigma to get alpha
    noise = 0.0
    if noise_std > 0.0:
        noise = torch.randn(sigma.shape, dtype=dtype, device=device) * noise_std

    # alpha_i = (1 - exp(- relu(sigma + noise) * delta_i)
    alpha = 1 - torch.exp(-torch.relu(sigma + noise) * deltas)  # (N_rays, N_p)

    trans_shift, weights = alpha_to_weights(alpha)  # (N_rays, N_p) * 2

    if weights_only:
        output = {'weights': weights}
        return output

    # depth = sum(weight_i * zvals_i)
    depth = torch.sum(weights * zvals, -1)  # (N_rays)
    # accumulated weight(mask)
    mask = torch.sum(weights, -1)  # (N_rays)

    # rgb = sum(weight_i * radiance_i)
    rgb = torch.sum(weights.unsqueeze(-1) * radiance, -2)  # (N_rays, 3)
    if white_bkg:  # where mask = 0, rgb = 1
        rgb = rgb + (1.0 - mask[:, None])

    output = {
        'rgb': rgb,  # (N_rays, 3)
        'depth': depth,  # (N_rays)
        'mask': mask,  # (N_rays)
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

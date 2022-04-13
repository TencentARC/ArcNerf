# -*- coding: utf-8 -*-

import numpy as np
import torch

from .transformation import batch_dot_product
from common.utils.torch_utils import set_tensor_to_zeros


def get_ray_points_by_zvals(rays_o: torch.Tensor, rays_d: torch.Tensor, zvals: torch.Tensor):
    """Get ray points by zvals. Each ray can be sampled by N_pts.
        rays_d is assumed to be normalized.

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        zvals: depth values, (N_rays, N_pts)

    Returns:
        rays_pts: final rays points (N_rays, N_pts, 3)
    """
    n_rays = rays_o.shape[0]
    n_pts = zvals.shape[1]
    assert zvals.shape[0] == n_rays, 'Invalid shape for zvals... Should be (N_rays, N_pts)'

    rays_pts = torch.repeat_interleave(rays_o.unsqueeze(1), n_pts, 1)
    rays_pts += torch.einsum('bi, bk->bki', rays_d, zvals)

    return rays_pts


def closest_point_on_ray(rays_o: torch.Tensor, rays_d: torch.Tensor, pts: torch.Tensor):
    """Closest point on ray. Allow batched processing.
    If point if projected on negative direction of ray, choose rays_o.
    rays_d is assumed to be normalized.

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        pts: point (N_pts, 3)

    Returns:
        pts_closest: (N_rays, N_pts, 3), each ray has N_pts
        zvals: (N_rays, N_pts), zvals for each ray
    """
    n_rays = rays_o.shape[0]
    n_pts = pts.shape[0]

    CA = pts.unsqueeze(0) - rays_o.unsqueeze(1)  # (N_rays, N_pts, 3)
    AB = torch.repeat_interleave(rays_d.unsqueeze(1), n_pts, 1)  # (N_rays, N_pts, 3)
    zvals = batch_dot_product(CA.view(-1, 3), AB.view(-1, 3))  # (N_rays, N_pts)
    zvals = zvals / batch_dot_product(AB.view(-1, 3), AB.view(-1, 3))
    zvals = torch.clamp_min(zvals, 0.0).view(n_rays, n_pts)

    pts_closest = get_ray_points_by_zvals(rays_o, rays_d, zvals)

    return pts_closest, zvals


def closest_point_to_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, n_init_pairs=10, n_iter=100, thres=1e-2, lr=1e-3):
    """Closest point to multiple rays. Use optimizer for loss optimization.
        rays_d is assumed to be normalized.

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        n_init_pairs: num of pair of rays to be selected at first to setup the init point.
                        More pair more robustness. By default 10.
        n_iter: max num of iter, by default 100
        thres: error thres(min mean distance), by default 1e-2
        lr: init lr, by default 1e-3

    Returns:
        pts: (1, 3), Only one point is produced
        distance: (10, ), shortest distance on all rays
        zvals: (n_rays, 1), zval of all rays
    """
    n_rays = rays_o.shape[0]
    assert n_rays > 1, 'At least has two rays...'

    # select several random pairs to find mean point
    pts_init_all = []
    for _ in range(n_init_pairs):
        pair_idx = np.random.choice(range(n_rays), 2, replace=False)
        pts_init, _, _ = closest_point_to_two_rays(rays_o[pair_idx, :], rays_d[pair_idx, :])  # (1, 3)
        pts_init_all.append(pts_init)
    pts_init_all = torch.cat(pts_init_all).mean(0)[None, :]
    pts_optim = torch.nn.Parameter(pts_init_all).to(pts_init.device)  # (1, 3)
    zvals = torch.zeros(size=(n_rays, 1), dtype=rays_o.dtype, device=rays_o.device)
    distance = torch.zeros(size=(n_rays, 1), dtype=rays_o.dtype, device=rays_o.device)

    optimizer = torch.optim.Adam([pts_optim], lr=lr)
    for i in range(n_iter):
        pts_on_rays, zvals = closest_point_on_ray(rays_o, rays_d, pts_optim)  # (10, 1, 3), (10, 1)
        distance = torch.norm(pts_on_rays[:, 0, :] - pts_optim, dim=-1)  # (10, )
        mean_dist = distance.mean()
        if mean_dist < thres:
            break
        mean_dist.backward()
        optimizer.step()

    return pts_optim.detach(), distance.detach(), zvals.detach()


def closest_point_to_two_rays(rays_o: torch.Tensor, rays_d: torch.Tensor):
    """Closest point to two rays. Min distance vec should be perpendicular to both rays.
        rays_d is assumed to be normalized.
    ref: https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
        https://math.stackexchange.com/questions/1036959/midpoint-of-the-shortest-distance-between-2-rays-in-3d

    Args:
        rays_o: ray origin, (2, 3)
        rays_d: ray direction, assume normalized, (2, 3)

    Returns:
        pts: (1, 3), Only one point is produced
        distance: shortest distance on rays
        zvals: (2, 1), zval of two rays
    """
    n_rays = rays_o.shape[0]
    assert n_rays == 2, 'Only two rays allows...'

    r1_o, r1_d = rays_o[0], rays_d[0]  # (3, ) * 2
    r2_o, r2_d = rays_o[1], rays_d[1]  # (3, ) * 2

    dot = torch.dot
    z1 = (dot(r2_o - r1_o, r1_d) + dot(r1_d, r2_d) * dot(r1_o - r2_o, r2_d)) / (1 - dot(r1_d, r2_d)**2)
    z2 = (dot(r1_o - r2_o, r2_d) + dot(r1_d, r2_d) * dot(r2_o - r1_o, r1_d)) / (1 - dot(r1_d, r2_d)**2)

    zvals = torch.cat([z1[None], z2[None]])[:, None]  # (2, 1)
    if torch.any(torch.isnan(zvals)) or not torch.all(zvals >= 0):  # not on the ray, or ray parallel
        r1_o_on_r2, z_r2 = closest_point_on_ray(rays_o[1:2], rays_d[1:2], rays_o[0:1])  # (1, 1, 3), (1, 1)
        r2_o_on_r1, z_r1 = closest_point_on_ray(rays_o[0:1], rays_d[0:1], rays_o[1:2])  # (1, 1, 3), (1, 1)
        r1_o_dist_to_r2 = torch.norm(r1_o - r1_o_on_r2[0, 0])
        r2_o_dist_to_r1 = torch.norm(r2_o - r2_o_on_r1[0, 0])
        if r1_o_dist_to_r2 < r2_o_dist_to_r1:  # close of r1 is z
            distance = r1_o_dist_to_r2
            zvals = torch.zeros_like(z_r2, dtype=z_r2.dtype, device=z_r2.device)
            zvals = torch.cat([zvals, z_r2], dim=0)  # (2, 1)
            pts = 0.5 * (rays_o[0:1] + r1_o_on_r2[0])  # (1, 3)
        else:
            distance = r2_o_dist_to_r1
            zvals = torch.zeros_like(z_r1, dtype=z_r1.dtype, device=z_r1.device)
            zvals = torch.cat([z_r1, zvals], dim=0)  # (2, 1)
            pts = 0.5 * (rays_o[1:2] + r2_o_on_r1[0])  # (1, 3)
    else:
        pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)[:, 0, :]  # (2, 3)
        distance = torch.norm(pts[0] - pts[1])
        pts = 0.5 * (pts[0:1] + pts[1:2])  # (1, 3)

    return pts, distance, zvals


def closest_distance_of_two_rays(rays_o: torch.Tensor, rays_d: torch.Tensor):
    """Closest point to two rays. Min distance vec should be perpendicular to both rays.
        rays_d is assumed to be normalized.
        This function is only distance when zvals > 0.
    ref: https://math.stackexchange.com/questions/13734/how-to-find-shortest-distance-between-two-skew-lines-in-3d

    Args:
        rays_o: ray origin, (2, 3)
        rays_d: ray direction, assume normalized, (2, 3)

    Returns:
        distance: shortest distance on rays
    """
    n_rays = rays_o.shape[0]
    assert n_rays == 2, 'Only two rays allows...'

    r1_o, r1_d = rays_o[0], rays_d[0]  # (3, ) * 2
    r2_o, r2_d = rays_o[1], rays_d[1]  # (3, ) * 2

    distance = torch.norm(torch.dot(torch.cross(r1_d, r2_d), r1_o - r2_o))
    distance = distance / torch.norm(torch.cross(r1_d, r2_d))

    return distance


def sphere_ray_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, origin=(0, 0, 0), radius=1.0):
    """Get intersection of ray with sphere surface and the near/far zvals.
    This will be 4 cases: (1)outside no intersection -> near/far: 0, mask = 0
                          (2)outside 1 intersection  -> near = far, mask = 1
                          (3)outside 2 intersections -> near=near>0, far=far
                          (4)inside 1 intersection -> near=0, far=far
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/definition-ray
    Since floating point error exists, we set torch.tensor as 0 for small values, used for tangent case

     Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        origin: sphere origin, by default (0, 0, 0)
        radius: sphere radius, by default 1.0

    Returns:
        near: near intersection zvals. (N_rays, 1)
              If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
        far:  far intersection zvals. (N_rays, 1)
              If only 1 intersection: if not tangent, same as far; else 0.
        pts: (N_rays, 2, 3), each ray has near/far two points, if nan, means no intersection at this ray
        mask: (N_rays,), show whether each ray has intersection with the sphere, BoolTensor
     """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]

    mask = torch.ones(size=(n_rays, 1)).type(torch.BoolTensor).to(device)

    C = torch.tensor([origin], dtype=dtype).to(device)  # (1, 3)
    r = torch.tensor([radius], dtype=dtype).to(device)  # (1, )

    OC = C - rays_o  # (N_rays, 3)
    z_half = batch_dot_product(OC, rays_d).unsqueeze(1)  # (N_rays, 1)
    z_half = set_tensor_to_zeros(z_half)
    rays_o_in_sphere = torch.norm(OC, dim=-1) <= radius
    rays_o_in_sphere = rays_o_in_sphere.unsqueeze(-1)  # (N_rays, 1)
    mask = torch.logical_and(mask, torch.logical_or(z_half > 0, rays_o_in_sphere))

    d_2 = batch_dot_product(OC, OC) - batch_dot_product(z_half, z_half)
    d_2 = d_2.unsqueeze(1)  # (N_rays, 1)
    d_2 = set_tensor_to_zeros(d_2)
    mask = torch.logical_and(mask, (d_2 >= 0))

    z_offset = r**2 - d_2  # (N_rays, 1)
    z_offset = set_tensor_to_zeros(z_offset)
    mask = torch.logical_and(mask, (z_offset >= 0))
    z_offset = torch.sqrt(z_offset)

    near = z_half - z_offset
    near = torch.clamp_min(near, 0.0)
    far = z_half + z_offset
    far = torch.clamp_min(far, 0.0)
    near[~mask], far[~mask] = 0.0, 0.0

    zvals = torch.cat([near, far], dim=1)  # (N_rays, 2)
    pts = get_ray_points_by_zvals(rays_o, rays_d, zvals)

    return near, far, pts, mask[:, 0]

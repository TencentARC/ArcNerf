# -*- coding: utf-8 -*-

import numpy as np
import torch

from .transformation import batch_dot_product
from arcnerf.ops.volume_func import ray_aabb_intersection_cuda, CUDA_BACKEND_AVAILABLE
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


def sphere_ray_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, radius: torch.Tensor, origin=(0, 0, 0)):
    """Get intersection of ray with sphere surface and the near/far zvals.
    This will be 6 cases: (1)outside no intersection -> near/far: 0, mask = 0
                          (2)outside 1 intersection  -> near = far, mask = 1
                          (3)outside 2 intersections -> near=near>0, far=far
                          (4)inside 1 intersection -> near=0, far=far
                          (5)on surface 1 intersection -> near=0=far=0
                          (6)on surface 2 intersection -> near=0, far=far (tangent/not tangent)
    www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    Since floating point error exists, we set torch.tensor as 0 for small values, used for tangent case

     Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        radius: sphere radius in (N_r, ) or a single value.
        origin: sphere origin, by default (0, 0, 0). Support only one origin now

    Returns:
        near: near intersection zvals. (N_rays, N_r)
              If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
        far:  far intersection zvals. (N_rays, N_r)
              If only 1 intersection: if not tangent, same as far; else 0.
        pts: (N_rays, N_r, 2, 3), each ray has near/far two points with each sphere.
        mask: (N_rays, N_r), show whether each ray has intersection with the sphere, BoolTensor
     """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]
    # read radius
    if not isinstance(radius, torch.Tensor):
        assert isinstance(radius, float) or isinstance(radius, int), 'Invalid type'
        radius = torch.tensor([radius], dtype=dtype, device=device)
    n_sphere = radius.shape[0]

    rays_o_repeat = torch.repeat_interleave(rays_o, n_sphere, 0)  # (N_rays*N_r, 3)
    rays_d_repeat = torch.repeat_interleave(rays_d, n_sphere, 0)  # (N_rays*N_r, 3)
    r = torch.repeat_interleave(radius.unsqueeze(0), n_rays, 0).view(-1, 1)  # (N_rays*N_r, 3)

    mask = torch.ones(size=(n_rays * n_sphere, 1), dtype=torch.bool, device=device)

    C = torch.tensor([origin], dtype=dtype, device=device)  # (1, 3)
    C = torch.repeat_interleave(C, n_rays * n_sphere, 0)  # (N_rays*N_r, 3)

    OC = C - rays_o_repeat  # (N_rays*N_r, 3)
    z_half = batch_dot_product(OC, rays_d_repeat).unsqueeze(1)  # (N_rays*N_r, 1)
    z_half = set_tensor_to_zeros(z_half)
    rays_o_in_sphere = torch.norm(OC, dim=-1) <= r[:, 0]  # (N_rays*N_r, )
    rays_o_in_sphere = rays_o_in_sphere.unsqueeze(1)  # (N_rays*N_r, 1)
    mask = torch.logical_and(mask, torch.logical_or(z_half > 0, rays_o_in_sphere))  # (N_rays*N_r, 1)

    d_2 = batch_dot_product(OC, OC) - batch_dot_product(z_half, z_half)  # (N_rays*N_r,)
    d_2 = d_2.unsqueeze(1)
    d_2 = set_tensor_to_zeros(d_2)  # (N_rays*N_r, 1)
    mask = torch.logical_and(mask, (d_2 >= 0))  # (N_rays*N_r, 1)

    z_offset = r**2 - d_2  # (N_rays*N_r, 1)
    z_offset = set_tensor_to_zeros(z_offset)
    mask = torch.logical_and(mask, (z_offset >= 0))
    z_offset = torch.sqrt(z_offset)

    near = z_half - z_offset
    near = torch.clamp_min(near, 0.0)
    far = z_half + z_offset
    far = torch.clamp_min(far, 0.0)
    near[~mask], far[~mask] = 0.0, 0.0  # (N_rays*N_r, 1) * 2

    zvals = torch.cat([near, far], dim=1)  # (N_rays*N_r, 2)
    pts = get_ray_points_by_zvals(rays_o_repeat, rays_d_repeat, zvals)  # (N_rays*N_r, 2, 3)

    # reshape
    near = near.contiguous().view(n_rays, n_sphere)
    far = far.contiguous().view(n_rays, n_sphere)
    mask = mask.contiguous().view(n_rays, n_sphere)
    pts = pts.contiguous().view(n_rays, n_sphere, 2, 3)

    return near, far, pts, mask


def aabb_ray_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_range: torch.Tensor, eps=1e-7):
    """Get intersection of ray with volume outside surface and the near/far zvals.
    This will be 6 cases: (1)outside no intersection -> near/far: 0, mask = 0
                          (2)outside 1 intersection  -> near = far, mask = 1
                          (3)outside 2 intersections -> near=near>0, far=far (tangent/not tangent)
                          (4)inside 1 intersection -> near=0, far=far
                          (5)on surface 1 intersection -> near=0=far=0
                          (6)on surface 2 intersection -> near=0, far=far (tangent/not tangent)
    www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    Since floating point error exists, we set torch.tensor as 0 for small values, used for tangent case

     Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        aabb_range: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
        eps: error threshold for parallel comparison, by default 1e-7

    Returns:
        near: near intersection zvals. (N_rays, N_v)
              If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
        far:  far intersection zvals. (N_rays, N_v)
              If only 1 intersection: if not tangent, same as far; else 0.
        pts: (N_rays, N_v, 2, 3), each ray has near/far two points with each volume.
        mask: (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
    """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]
    n_volume = aabb_range.shape[0]
    assert aabb_range.shape[1] == 3 and aabb_range.shape[2] == 2, 'AABB range must be (N, 3, 2)'

    if CUDA_BACKEND_AVAILABLE and rays_o.is_cuda:
        near, far, pts, mask = ray_aabb_intersection_cuda(rays_o, rays_d, aabb_range, eps)
    else:
        near = torch.zeros((n_rays * n_volume, ), dtype=dtype, device=device)  # (N_rays*N_v,)
        far = torch.ones((n_rays * n_volume, ), dtype=dtype, device=device) * 10000.0  # (N_rays*N_v,)
        aabb_range_repeat = torch.repeat_interleave(aabb_range.unsqueeze(0), n_rays, 0).view(-1, 3, 2)  # (*, 3, 2)
        min_range, max_range = aabb_range_repeat[..., 0], aabb_range_repeat[..., 1]  # (N_rays*N_v, 3)
        mask = torch.ones(size=(n_rays * n_volume, ), dtype=torch.bool, device=device)

        rays_o_repeat = torch.repeat_interleave(rays_o, n_volume, 0)  # (N_rays*N_v, 3)
        rays_d_repeat = torch.repeat_interleave(rays_d, n_volume, 0)  # (N_rays*N_v, 3)

        def update_bound(_rays_o, _rays_d, _min_range, _max_range, _mask, _near, _far, dim=0):
            """Update bound and mask on each dim"""
            _mask_axis = (torch.abs(_rays_d[..., dim]) < eps)  # (N_rays*N_v,)
            _mask_axis_out = torch.logical_or((_rays_o[..., dim] < _min_range[..., dim]),
                                              (_rays_o[..., dim] > _max_range[..., dim]))  # outside the plane
            _mask[torch.logical_and(_mask_axis, _mask_axis_out)] = False

            t1 = (_min_range[..., dim] - _rays_o[..., dim]) / _rays_d[..., dim]
            t2 = (_max_range[..., dim] - _rays_o[..., dim]) / _rays_d[..., dim]
            t = torch.cat([t1[:, None], t2[:, None]], dim=-1)
            t1, _ = torch.min(t, dim=-1)
            t2, _ = torch.max(t, dim=-1)
            update_near = torch.logical_and(_mask, t1 > _near)
            _near[update_near] = t1[update_near]
            update_far = torch.logical_and(_mask, t2 < _far)
            _far[update_far] = t2[update_far]
            _mask[_near > _far] = False

            return _mask, _near, _far

        # x plane
        mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 0)
        # y plane
        mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 1)
        # z plane
        mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 2)

        near, far, mask = near[:, None], far[:, None], mask[:, None]  # (N_rays*N_v, 1)

        near = torch.clamp_min(near, 0.0)
        far = torch.clamp_min(far, 0.0)
        near[~mask], far[~mask] = 0.0, 0.0  # (N_rays*N_v, 1) * 2

        # add some eps for reduce the rounding error
        near[mask] += eps
        far[mask] -= eps

        zvals = torch.cat([near, far], dim=1)  # (N_rays*N_v, 2)
        pts = get_ray_points_by_zvals(rays_o_repeat, rays_d_repeat, zvals)  # (N_rays*N_v, 2, 3)

        # reshape
        near = near.contiguous().view(n_rays, n_volume)
        far = far.contiguous().view(n_rays, n_volume)
        mask = mask.contiguous().view(n_rays, n_volume)
        pts = pts.contiguous().view(n_rays, n_volume, 2, 3)

    return near, far, pts, mask


def surface_ray_intersection(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    geo_func,
    method='sphere_tracing',
    near=0.0,
    far=10.0,
    n_step=128,
    n_iter=100,
    threshold=0.001,
    level=0.0,
    grad_dir='ascent',
):
    """Finding the surface-ray intersection given geo_func(like sdf func)

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        geo_func: input a point with (N_pts, 3), return (N_pts) as sdf value
        method: method used to find the intersection. support
                ['sphere_tracing', 'secant_root_finding']
        near: near distance to start the searching. By default 0.0
        far: far distance to end the searching, after that are background. By default 10.0
            near/far can be single value or tensor in (N_rays, 1)
        n_step: used for secant_root_finding, split the whole ray into intervals. By default 128
        n_iter: num of iter to run finding algorithm. By default 100, large enough to escape
        threshold: error bounding to stop the iteration. By default 0.001 (1mm)
        level: the surface pts geo_value offset. 0.0 is for sdf. some positive value may be for density.
        grad_dir: If descent, the inner obj has geo_value > level,
                            find the root where geo_value first meet ---level+++
                  If ascent, the inner obj has geo_value < level(like sdf),
                            find the root where geo_value first meet +++level---

    Returns:
        zvals: (N_rays, 1), each ray intersection zvals. If no intersection, use the zvals after far
        pts: (N_rays, 3), each ray with a point intersected with surface.
        mask: (N_rays,), show whether each ray has intersection with the surface, BoolTensor
    """
    if method == 'sphere_tracing':
        zvals, pts, mask = sphere_tracing(rays_o, rays_d, geo_func, near, far, n_iter, threshold)
    elif method == 'secant_root_finding':
        zvals, pts, mask = secant_root_finding(
            rays_o, rays_d, geo_func, near, far, n_step, n_iter, threshold, level, grad_dir
        )
    else:
        raise NotImplementedError('Method {} not support for surface-ray intersection'.format(method))

    return zvals, pts, mask


def sphere_tracing(
    rays_o: torch.Tensor, rays_d: torch.Tensor, sdf_func, near=0.0, far=10.0, n_iter=100, threshold=0.001
):
    """Finding the surface-ray intersection by sphere_tracing using sdf_func
       If the pts is inside the obj (sdf < 0) or more than far, do not find its intersection pts.

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        sdf_func: input a point with (N_pts, 3), return (N_pts) as sdf value
        near: near distance to start the searching. By default 0.0
        far: far distance to end the searching, after that are background. By default 10.0
            near/far can be single value or tensor in (N_rays, 1)
        n_iter: num of iter to run sphere_tracing algorithm. By default 100, large enough to escape
        threshold: error bounding to stop the iteration. By default 0.001 (1mm)

    Returns:
        zvals: (N_rays, 1), each ray intersection zvals. If no intersection, use the zvals after far
        pts: (N_rays, 3), each ray with a point intersected with surface.
        mask: (N_rays,), show whether each ray has intersection with the surface, BoolTensor
    """
    dtype = rays_o.dtype
    device = rays_o.device
    n_rays = rays_o.shape[0]

    # set near far
    if isinstance(near, torch.Tensor) and near.shape == (n_rays, 1):
        _near = near
    else:
        _near = torch.ones((n_rays, 1), dtype=dtype, device=device) * near
    if isinstance(far, torch.Tensor) and far.shape == (n_rays, 1):
        _far = far
    else:
        _far = torch.ones((n_rays, 1), dtype=dtype, device=device) * far

    zvals = torch.ones((n_rays, 1), dtype=dtype, device=device) * _near  # (N_rays, 1), start from rays_o
    mask = torch.ones(n_rays, dtype=torch.bool, device=device)  # (N_rays)
    obj_mask = torch.zeros(n_rays, dtype=torch.bool, device=device)  # (N_rays)
    sdf = torch.zeros(n_rays, dtype=dtype, device=device)  # (N_rays)

    for _ in range(n_iter):
        # only update for the valid pts
        valid_mask = torch.logical_and(~obj_mask, mask)  # (N_valid,)
        pts = get_ray_points_by_zvals(rays_o[valid_mask], rays_d[valid_mask], zvals[valid_mask]).view(-1, 3)
        # all pts are invalid
        if pts.shape[0] == 0:
            break
        with torch.no_grad():
            sdf[valid_mask] = sdf_func(pts)  # (N_valid)
        # stop if all valid update sdf is small
        if torch.all(torch.abs(sdf) < threshold):
            break
        # update obj mask if sdf is small
        obj_mask[torch.abs(sdf) < threshold] = True
        # update only not converge rays
        zvals[torch.logical_and(~obj_mask, mask)] += sdf[torch.logical_and(~obj_mask, mask)][:, None]  # (N_valid, 1)
        # update mask
        mask[zvals[:, 0] > _far[:, 0]] = False
        mask[zvals[:, 0] < _near[:, 0]] = False

    zvals[zvals <= near] = 0.0  # set min distance as 0.0

    pts = get_ray_points_by_zvals(rays_o, rays_d, zvals).view(-1, 3)

    return zvals, pts, mask


def secant_root_finding(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    geo_func,
    near=0.0,
    far=10.0,
    n_step=128,
    n_iter=20,
    threshold=0.001,
    level=0.0,
    grad_dir='ascent'
):
    """Finding the surface-ray intersection by root finding using secant method. It does not require sdf

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        geo_func: input a point with (N_pts, 3), return (N_pts) as geo value(density, sdf)
        near: near distance to start the searching. By default 0.0
        far: far distance to end the searching, after that are background. By default 10.0
            near/far can be single value or tensor in (N_rays, 1)
        n_step: used for secant_root_finding, split the whole ray into intervals. By default 128
        n_iter: num of iter to run finding algorithm. By default 20
        threshold: error bounding to stop the iteration. By default 0.001 (1mm)
        level: the surface pts geo_value offset. 0.0 is for sdf. some positive value may be for density.
        grad_dir: If descent, the inner obj has geo_value > level,
                            find the root where geo_value first meet ---level+++
                  If ascent, the inner obj has geo_value < level(like sdf),
                            find the root where geo_value first meet +++level---

    Returns:
        zvals: (N_rays, 1), each ray intersection zvals. If no intersection, use the zvals as far
        pts: (N_rays, 3), each ray with a point intersected with surface.
        mask: (N_rays,), show whether each ray has intersection with the surface, BoolTensor
    """
    dtype = rays_o.dtype
    device = rays_o.device
    n_rays = rays_o.shape[0]

    zvals = torch.zeros((n_rays, 1), dtype=dtype, device=device)  # (N_rays, 1), start from rays_o

    t = torch.linspace(0., 1., n_step, device=device)[None, :]  # (N_pts, 1)
    if isinstance(near, torch.Tensor) and near.shape == (n_rays, 1):
        _near = near
    else:
        _near = torch.ones((n_rays, 1), dtype=dtype, device=device) * near
    if isinstance(far, torch.Tensor) and far.shape == (n_rays, 1):
        _far = far
    else:
        _far = torch.ones((n_rays, 1), dtype=dtype, device=device) * far
    step = _near * (1 - t) + _far * t  # (N_rays, N_pts)

    pts = get_ray_points_by_zvals(rays_o, rays_d, step).view(-1, 3)  # (N_rays*N_pts, 3)
    with torch.no_grad():
        geo_value = geo_func(pts).view(n_rays, -1)  # (N_rays, N_pts)
    geo_value_diff = geo_value - level
    if grad_dir == 'descent':
        geo_value_diff *= -1  # from ---+++ to +++---

    # if the first one is inside, the value is negative
    mask_not_occ = (geo_value_diff[..., 0] > 0)  # (N_rays)

    # capture the sign change from +++ to ---
    sign_matrix = torch.cat(
        [
            torch.sign(geo_value_diff[..., :-1] * geo_value_diff[..., 1:]),  # (N_rays, N_pts-1)
            torch.ones([n_rays, 1], device=device)  # (N_rays, 1)
        ],
        dim=-1
    )  # (N_rays, N_pts)

    # first change gives higher weights
    cost_matrix = sign_matrix * torch.arange(n_step, 0, -1, dtype=dtype, device=device)  # (N_rays, N_pts)
    min_cost, index = torch.min(cost_matrix, -1)  # (N_rays) * 2

    # at least one sign change in (0, far)
    mask_sign_change = (min_cost < 0)

    # mask change from +++ to ---
    mask_pos_to_neg = (geo_value_diff[torch.arange(n_rays), index] > 0)

    # all the mask
    mask = (mask_not_occ & mask_sign_change & mask_pos_to_neg)

    # run secant method, just run on the rays with intersection
    z_high = step[torch.arange(n_rays), index][mask]  # (N_valid)
    geo_high = geo_value_diff[torch.arange(n_rays), index][mask]  # (N_valid)
    index = torch.clamp(index + 1, max=n_step - 1)
    z_low = step[torch.arange(n_rays), index][mask]  # (N_valid)
    geo_low = geo_value_diff[torch.arange(n_rays), index][mask]  # (N_valid)

    rays_o_mask = rays_o[mask]
    rays_d_mask = rays_d[mask]
    n_rays_valid = rays_o_mask.shape[0]

    # valid
    if n_rays_valid > 0:
        # weight zvals near surface
        z_mid = -geo_low * (z_high - z_low) / (geo_high - geo_low) + z_low  # (N_valid)
        z_mid_init = z_mid.clone()
        for i in range(n_iter):
            # stop if all valid update zval is small
            if i > 0 and torch.all(torch.abs(z_mid_init - z_mid) < threshold):
                break
            pts_mid = get_ray_points_by_zvals(rays_o_mask, rays_d_mask, z_mid.unsqueeze(1)).view(-1, 3)  # (N_valid, 3)
            with torch.no_grad():
                geo_mid_value = geo_func(pts_mid)  # (N_valid,)
            geo_mid_value_diff = geo_mid_value - level
            if grad_dir == 'descent':
                geo_mid_value_diff *= -1  # from ---+++ to +++---

            ind_low = (geo_mid_value_diff < 0)  # (N_valid)
            if ind_low.sum() > 0:
                z_low[ind_low] = z_mid[ind_low]
                geo_low[ind_low] = geo_mid_value_diff[ind_low]

            if ~ind_low.sum() > 0:
                z_high[~ind_low] = z_mid[~ind_low]
                geo_high[~ind_low] = geo_mid_value_diff[~ind_low]

            z_mid = -geo_low * (z_high - z_low) / (geo_high - geo_low) + z_low  # (N_valid)

    # update zvals for different case
    if n_rays_valid > 0:
        zvals[mask, 0] = z_mid
    zvals[~mask] = _far[~mask]  # if no change, too far
    zvals[~mask_not_occ] = 0.0  # inside obj
    zvals[zvals <= _near] = 0.0  # set min distance

    pts = get_ray_points_by_zvals(rays_o, rays_d, zvals).view(-1, 3)

    return zvals, pts, mask

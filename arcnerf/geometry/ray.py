# -*- coding: utf-8 -*-

import torch


def get_ray_points_by_zvals(rays_o: torch.Tensor, rays_d: torch.Tensor, zvals: torch.Tensor):
    """Get ray points by zvals. Each ray can be sampled by N_pts

    Args:
        rays_o: ray origin, (N_rays, 3), torch
        rays_d: ray direction, generally normalized, (N_rays, 3), np or direction
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

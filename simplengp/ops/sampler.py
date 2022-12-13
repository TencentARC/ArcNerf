# -*- coding: utf-8 -*-

import torch

try:
    import _sampler
except ImportError:
    raise NotImplementedError("You have not build the customized ops...run `sh scripts/install_ops.sh`...")


# -------------------------------------------------- ------------------------------------ #


class RaysSample(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, rays_o, rays_d, density_grid_bitfield, near_distance, n_sample,
            min_step, max_step, cone_angle, aabb_range, n_grid, n_cascades
    ):
        device = rays_o.device
        n_rays_per_batch = rays_o.shape[0]
        num_coords_elements = n_rays_per_batch * n_sample
        # output
        pts = torch.zeros((num_coords_elements, 3), dtype=rays_o.dtype, device=device)
        dirs = torch.zeros((num_coords_elements, 3), dtype=rays_o.dtype, device=device)
        dt = torch.zeros((num_coords_elements,), dtype=rays_o.dtype, device=device)

        rays_numsteps = torch.zeros((n_rays_per_batch, 2), dtype=torch.int32, device=device)
        ray_numstep_counter = torch.zeros((1,), dtype=torch.int32, device=device)

        _sampler.rays_sampler(rays_o, rays_d, density_grid_bitfield, near_distance, n_sample,
                              min_step, max_step, cone_angle, aabb_range[0], aabb_range[1], n_grid, n_cascades,
                              pts, dirs, dt, rays_numsteps, ray_numstep_counter)

        # keep only valid samples
        n_samples = ray_numstep_counter[0].item()
        pts, dirs = pts.detach()[:n_samples].contiguous(), dirs.detach()[:n_samples].contiguous()
        dt = dt.detach()[:n_samples].contiguous()
        rays_numsteps = rays_numsteps.detach()
        ray_numstep_counter = ray_numstep_counter.detach()

        return pts, dirs, dt, rays_numsteps, ray_numstep_counter


@torch.no_grad()
def rays_sampler(
        rays_o, rays_d, density_grid_bitfield, near_distance, n_sample,
        min_step, max_step, cone_angle, aabb_range, n_grid, n_cascades
):
    return RaysSample.apply(
        rays_o, rays_d, density_grid_bitfield, near_distance, n_sample, min_step, max_step, cone_angle,
        aabb_range, n_grid, n_cascades
    )

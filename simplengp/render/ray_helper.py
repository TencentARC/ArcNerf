# -*- coding: utf-8 -*-

import torch

from simplengp.geometry.projection import pixel_to_world
from simplengp.geometry.transformation import normalize


def get_rays(W, H, intrinsic: torch.Tensor, c2w: torch.Tensor, wh_order=True, center_pixel=False):
    """Get rays in world coord from camera.
    No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        c2w: torch.tensor(4, 4) cam pose. cam_to_world transform
        wh_order: If True, the rays are flatten in column-major. If False, in row-major. By default True
        center_pixel: If True, use the center pixel from (0.5, 0.5) instead of corner(0, 0)

    Returns:
        rays_o: origin (N_ray, 3) tensor. If no sampler is used, return (WH, 3) num of rays
        rays_d: direction (N_ray, 3) tensor. If no sampler is used, return (WH, 3) num of rays
    """
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype, device=device),
        torch.linspace(0, H - 1, H, dtype=dtype, device=device)
    )  # i, j: (W, H)
    pixels = torch.stack([i, j], dim=-1).view(-1, 2).unsqueeze(0)  # (1, WH, 2)

    if center_pixel:
        pixels += 0.5

    # reorder if full rays
    if not wh_order:  # (HW, 2)
        pixels = pixels.squeeze(0).contiguous().view(W, H, 2).permute(1, 0, 2).contiguous().view(-1, 2).unsqueeze(0)

    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype, device=device)  # (1, WH/N_rays)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH/N_rays, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH/N_rays, 3)
    rays_d = rays_d[0]  # (WH/N_rays, 3)
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH/N_rays, 3)

    # normalize rays for non_ndc case
    rays_d = normalize(rays_d)  # (WH/N_rays, 3)

    return rays_o, rays_d

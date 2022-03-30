# -*- coding: utf-8 -*-

import torch

from .transformation import rotate_points


def pixel_to_cam(pixels, z, intrinsic):
    """Pixel to cam space xyz

    Args:
        pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
        z: depth. torch.tensor(B, N)
        intrinsic: intrinsic, torch.tensor(B, 3, 3)

    Returns:
        xyz_cam: torch.tensor(B, N, 3), pixel lift to cam coord position
    """
    fx = intrinsic[..., 0, 0].unsqueeze(-1)
    fy = intrinsic[..., 1, 1].unsqueeze(-1)
    cx = intrinsic[..., 0, 2].unsqueeze(-1)
    cy = intrinsic[..., 1, 2].unsqueeze(-1)
    s = intrinsic[..., 0, 1].unsqueeze(-1)  # all in (B, 1) shape

    i = pixels[:, :, 0]
    j = pixels[:, :, 1]

    x_cam = (i - (s * (j - cy) / fy) - cx) / fx * z  # (B, N)
    y_cam = (j - cy) / fy * z  # (B, N)

    xyz_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (B, N, 3)

    return xyz_cam


def cam_to_world(points, c2w):
    """points in camera to world coord

    Args:
        points: points in cam_coord, torch.tensor(B, N, 3)
        c2w: cam_to_world transformation, torch.tensor(B, 4, 4)

    Returns:
        xyz_world: torch.tensor(B, N, 3), pixel lift to world coord position
    """
    xyz_world = rotate_points(points, c2w)

    return xyz_world


def pixel_to_world(pixels, z, intrinsic, c2w):
    """pixel to world coord

    Args:
        pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
        z: depth. torch.tensor(B, N)
        intrinsic: intrinsic, torch.tensor(B, 3, 3)
        c2w: cam_to_world transformation, torch.tensor(B, 4, 4)
    Returns:
         xyz_world: torch.tensor(B, N, 3), pixel lift to world coord position
    """
    xyz_cam = pixel_to_cam(pixels, z, intrinsic)
    xyz_world = cam_to_world(xyz_cam, c2w)

    return xyz_world


def world_to_cam(points, w2c):
    """points in world to cam coord

    Args:
        points: points in world_coord, torch.tensor(B, N, 3)
        w2c: world_to_cam transformation, torch.tensor(B, 4, 4)

    Returns:
         xyz_cam: torch.tensor(B, N, 3), xyz in world coord
    """
    xyz_cam = rotate_points(points, w2c)

    return xyz_cam


def cam_to_pixel(points, intrinsic):
    """Points in cam coord to pixels locations

    Args:
        points: points in cam_coord, torch.tensor(B, N, 3)
        intrinsic: intrinsic, torch.tensor(B, 3, 3)

    Returns:
        pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    """
    pixels = torch.einsum('bki,bji->bjk', intrinsic, points)
    pixels = torch.div(pixels[:, :, :2], pixels[:, :, 2].unsqueeze(dim=-1) + 1e-8)

    return pixels[:, :, :2]


def world_to_pixel(points, intrinsic, w2c, distort=None):
    """points in world to cam coord. Distortion is allowed to be adjusted in cam space.

    Args:
        points: points in world_coord, torch.tensor(B, N, 3)
        intrinsic: intrinsic, torch.tensor(B, 3, 3)
        w2c: world_to_cam transformation, torch.tensor(B, 4, 4)
        distort: a tuple of distortion (radial(B, 3), tan(B, 2))
    Returns:
        pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    """
    xyz_cam = world_to_cam(points, w2c)

    if distort:
        xyz_cam = apply_distortion(xyz_cam, distort[0], distort[1])

    pixels = cam_to_pixel(xyz_cam, intrinsic)

    return pixels


def apply_distortion(points, radial, tan):
    """Apply distortion to points in cam space.

    Args:
        points - (B, N, 3)
        radial - (B, 3)
        tan - (B, 2)
    """
    norm = torch.clamp(points[..., :2] / points[..., 2:], min=-1, max=1)
    r2 = torch.sum(norm[..., :2]**2, dim=-1, keepdim=True)

    radial_dist = 1 + torch.sum(radial * torch.cat((r2, r2**2, r2**3), dim=-1), dim=-1, keepdim=True)
    tan_norm = torch.sum(tan * norm, dim=-1, keepdim=True)

    points_dist = points.clone()
    points_dist[..., :2] += points_dist[..., 2:] * (tan * r2) / (radial_dist + tan_norm)
    points_dist[..., 2:] /= (radial_dist + tan_norm)

    return points_dist

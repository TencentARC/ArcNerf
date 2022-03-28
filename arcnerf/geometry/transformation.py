# -*- coding: utf-8 -*-

import numpy as np
import torch


def invert_pose(poses):
    """ Change poses between c2w and w2c. Support numpy and torch
    :params: poses: (N, 4, 4)
    :return: poses: (N, 4, 4)
    """
    if isinstance(poses, torch.FloatTensor):
        return torch.inverse(poses)
    elif isinstance(poses, np.ndarray):
        return np.linalg.inv(poses)


def normalize(vec):
    """Normalize vector. Support numpy and torch
    :params: vec: (B, N, 3)
    :return: vec: (B, N, 3)
    """
    if isinstance(vec, torch.FloatTensor):
        vec = vec / torch.norm(vec, dim=-1).unsqueeze(-1)
    elif isinstance(vec, np.ndarray):
        vec = vec / np.linalg.norm(vec, axis=-1)[..., None]

    return vec


def pixel_to_cam(pixels, z, intrinsic):
    """Pixel to cam space xyz
    :param: pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    :param: z: depth. torch.tensor(B, N)
    :param: intrinsic: intrinsic, torch.tensor(B, 3, 3)
    :return: xyz_cam: torch.tensor(B, N, 3), pixel lift to cam coord position
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
    :param: points: points in cam_coord, torch.tensor(B, N, 3)
    :param: c2w: cam_to_world transformation, torch.tensor(B, 4, 4)
    :return: xyz_world: torch.tensor(B, N, 3), pixel lift to world coord position
    """
    xyz_world = torch.einsum('bki, bji->bjk', c2w[:, :3, :3], points)

    return xyz_world


def pixel_to_world(pixels, z, intrinsic, c2w):
    """pixel to world coord
    :param: pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    :param: z: depth. torch.tensor(B, N)
    :param: intrinsic: intrinsic, torch.tensor(B, 3, 3)
    :param: c2w: cam_to_world transformation, torch.tensor(B, 4, 4)
    :return: xyz_world: torch.tensor(B, N, 3), pixel lift to world coord position
    """
    xyz_cam = pixel_to_cam(pixels, z, intrinsic)
    xyz_world = cam_to_world(xyz_cam, c2w)

    return xyz_world


def world_to_cam(points, w2c):
    """points in world to cam coord
    :param: points: points in world_coord, torch.tensor(B, N, 3)
    :param: w2c: world_to_cam transformation, torch.tensor(B, 4, 4)
    :return: xyz_cam: torch.tensor(B, N, 3), xyz in world coord
    """
    xyz_cam = torch.einsum('bki, bji->bjk', w2c[:, :3, :3], points)

    return xyz_cam


def cam_to_pixel(points, intrinsic):
    """Points in cam coord to pixels locations
    :param: points: points in cam_coord, torch.tensor(B, N, 3)
    :param: intrinsic: intrinsic, torch.tensor(B, 3, 3)
    :return: pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    """
    pixels = torch.einsum('bki,bji->bjk', intrinsic, points)
    pixels = torch.div(pixels[:, :, :2], pixels[:, :, 2].unsqueeze(dim=-1))

    return pixels[:, :, :2]


def world_to_pixel(points, intrinsic, w2c, distort=None):
    """points in world to cam coord. Distortion is allowed to be adjusted in cam space.
    :param: points: points in world_coord, torch.tensor(B, N, 3)
    :param: intrinsic: intrinsic, torch.tensor(B, 3, 3)
    :param: w2c: world_to_cam transformation, torch.tensor(B, 4, 4)
    :param: distort: a tuple of distortion (radial(B, 3), tan(B, 2))
    :return: pixels: index in x(horizontal)/y(vertical), torch.tensor (B, N, 2)
    """
    xyz_cam = world_to_cam(points, w2c)
    if distort:
        xyz_cam = apply_distortion(xyz_cam, distort[0], distort[1])
    pixels = cam_to_pixel(xyz_cam, intrinsic)

    return pixels


def apply_distortion(points, radial, tan):
    """Apply distortion to points in cam space.
    :param: points - (B, N, 3)
    :param: radial - (B, 3)
    :param: tan - (B, 2)
    """
    norm = torch.clamp(points[..., :2] / points[..., 2:], min=-1, max=1)
    r2 = torch.sum(norm[..., :2]**2, dim=-1, keepdim=True)

    radial_dist = 1 + torch.sum(radial * torch.cat((r2, r2**2, r2**3), dim=-1), dim=-1, keepdim=True)
    tan_norm = torch.sum(tan * norm, dim=-1, keepdim=True)

    points_dist = points.clone()
    points_dist[..., :2] += points_dist[..., 2:] * (tan * r2) / (radial_dist + tan_norm)
    points_dist[..., 2:] /= (radial_dist + tan_norm)

    return points_dist


def rotate_matrix(rot, source):
    """Rotate a matrix by a rot
    :param: source: the origin matrix in (B, i, j)
    :param: rot: the applied transformation in (B, k, j)
    :return: rotated matrix in (B, k, j)
    """
    rot_mat = torch.einsum('bki,bij->bkj', rot, source)

    return rot_mat


def np_wrapper(func, *args):
    """ Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    Reference from VideoPose3d: https://github.com/facebookresearch/VideoPose3D/blob/master/common/utils.py
    """
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        return result.numpy()
    else:
        return result

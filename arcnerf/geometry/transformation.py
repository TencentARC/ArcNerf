# -*- coding: utf-8 -*-

import numpy as np
import torch

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
)


def normalize(vec):
    """Normalize vector. Support numpy and torch

    Args:
        vec: (B, N, 3) or (B, 3)

    Returns:
        vec: (B, N, 3) or (B, 3)
    """
    if isinstance(vec, torch.FloatTensor):
        vec = vec / (torch.norm(vec, dim=-1).unsqueeze(-1) + 1e-8)
    elif isinstance(vec, np.ndarray):
        vec = vec / (np.linalg.norm(vec, axis=-1)[..., None] + 1e-8)

    return vec


def batch_dot_product(a: torch.Tensor, b: torch.Tensor):
    """Dot product in batch

    Args:
        a: (B, v)
        b: (B, v)

    Returns:
        dot_prod: (B,)
    """
    assert len(a.shape) == 2 and a.shape == b.shape
    dot_prod = torch.bmm(a.unsqueeze(1), b.unsqueeze(-1))[:, 0, 0]

    return dot_prod


def rotate_points(points: torch.Tensor, rot: torch.Tensor):
    """Rotate points by a rot

    Args:
        points: points, torch.tensor(B, N, 3)
        rot: rot matrix, torch.tensor(B, 4, 4)

    Returns:
        rotated points in (B, N, 3)
    """
    # convert points to home
    homo_coord = torch.ones(list(points.shape)[:-1] + [1], dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, homo_coord], dim=-1)

    proj_points = torch.einsum('bki,bji->bjk', rot, points_h)  # (B, N, 4)
    proj_points = torch.div(proj_points[..., :3], proj_points[..., 3].unsqueeze(-1) + 1e-8)

    return proj_points[..., :3]


def rotate_matrix(rot: torch.Tensor, source: torch.Tensor):
    """Rotate a matrix by a rot

    Args:
        source: the origin matrix in (B, i, j)
        rot: the applied transformation in (B, k, j)

    Returns:
        rotated matrix in (B, k, j)
    """
    rot_mat = torch.einsum('bki,bij->bkj', rot, source)

    return rot_mat


def axis_angle_to_rot_6d(axis_angle):
    """Turn axis_angle representation to rot 6d representation

    Args:
        axis_angle: axis angle representation in (B, 3) or (B, n, 3) shape

    Returns:
        rot_6d: rot 6d representation in (B, 6) or (B, n, 6) shape
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rot_6d_to_axis_angle(rot_6d):
    """Turn rot_6d representation to axis_angle_representation

    Args:
        rot_6d: rot 6d representation in (B, 6) or (B, n, 6) shape

    Returns:
        axis_angle: axis angle representation in (B, 3) or (B, n, 3) shape
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot_6d))


def qinverse(q, inplace=False):
    """Inverse quaternion"""
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.

    Args:
        q: quaternion (N, 4)
        v: vector (N, 3)

    Returns:
         a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)

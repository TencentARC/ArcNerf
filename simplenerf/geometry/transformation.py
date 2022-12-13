# -*- coding: utf-8 -*-

import numpy as np
import torch


def normalize(vec):
    """Normalize vector. Support numpy and torch

    Args:
        vec: (B, N, 3) or (B, 3)

    Returns:
        vec: (B, N, 3) or (B, 3)
    """
    if isinstance(vec, torch.Tensor):
        vec = vec / (torch.norm(vec, dim=-1).unsqueeze(-1) + 1e-8)
    elif isinstance(vec, np.ndarray):
        vec = vec / (np.linalg.norm(vec, axis=-1)[..., None] + 1e-8)

    return vec


def rotate_points(points: torch.Tensor, rot: torch.Tensor, rotate_only=False):
    """Rotate points by a rot

    Args:
        points: points, torch.tensor(B, N, 3)
        rot: rot matrix, torch.tensor(B, 4, 4). If rotate_only, can be (B, 3, 3)
        rotate_only: If True, only do the rotation using rot (B, 3, 3)

    Returns:
        rotated points in (B, N, 3)
    """
    proj_points = torch.einsum('bki,bji->bjk', rot[:, :3, :3], points)
    if not rotate_only:
        proj_points += rot[:, :3, 3]

    return proj_points

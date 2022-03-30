# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import pixel_to_world, world_to_pixel
from arcnerf.geometry.transformation import normalize


class PerspectiveCamera(object):
    """A camera with intrinsic and c2w pose"""

    def __init__(self, intrinsic, c2w, H=None, W=None, dtype=torch.float32):
        """
        Args:
            Intrinsic: (3, 3) numpy array
            cw2: (4, 4) numpy array
            H, W: height/width
            dtype: torch tensor type. controls for outputs like intrinsic/pose/ray_bundle.
                    By default is torch.float32
        """
        self.dtype = dtype
        self.intrinsic = intrinsic.copy()
        self.c2w = c2w.copy()
        self.W = W if W is not None else int(c2w[0, 2] / 2.0)
        self.H = H if H is not None else int(c2w[1, 2] / 2.0)

    def rescale(self, scale):
        """Scale intrinsic and hw. scale < 1.0 means scale_down"""
        self.intrinsic[0, 0] *= scale
        self.intrinsic[1, 1] *= scale
        self.intrinsic[0, 2] *= scale
        self.intrinsic[1, 2] *= scale
        self.intrinsic[0, 1] *= scale
        self.H = int(self.H * scale)
        self.W = int(self.W * scale)

    def get_cam_pose_norm(self):
        """Get the camera distance from origin in world coord"""
        pose_norm = np.linalg.norm(self.c2w[:3, 3])

        return pose_norm

    def get_wh(self):
        """Get camera width and height"""
        return self.W, self.H

    def rescale_pose(self, scale):
        """scale the pose."""
        self.c2w[:3, 3] *= scale

    def get_intrinsic(self, torch_tensor=True):
        """Get intrinsic. return numpy array by default"""
        if torch_tensor:
            return torch.tensor(self.intrinsic, dtype=self.dtype)
        else:
            return self.intrinsic

    def get_pose(self, torch_tensor=True, w2c=False):
        """Get pose, return numpy array by default. Support w2c transformation"""
        pose = self.c2w.copy()
        if w2c:
            pose = invert_poses(pose)
        if torch_tensor:
            pose = torch.tensor(pose, dtype=self.dtype)

        return pose

    def get_rays(self, index=None):
        """Get camera rays by intrinsic and c2w, in world coord"""
        return get_rays(self.H, self.W, self.get_intrinsic(), self.get_pose(), index=index)

    def proj_world_to_pixel(self, points):
        """Project points onto image plane.

        Args:
            points: pts in world coord, torch.Tensor(N, 3)

        Returns:
            pixels: pixel loc, torch.Tensor(N, 2)
        """
        pixel = world_to_pixel(
            points.unsqueeze(0),
            self.get_intrinsic().unsqueeze(0),
            self.get_pose(w2c=True).unsqueeze(0)
        )

        return pixel[0]


def load_K_Rt_from_P(proj_mat):
    """ Get intrinsic and extrinsic Rt from proj_matrix
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(proj_mat)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_rays(H, W, intrinsic, c2w, index=None):
    """Get rays in world coord. No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        H: img_height
        W: img_width
        intrinsic: (3, 3) intrinsic matrix
        c2w: (4, 4) cam pose. cam_to_world transform

    Returns:
        a ray_bundle with rays_o and rays_d. Each is in dim (N_ray, 3).
            If no sampler is used, return (WH, 3) num of rays
    """
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype), torch.linspace(0, H - 1, H, dtype=dtype)
    )  # i, j: (W, H)

    pixels = torch.stack([i, j], dim=-1).reshape(-1, 2).unsqueeze(0)  # (1, WH, 2)
    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype).to(device)  # (1, WH)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH, 3)
    rays_d = normalize(rays_d)[0]  # (WH, 3)
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH, 3)

    return rays_o, rays_d

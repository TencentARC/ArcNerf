# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

from arcnerf.geometry import torch_to_np
from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import pixel_to_world, world_to_pixel, world_to_cam
from arcnerf.geometry.transformation import normalize


class PerspectiveCamera(object):
    """A camera with intrinsic and c2w pose. All calculation is on cpu."""

    def __init__(self, intrinsic: np.ndarray, c2w: np.ndarray, W=None, H=None, dtype=torch.float32):
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
        self.W = int(self.W * scale)
        self.H = int(self.H * scale)

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

    def reset_pose(self, c2w):
        """reset the c2w (4, 4)"""
        self.c2w = c2w.copy()

    def reset_intrinsic(self, intrinsic):
        """reset_pose the intrinsic (3, 3)"""
        self.intrinsic = intrinsic.copy()

    def get_pose(self, torch_tensor=True, w2c=False):
        """Get pose, return numpy array by default. Support w2c transformation"""
        pose = self.c2w.copy()
        if w2c:
            pose = invert_poses(pose)
        if torch_tensor:
            pose = torch.tensor(pose, dtype=self.dtype)

        return pose

    def get_rays(self, index: np.ndarray = None, N_rays=-1, to_np=False):
        """Get camera rays by intrinsic and c2w, in world coord"""
        return get_rays(self.W, self.H, self.get_intrinsic(), self.get_pose(), index=index, N_rays=N_rays, to_np=to_np)

    def proj_world_to_pixel(self, points: torch.Tensor):
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

    def proj_world_to_cam(self, points: torch.Tensor):
        """Project points onto cam space. Help to find near/far bounds

        Args:
            points: pts in world coord, torch.Tensor(N, 3)

        Returns:
            pts_cam: pts in cam space, torch.Tensor(N, 3)
        """
        pixel = world_to_cam(points.unsqueeze(0), self.get_pose(w2c=True).unsqueeze(0))

        return pixel[0]


def load_K_Rt_from_P(proj_mat: np.ndarray):
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

    pose = np.eye(4, dtype=proj_mat.dtype)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_rays(W, H, intrinsic: torch.Tensor, c2w: torch.Tensor, index: np.ndarray = None, N_rays=-1, to_np=False):
    """Get rays in world coord from camera.
    No batch processing allow. Rays are produced by setting z=1 and get location.
    You can select index by a tuple, a list of tuple or a list of index

    Args:
        W: img_width
        H: img_height
        intrinsic: torch.tensor(3, 3) intrinsic matrix
        c2w: torch.tensor(4, 4) cam pose. cam_to_world transform
        index: sample ray by (i, j) index from (W, H), np.array/torch.tensor(N_ind, 2) for (i, j) index
                first index is X and second is Y, any index should be in range (0, W-1) and (0, H-1)
        N_rays: random sample ray by such num if it > 0
        to_np: if to np, return np array instead of torch.tensor

    Returns:
        a ray_bundle with rays_o and rays_d. Each is in dim (N_ray, 3).
             If no sampler is used, return (WH, 3) num of rays
        ind_unroll: sample index in list of (N_ind, ) for index in (WH, ) range
    """
    assert (index is None) or N_rays <= 0, 'You are not allowed to sampled both by index and N_ray'
    device = intrinsic.device
    dtype = intrinsic.dtype
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=dtype), torch.linspace(0, H - 1, H, dtype=dtype)
    )  # i, j: (W, H)
    pixels = torch.stack([i, j], dim=-1).view(-1, 2).unsqueeze(0).to(device)  # (1, WH, 2)

    # index unroll
    if index is not None:
        assert len(index.shape) == 2 and index.shape[-1] == 2, 'invalid shape, should be (N_rays, 2)'
        if isinstance(index, np.ndarray):
            index = torch.tensor(index, dtype=torch.long).to(device)
        else:
            index = index.type(torch.long).to(device)
        index = index[:, 0] * H + index[:, 1]  # (N_rays, ) unroll from (i, j)
    # sample by N_rays
    if N_rays > 0:
        index = np.random.choice(range(0, W * H), N_rays, replace=False)  # (N_rays, )
        index = torch.tensor(index, dtype=torch.long).to(device)
    # sampled by index
    if index is not None:
        pixels = pixels[:, index, :]
        index = torch_to_np(index).tolist()

    z = torch.ones(size=(1, pixels.shape[1]), dtype=dtype).to(device)  # (1, WH/N_rays)
    xyz_world = pixel_to_world(pixels, z, intrinsic.unsqueeze(0), c2w.unsqueeze(0))  # (1, WH/N_rays, 3)

    cam_loc = c2w[:3, 3].unsqueeze(0)  # (1, 3)
    rays_d = xyz_world - cam_loc.unsqueeze(0)  # (1, WH/N_rays, 3)
    rays_d = normalize(rays_d)[0]  # (WH/N_rays, 3)
    rays_o = torch.repeat_interleave(cam_loc, rays_d.shape[0], dim=0)  # (WH/N_rays, 3)

    if to_np:
        rays_o = torch_to_np(rays_o)
        rays_d = torch_to_np(rays_d)

    return rays_o, rays_d, index


def equal_sample(n_rays_w, n_rays_h, W, H):
    """Eqaul sample i,j index on img with (W, H)

    Args:
        n_rays_w: num of samples on each row (x direction)
        n_rays_h: num of samples on each col (y direction)
        W: image width
        H: image height

    Returns:
        index: np.array(n_rays_w*n_rays_h, 2) equally sampled grid
    """

    i, j = np.meshgrid(np.linspace(0, W - 1, n_rays_w), np.linspace(0, H - 1, n_rays_h))
    index = np.stack([i, j], axis=-1).reshape(-1, 2)

    return index

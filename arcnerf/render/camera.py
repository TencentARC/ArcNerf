# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

from arcnerf.geometry.poses import invert_poses
from arcnerf.geometry.projection import world_to_pixel, world_to_cam
from .ray_helper import get_rays


class PerspectiveCamera(object):
    """A camera with intrinsic and c2w pose. All calculation is on cpu."""

    def __init__(self, intrinsic: np.ndarray, c2w: np.ndarray, W=None, H=None, dtype=torch.float32, device='cpu'):
        """
        Args:
            Intrinsic: (3, 3) numpy array
            cw2: (4, 4) numpy array
            H, W: height/width
            dtype: torch tensor type. controls for outputs like intrinsic/pose/ray_bundle.
                    By default is torch.float32
            device: If device is 'gpu', will return the tensor(intrinsic/c2w/rays) one gpu
        """
        self.dtype = dtype
        self.intrinsic = intrinsic.copy()
        self.c2w = c2w.copy()
        self.W = W if W is not None else int(c2w[0, 2] / 2.0)
        self.H = H if H is not None else int(c2w[1, 2] / 2.0)
        self.device = device

    def set_device(self, device):
        """Manually set the device"""
        self.device = device

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
            intrinsic = torch.tensor(self.intrinsic, dtype=self.dtype)
            if self.device == 'gpu':
                intrinsic = intrinsic.cuda()
            return intrinsic
        else:
            return self.intrinsic

    def exchange_coord(self, src, dst, flip):
        """exchange any two of xyz coord, negative is allow in dst
        eg: x->y or x->-y

        Args:
            src: any of x, y, z
            dst: any of x, y, z
            flip: any of xyz or nan
        """
        assert src in ['x', 'y', 'z'], 'Only xyz allow'
        assert dst in ['x', 'y', 'z'], 'Only xyz allow'

        _flip = (flip != 'nan')
        if _flip:
            assert flip in ['x', 'y', 'z'], 'Only xyz allow'

        mapping = {'x': 0, 'y': 1, 'z': 2}
        new_order = [0, 1, 2, 3]
        new_order[mapping[src]], new_order[mapping[dst]] = new_order[mapping[dst]], new_order[mapping[src]]
        # change order
        self.c2w = self.c2w[new_order, :]
        # change sign
        if _flip:
            self.c2w[mapping[flip], :] *= -1

    def reset_pose(self, c2w):
        """reset the c2w (4, 4)"""
        self.c2w = c2w.copy()

    def reset_intrinsic(self, intrinsic):
        """reset_pose the intrinsic (3, 3)"""
        self.intrinsic = intrinsic.copy()

    def apply_transform(self, rot):
        """Rotate a pose by rot (4, 4)"""
        self.c2w = np.matmul(rot, self.c2w)

    def get_pose(self, torch_tensor=True, w2c=False):
        """Get pose, return numpy array by default. Support w2c transformation"""
        pose = self.c2w.copy()
        if w2c:
            pose = invert_poses(pose)
        if torch_tensor:
            pose = torch.tensor(pose, dtype=self.dtype)
            if self.device == 'gpu':
                pose = pose.cuda()

        return pose

    def get_rays(
        self,
        wh_order=True,
        index: np.ndarray = None,
        n_rays=-1,
        to_np=False,
        ndc=False,
        center_pixel=False,
        normalize_rays_d=True
    ):
        """Get camera rays by intrinsic and c2w, in world coord"""
        return get_rays(
            self.W,
            self.H,
            self.get_intrinsic(),
            self.get_pose(),
            wh_order,
            index,
            n_rays,
            to_np,
            ndc,
            center_pixel=center_pixel,
            normalize_rays_d=normalize_rays_d
        )

    def proj_world_to_pixel(self, points: torch.Tensor):
        """Project points onto image plane.

        Args:
            points: pts in world coord, torch.Tensor(N, 3)

        Returns:
            pixels: pixel loc, torch.Tensor(N, 2)
        """
        if self.device == 'gpu':
            assert points.is_cuda, 'You must have the points on device {}'.format(self.device)

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
        if self.device == 'gpu':
            assert points.is_cuda, 'You must have the points on device {}'.format(self.device)

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

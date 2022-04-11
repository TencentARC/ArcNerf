# -*- coding: utf-8 -*-

import numpy as np
import torch

from arcnerf.geometry import np_wrapper
from arcnerf.geometry.poses import center_poses
from arcnerf.geometry.ray import closest_point_to_rays
from common.datasets.base_dataset import BaseDataset
from common.utils.cfgs_utils import valid_key_in_cfgs
from common.utils.img_utils import img_scale, read_img


class Base3dDataset(BaseDataset):
    """Base 3d dataset with images and cameras, etc"""

    def __init__(self, cfgs, data_dir, mode, transforms):
        """For any of 3d dataset, images/intrinsics/c2w are required. mask is optional"""
        super(Base3dDataset, self).__init__(cfgs, data_dir, mode, transforms)
        self.images = []
        self.n_imgs = 0
        self.cam_file = None
        self.cameras = []
        self.H, self.W = 0, 0
        self.precache = False
        self.ray_bundles = None
        self.identifier = ''
        # below are optional
        self.masks = []
        self.point_cloud = None

    def get_identifier(self):
        """string identifier of a dataset like scan_id/scene_name"""
        return self.identifier

    def rescale_img_and_pose(self):
        """Rescale image/mask and pose if needed. It affects intrinsic only. """
        if valid_key_in_cfgs(self.cfgs, 'img_scale'):
            scale = self.cfgs.img_scale
            if scale != 1:
                for i in range(len(self.images)):
                    self.images[i] = img_scale(self.images[i], scale)
                for i in range(len(self.masks)):
                    self.masks[i] = img_scale(self.masks[i], scale)
                self.H, self.W = self.images[0].shape[:2]

                for camera in self.cameras:
                    camera.rescale(scale)

    @staticmethod
    def read_image_list(img_list):
        """Read image from list."""
        images = [read_img(path, norm_by_255=True) for path in img_list]

        return images

    @staticmethod
    def read_mask_list(mask_list):
        """Read mask from list. can be emtpy list if not file needed"""
        masks = [read_img(path, norm_by_255=True, gray=True) for path in mask_list]

        return masks

    def read_cameras(self):
        """Return a list of render.camera with c2w and intrinsic"""
        raise NotImplementedError('Please implement the detail function in child class....')

    def get_poses(self, torch_tensor=True, w2c=False, concat=False):
        """Get the a list of poses of all cameras. If concat, get (n_cam, 4, 4)"""
        extrinsic = []
        for cam in self.cameras:
            extrinsic.append(cam.get_pose(torch_tensor=torch_tensor, w2c=w2c))
        if concat:
            extrinsic = [ext[None] for ext in extrinsic]
            if torch_tensor:
                extrinsic = torch.cat(extrinsic, dim=0)
            else:
                extrinsic = np.concatenate(extrinsic, axis=0)

        return extrinsic

    def norm_cam_pose(self):
        """Normalize camera pose by scale_radius, place camera near a sphere surface. It affects extrinsic"""
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
        if hasattr(self.cfgs, 'scale_radius') and self.cfgs.scale_radius > 0:
            cam_norm_t = []
            for camera in self.cameras:
                cam_norm_t.append(camera.get_cam_pose_norm())
            max_cam_norm_t = max(cam_norm_t)

            for camera in self.cameras:
                camera.rescale_pose(scale=self.cfgs.scale_radius / (max_cam_norm_t * 1.1))

    def center_cam_poses_by_view_dir(self):
        """Recenter camera pose by setting the common view point center at (0,0,0)
        The common view point is the closest point to all rays.
        """
        c2ws = self.get_poses(torch_tensor=False, concat=True)
        # use ray from image center to represent cam view dir
        center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
        rays_o = []
        rays_d = []
        for idx in range(len(self.cameras)):
            ray = self.cameras[idx].get_rays(index=center_idx, to_np=True)
            rays_o.append(ray[0])
            rays_d.append(ray[1])
        rays = (np.concatenate(rays_o, axis=0), np.concatenate(rays_d, axis=0))
        # calculate mean view point
        view_point_mean, _, _ = np_wrapper(closest_point_to_rays, rays[0], rays[1])  # (1, 3)
        center_c2w = center_poses(c2ws, view_point_mean[0])
        for idx in range(len(self.cameras)):
            self.cameras[idx].reset_pose(center_c2w[idx])

    def precache_ray(self):
        """Precache all the rays for all images first"""
        if self.ray_bundles is None:
            self.ray_bundles = []
            for i in range(self.n_imgs):
                self.ray_bundles.append(self.cameras[i].get_rays())

    def __len__(self):
        """Len of dataset"""
        return self.n_imgs

    def __getitem__(self, idx):
        """Get the image, mask and rays"""
        img = self.images[idx].reshape(-1, 3)
        mask = self.masks[idx].reshape(-1) if len(self.masks) > 0 else None  # (hw)
        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask) if mask is not None else None
        c2w = self.cameras[idx].get_pose()
        intrinsic = self.cameras[idx].get_intrinsic()

        if self.precache:
            ray_bundle = self.ray_bundles[idx]
        else:
            ray_bundle = self.cameras[idx].get_rays()  # We don't sample rays here, although you can do that

        inputs = {
            'img': img,  # (hw, 3), in rgb order / (n_rays, 3) if sample rays
            'mask': mask,  # (hw,) / (n_rays,) if sample rays
            'c2w': c2w,  # (4, 4)
            'intrinsic': intrinsic,  # (3, 3)
            'rays_o': ray_bundle[0],  # (hw, 3) / (n_rays, 3) if sample rays
            'rays_d': ray_bundle[1],  # (hw, 3) / (n_rays, 3) if sample rays
            'H': self.H,
            'W': self.W,
            'pc': self.point_cloud,  # a dict contains['pts', 'color', 'vis']. Same for all cam
        }

        pop_k = []
        for k, v in inputs.items():
            if v is None:  # in case can not collate
                pop_k.append(k)
        for k in pop_k:
            inputs.pop(k)

        if self.transforms is not None:
            inputs = self.transforms(inputs)

        return inputs

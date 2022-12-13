# -*- coding: utf-8 -*-

import numpy as np
import torch

from simplenerf.geometry.poses import average_poses
from common.datasets.base_dataset import BaseDataset
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field, pop_none_item
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
        self.bounds = []
        # set for skip
        self.skip = get_value_from_cfgs_field(cfgs, 'skip', 1)
        # set for eval
        self.eval_max_sample = get_value_from_cfgs_field(cfgs, 'eval_max_sample')

        # whether in ndc space
        self.ndc_space = get_value_from_cfgs_field(cfgs, 'ndc_space', False)

        # device
        self.device = get_value_from_cfgs_field(cfgs, 'device', 'cpu')

    def get_identifier(self):
        """string identifier of a dataset like scan_id/scene_name"""
        return self.identifier

    def get_wh(self):
        """Get the image shape"""
        return self.W, self.H

    def set_device(self, device):
        """Set the device for the tensors"""
        self.device = device
        for cam in self.cameras:
            cam.set_device(self.device)

    def skip_samples(self):
        """For any mode, you can skip the samples in order."""
        if self.skip > 1:
            self.images = self.images[::self.skip]
            self.cameras = self.cameras[::self.skip]
            self.masks = self.masks[::self.skip]
            self.bounds = self.bounds[::self.skip]
            self.n_imgs = len(self.images)

    def keep_eval_samples(self):
        """For eval model, only keep a small number of samples. Which are closer to the avg pose
         It should be done before precache_rays in child class to avoid full precache.
         """
        ind = None
        if self.eval_max_sample is not None:
            n_imgs = min(self.eval_max_sample, self.n_imgs)
            self.n_imgs = n_imgs
            ind = self.find_closest_cam_ind(n_imgs)
            self.images = [self.images[i] for i in ind]
            self.cameras = [self.cameras[i] for i in ind]
            self.masks = [self.masks[i] for i in ind] if len(self.masks) > 0 else []
            self.bounds = [self.bounds[i] for i in ind] if len(self.bounds) > 0 else []

        return ind

    def find_closest_cam_ind(self, n_close):
        """Find the closest cam ind to the avg pose, return a list of index"""
        c2ws = self.get_poses(torch_tensor=False, concat=True)  # (N_total_cam, 4, 4)
        if n_close >= c2ws.shape[0]:
            return range(c2ws.shape[0])
        avg_pose = average_poses(c2ws)[None, :]  # (1, 4, 4)
        dist = np.linalg.norm(c2ws[:, :3, 3] - avg_pose[:, :3, 3], axis=-1)  # (N_total_cam)
        min_dist = np.argsort(dist)  # (N_total_cam)
        ind = min_dist[:n_close].tolist()  # (N_total_cam)

        return ind

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
        """Read image from list, all image should be in `rgb` order"""
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

    def read_cameras_by_mode(self, mode):
        """Return a list of render.camera with c2w and intrinsic in all split. Return the camera index of split"""
        raise NotImplementedError('Please implement the detail function in child class....')

    def get_intrinsic(self, torch_tensor=True):
        """Get the intrinsic from camera_0, (3, 3)"""
        intrinsic = self.cameras[0].get_intrinsic(torch_tensor)

        return intrinsic

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

    def precache_ray(self):
        """Precache all the rays for all images first"""
        if self.ray_bundles is None:
            self.ray_bundles = []
            for i in range(self.n_imgs):
                self.ray_bundles.append(self.cameras[i].get_rays(wh_order=False, ndc=self.ndc_space))

    def __len__(self):
        """Len of dataset"""
        return self.n_imgs

    def __getitem__(self, idx):
        """Get the image, mask and rays"""
        img = self.images[idx].reshape(-1, 3)  # (hw, 3)
        mask = self.masks[idx].reshape(-1) if len(self.masks) > 0 else None  # (hw)
        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask) if mask is not None else None
        if self.device == 'gpu':
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True) if mask is not None else None

        c2w = self.cameras[idx].get_pose()
        intrinsic = self.cameras[idx].get_intrinsic()
        bounds = None
        if len(self.bounds) > 0:
            bounds = self.bounds[idx]  # (2,)
            bounds = torch.FloatTensor(bounds).unsqueeze(0)
            if self.device == 'gpu':
                bounds = bounds.cuda(non_blocking=True)
            bounds = torch.repeat_interleave(bounds, img.shape[0], dim=0)

        # force the bounds to be 0-1 in ndc_space
        if self.ndc_space:
            bounds = torch.FloatTensor([[0.0, 1.0]])  # (1, 2)
            if self.device == 'gpu':
                bounds = bounds.cuda(non_blocking=True)
            bounds = torch.repeat_interleave(bounds, img.shape[0], dim=0)

        if self.precache:
            ray_bundle = self.ray_bundles[idx]
        else:
            ray_bundle = self.cameras[idx].get_rays(wh_order=False, ndc=self.ndc_space)  # We don't sample rays here

        inputs = {
            'img': img,  # (hw, 3), in rgb order / (n_rays, 3) if sample rays
            'mask': mask,  # (hw,) / (n_rays,) if sample rays
            'c2w': c2w,  # (4, 4)
            'intrinsic': intrinsic,  # (3, 3)
            'rays_o': ray_bundle[0],  # (hw, 3) / (n_rays, 3) if sample rays
            'rays_d': ray_bundle[1],  # (hw, 3) / (n_rays, 3) if sample rays
            'view_dirs': ray_bundle[2],  # (hw, 3) / (n_rays, 3) if sample rays, the real view dir in non-ndc space
            'H': self.H,
            'W': self.W,
            'bounds': bounds  # (hw, 2) for (near, far), if set bounds(generally for pc)
        }

        # in case can not collate, pop none item
        pop_none_item(inputs)

        if self.transforms is not None:
            inputs = self.transforms(inputs)

        return inputs

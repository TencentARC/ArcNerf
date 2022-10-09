# -*- coding: utf-8 -*-

import numpy as np
import torch

from arcnerf.geometry.poses import center_poses, average_poses
from arcnerf.geometry.ray import closest_point_to_rays
from common.datasets.base_dataset import BaseDataset
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field, pop_none_item
from common.utils.img_utils import img_scale, read_img
from common.utils.torch_utils import np_wrapper


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
        self.bounds = []
        # set for skip
        self.skip = get_value_from_cfgs_field(cfgs, 'skip', 1)
        self.test_holdout = get_value_from_cfgs_field(cfgs, 'test_holdout', 8)
        # set for eval
        self.eval_max_sample = get_value_from_cfgs_field(cfgs, 'eval_max_sample')

        # whether in ndc space
        self.ndc_space = get_value_from_cfgs_field(cfgs, 'ndc_space', False)
        # center pixel
        self.center_pixel = get_value_from_cfgs_field(cfgs, 'center_pixel', False)

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

    def get_holdout_index(self):
        """Keep samples by mode and test_holdout. This can split the train/(val/eval).
        To use that, you should have n_imgs representing all the images
        """
        holdout_index = list(range(self.n_imgs))
        if self.test_holdout > 1:
            full_idx = list(range(self.n_imgs))
            skip_idx = full_idx[::self.test_holdout]
            if self.mode == 'train':
                holdout_index = [idx for idx in full_idx if idx not in skip_idx]
            else:
                holdout_index = skip_idx

        return holdout_index

    def get_holdout_samples(self, holdout_index):
        """Get the holdout split for images/camera, etc"""
        self.n_imgs = len(holdout_index)
        self.images = [self.images[idx] for idx in holdout_index]
        self.masks = [self.masks[i] for i in holdout_index] if len(self.masks) > 0 else []
        self.cameras = [self.cameras[idx] for idx in holdout_index]
        self.bounds = [self.bounds[i] for i in holdout_index] if len(self.bounds) > 0 else []

    def get_holdout_samples_with_list(self, holdout_index, img_list, mask_list=None):
        """Get the holdout split for images/camera, etc, img are not read but given list"""
        self.n_imgs = len(holdout_index)
        self.cameras = [self.cameras[idx] for idx in holdout_index]
        self.bounds = [self.bounds[i] for i in holdout_index] if len(self.bounds) > 0 else []
        img_list = [img_list[idx] for idx in holdout_index]
        if mask_list is not None:
            mask_list = [mask_list[idx] for idx in holdout_index]

        return img_list, mask_list

    def skip_samples(self):
        """For any mode, you can skip the samples in order."""
        if self.skip > 1:
            self.images = self.images[::self.skip]
            self.masks = self.masks[::self.skip]
            self.cameras = self.cameras[::self.skip]
            self.bounds = self.bounds[::self.skip]
            self.n_imgs = len(self.images)

    def skip_samples_with_list(self, img_list, mask_list=None):
        """Do not real image at the beginning, skip the img_list/mask_list for further loading."""
        if self.skip > 1:
            self.cameras = self.cameras[::self.skip]
            self.bounds = self.bounds[::self.skip] if len(self.bounds) > 0 else []
            img_list = img_list[::self.skip]
            self.n_imgs = len(img_list)
            if mask_list is not None:
                mask_list = mask_list[::self.skip]

        return img_list, mask_list

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

    def norm_cam_pose(self):
        """Normalize camera pose by scale_radius, place camera near a sphere surface. It affects extrinsic"""
        max_cam_norm_t = None
        assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
        if valid_key_in_cfgs(self.cfgs, 'scale_radius') and self.cfgs.scale_radius > 0:
            cam_norm_t = []
            for camera in self.cameras:
                cam_norm_t.append(camera.get_cam_pose_norm())
            max_cam_norm_t = max(cam_norm_t)

            for camera in self.cameras:
                camera.rescale_pose(scale=self.cfgs.scale_radius / (max_cam_norm_t * 1.05))

            if len(self.bounds) > 0:
                self.bounds = [bound / max_cam_norm_t for bound in self.bounds]

        return max_cam_norm_t

    def center_cam_poses_by_view_dirs(self):
        """Recenter camera pose by setting the common view point center at (0,0,0)
        The common view point is the closest point to all rays.
        """
        view_point_mean = None
        if get_value_from_cfgs_field(self.cfgs, 'center_by_view_dirs', False):
            assert len(self.cameras) > 0, 'Not camera in dataset, do not use this func'
            c2ws = self.get_poses(torch_tensor=False, concat=True)
            # use ray from image center to represent cam view dir
            center_idx = np.array([[int(self.W / 2.0), int(self.H / 2.0)]])
            rays_o = []
            rays_d = []
            for idx in range(len(self.cameras)):
                ray = self.cameras[idx].get_rays(index=center_idx, to_np=True, center_pixel=self.center_pixel)
                rays_o.append(ray[0])
                rays_d.append(ray[1])
            rays = (np.concatenate(rays_o, axis=0), np.concatenate(rays_d, axis=0))
            # calculate mean view point
            view_point_mean, _, _ = np_wrapper(closest_point_to_rays, rays[0], rays[1])  # (1, 3)
            center_c2w = center_poses(c2ws, view_point_mean[0])
            for idx in range(len(self.cameras)):
                self.cameras[idx].reset_pose(center_c2w[idx])

        return view_point_mean

    def align_cam_horizontal(self):
        """Align all camera direction and position to up.
        Use it only when camera are not horizontally around the object
        """
        rot_mat = None
        if valid_key_in_cfgs(self.cfgs, 'align_cam') and self.cfgs.align_cam is True:
            c2ws = self.get_poses(torch_tensor=False, concat=True)
            dtype = c2ws.dtype
            avg_pose = average_poses(c2ws)
            rot_mat = np.eye(4, dtype=dtype)
            rot_mat[:3, :3] = np.linalg.inv(avg_pose)[:3, :3]
            for idx in range(len(self.cameras)):
                self.cameras[idx].apply_transform(rot_mat)

        return rot_mat

    def exchange_coord(self):
        """Exchange any two coord"""
        if get_value_from_cfgs_field(self.cfgs, 'exchange_coord', None) is not None:
            src = get_value_from_cfgs_field(self.cfgs, 'exchange_coord')[0]
            dst = get_value_from_cfgs_field(self.cfgs, 'exchange_coord')[1]
            flip = get_value_from_cfgs_field(self.cfgs, 'exchange_coord')[2]
            for idx in range(len(self.cameras)):
                self.cameras[idx].exchange_coord(src, dst, flip)

    def precache_ray(self):
        """Precache all the rays for all images first"""
        if self.ray_bundles is None:
            self.ray_bundles = []
            for i in range(self.n_imgs):
                self.ray_bundles.append(
                    self.cameras[i].get_rays(wh_order=False, ndc=self.ndc_space, center_pixel=self.center_pixel)
                )

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
        else:  # We don't sample rays here
            ray_bundle = self.cameras[idx].get_rays(wh_order=False, ndc=self.ndc_space, center_pixel=self.center_pixel)

        inputs = {
            'img': img,  # (hw, 3), in rgb order / (n_rays, 3) if sample rays
            'mask': mask,  # (hw,) / (n_rays,) if sample rays
            'c2w': c2w,  # (4, 4)
            'intrinsic': intrinsic,  # (3, 3)
            'rays_o': ray_bundle[0],  # (hw, 3) / (n_rays, 3) if sample rays
            'rays_d': ray_bundle[1],  # (hw, 3) / (n_rays, 3) if sample rays
            'rays_r': ray_bundle[3],  # (hw, 1) / (n_rays, 1) if sample rays
            'H': self.H,
            'W': self.W,
            'pc': self.point_cloud,  # a dict contains['pts', 'color', 'vis']. Same for all cam
            'bounds': bounds  # (hw, 2) for (near, far), if set bounds(generally for pc)
        }

        # in case can not collate, pop none item
        pop_none_item(inputs)

        if self.transforms is not None:
            inputs = self.transforms(inputs)

        return inputs

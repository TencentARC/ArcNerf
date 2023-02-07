# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np
import torch

from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .llff_dataset import LLFF


@DATASET_REGISTRY.register()
class HDRReal(LLFF):
    """HDR Real Dataset. From paper HDR-NeRF. It uses the same processing skill as LLFF dataset
    This dataset do not have a foreground object, only used for view synthesis.
    Different from LLFF, it considers exposure time delta_t as well.
    Ref: https://github.com/shsf0817/hdr-nerf
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(LLFF, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'HDRReal', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image for all case
        img_list, self.n_imgs = self.get_image_list(mode)
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'poses_bounds_exps.npy')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.poses = np.load(self.cam_file, allow_pickle=True)  # (N_imgs, 18)
        self.cameras, self.bounds, self.exp_time = self.read_cameras()
        for cam in self.cameras:
            cam.set_device(self.device)

        # split dataset for train/eval and read
        img_list = self.split_dataset(img_list, mode)
        # skip image and keep less samples
        img_list, _ = self.skip_samples_with_list(img_list)
        # read the real image after skip
        self.images = self.read_image_list(img_list)
        # keep close-to-mean samples if set
        self.keep_eval_samples()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = get_value_from_cfgs_field(self.cfgs, 'precache', False)

        if self.precache:
            self.precache_ray()

    def get_image_list(self, mode=None):
        """Get image list for all that match the pose order. """
        img_dir = osp.join(self.data_spec_dir, 'input_images')
        img_list = sorted(glob.glob(img_dir + '/*.jpg'))
        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def split_dataset(self, img_list, mode='train'):
        """Split the dataset by mode ('train'/'val'/'eval'). We follow the original repo.
        For HDRReal dataset, {1, 3, 5, ..., 35} is for train, {2, 4, 6, ..., 34} is for val/eval.
        One random exposure in {t1, t3, t5} is for train/val. And we set {t2, t4} for eval.
        """
        train_idx, val_idx, eval_idx = [], [], []
        for i in range(self.n_imgs // 10 + 1):  # keep only random {t1, t3, t5} from each {1,3,5...,35} images
            step = i * 10
            train_idx.append(np.random.choice([0 + step, 2 + step, 4 + step], 1, replace=False).item())
        for i in range(self.n_imgs // 10):
            step = i * 10
            # keep only random {t1, t3, t5} from each {2,4,6...,34} images
            val_idx.append(np.random.choice([5 + step, 7 + step, 9 + step], 1, replace=False).item())
            # keep only all {t2, t4} from each {2,4,6...,34} images
            eval_idx.extend([6 + step, 8 + step])

        idx = None
        if mode == 'train':
            idx = train_idx
        elif mode == 'val':
            idx = val_idx
        elif mode == 'eval':
            idx = eval_idx

        # collect from the group
        img_list = [img_list[i] for i in idx]
        self.cameras = [self.cameras[i] for i in idx]
        self.bounds = [self.bounds[i] for i in idx]
        self.exp_time = [self.exp_time[i] for i in idx]
        self.n_imgs = len(img_list)

        return img_list

    def skip_samples_with_list(self, img_list, mask_list=None):
        """Modify it for exp_time key"""
        img_list, _ = super().skip_samples_with_list(img_list)
        if self.skip > 1:
            self.exp_time = self.exp_time[::self.skip]

        return img_list, None

    def keep_eval_samples(self):
        """Modify it for exp_time key"""
        ind = super().keep_eval_samples()
        if ind is not None:
            self.exp_time = [self.exp_time[i] for i in ind]

    def read_cameras(self):
        """Read camera from pose file"""
        poses = self.poses[:, :-3].reshape(-1, 3, 5)  # (N, 3, 5)
        hwf = poses[0, :, -1]  # (3)
        intrinsic = self.get_llff_intrinsic(hwf)
        # Exposure time
        exps = self.poses[:, -1:]  # (N, 1)

        c2w = poses[:, :, :4]  # (N, 3, 4)
        bottom = np.repeat(np.array([0, 0, 0, 1.]).reshape([1, 4])[None, ...], c2w.shape[0], axis=0)  # (N, 1, 4)
        c2w = np.concatenate([c2w, bottom], axis=1)  # (N, 4, 4)
        # correct the pose in our system
        c2w = c2w[:, :, [1, 0, 2, 3]]
        c2w[:, :, 1] *= -1

        # bounds
        bounds = self.poses[:, -3:-1]  # (N, 2)

        # norm by bound. This make the near zvals as 1.0
        factor = 1.0 / (bounds.min() * 0.75)
        c2w[:, :3, 3] *= factor
        bounds *= factor

        # center pose
        c2w = self.center_pose(c2w)

        # adjust the system for get_rays
        c2w[:, :, 1:3] *= -1.0

        cameras = []
        for idx in range(self.n_imgs):
            cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2w[idx], W=self.W, H=self.H))

        return cameras, bounds, exps

    def __getitem__(self, idx):
        """Get the image, mask and rays. For HDRReal, You have one more key exp_time for modeling."""
        inputs = super().__getitem__(idx)

        # load the exp_time key
        exp_time = self.exp_time[idx]  # (1,)
        exp_time = torch.FloatTensor(exp_time).unsqueeze(0)
        if self.device == 'gpu':
            exp_time = exp_time.cuda(non_blocking=True)
        exp_time = exp_time.repeat_interleave(int(self.H * self.W), dim=0)

        inputs['exp_time'] = exp_time  # (hw, 1)

        return inputs

# -*- coding: utf-8 -*-

import glob
import os.path as osp

import cv2
import numpy as np

from .base_3d_dataset import Base3dDataset
from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RTMV(Base3dDataset):
    """RTMV synthetic Dataset.
    Ref: http://www.cs.umd.edu/~mmeshry/projects/rtmv/
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(RTMV, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        split_name, scene_name = self.cfgs.scene_name.split('/')[0], self.cfgs.scene_name.split('/')[1]
        self.data_spec_dir = osp.join(self.data_dir, 'RTMV', split_name, scene_name)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, mask_list, self.n_imgs = self.get_image_list(mode)
        self.images = self.read_image_list(img_list)
        self.masks = self.read_mask_list(mask_list)
        self.H, self.W = self.images[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'Truck_COLMAP_SfM.log')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.cameras = self.read_cameras()

        for cam in self.cameras:
            cam.set_device(self.device)

        # norm camera_pose to restrict pc range
        self.norm_cam_pose()

        # to make fair comparison, remove test file from train
        self.test_holdout = get_value_from_cfgs_field(self.cfgs, 'test_holdout', 8)
        holdout_index = self.get_holdout_index()
        # keep correct sample
        img_list = [img_list[idx] for idx in holdout_index]
        self.cameras = [self.cameras[idx] for idx in holdout_index]
        self.n_imgs = len(holdout_index)

        # skip image and keep less samples
        img_list = self.skip_samples_no_images(img_list)
        # read the real image after skip
        self.images = self.read_image_list(img_list)
        self.keep_eval_samples()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = get_value_from_cfgs_field(self.cfgs, 'precache', False)

        if self.precache:
            self.precache_ray()

    def skip_samples_no_images(self, img_list):
        """do not read all images at first."""
        if self.skip > 1:
            self.cameras = self.cameras[::self.skip]
            img_list = img_list[::self.skip]
            self.n_imgs = len(img_list)

        return img_list

    def get_holdout_index(self):
        """Keep samples by mode and skip"""
        holdout_index = list(range(self.n_imgs))
        if self.test_holdout > 1:
            if self.mode == 'train':
                full_idx = list(range(self.n_imgs))
                skip_idx = full_idx[::self.test_holdout]
                holdout_index = [idx for idx in full_idx if idx not in skip_idx]

            else:
                holdout_index = holdout_index[::self.test_holdout]

        return holdout_index

    def get_image_list(self, mode=None):
        """Get image list."""
        img_list = sorted(glob.glob(self.data_spec_dir + '/*.exr'))
        img_list = [file for file in img_list if 'seg' not in file and 'depth' not in file]
        mask_list = sorted(glob.glob(self.data_spec_dir + '/*.seg.exr'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(self.data_spec_dir)

        return img_list, mask_list, n_imgs

    @staticmethod
    def read_image_list(img_list):
        """Read image from list. Original bkg is black, change it to white"""
        images = []
        for path in img_list:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 255.0
            images.append(img)

        return images

    @staticmethod
    def read_mask_list(mask_list):
        """Read mask from list"""
        masks = []
        for path in mask_list:
            mask = cv2.imread(path)[..., 0].astype(np.float32) / 255.0
            mask = mask / mask.max() * 255.0
            masks.append(mask)

        return masks

    def read_cameras(self):
        """Read camera from pose file"""
        with open(self.cam_file, 'r') as f:
            lines = f.readlines()
        n_cam = int(len(lines) / 5.0)
        assert n_cam == self.n_imgs, 'Num of images not match num of cam...Check it...'

        c2ws = []
        for idx in range(n_cam):
            c2w_lines = lines[idx * 5 + 1:(idx + 1) * 5]
            c2w_lines = [line.strip().split() for line in c2w_lines]
            c2w = np.array(c2w_lines, dtype=np.float32)
            c2ws.append(c2w)

        intrinsic = self.get_est_intrinsic()

        cameras = []
        for idx in range(self.n_imgs):
            cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2ws[idx], W=self.W, H=self.H))

        return cameras

    def get_est_intrinsic(self):
        """Get intrinsic (3, 3) from hwf"""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 0.7 * self.W  # approximate focal
        intrinsic[1, 1] = 0.7 * self.W
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0

        return intrinsic

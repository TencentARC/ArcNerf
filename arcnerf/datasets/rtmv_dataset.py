# -*- coding: utf-8 -*-

import glob
import json
import os.path as osp

import cv2
import numpy as np

from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .base_3d_dataset import Base3dDataset


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
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # get cameras
        self.cameras = self.read_cameras()

        for cam in self.cameras:
            cam.set_device(self.device)

        # norm camera_pose to restrict pc range
        self.norm_cam_pose()

        # to make fair comparison, remove test file from train
        holdout_index = self.get_holdout_index()
        img_list, mask_list = self.get_holdout_samples_with_list(holdout_index, img_list, mask_list)

        # skip image and keep less samples
        img_list, mask_list = self.skip_samples_with_list(img_list, mask_list)
        # read the real image after skip
        self.images = self.read_image_list(img_list)
        self.masks = self.read_mask_list(mask_list)
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
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # already float32
            img /= img.max()  # restrict in (0, 1), Need to check with origin repo
            images.append(img)

        return images

    @staticmethod
    def read_mask_list(mask_list):
        """Read mask from list"""
        masks = []
        for path in mask_list:
            mask = cv2.imread(path)[..., 0].astype(np.float32)
            mask[mask > 0.0] = 1.0
            masks.append(mask)

        return masks

    def read_cameras(self):
        """Read camera from pose files"""
        cam_files = sorted(glob.glob(self.data_spec_dir + '/*.json'))
        assert len(cam_files) == self.n_imgs, 'Num of image not match num of cam...'

        cameras = []
        for file in cam_files:
            with open(file, 'r') as f:
                cam_info = json.load(f)
                c2w = cam_info['camera_data']['cam2world']
                intrinsic = cam_info['camera_data']['intrinsics']
                c2w = np.array(c2w).transpose((1, 0))
                # correct the poses in our system
                c2w = c2w[:, [1, 0, 2, 3]]
                c2w[:, 2] *= -1.0
                c2w = c2w[[0, 2, 1, 3], :]
                c2w[1, :] *= -1

                intrinsic = self.get_intrinsic_from_file(intrinsic)

                cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2w, W=self.W, H=self.H))

        return cameras

    def get_intrinsic_from_file(self, info):
        """Get intrinsic (3, 3) from hwf"""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = info['fx']  # approximate focal
        intrinsic[1, 1] = info['fy']
        intrinsic[0, 2] = info['cx']
        intrinsic[1, 2] = info['cy']

        return intrinsic

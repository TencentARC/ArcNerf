# -*- coding: utf-8 -*-

import glob
import json
import os.path as osp
import re

import cv2
import numpy as np

from .base_3d_dataset import Base3dDataset
from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NeRF(Base3dDataset):
    """Nerf synthetic dataset introduced in the original paper.
    Ref: https://github.com/bmild/nerf
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(NeRF, self).__init__(cfgs, data_dir, mode, transforms)

        # nerf dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'NeRF', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # read image in the split
        img_list, self.n_imgs = self.get_image_list(mode)
        self.images, self.masks = self.read_image_list(img_list, mode)
        self.H, self.W = self.images[0].shape[:2]

        # load all camera together in all split for consistent camera normalization
        self.cam_file = osp.join(self.data_spec_dir, 'transforms_{}.json'.format(self.convert_mode(mode)))
        assert osp.exists(self.cam_file), 'Camera file {} not exist...'.format(self.cam_file)
        self.cameras, cam_split_idx = self.read_cameras_by_mode(mode)  # get the index for final selection
        for cam in self.cameras:
            cam.set_device(self.device)

        # handle the camera in all split to make consistent
        # norm camera_pose
        self.norm_cam_pose()
        # align if required
        self.align_cam_horizontal()

        # keep only the camera in certain split
        self.cameras = [self.cameras[idx] for idx in cam_split_idx]
        assert self.n_imgs == len(self.cameras), 'Camera num not match the image number'

        # skip image and keep less samples
        self.skip_samples()
        # keep close-to-mean samples if set
        self.keep_eval_samples()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = get_value_from_cfgs_field(self.cfgs, 'precache', False)

        if self.precache:
            self.precache_ray()

    @staticmethod
    def convert_mode(mode):
        """Convert mode train/val/eval to dataset name"""
        if mode == 'train' or mode == 'val':
            return mode
        elif mode == 'eval':
            return 'test'
        else:
            raise NotImplementedError('Not such mode {}...'.format(mode))

    def get_image_list(self, mode):
        """Get image list"""
        img_dir = osp.join(self.data_spec_dir, self.convert_mode(mode))
        img_list = glob.glob(img_dir + '/r_*.png')
        img_list = [f for f in img_list if re.search('r_[0-9]{1,}.png', f)]

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        # sort by name
        img_list = [osp.join(img_dir, 'r_{}.png'.format(i)) for i in range(n_imgs)]

        return img_list, n_imgs

    def get_image_list_with_key(self, mode, key='depth'):
        """Get the images for depth/normal"""
        assert mode == 'eval', 'only provide in eval mode'

        img_dir = osp.join(self.data_spec_dir, self.convert_mode(mode))
        img_list = glob.glob(img_dir + '/r_*.png')
        img_list = [f for f in img_list if key in f]

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No {} image exists in {}'.format(key, img_dir)

        # sort by name
        img_list = [osp.join(img_dir, 'r_{}_{}_0001.png'.format(i, key)) for i in range(n_imgs)]

        return img_list, n_imgs

    @staticmethod
    def read_image_list(img_list, mode):
        """Read image from list. Original bkg is black, change it to white"""
        images, masks = [], []
        for path in img_list:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32) / 255.0  # rgba
            mask = img[:, :, -1]
            img = img[..., :3]

            images.append(img)
            masks.append(mask)

        return images, masks

    def load_cam_json(self, mode):
        """Load the camera json file in any split"""
        json_file = osp.join(self.data_spec_dir, 'transforms_{}.json'.format(self.convert_mode(mode)))
        assert osp.exists(json_file), 'Camera file {} not exist...'.format(json_file)

        with open(json_file, 'r') as f:
            cam_file = json.load(f)

        return cam_file

    def read_cameras_by_mode(self, mode):
        """Read in all the camera file and keep the index of split"""
        # read cam on all split
        all_mode = ['train', 'val', 'eval']
        cam_json = {}
        idx = [[-1]]
        for i, m in enumerate(all_mode):
            cam_json[m] = self.load_cam_json(m)
            last_idx = idx[i][-1] + 1
            idx.append(list(range(last_idx, last_idx + len(cam_json[m]['frames']))))

        split_idx = idx[all_mode.index(mode) + 1]

        # concat all camera
        cameras = []
        for m in all_mode:
            for cam_idx in range(len(cam_json[m]['frames'])):
                poses = np.array(cam_json[m]['frames'][cam_idx]['transform_matrix']).astype(np.float32)  # (4, 4)
                # correct the poses in our system
                poses[:, 1:3] *= -1.0
                poses = poses[[0, 2, 1, 3], :]
                poses[1, :] *= -1

                cameras.append(
                    PerspectiveCamera(
                        intrinsic=self.get_intrinsic_by_angle(float(cam_json[m]['camera_angle_x'])),
                        c2w=poses,
                        W=self.W,
                        H=self.H
                    )
                )

        return cameras, split_idx

    def get_intrinsic_by_angle(self, camera_angle_x):
        """Get the (3, 3) intrinsic"""
        focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = float(self.W) / 2.0
        intrinsic[1, 2] = float(self.H) / 2.0

        return intrinsic

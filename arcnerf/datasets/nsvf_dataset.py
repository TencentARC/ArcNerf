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
class NSVF(Base3dDataset):
    """NSVF synthetic dataset introduced in the NSVF paper.
    Ref: https://lingjie0206.github.io/papers/NSVF/
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(NSVF, self).__init__(cfgs, data_dir, mode, transforms)

        # nsvf dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'NSVF', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # read image in the split
        img_list, self.n_imgs = self.get_image_list(mode)
        self.images, self.masks = self.read_image_list(img_list)
        self.H, self.W = self.images[0].shape[:2]

        # load all camera together in all split for consistent camera normalization
        self.cam_folder = osp.join(self.data_spec_dir, 'pose')
        assert osp.exists(self.cam_folder), 'Camera folder {} not exist...'.format(self.cam_folder)
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
        if mode == 'train':
            return 0, mode
        elif mode == 'val':
            return 1, mode
        elif mode == 'eval':
            return 2, 'test'
        else:
            raise NotImplementedError('Not such mode {}...'.format(mode))

    def get_image_list(self, mode):
        """Get image list"""
        img_dir = osp.join(self.data_spec_dir, 'rgb')
        split_id, split_mode = self.convert_mode(mode)
        img_list = glob.glob(img_dir + '/{}_cam_{}_*.png'.format(split_id, split_mode))
        img_list = sorted(img_list)

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    @staticmethod
    def read_image_list(img_list):
        """Read image from list. Original bkg is black, change it to white"""
        images, masks = [], []
        for path in img_list:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32) / 255.0  # rgba
            mask = img[:, :, -1]
            img = img[..., :3]

            images.append(img)
            masks.append(mask)

        return images, masks

    def load_cam_files(self, mode):
        """Load the camera file in any split"""
        split_id, split_mode = self.convert_mode(mode)
        cam_files = glob.glob(self.cam_folder + '/{}_cam_{}_*.txt'.format(split_id, split_mode))
        cam_files = sorted(cam_files)

        c2ws = []
        for cam_file in cam_files:
            with open(cam_file, 'r') as f:
                c2w = []
                for line in f.readlines():
                    c2w.append(line.strip().split())
            c2w = np.array(c2w, dtype=np.float32)
            c2ws.append(c2w)

        return c2ws

    def read_cameras_by_mode(self, mode):
        """Read in all the camera file and keep the index of split"""
        # read cam on all split
        all_mode = ['train', 'val', 'eval']
        c2ws = {}
        idx = [[-1]]
        for i, m in enumerate(all_mode):
            c2ws[m] = self.load_cam_files(m)
            last_idx = idx[i][-1] + 1
            idx.append(list(range(last_idx, last_idx + len(c2ws[m]))))

        split_idx = idx[all_mode.index(mode) + 1]

        # concat all camera
        cameras = []
        for m in all_mode:
            for poses in c2ws[m]:
                # correct the poses in our system
                poses = poses[[0, 2, 1, 3], :]
                poses[1, :] *= -1

                cameras.append(PerspectiveCamera(intrinsic=self.read_intrinsic(), c2w=poses, W=self.W, H=self.H))

        return cameras, split_idx

    def read_intrinsic(self):
        """Get the (3, 3) intrinsic"""
        intrinsic = osp.join(self.data_spec_dir, 'intrinsics.txt')
        with open(intrinsic, 'r') as file:
            focal, cx, cy, _ = map(float, file.readline().split())

        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy

        return intrinsic

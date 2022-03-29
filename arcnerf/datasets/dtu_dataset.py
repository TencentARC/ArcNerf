# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from .base_3d_dataset import Base3dDataset
from ..datasets import DATASET_REGISTRY
from arcnerf.render.camera import load_K_Rt_from_P, PerspectiveCamera
from common.utils.img_utils import read_img


@DATASET_REGISTRY.register()
class DTU(Base3dDataset):
    """DTU dataset
    Refer: https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(Base3dDataset, self).__init__(cfgs, data_dir, mode, transforms)

        # real DTU dataset with scan_id
        self.data_spec_dir = osp.join(self.data_dir, 'DTU', 'scan{}'.format(self.cfgs.scan_id))

        # get image and mask
        img_list, mask_list, self.n_imgs = self.get_image_mask_list()
        self.images, self.masks = self.read_image_mask(img_list, mask_list)
        self.H, self.W = self.images[0].shape[:2]

        # get cameras
        cam_file = osp.join(self.data_spec_dir, 'cameras.npz')
        assert osp.exists(cam_file), 'Camera file {} not exist...'.format(cam_file)
        self.cameras = self.read_cameras(cam_file)

        # norm camera_pose
        self.norm_cam_pose()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # precache_all rays
        self.ray_bundles = None
        self.precache = self.cfgs.precache

        if self.precache:
            self.precache_ray()

    def get_image_mask_list(self):
        """Get image and mask list"""
        img_dir = osp.join(self.data_spec_dir, 'image')
        mask_dir = osp.join(self.data_spec_dir, 'mask')
        img_list = sorted(glob.glob(img_dir + '/*.png'))
        mask_list = sorted(glob.glob(mask_dir + '/*.png'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, mask_list, n_imgs

    @staticmethod
    def read_image_mask(img_list, mask_list):
        """Read image and mask from list"""

        images = [read_img(path, norm_by_255=True) for path in img_list]
        masks = [read_img(path, norm_by_255=True, gray=True) for path in mask_list]

        for i in range(len(masks)):
            masks[i][masks[i] > 0.5] = 1.0

        return images, masks

    def read_cameras(self, cam_file):
        """Get cameras with pose and intrinsic from cam_file.npz. Detail information is here
        https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/FAQ.md

        In DTU, camera mat is [2/w, 0, -1,
                               0, 2/h, -1;
                               0, 0, 1], which transfer point in range (w,h) to (-1, 1).
        Scale_mat and world_mats transfer the image into correct image_plane within range (w, h)
        """
        cam_dict = np.load(cam_file)
        scale_mats = [cam_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]
        world_mats = [cam_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]

        cameras = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            proj_mat = world_mat @ scale_mat
            proj_mat = proj_mat[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(proj_mat)  # (4,4), (4,4)
            cameras.append(PerspectiveCamera(intrinsic=intrinsic[:3, :3], c2w=pose, H=self.H, W=self.W))

        return cameras

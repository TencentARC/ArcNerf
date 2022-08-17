# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from .base_3d_dataset import Base3dDataset
from arcnerf.render.camera import load_K_Rt_from_P, PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DTU(Base3dDataset):
    """DTU dataset
    Refer: https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(DTU, self).__init__(cfgs, data_dir, mode, transforms)

        # real DTU dataset with scan_id
        self.data_spec_dir = osp.join(self.data_dir, 'DTU', 'dtu_scan{}'.format(self.cfgs.scan_id))
        self.identifier = str(self.cfgs.scan_id)

        # get image and mask
        img_list, self.n_imgs = self.get_image_list()
        mask_list = self.get_mask_list()
        self.images = self.read_image_list(img_list)
        self.masks = self.read_mask_list(mask_list)
        self.H, self.W = self.images[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'cameras_sphere.npz')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...'.format(self.cam_file)
        self.cameras = self.read_cameras()
        for cam in self.cameras:
            cam.set_device(self.device)

        # norm camera_pose
        self.norm_cam_pose()
        # align if required
        self.align_cam_horizontal()

        # to make fair comparison, remove test file from train
        holdout_index = self.get_holdout_index()
        self.get_holdout_samples(holdout_index)
        # skip samples
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

    def get_image_list(self, mode=None):
        """Get image list."""
        img_dir = osp.join(self.data_spec_dir, 'image')
        img_list = sorted(glob.glob(img_dir + '/*.png'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def get_mask_list(self, mode=None):
        """Get image list."""
        mask_dir = osp.join(self.data_spec_dir, 'mask')
        mask_list = sorted(glob.glob(mask_dir + '/*.png'))

        return mask_list

    def read_cameras(self):
        """Get cameras with pose and intrinsic from cam_file.npz. Detail information is here
        https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/FAQ.md

        In DTU, camera mat is [2/w, 0, -1,
                               0, 2/h, -1;
                               0, 0, 1], which transfer point in range (w,h) to (-1, 1).
        Scale_mat and world_mats transfer the image into correct image_plane within range (w, h)
        """
        cam_dict = np.load(self.cam_file)
        scale_mats = [cam_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]
        world_mats = [cam_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_imgs)]

        cameras = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            proj_mat = world_mat @ scale_mat
            proj_mat = proj_mat[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(proj_mat)  # (4,4), (4,4)
            cameras.append(PerspectiveCamera(intrinsic=intrinsic[:3, :3], c2w=pose, W=self.W, H=self.H))

        return cameras

# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from arcnerf.render.camera import load_K_Rt_from_P, PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .base_3d_dataset import Base3dDataset


@DATASET_REGISTRY.register()
class BlendedMVS(Base3dDataset):
    """BlendedMVS Dataset.
    Ref: https://github.com/YoYo000/BlendedMVS and https://lioryariv.github.io/volsdf/
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(BlendedMVS, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'BlendedMVS', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, self.n_imgs = self.get_image_list(mode)
        self.images = self.read_image_list(img_list)
        self.H, self.W = self.images[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'cameras.npz')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...'.format(self.cam_file)
        self.cameras = self.read_cameras()
        for cam in self.cameras:
            cam.set_device(self.device)

        # norm camera_pose
        self.norm_cam_pose()
        # align if required
        self.align_cam_horizontal()

        # exchange pose coord
        self.exchange_coord()

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
        img_list = sorted(glob.glob(img_dir + '/*.jpg'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def read_cameras(self):
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

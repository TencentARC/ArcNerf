# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from simplengp.geometry.poses import average_poses
from simplengp.render.camera import PerspectiveCamera
from .base_3d_dataset import Base3dDataset


@DATASET_REGISTRY.register()
class MipNeRF360(Base3dDataset):
    """MipNeRF Dataset. Use colmap to process but do not save pointcloud.
    Ref: https://jonbarron.info/mipnerf360/
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(MipNeRF360, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'MipNeRF360', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, self.n_imgs = self.get_image_list(mode)
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'poses_bounds.npy')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.poses = np.load(self.cam_file, allow_pickle=True)  # (N_imgs, 17)
        self.cameras, self.bounds = self.read_cameras()

        for cam in self.cameras:
            cam.set_device(self.device)

        # roughly center the cameras by common view point
        self.center_cam_poses_by_view_dirs()
        # norm camera_pose to restrict pc range
        self.norm_cam_pose()
        # align if required
        self.align_cam_horizontal()

        # align to ngp coord
        self.adjust_cam_translation()

        # to make fair comparison, remove test file from train
        holdout_index = self.get_holdout_index()
        img_list, _ = self.get_holdout_samples_with_list(holdout_index, img_list)

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
        """Get image list."""
        img_dir = osp.join(self.data_spec_dir, 'images')
        img_list = sorted(glob.glob(img_dir + '/*.JPG'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def read_cameras(self):
        """Read camera from pose file"""
        poses = self.poses[:, :-2].reshape(-1, 3, 5)  # (N, 3, 5)
        hwf = poses[0, :, -1]  # (3)
        intrinsic = self.get_mipnerf_intrinsic(hwf)
        c2w = poses[:, :, :4]  # (N, 3, 4)
        bottom = np.repeat(np.array([0, 0, 0, 1.]).reshape([1, 4])[None, ...], c2w.shape[0], axis=0)  # (N, 1, 4)
        c2w = np.concatenate([c2w, bottom], axis=1)  # (N, 4, 4)
        # correct the pose in our system
        c2w = c2w[:, :, [1, 0, 2, 3]]
        c2w[:, :, 2] *= -1.0
        c2w = self.center_pose(c2w)
        c2w = c2w[:, [0, 2, 1, 3], :]

        # bounds
        bounds = self.poses[:, -2:]  # (N, 2)

        cameras = []
        for idx in range(self.n_imgs):
            cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2w[idx], W=self.W, H=self.H))

        return cameras, bounds

    @staticmethod
    def get_mipnerf_intrinsic(hwf):
        """Get intrinsic (3, 3) from hwf"""
        intrinsic = np.eye(3)
        intrinsic[0, 0] = hwf[2]
        intrinsic[1, 1] = hwf[2]
        intrinsic[0, 2] = hwf[1] / 2.0
        intrinsic[1, 2] = hwf[0] / 2.0

        return intrinsic

    @staticmethod
    def center_pose(c2w):
        """Center the pose of (N, 4, 4) """
        c2w_avg = average_poses(c2w)  # (4, 4)
        c2w = np.linalg.inv(c2w_avg) @ c2w  # (N, 4, 4)

        return c2w

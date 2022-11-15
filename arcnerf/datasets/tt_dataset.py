# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .base_3d_dataset import Base3dDataset


@DATASET_REGISTRY.register()
class TanksAndTemples(Base3dDataset):
    """TanksAndTemples Dataset. Use colmap to process but do not save pointcloud.
    The poses are not that accurate which needs your optimization.
    Ref: https://www.tanksandtemples.org/
    """

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(TanksAndTemples, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'TanksAndTemples', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image
        img_list, self.n_imgs = self.get_image_list(mode)
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'Truck_COLMAP_SfM.log')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.cameras = self.read_cameras()

        for cam in self.cameras:
            cam.set_device(self.device)

        # norm camera_pose to restrict pc range
        self.norm_cam_pose()

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
        img_list = sorted(glob.glob(img_dir + '/*.jpg'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

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
        """Get intrinsic (3, 3) from hwf
        TanksAndTemplates do not provide exact focal, the num is estimated and affects the result.
        """
        intrinsic = np.eye(3)
        intrinsic[0, 0] = 0.59365 * self.W  # approximate focal
        intrinsic[1, 1] = 0.59365 * self.W
        intrinsic[0, 2] = self.W / 2.0
        intrinsic[1, 2] = self.H / 2.0

        return intrinsic

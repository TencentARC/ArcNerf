# -*- coding: utf-8 -*-

import glob
import os.path as osp

import numpy as np

from arcnerf.geometry.poses import invert_poses
from arcnerf.render.camera import PerspectiveCamera
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import DATASET_REGISTRY
from .base_3d_pc_dataset import Base3dPCDataset


@DATASET_REGISTRY.register()
class Capture(Base3dPCDataset):
    """A dataset class for self-capture images with colmap pose estimation"""

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(Capture, self).__init__(cfgs, data_dir, mode, transforms)

        # real capture dataset with scene_name
        self.data_spec_dir = osp.join(self.data_dir, 'Capture', self.cfgs.scene_name)
        self.identifier = self.cfgs.scene_name

        # get image, read image list first, then read skip images in case too many images.
        img_list, self.n_imgs = self.get_image_list()
        mask_list, n_mask = self.get_mask_list()
        if mask_list is not None:
            assert self.n_imgs == n_mask, 'Num of image not match num of mask {}-vs-{}'.format(self.n_imgs, n_mask)
        self.H, self.W = self.read_image_list(img_list[:1])[0].shape[:2]

        # get cameras
        self.cam_file = osp.join(self.data_spec_dir, 'poses_bounds.npy')
        assert osp.exists(self.cam_file), 'Camera file {} not exist...Please run colmap first...'.format(self.cam_file)
        self.poses = np.load(self.cam_file, allow_pickle=True).item()
        self.cameras = self.read_cameras()
        for cam in self.cameras:
            cam.set_device(self.device)

        # get pointcloud
        self.point_cloud = self.get_sparse_point_cloud()

        # roughly center the cameras by common view point
        self.center_cam_poses_by_view_dirs()
        # norm camera_pose to restrict pc range
        self.norm_cam_pose()
        # filter point outside sphere
        self.filter_point_cloud()
        # recenter the cameras by remaining point cloud
        self.center_cam_poses_by_pc_mean()
        # re-norm again
        self.norm_cam_pose()
        # align if required
        self.align_cam_horizontal()
        # adjust translation if necessary to put obj in the center
        self.adjust_cam_translation()

        # to make fair comparison, remove test file from train
        holdout_index = self.get_holdout_index()
        img_list, mask_list = self.get_holdout_samples_with_list(holdout_index, img_list, mask_list)

        # skip image and keep less samples
        img_list, mask_list = self.skip_samples_with_list(img_list, mask_list)
        # read the real image after skip
        self.images = self.read_image_list(img_list)
        # read mask, it is optional
        self.masks = self.read_mask_list(mask_list) if mask_list is not None else []
        # keep close-to-mean samples if set
        self.keep_eval_samples()

        # rescale image, call from parent class
        self.rescale_img_and_pose()

        # get bounds
        self.bounds = self.get_bounds_from_pc()

        # precache_all rays
        self.ray_bundles = None
        self.precache = get_value_from_cfgs_field(self.cfgs, 'precache', False)

        if self.precache:
            self.precache_ray()

    def get_image_list(self):
        """Get image list."""
        img_dir = osp.join(self.data_spec_dir, 'images')
        img_list = sorted(glob.glob(img_dir + '/*.png'))

        n_imgs = len(img_list)
        assert n_imgs > 0, 'No image exists in {}'.format(img_dir)

        return img_list, n_imgs

    def get_mask_list(self):
        """Get mask list. Return None if mask list is empty"""
        mask_dir = osp.join(self.data_spec_dir, 'mask')
        mask_list = sorted(glob.glob(mask_dir + '/*.png'))

        n_imgs = len(mask_list)
        if n_imgs == 0:
            return None, None

        return mask_list, n_imgs

    def read_cameras(self):
        """Read camera from pose file"""
        assert self.poses['h'] == self.H and self.poses['w'] == self.W,\
            'Cam poses not match image size...image {}/{} - cam {}/{}'.format(self.W, self.H,
                                                                              self.poses['w'], self.poses['h'])

        w2c = np.concatenate([self.poses['R'], self.poses['T']], axis=-1)  # (N, 3, 4)
        bottom = np.repeat(np.array([0, 0, 0, 1.]).reshape([1, 4])[None, ...], w2c.shape[0], axis=0)  # (N, 1, 4)
        w2c = np.concatenate([w2c, bottom], axis=1)  # (N, 4, 4)
        c2w = invert_poses(w2c)
        intrinsic = self.get_colmap_intrinsic()

        cameras = []
        for idx in range(self.n_imgs):  # read only first n_imgs
            cameras.append(PerspectiveCamera(intrinsic=intrinsic, c2w=c2w[idx], W=self.W, H=self.H))

        return cameras

    def get_colmap_intrinsic(self):
        """Get intrinsic (3, 3) from pose file"""
        cam_type = self.poses['cam_type']
        cam_params = self.poses['cam_params']
        if cam_type == 'SIMPLE_RADIAL':  # f, cx, cy, k. Ignore k for simplicity
            intrinsic = np.eye(3)
            intrinsic[0, 0] = cam_params[0]
            intrinsic[1, 1] = cam_params[0]
            intrinsic[0, 2] = cam_params[1]
            intrinsic[1, 2] = cam_params[2]
        else:
            raise NotImplementedError('Not support cam mode {} from colmap reconstruction yet...'.format(cam_type))

        return intrinsic

    def get_sparse_point_cloud(self, dtype=np.float32):
        """Get sparse point cloud as the point cloud. color should be normed in (0,1)"""
        pc = {
            'pts': self.poses['pts'].astype(dtype),
            'color': self.poses['rgb'].astype(dtype) / 255.0,
            'vis': self.poses['vis'][:self.n_imgs].astype(dtype)
        }

        return pc
